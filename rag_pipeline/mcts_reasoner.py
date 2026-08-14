# RAG Pipeline - General MCTS Reasoning Engine
"""
General Monte Carlo Tree Search reasoning engine for legal RAG.

MCTS as the algorithm, not a feature. Unlike the classification-specific
LegalReasoningAgent (hardcoded Sagaz factors), this engine reasons over
ANY legal question:

1. SELECT:   UCB1 over reasoning states
2. EXPAND:   LLM decomposes the question into issues / sub-questions
3. SIMULATE: retrieve evidence, score with cheap proxies first,
             judge LLM only when proxies are ambiguous
4. BACKPROP: standard reward propagation

Design:
- ReasoningState is general (question, issue, evidence, partial_answer)
- Expansion is LLM-driven, not factor-hardcoded
- Value estimation is tiered: cheap heuristic -> judge LLM
- Simulation budget adapts to question complexity
- Dependencies injected (rag_query, judge) for testability
"""

import sys
import math
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MCTS_N_SIMULATIONS,
    MCTS_EXPLORATION_CONSTANT,
    MCTS_MAX_DEPTH,
    MCTS_MIN_SCORE_THRESHOLD,
    MCTS_REWARD_WEIGHTS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robust JSON extraction from LLM output
# ---------------------------------------------------------------------------

def _extract_first_json(text: str) -> Optional[List[Any]]:
    """Find the first '[' ... ']' span and try to parse it.

    LLM output frequently wraps JSON in prose or truncates mid-array.
    Returns a list on success, None otherwise.
    """
    start = text.find("[")
    if start == -1:
        return None
    # Walk backwards from the end to the last ']' that still leaves a
    # balanced-enough payload.
    for end in range(len(text) - 1, start, -1):
        if text[end] != "]":
            continue
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """Find the first '{' ... '}' span and try to parse it as an object."""
    start = text.find("{")
    if start == -1:
        return None
    for end in range(len(text) - 1, start, -1):
        if text[end] != "}":
            continue
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _repair_truncated_obj(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to salvage a truncated JSON object (missing closing brace)."""
    stripped = text.strip()
    for brace in range(len(stripped) - 1, -1, -1):
        if stripped[brace] == "{":
            candidate = stripped[brace:]
            candidate = candidate.rstrip().rstrip(",")
            if candidate.count('"') % 2 == 1:
                candidate += '"'
            candidate += "}"
            for trial in (
                candidate,
                candidate.replace("\n", " "),
            ):
                try:
                    parsed = json.loads(trial)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    continue
            break
    return None


def _extract_json_obj(response: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from an LLM response.

    Handles markdown fences and surrounding prose. Returns None when
    nothing valid can be parsed.
    """
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n", 1)
        if len(lines) > 1:
            cleaned = lines[1]
        idx = cleaned.rfind("```")
        if idx != -1:
            cleaned = cleaned[:idx]
        cleaned = cleaned.strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    parsed = _extract_first_json_obj(cleaned)
    if parsed is not None:
        return parsed
    return _repair_truncated_obj(cleaned)


def _repair_truncated(text: str) -> Optional[List[Any]]:
    """Attempt to salvage a truncated JSON array (missing closing bracket)."""
    stripped = text.strip()
    for backtick in range(len(stripped) - 1, -1, -1):
        if stripped[backtick] == "[":
            candidate = stripped[backtick:]
            candidate = candidate.rstrip().rstrip(",")
            # If the array was cut off inside a string literal, close it.
            if candidate.count('"') % 2 == 1:
                candidate += '"'
            candidate += "]"
            for trial in (
                candidate,
                candidate.replace("\n", " "),
            ):
                try:
                    parsed = json.loads(trial)
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    continue
            break
    return None


def extract_json_array(response: str) -> List[Any]:
    """Best-effort extraction of a JSON array from an LLM response.

    Handles markdown fences, surrounding prose, and truncated arrays.
    Returns an empty list when nothing valid can be parsed (callers treat
    that as a soft failure and fall back to their own defaults).
    """
    cleaned = response.strip()
    if cleaned.startswith("```"):
        # Strip fenced code block
        lines = cleaned.split("\n", 1)
        if len(lines) > 1:
            cleaned = lines[1]
        # Remove trailing fence if present
        idx = cleaned.rfind("```")
        if idx != -1:
            cleaned = cleaned[:idx]
        cleaned = cleaned.strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    parsed = _extract_first_json(cleaned)
    if parsed is not None:
        return parsed

    parsed = _repair_truncated(cleaned)
    if parsed is not None:
        return parsed

    logger.warning(
        "Could not extract JSON array from LLM output (%d chars): %r",
        len(response), response[:120],
    )
    return []


# ---------------------------------------------------------------------------
# Reasoning State
# ---------------------------------------------------------------------------

@dataclass
class ReasoningState:
    """A node in the MCTS reasoning tree.

    Represents a legal reasoning state: a sub-question, the issue it
    addresses, evidence gathered, and a partial answer.
    """
    id: str
    question: str
    issue: str = ""
    evidence: List[str] = field(default_factory=list)
    partial_answer: str = ""
    depth: int = 0
    parent_id: Optional[str] = None
    visits: int = 0
    total_reward: float = 0.0
    children: List['ReasoningState'] = field(default_factory=list)

    @property
    def average_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb1(self, parent_visits: int, exploration_constant: float = MCTS_EXPLORATION_CONSTANT) -> float:
        """UCB1 selection criterion: exploitation + exploration."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.average_reward
        exploration = exploration_constant * math.sqrt(
            math.log(max(parent_visits, 1)) / self.visits
        )
        return exploitation + exploration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "issue": self.issue,
            "evidence_count": len(self.evidence),
            "evidence": self.evidence[:3],
            "partial_answer": self.partial_answer[:200],
            "depth": self.depth,
            "visits": self.visits,
            "average_reward": round(self.average_reward, 4),
        }


@dataclass
class ReasoningResult:
    """Final result of MCTS reasoning."""
    question: str
    answer: str
    confidence: float
    best_path: List[Dict[str, Any]]
    tree_statistics: Dict[str, Any]
    duration_ms: float
    status: str = "ok"
    error_type: Optional[str] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Value estimation helpers (pure functions)
# ---------------------------------------------------------------------------

def evidence_heuristic(evidence: List[str], question: str) -> float:
    """Cheap value proxy: score evidence without an LLM call.

    Combines coverage (how many question terms appear in evidence) with
    volume. Returns 0.0-1.0. Used to skip the judge when evidence is
    clearly strong or clearly weak.
    """
    if not evidence:
        return 0.0
    terms = [t for t in question.lower().split() if len(t) > 3]
    if not terms:
        return min(1.0, len(evidence) / 3.0)
    covered = sum(
        1 for t in terms
        if any(t in e.lower() for e in evidence)
    )
    coverage = covered / len(terms)
    volume = min(1.0, len(evidence) / 3.0)
    return 0.7 * coverage + 0.3 * volume


def parse_judge_scores(response: str) -> Optional[Dict[str, float]]:
    """Parse judge LLM JSON output into a score dict. Pure function."""
    scores = _extract_json_obj(response)
    if scores is None:
        return None
    return {
        k: float(scores.get(k, 0.5))
        for k in ("precedent_alignment", "factor_completeness",
                  "logical_consistency", "evidence_strength")
    }


def weighted_reward(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Combine judge dimension scores into a single reward. Pure function."""
    return sum(
        weights.get(k, 0.25) * scores.get(k, 0.5)
        for k in ("precedent_alignment", "factor_completeness",
                  "logical_consistency", "evidence_strength")
    )


# ---------------------------------------------------------------------------
# MCTS Reasoner
# ---------------------------------------------------------------------------

class MCTSReasoner:
    """General MCTS reasoning engine over any legal question.

    Dependencies injected: rag_query (LegalRAGQuery), judge (any chat
    with .generate — MultiModelChat recommended).
    """

    def __init__(
        self,
        rag_query,
        judge,
        n_simulations: int = MCTS_N_SIMULATIONS,
        exploration_constant: float = MCTS_EXPLORATION_CONSTANT,
        max_depth: int = MCTS_MAX_DEPTH,
        min_score_threshold: float = MCTS_MIN_SCORE_THRESHOLD,
        judge_threshold: float = 0.35,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
    ):
        self.rag = rag_query
        self.judge = judge
        self.n_simulations = n_simulations
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.min_score_threshold = min_score_threshold
        self.judge_threshold = judge_threshold
        self.namespace = namespace
        self.filter = filter

        self._states: Dict[str, ReasoningState] = {}
        self._counter = 0

    # -- Public API --------------------------------------------------------

    def reason(self, question: str, n_simulations: Optional[int] = None) -> ReasoningResult:
        """Run MCTS reasoning over a question. Returns best-path answer."""
        start = datetime.now()
        n_sims = n_simulations or self.n_simulations

        self._states = {}
        self._counter = 0

        root = self._new_state(question, issue="root", depth=0)
        self._states[root.id] = root

        # EXPAND root: LLM decomposes the question into issues
        issues = self._decompose_question(question)
        for issue in issues[:5]:
            child = self._new_state(
                question=issue,
                issue=issue,
                depth=1,
                parent_id=root.id,
            )
            self._states[child.id] = child
            root.children.append(child)

        # MCTS loop
        for sim in range(n_sims):
            selected = self._select(root)
            if selected.depth < self.max_depth and selected.visits > 0:
                self._expand(selected, question)
                if selected.children:
                    selected = selected.children[0]

            reward = self._simulate(selected, question)
            self._backpropagate(selected, reward)

            if self._should_stop_early(root):
                logger.info(f"MCTS early stop at simulation {sim + 1}")
                break

        best_path = self._best_path(root)
        answer = self._synthesize(question, best_path)
        confidence = self._path_confidence(best_path)

        duration_ms = (datetime.now() - start).total_seconds() * 1000

        return ReasoningResult(
            question=question,
            answer=answer,
            confidence=round(confidence, 3),
            best_path=[s.to_dict() for s in best_path],
            tree_statistics={
                "total_states": len(self._states),
                "total_simulations": sim + 1,
                "max_depth_reached": max(s.depth for s in self._states.values()),
                "root_visits": root.visits,
            },
            duration_ms=round(duration_ms, 1),
        )

    # -- Tree construction -------------------------------------------------

    def _new_state(self, question: str, issue: str, depth: int,
                   parent_id: Optional[str] = None) -> ReasoningState:
        self._counter += 1
        return ReasoningState(
            id=f"state_{self._counter}",
            question=question,
            issue=issue,
            depth=depth,
            parent_id=parent_id,
        )

    def _decompose_question(self, question: str) -> List[str]:
        """LLM-driven expansion: break a question into legal issues."""
        prompt = (
            "You are a legal research planner. Break the following legal question "
            "into the 3-5 key sub-questions a lawyer would need to answer to "
            "resolve it fully.\n\n"
            f"QUESTION: {question}\n\n"
            "Respond with ONLY a JSON array of strings, e.g. "
            '["sub-question 1", "sub-question 2"].'
        )
        try:
            response = self.judge.generate(prompt, temperature=0.2, max_tokens=512)
            issues = extract_json_array(response)
            filtered = [str(i) for i in issues if str(i).strip()]
            if filtered:
                return filtered
        except Exception as e:
            logger.warning(f"Question decomposition failed: {e}")
        return [question]

    def _expand(self, state: ReasoningState, question: str):
        """Generate child states: refine the current sub-question."""
        prompt = (
            "You are a legal research planner. Given a legal question and a "
            "sub-question already being explored, propose 2-3 more specific "
            "sub-questions that would deepen the analysis.\n\n"
            f"MAIN QUESTION: {question}\n"
            f"CURRENT SUB-QUESTION: {state.question}\n\n"
            "Respond with ONLY a JSON array of strings."
        )
        try:
            response = self.judge.generate(prompt, temperature=0.2, max_tokens=512)
            children = extract_json_array(response)
            for c in children[:3]:
                child = self._new_state(
                    question=str(c),
                    issue=state.issue,
                    depth=state.depth + 1,
                    parent_id=state.id,
                )
                self._states[child.id] = child
                state.children.append(child)
        except Exception as e:
            logger.warning(f"Expansion failed for {state.id}: {e}")

    # -- MCTS core ---------------------------------------------------------

    def _select(self, root: ReasoningState) -> ReasoningState:
        current = root
        while current.children:
            best = max(
                current.children,
                key=lambda c: c.ucb1(current.visits, self.exploration_constant),
            )
            current = best
        return current

    def _simulate(self, state: ReasoningState, question: str) -> float:
        """Score a state: retrieve evidence, cheap proxy, judge if needed."""
        # Retrieve evidence for this sub-question
        try:
            rag_response = self.rag.query(
                state.question,
                top_k=3,
                include_analysis=False,
                namespace=self.namespace,
                filter=self.filter,
                user_id=None,
            )
            state.evidence = [s.get("excerpt", "") for s in rag_response.sources]
        except Exception as e:
            logger.warning(f"Retrieval failed for {state.id}: {e}")
            state.evidence = []
            return 0.2

        # Tier 1: cheap heuristic
        heuristic = evidence_heuristic(state.evidence, state.question)
        if heuristic >= 0.7:
            state.partial_answer = "Evidence strongly supports this sub-question."
            return heuristic
        if heuristic <= self.judge_threshold:
            state.partial_answer = "Insufficient evidence for this sub-question."
            return heuristic

        # Tier 2: judge LLM
        reward = self._judge_state(state, question)
        return reward

    def _judge_state(self, state: ReasoningState, question: str) -> float:
        """Judge LLM scores a single state on 4 dimensions."""
        evidence_text = "\n".join(state.evidence[:3]) if state.evidence else "No evidence retrieved."
        prompt = (
            "You are a legal analysis judge. Score this sub-question's analysis.\n\n"
            f"MAIN QUESTION: {question}\n"
            f"SUB-QUESTION: {state.question}\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            "Score 4 dimensions (0.0-1.0): precedent_alignment, "
            "factor_completeness, logical_consistency, evidence_strength.\n"
            'Respond with ONLY JSON: {"precedent_alignment": 0.0, '
            '"factor_completeness": 0.0, "logical_consistency": 0.0, '
            '"evidence_strength": 0.0}'
        )
        try:
            response = self.judge.generate(prompt, temperature=0.1, max_tokens=256)
            scores = parse_judge_scores(response)
            if scores is None:
                return 0.4
            reward = weighted_reward(scores, MCTS_REWARD_WEIGHTS)
            state.partial_answer = f"Scored {reward:.2f} by judge."
            return reward
        except Exception as e:
            logger.warning(f"Judge scoring failed for {state.id}: {e}")
            return 0.4

    def _backpropagate(self, state: ReasoningState, reward: float):
        current_id = state.id
        while current_id is not None:
            current = self._states[current_id]
            current.visits += 1
            current.total_reward += reward
            current_id = current.parent_id

    # -- Result extraction -------------------------------------------------

    def _should_stop_early(self, root: ReasoningState) -> bool:
        """Stop when the best child's reward plateaus across recent visits."""
        if root.visits < 10:
            return False
        best = max(root.children, key=lambda c: c.average_reward, default=None)
        if best is None or best.visits < 5:
            return False
        return best.average_reward >= 0.8

    def _best_path(self, root: ReasoningState) -> List[ReasoningState]:
        path = [root]
        current = root
        while current.children:
            best = max(
                current.children,
                key=lambda c: c.average_reward if c.visits > 0 else -1,
            )
            if best.visits == 0:
                break
            path.append(best)
            current = best
        return path

    def _path_confidence(self, path: List[ReasoningState]) -> float:
        if len(path) <= 1:
            return 0.0
        rewards = [s.average_reward for s in path[1:] if s.visits > 0]
        if not rewards:
            return 0.0
        return sum(rewards) / len(rewards)

    def _synthesize(self, question: str, path: List[ReasoningState]) -> str:
        """Generate final answer from the best reasoning path."""
        steps = []
        for i, s in enumerate(path[1:], 1):
            steps.append(f"Step {i}: {s.question} (score {s.average_reward:.3f})")
        evidence = []
        for s in path[1:]:
            evidence.extend(s.evidence[:2])

        prompt = (
            "You are a senior employment law analyst. Synthesize a final answer "
            "from the reasoning path below.\n\n"
            f"QUESTION: {question}\n\n"
            "REASONING PATH:\n" + "\n".join(steps) + "\n\n"
            "KEY EVIDENCE:\n" + "\n".join(evidence[:6]) + "\n\n"
            "Provide a structured, well-cited answer. Note any ambiguity."
        )
        try:
            return self.judge.generate(
                prompt,
                system_instruction=(
                    "You are a senior employment law analyst providing rigorous "
                    "legal reasoning grounded in the provided evidence."
                ),
                temperature=0.4,
                max_tokens=2048,
            )
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return "Reasoning completed but answer synthesis failed. Review the reasoning path above."