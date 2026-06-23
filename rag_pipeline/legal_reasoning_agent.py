# RAG Pipeline - MCTS-Based Legal Reasoning Agent
"""
Monte Carlo Tree Search (MCTS) based legal reasoning agent for worker classification.

This module implements the "推理期计算资源扩展技术" (inference-time compute scaling)
pattern described in the article. Instead of a single-shot classification, the agent
explores a tree of legal reasoning paths using MCTS:

1. SELECT: Choose the most promising unexplored reasoning path (UCB1)
2. EXPAND: Generate sub-hypotheses by examining different Sagaz factors
3. SIMULATE: Score hypothesis using RAG evidence + judge model
4. BACKPROPAGATE: Update scores up the tree

Key research contributions:
- Reward function design: combining precedent alignment, factor completeness,
  logical consistency, and evidence strength
- Pruning boundaries: when to abandon a reasoning path
- Dynamic backtracking: when to backtrack vs. continue exploring

This is the legal domain equivalent of the medical diagnosis agent described
in the article — structured hypothesis exploration with evidence-based scoring.
"""

import sys
import math
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.embeddings import GeminiChat
from rag_pipeline.rag_query import LegalRAGQuery
from config import (
    MCTS_N_SIMULATIONS,
    MCTS_EXPLORATION_CONSTANT,
    MCTS_MAX_DEPTH,
    MCTS_MIN_SCORE_THRESHOLD,
    MCTS_REWARD_WEIGHTS,
)

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================
# Legal Domain Constants for Worker Classification
# ============================================================

SAGAZ_FACTORS = [
    {
        "name": "control",
        "description": "Degree of control the payer has over the worker's activities",
        "employee_indicators": [
            "payer sets work hours", "payer directs methods", "payer supervises daily work",
            "worker cannot delegate", "worker reports to manager",
        ],
        "contractor_indicators": [
            "worker sets own hours", "worker chooses methods", "worker hires helpers",
            "worker decides work sequence", "minimal supervision",
        ],
    },
    {
        "name": "ownership_of_tools",
        "description": "Who owns the tools and equipment used in the work",
        "employee_indicators": [
            "employer provides all tools", "employer provides vehicle",
            "employer provides office space", "employer provides technology",
        ],
        "contractor_indicators": [
            "worker owns tools", "worker provides vehicle",
            "worker has own office", "worker invests in equipment",
        ],
    },
    {
        "name": "chance_of_profit",
        "description": "Whether the worker has an opportunity for profit beyond fixed wages",
        "employee_indicators": [
            "fixed salary", "hourly wage", "no commission structure",
            "no ability to increase earnings through efficiency",
        ],
        "contractor_indicators": [
            "payment per project", "can negotiate rates", "serves multiple clients",
            "profit from efficiency", "can subcontract for margin",
        ],
    },
    {
        "name": "risk_of_loss",
        "description": "Whether the worker bears any financial risk of loss",
        "employee_indicators": [
            "no financial risk", "guaranteed minimum pay",
            "employer covers expenses", "employer covers liability insurance",
        ],
        "contractor_indicators": [
            "risk of non-payment", "warranty obligations", "liable for mistakes",
            "carries own insurance", "unpaid if client dissatisfied",
        ],
    },
    {
        "name": "integration",
        "description": "How integrated the worker is into the payer's business",
        "employee_indicators": [
            "part of organizational structure", "has company email",
            "attends company meetings", "wears company uniform",
            "exclusively works for payer",
        ],
        "contractor_indicators": [
            "operates independently", "has own business name",
            "advertises services", "works for multiple clients",
            "not integrated into management structure",
        ],
    },
]


@dataclass
class ReasoningNode:
    """
    A node in the MCTS reasoning tree.
    
    Each node represents a legal hypothesis about a specific aspect
    of the worker classification analysis.
    """
    id: str
    hypothesis: str
    factor: str  # Which Sagaz factor this hypothesis addresses
    evidence: List[str] = field(default_factory=list)
    score: float = 0.0
    children: List['ReasoningNode'] = field(default_factory=list)
    parent_id: Optional[str] = None
    depth: int = 0
    
    # MCTS statistics
    visits: int = 0
    total_reward: float = 0.0
    
    @property
    def average_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def ucb1(self, parent_visits: int, exploration_constant: float = MCTS_EXPLORATION_CONSTANT) -> float:
        """
        Upper Confidence Bound 1 (UCB1) formula for node selection.
        
        Balances exploitation (high average reward) with exploration
        (under-visited nodes). This is the classic MCTS selection criterion.
        
        UCB1 = avg_reward + C * sqrt(ln(parent_visits) / node_visits)
        """
        if self.visits == 0:
            return float('inf')  # Always explore unvisited nodes
        
        exploitation = self.average_reward
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits) / self.visits
        )
        return exploitation + exploration
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "factor": self.factor,
            "score": round(self.score, 4),
            "visits": self.visits,
            "average_reward": round(self.average_reward, 4),
            "evidence_count": len(self.evidence),
            "children_count": len(self.children),
            "depth": self.depth,
        }


@dataclass
class ReasoningResult:
    """Final result of MCTS-based legal reasoning"""
    classification: str  # "Employee" or "Independent Contractor"
    confidence: float
    best_reasoning_path: List[Dict[str, Any]]
    tree_statistics: Dict[str, Any]
    factor_analysis: Dict[str, Dict[str, Any]]
    full_reasoning_text: str
    duration_ms: float


class LegalReasoningAgent:
    """
    MCTS-based legal reasoning agent for worker classification.
    
    Explores different classification hypotheses through tree search,
    scoring each hypothesis using RAG-retrieved legal precedents and
    a judge model. This is inference-time compute scaling applied to
    legal reasoning.
    
    The agent's reasoning tree structure:
    - Root: "Classify worker based on given facts"
    - Level 1: Factor-specific hypotheses (one per Sagaz factor)
    - Level 2+: Refined sub-hypotheses examining specific indicators
    """
    
    def __init__(
        self,
        rag_query: Optional[LegalRAGQuery] = None,
        judge: Optional[GeminiChat] = None,
        n_simulations: int = MCTS_N_SIMULATIONS,
        exploration_constant: float = MCTS_EXPLORATION_CONSTANT,
        max_depth: int = MCTS_MAX_DEPTH,
        min_score_threshold: float = MCTS_MIN_SCORE_THRESHOLD,
    ):
        self.rag = rag_query or LegalRAGQuery()
        self.judge = judge or GeminiChat()
        self.n_simulations = n_simulations
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.min_score_threshold = min_score_threshold
        
        # Node registry for fast lookup
        self._nodes: Dict[str, ReasoningNode] = {}
        self._node_counter = 0
    
    def _new_node_id(self) -> str:
        self._node_counter += 1
        return f"node_{self._node_counter}"
    
    def classify_with_reasoning(
        self,
        facts: str,
        n_simulations: Optional[int] = None,
    ) -> ReasoningResult:
        """
        Perform MCTS-based worker classification with full reasoning trace.
        
        Args:
            facts: Description of the working relationship
            n_simulations: Number of MCTS simulations (overrides default)
            
        Returns:
            ReasoningResult with classification, confidence, and full reasoning tree
        """
        start_time = datetime.now()
        n_sims = n_simulations or self.n_simulations
        
        logger.info(f"Starting MCTS reasoning with {n_sims} simulations...")
        
        # Reset node registry
        self._nodes = {}
        self._node_counter = 0
        
        # Create root node
        root = ReasoningNode(
            id=self._new_node_id(),
            hypothesis="Analyze worker classification based on provided facts",
            factor="root",
            depth=0,
        )
        self._nodes[root.id] = root
        
        # Initial expansion: create factor-level hypotheses
        self._initial_expand(root, facts)
        
        # MCTS loop
        for sim in range(n_sims):
            if (sim + 1) % 10 == 0:
                logger.info(f"  Simulation {sim + 1}/{n_sims}")
            
            # SELECT
            selected = self._select(root)
            
            # EXPAND (if not at max depth and not already expanded)
            if selected.depth < self.max_depth and selected.is_leaf and selected.visits > 0:
                self._expand(selected, facts)
                if selected.children:
                    selected = selected.children[0]  # Evaluate first new child
            
            # SIMULATE
            reward = self._simulate(selected, facts)
            
            # BACKPROPAGATE
            self._backpropagate(selected, reward)
        
        # Extract results
        best_path = self._get_best_path(root)
        factor_analysis = self._build_factor_analysis(root)
        classification, confidence = self._determine_classification(root, facts)
        full_text = self._generate_full_reasoning(facts, best_path, factor_analysis, classification)
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return ReasoningResult(
            classification=classification,
            confidence=confidence,
            best_reasoning_path=[n.to_dict() for n in best_path],
            tree_statistics={
                "total_nodes": len(self._nodes),
                "total_simulations": n_sims,
                "max_depth_reached": max(n.depth for n in self._nodes.values()),
                "root_visits": root.visits,
            },
            factor_analysis=factor_analysis,
            full_reasoning_text=full_text,
            duration_ms=round(duration_ms, 1),
        )
    
    def _initial_expand(self, root: ReasoningNode, facts: str):
        """Create initial factor-level hypotheses from the Sagaz test factors."""
        for factor in SAGAZ_FACTORS:
            # Employee hypothesis for this factor
            emp_node = ReasoningNode(
                id=self._new_node_id(),
                hypothesis=f"Under the {factor['name']} factor, the evidence suggests EMPLOYEE status",
                factor=factor["name"],
                parent_id=root.id,
                depth=1,
            )
            self._nodes[emp_node.id] = emp_node
            root.children.append(emp_node)
            
            # Contractor hypothesis for this factor
            con_node = ReasoningNode(
                id=self._new_node_id(),
                hypothesis=f"Under the {factor['name']} factor, the evidence suggests INDEPENDENT CONTRACTOR status",
                factor=factor["name"],
                parent_id=root.id,
                depth=1,
            )
            self._nodes[con_node.id] = con_node
            root.children.append(con_node)
    
    def _select(self, node: ReasoningNode) -> ReasoningNode:
        """
        Select the most promising node for expansion using UCB1.
        
        Traverses the tree from root to a leaf, always choosing the child
        with the highest UCB1 score. This balances exploration (visiting
        under-explored nodes) with exploitation (visiting high-reward nodes).
        """
        current = node
        
        while not current.is_leaf:
            # Select child with highest UCB1
            best_child = max(
                current.children,
                key=lambda c: c.ucb1(current.visits, self.exploration_constant),
            )
            current = best_child
        
        return current
    
    def _expand(self, node: ReasoningNode, facts: str):
        """
        Generate child hypotheses for a selected node.
        
        Creates more specific sub-hypotheses by examining specific
        indicators within the factor that the parent node addresses.
        """
        # Find the relevant factor
        factor_info = next(
            (f for f in SAGAZ_FACTORS if f["name"] == node.factor),
            None,
        )
        
        if not factor_info:
            return
        
        # Determine which indicators to explore based on parent hypothesis
        is_employee = "EMPLOYEE" in node.hypothesis.upper()
        indicators = (factor_info["employee_indicators"] if is_employee
                      else factor_info["contractor_indicators"])
        
        # Create child nodes for specific indicators
        for indicator in indicators[:3]:  # Cap at 3 to control branching
            child = ReasoningNode(
                id=self._new_node_id(),
                hypothesis=f"Regarding '{indicator}': analyzing whether the facts support this indicator under the {node.factor} factor",
                factor=node.factor,
                parent_id=node.id,
                depth=node.depth + 1,
            )
            self._nodes[child.id] = child
            node.children.append(child)
    
    def _simulate(self, node: ReasoningNode, facts: str) -> float:
        """
        Score a hypothesis using RAG evidence and judge model.
        
        This is the core evaluation function that determines the quality
        of a reasoning path. Uses the weighted reward function:
        
        reward = w1 * precedent_alignment 
               + w2 * factor_completeness
               + w3 * logical_consistency
               + w4 * evidence_strength
        
        Research contribution: the mathematical construction of this
        reward function ("奖励函数的数学构建").
        """
        # Step 1: Retrieve relevant precedents for this hypothesis
        try:
            rag_response = self.rag.query(
                f"{node.hypothesis} given these facts: {facts[:500]}",
                top_k=3,
                include_analysis=False,
            )
            
            # Store evidence in node
            node.evidence = [s.get("excerpt", "") for s in rag_response.sources]
            
        except Exception as e:
            logger.warning(f"RAG retrieval failed for node {node.id}: {e}")
            node.evidence = []
            return 0.3  # Default low reward on failure
        
        # Step 2: Judge model scores the hypothesis
        evidence_text = "\n".join(node.evidence[:3]) if node.evidence else "No evidence retrieved."
        
        prompt = f"""You are a legal analysis judge evaluating a worker classification hypothesis.

WORKER FACTS:
{facts[:800]}

HYPOTHESIS BEING EVALUATED:
{node.hypothesis}

SUPPORTING EVIDENCE FROM LEGAL PRECEDENTS:
{evidence_text}

Score this hypothesis on 4 dimensions (each 0.0 to 1.0):

1. precedent_alignment: How well does the evidence from precedents support this hypothesis?
2. factor_completeness: Does the hypothesis thoroughly analyze the relevant classification factor?
3. logical_consistency: Is the reasoning logically sound and free of contradictions?
4. evidence_strength: How strong and relevant is the retrieved evidence?

Respond in EXACTLY this JSON format:
{{
    "precedent_alignment": <float>,
    "factor_completeness": <float>,
    "logical_consistency": <float>,
    "evidence_strength": <float>,
    "brief_justification": "<one sentence>"
}}"""
        
        try:
            response = self.judge.generate(prompt, temperature=0.1, max_tokens=512)
            
            # Parse scores
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            
            scores = json.loads(cleaned)
            
            # Calculate weighted reward
            weights = MCTS_REWARD_WEIGHTS
            reward = (
                weights["precedent_alignment"] * float(scores.get("precedent_alignment", 0.5))
                + weights["factor_completeness"] * float(scores.get("factor_completeness", 0.5))
                + weights["logical_consistency"] * float(scores.get("logical_consistency", 0.5))
                + weights["evidence_strength"] * float(scores.get("evidence_strength", 0.5))
            )
            
            node.score = reward
            return reward
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Judge scoring error for node {node.id}: {e}")
            node.score = 0.4
            return 0.4  # Default moderate reward on parsing error
    
    def _backpropagate(self, node: ReasoningNode, reward: float):
        """
        Update visit counts and rewards from the evaluated node back to root.
        
        Standard MCTS backpropagation: increment visit count and add reward
        at each node along the path from the evaluated node to the root.
        """
        current_id = node.id
        
        while current_id is not None:
            current = self._nodes[current_id]
            current.visits += 1
            current.total_reward += reward
            current_id = current.parent_id
    
    def _get_best_path(self, root: ReasoningNode) -> List[ReasoningNode]:
        """
        Extract the highest-scoring reasoning chain from root to leaf.
        
        At each level, choose the child with the highest average reward.
        """
        path = [root]
        current = root
        
        while current.children:
            # Choose child with highest average reward
            best_child = max(
                current.children,
                key=lambda c: c.average_reward if c.visits > 0 else -1,
            )
            
            if best_child.visits == 0:
                break
            
            path.append(best_child)
            current = best_child
        
        return path
    
    def _build_factor_analysis(self, root: ReasoningNode) -> Dict[str, Dict[str, Any]]:
        """
        Build per-factor analysis from the MCTS tree.
        
        For each Sagaz factor, aggregate scores from the Employee and
        Contractor hypotheses to determine which direction the evidence leans.
        """
        factor_analysis = {}
        
        for child in root.children:
            factor = child.factor
            
            if factor not in factor_analysis:
                factor_analysis[factor] = {
                    "employee_score": 0.0,
                    "contractor_score": 0.0,
                    "employee_visits": 0,
                    "contractor_visits": 0,
                    "lean_direction": "undetermined",
                    "evidence": [],
                }
            
            if "EMPLOYEE" in child.hypothesis.upper():
                factor_analysis[factor]["employee_score"] = child.average_reward
                factor_analysis[factor]["employee_visits"] = child.visits
            else:
                factor_analysis[factor]["contractor_score"] = child.average_reward
                factor_analysis[factor]["contractor_visits"] = child.visits
            
            factor_analysis[factor]["evidence"].extend(child.evidence[:2])
        
        # Determine lean direction
        for factor, data in factor_analysis.items():
            emp = data["employee_score"]
            con = data["contractor_score"]
            
            if emp > con + 0.1:
                data["lean_direction"] = "employee"
            elif con > emp + 0.1:
                data["lean_direction"] = "contractor"
            else:
                data["lean_direction"] = "neutral"
        
        return factor_analysis
    
    def _determine_classification(
        self,
        root: ReasoningNode,
        facts: str,
    ) -> Tuple[str, float]:
        """
        Determine final classification from factor analysis.
        
        Counts how many factors lean toward employee vs. contractor,
        weighted by the strength of evidence for each factor.
        """
        factor_analysis = self._build_factor_analysis(root)
        
        employee_weighted_score = 0.0
        contractor_weighted_score = 0.0
        total_weight = 0.0
        
        for factor, data in factor_analysis.items():
            # Weight by total visits (more visits = more explored = more confident)
            weight = data["employee_visits"] + data["contractor_visits"]
            if weight == 0:
                weight = 1
            total_weight += weight
            
            employee_weighted_score += data["employee_score"] * weight
            contractor_weighted_score += data["contractor_score"] * weight
        
        if total_weight > 0:
            employee_weighted_score /= total_weight
            contractor_weighted_score /= total_weight
        
        # Classification and confidence
        if employee_weighted_score > contractor_weighted_score:
            classification = "Employee"
            confidence = employee_weighted_score / (employee_weighted_score + contractor_weighted_score + 1e-8)
        else:
            classification = "Independent Contractor"
            confidence = contractor_weighted_score / (employee_weighted_score + contractor_weighted_score + 1e-8)
        
        return classification, round(confidence, 3)
    
    def _generate_full_reasoning(
        self,
        facts: str,
        best_path: List[ReasoningNode],
        factor_analysis: Dict[str, Dict[str, Any]],
        classification: str,
    ) -> str:
        """Generate a comprehensive reasoning narrative from the MCTS results."""
        
        # Build factor summary
        factor_lines = []
        for factor, data in factor_analysis.items():
            direction = data["lean_direction"]
            emp_score = data["employee_score"]
            con_score = data["contractor_score"]
            factor_lines.append(
                f"  • {factor.replace('_', ' ').title()}: "
                f"leans {direction} (Employee: {emp_score:.2f}, Contractor: {con_score:.2f})"
            )
        
        prompt = f"""Based on the following MCTS analysis of a worker classification case,
generate a comprehensive legal reasoning narrative.

FACTS:
{facts[:1000]}

FACTOR-BY-FACTOR ANALYSIS:
{chr(10).join(factor_lines)}

BEST REASONING PATH:
{chr(10).join([f"  Step {i+1}: {n.hypothesis} (score: {n.score:.3f})" for i, n in enumerate(best_path)])}

PRELIMINARY CLASSIFICATION: {classification}

Generate a structured legal analysis that:
1. Summarizes each Sagaz factor's findings
2. Explains the reasoning behind the classification
3. Notes any factors that were ambiguous or conflicting
4. Provides a confidence-qualified conclusion
5. Suggests next steps or additional information needed"""
        
        try:
            return self.judge.generate(
                prompt,
                system_instruction="You are a senior employment law analyst providing rigorous legal reasoning.",
                temperature=0.4,
                max_tokens=2048,
            )
        except Exception as e:
            logger.error(f"Reasoning generation error: {e}")
            return f"Classification: {classification}\n\nFactor Analysis:\n" + "\n".join(factor_lines)


def main():
    """Test the MCTS reasoning agent"""
    print("\n" + "=" * 60)
    print("TESTING MCTS LEGAL REASONING AGENT")
    print("=" * 60)
    
    try:
        agent = LegalReasoningAgent(n_simulations=10)  # Small for testing
        
        test_facts = """
        Sarah has been working for TechCorp for 3 years. She uses company-provided 
        laptop and works from the company office. Her manager sets her daily tasks 
        and reviews her work weekly. She is paid a fixed monthly salary of $5,000 
        with no commission or bonus structure. She works exclusively for TechCorp 
        and cannot take on other clients. She wears a company uniform with the 
        TechCorp logo. However, she sets her own daily schedule within the 9-5 
        window and occasionally works from home.
        """
        
        result = agent.classify_with_reasoning(test_facts)
        
        print(f"\n🏷️  Classification: {result.classification}")
        print(f"📊 Confidence: {result.confidence:.1%}")
        print(f"🌳 Tree nodes: {result.tree_statistics['total_nodes']}")
        print(f"⏱️  Duration: {result.duration_ms:.0f}ms")
        
        print("\n📋 Factor Analysis:")
        for factor, data in result.factor_analysis.items():
            print(f"  {factor}: {data['lean_direction']} "
                  f"(E: {data['employee_score']:.2f}, C: {data['contractor_score']:.2f})")
        
        print(f"\n💬 Full Reasoning:\n{result.full_reasoning_text[:500]}...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
