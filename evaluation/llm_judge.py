# Evaluation Framework - LLM Judge with Bias Mitigation
"""
Debiased LLM-as-Judge system for scoring legal AI outputs.

The article identifies three critical biases in LLM judges:
1. POSITION BIAS: Judge prefers the answer presented first
2. LENGTH BIAS: Judge prefers longer, more verbose responses
3. SELF-ENHANCEMENT BIAS: Judge prefers outputs matching its own style

Mitigation strategies implemented:
- Position debiasing: Swap answer order and average scores
- Length debiasing: Normalize scores by response length
- Self-enhancement debiasing: Structured rubric decomposition instead
  of holistic scoring — score individual components, not overall quality

"不是让裁判模型直接给个综合分数，而是你设计算法让裁判模型先提取文章的
核心论点、逻辑推导链条、市场数据支撑，然后针对每一个细分的叶子节点
进行严格的规则化打分"
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from statistics import mean, stdev

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import JUDGE_MODEL, JUDGE_TEMPERATURE, JUDGE_POSITION_SWAP_TRIALS

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result of a single judge evaluation"""
    overall_score: float
    component_scores: Dict[str, float]
    feedback: str
    debiasing_applied: List[str]
    raw_scores: List[float] = field(default_factory=list)  # Before debiasing
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PairwiseResult:
    """Result of a pairwise comparison between two responses"""
    winner: str  # "A", "B", or "tie"
    score_a: float
    score_b: float
    confidence: float
    position_bias_detected: bool
    raw_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DebiasedLegalJudge:
    """
    LLM-as-Judge with active bias mitigation for legal output evaluation.
    
    Instead of a single holistic score, decomposes evaluation into
    structured components and applies explicit debiasing techniques.
    
    Debiasing techniques:
    1. Rubric decomposition: Score components independently
    2. Position swapping: For pairwise, swap order and average
    3. Length normalization: Divide quality signals by length
    4. Multi-trial averaging: Run multiple scoring passes
    """
    
    # Legal evaluation components for decomposition
    LEGAL_COMPONENTS = {
        "argument_identification": {
            "description": "Correctly identifies the core legal arguments and issues",
            "weight": 0.2,
        },
        "reasoning_chain": {
            "description": "Logical chain from facts → legal principles → conclusion",
            "weight": 0.25,
        },
        "evidence_usage": {
            "description": "Properly uses cited evidence/precedents to support claims",
            "weight": 0.2,
        },
        "legal_accuracy": {
            "description": "Correct application of legal tests and principles",
            "weight": 0.2,
        },
        "practical_utility": {
            "description": "Actionable guidance, clear conclusions, appropriate caveats",
            "weight": 0.15,
        },
    }
    
    def __init__(
        self,
        chat=None,
        temperature: float = JUDGE_TEMPERATURE,
        n_swap_trials: int = JUDGE_POSITION_SWAP_TRIALS,
    ):
        from rag_pipeline.embeddings import GeminiChat
        self.chat = chat or GeminiChat(model=JUDGE_MODEL)
        self.temperature = temperature
        self.n_swap_trials = n_swap_trials
    
    def score(
        self,
        question: str,
        response: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
    ) -> JudgeResult:
        """
        Score a legal response with bias mitigation via rubric decomposition.
        
        Instead of asking for a single score, decomposes the response into
        5 components and scores each independently. This mitigates
        self-enhancement bias by forcing granular, criteria-specific evaluation.
        
        Args:
            question: The question that was asked
            response: The AI-generated response to evaluate
            reference: Optional reference/gold standard answer
            context: Optional context (e.g., retrieved documents)
            
        Returns:
            JudgeResult with component scores and debiasing metadata
        """
        component_scores = {}
        component_feedback = {}
        debiasing_applied = ["rubric_decomposition"]
        raw_scores = []
        
        for component_name, component_info in self.LEGAL_COMPONENTS.items():
            score, feedback = self._score_component(
                question, response, component_name,
                component_info["description"], reference, context,
            )
            component_scores[component_name] = score
            component_feedback[component_name] = feedback
            raw_scores.append(score)
        
        # Length normalization
        response_length = len(response.split())
        if response_length > 500:
            # Apply mild penalty for very verbose responses
            length_penalty = min(1.0, 500 / response_length * 1.2)  # Soft cap
            debiasing_applied.append("length_normalization")
        else:
            length_penalty = 1.0
        
        # Weighted average of component scores
        weighted_sum = sum(
            component_scores[name] * info["weight"] * length_penalty
            for name, info in self.LEGAL_COMPONENTS.items()
        )
        total_weight = sum(info["weight"] for info in self.LEGAL_COMPONENTS.values())
        overall = weighted_sum / total_weight
        
        # Build feedback
        feedback_parts = []
        for comp, fb in component_feedback.items():
            score_val = component_scores[comp]
            feedback_parts.append(f"  [{comp}] ({score_val:.2f}): {fb}")
        
        return JudgeResult(
            overall_score=round(overall, 4),
            component_scores=component_scores,
            feedback="\n".join(feedback_parts),
            debiasing_applied=debiasing_applied,
            raw_scores=raw_scores,
            metadata={
                "response_length_words": response_length,
                "length_penalty": round(length_penalty, 3),
            },
        )
    
    def _score_component(
        self,
        question: str,
        response: str,
        component_name: str,
        component_description: str,
        reference: Optional[str],
        context: Optional[str],
    ) -> Tuple[float, str]:
        """Score a single evaluation component."""
        
        ref_section = ""
        if reference:
            ref_section = f"\nREFERENCE ANSWER:\n{reference[:500]}"
        
        ctx_section = ""
        if context:
            ctx_section = f"\nCONTEXT PROVIDED:\n{context[:300]}"
        
        prompt = f"""You are evaluating ONE specific component of a legal AI response.

COMPONENT: {component_name.replace('_', ' ').upper()}
DEFINITION: {component_description}

QUESTION ASKED:
{question[:300]}

AI RESPONSE:
{response[:1500]}
{ref_section}
{ctx_section}

Score ONLY on the "{component_name.replace('_', ' ')}" component.
Ignore all other quality aspects — focus narrowly on this one criterion.

Score from 0.0 to 1.0:
- 0.0: Component completely absent or wrong
- 0.5: Component present but with issues
- 1.0: Component excellent

Respond in EXACTLY this JSON format:
{{
    "score": <float>,
    "feedback": "<1 sentence about this specific component>"
}}"""
        
        try:
            result = self.chat.generate(
                prompt,
                temperature=self.temperature,
                max_tokens=200,
            )
            
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            return (
                max(0.0, min(1.0, float(data.get("score", 0.5)))),
                data.get("feedback", ""),
            )
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Component scoring error for {component_name}: {e}")
            return 0.5, f"Scoring error: {str(e)[:80]}"
    
    def pairwise_compare(
        self,
        question: str,
        response_a: str,
        response_b: str,
        label_a: str = "Response A",
        label_b: str = "Response B",
    ) -> PairwiseResult:
        """
        Compare two responses with position debiasing.
        
        Runs multiple trials with swapped presentation order.
        If the winner changes when order is swapped, position bias
        is detected and the result is marked accordingly.
        
        Args:
            question: The question both responses answer
            response_a: First response
            response_b: Second response
            label_a: Label for first response
            label_b: Label for second response
            
        Returns:
            PairwiseResult with debiased comparison
        """
        results = []
        
        for trial in range(self.n_swap_trials):
            # Alternate presentation order
            if trial % 2 == 0:
                # Normal order: A first, B second
                first, second = response_a, response_b
                first_label, second_label = label_a, label_b
                order = "normal"
            else:
                # Swapped order: B first, A second
                first, second = response_b, response_a
                first_label, second_label = label_b, label_a
                order = "swapped"
            
            score_first, score_second = self._pairwise_trial(
                question, first, second, first_label, second_label,
            )
            
            # Map scores back to A/B regardless of presentation order
            if order == "normal":
                score_a, score_b = score_first, score_second
            else:
                score_a, score_b = score_second, score_first
            
            results.append({
                "trial": trial,
                "order": order,
                "score_a": score_a,
                "score_b": score_b,
            })
        
        # Average scores across trials
        avg_score_a = mean(r["score_a"] for r in results)
        avg_score_b = mean(r["score_b"] for r in results)
        
        # Detect position bias: did the winner change when order swapped?
        normal_winners = [r for r in results if r["order"] == "normal"]
        swapped_winners = [r for r in results if r["order"] == "swapped"]
        
        position_bias = False
        if normal_winners and swapped_winners:
            normal_winner = "A" if normal_winners[0]["score_a"] > normal_winners[0]["score_b"] else "B"
            swapped_winner = "A" if swapped_winners[0]["score_a"] > swapped_winners[0]["score_b"] else "B"
            position_bias = normal_winner != swapped_winner
        
        # Determine winner
        margin = abs(avg_score_a - avg_score_b)
        if margin < 0.05:
            winner = "tie"
        elif avg_score_a > avg_score_b:
            winner = "A"
        else:
            winner = "B"
        
        # Confidence: higher when scores are consistent across trials
        if len(results) > 1:
            score_a_std = stdev(r["score_a"] for r in results) if len(results) > 1 else 0
            score_b_std = stdev(r["score_b"] for r in results) if len(results) > 1 else 0
            consistency = 1.0 - (score_a_std + score_b_std) / 2
            confidence = max(0.0, min(1.0, consistency * margin * 5))
        else:
            confidence = margin
        
        return PairwiseResult(
            winner=winner,
            score_a=round(avg_score_a, 4),
            score_b=round(avg_score_b, 4),
            confidence=round(confidence, 4),
            position_bias_detected=position_bias,
            raw_results=results,
        )
    
    def _pairwise_trial(
        self,
        question: str,
        first: str,
        second: str,
        first_label: str,
        second_label: str,
    ) -> Tuple[float, float]:
        """Run a single pairwise comparison trial."""
        
        prompt = f"""You are comparing two legal AI responses to the same question.
Score each response INDEPENDENTLY on quality (0.0 to 1.0).

QUESTION:
{question[:300]}

--- {first_label} ---
{first[:1000]}

--- {second_label} ---
{second[:1000]}

Score each response independently. Do NOT compare them to each other.
Evaluate each on its own merits against the question.

Respond in EXACTLY this JSON format:
{{
    "first_score": <float 0.0-1.0>,
    "second_score": <float 0.0-1.0>,
    "first_reasoning": "<1 sentence>",
    "second_reasoning": "<1 sentence>"
}}"""
        
        try:
            result = self.chat.generate(
                prompt,
                temperature=self.temperature,
                max_tokens=300,
            )
            
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            return (
                max(0.0, min(1.0, float(data.get("first_score", 0.5)))),
                max(0.0, min(1.0, float(data.get("second_score", 0.5)))),
            )
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Pairwise trial error: {e}")
            return 0.5, 0.5
    
    def _decompose_response(self, response: str) -> Dict[str, str]:
        """
        Break a legal response into scoreable components.
        
        Extracts:
        - Core arguments
        - Reasoning chain
        - Evidence citations
        - Conclusions
        - Hedging/uncertainty markers
        """
        prompt = f"""Decompose the following legal response into components.

RESPONSE:
{response[:2000]}

Extract these components as JSON:
{{
    "core_arguments": "<main legal arguments made>",
    "reasoning_chain": "<step-by-step logic: premise -> inference -> conclusion>",
    "citations": "<specific cases, statutes, or legal principles cited>",
    "conclusions": "<final classification or recommendation>",
    "hedging": "<uncertainty markers, caveats, qualifications>"
}}"""
        
        try:
            result = self.chat.generate(prompt, temperature=0.1, max_tokens=1024)
            
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            
            return json.loads(cleaned)
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Response decomposition error: {e}")
            return {
                "core_arguments": response[:200],
                "reasoning_chain": "Could not extract",
                "citations": "Could not extract",
                "conclusions": "Could not extract",
                "hedging": "Could not extract",
            }


def main():
    """Test the debiased judge"""
    print("\n" + "=" * 60)
    print("TESTING DEBIASED LEGAL JUDGE")
    print("=" * 60)
    
    judge = DebiasedLegalJudge()
    
    question = "Is Sarah an employee or independent contractor under the Sagaz test?"
    
    response_good = """
    Based on the Sagaz test (671122 Ontario Ltd. v. Sagaz Industries Canada Inc., 2001 SCC 59),
    Sarah is likely an Employee. The control factor strongly supports this: her manager assigns
    daily tasks and reviews her work. She uses company equipment (ownership of tools factor),
    receives a fixed salary (no chance of profit), bears no financial risk, and is fully
    integrated into the company's organizational structure. However, her flexibility in setting
    her daily schedule within the 9-5 window introduces some ambiguity in the control factor.
    Misclassification risks include back-payment of employment benefits and CPP/EI contributions.
    """
    
    response_bad = "She's probably an employee because she works at the company."
    
    # Score individual responses
    result_good = judge.score(question, response_good)
    print(f"\nGood response score: {result_good.overall_score:.3f}")
    print(f"Components: {result_good.component_scores}")
    print(f"Debiasing: {result_good.debiasing_applied}")
    
    result_bad = judge.score(question, response_bad)
    print(f"\nBad response score: {result_bad.overall_score:.3f}")
    print(f"Components: {result_bad.component_scores}")
    
    # Pairwise comparison
    pairwise = judge.pairwise_compare(question, response_good, response_bad)
    print(f"\nPairwise winner: {pairwise.winner}")
    print(f"Scores: A={pairwise.score_a:.3f}, B={pairwise.score_b:.3f}")
    print(f"Position bias detected: {pairwise.position_bias_detected}")
    print(f"Confidence: {pairwise.confidence:.3f}")


if __name__ == "__main__":
    main()
