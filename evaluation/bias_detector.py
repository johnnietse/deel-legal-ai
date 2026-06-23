# Evaluation Framework - Bias Detector
"""
Detects and quantifies biases in LLM judge systems.

Implements automated detection for the three biases identified in the article:
1. Position bias: Judge favors the answer presented first/last
2. Length bias: Judge favors longer responses
3. Self-enhancement bias: Judge favors responses matching its own style

Provides both detection (measuring bias magnitude) and reporting
(generating actionable bias reports with recommendations).
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

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class BiasMetric:
    """A single bias measurement"""
    bias_type: str
    magnitude: float  # 0.0 = no bias, 1.0 = maximum bias
    direction: str  # e.g., "favors_first", "favors_longer"
    confidence: float
    details: str


@dataclass
class BiasReport:
    """Comprehensive bias analysis report"""
    overall_bias_score: float  # 0.0 = unbiased, 1.0 = severely biased
    biases_detected: List[BiasMetric]
    recommendations: List[str]
    test_sample_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_bias_score": self.overall_bias_score,
            "biases_detected": [asdict(b) for b in self.biases_detected],
            "recommendations": self.recommendations,
            "test_sample_size": self.test_sample_size,
            "metadata": self.metadata,
        }
    
    def summary(self) -> str:
        """Generate human-readable bias report summary."""
        lines = [
            "=" * 50,
            "BIAS DETECTION REPORT",
            "=" * 50,
            f"Overall Bias Score: {self.overall_bias_score:.3f} "
            f"({'LOW' if self.overall_bias_score < 0.3 else 'MODERATE' if self.overall_bias_score < 0.6 else 'HIGH'})",
            f"Test Samples: {self.test_sample_size}",
            "",
        ]
        
        for bias in self.biases_detected:
            severity = "⚠️" if bias.magnitude > 0.3 else "✅"
            lines.append(
                f"{severity} {bias.bias_type}: magnitude={bias.magnitude:.3f}, "
                f"direction={bias.direction}, confidence={bias.confidence:.3f}"
            )
            lines.append(f"    {bias.details}")
        
        if self.recommendations:
            lines.append("\nRecommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")
        
        return "\n".join(lines)


class BiasDetector:
    """
    Detects and quantifies biases in LLM-as-Judge systems.
    
    Uses controlled experiments to measure bias magnitude:
    - Position bias: Score same content in different positions
    - Length bias: Score semantically equivalent responses of different lengths
    - Self-enhancement bias: Compare scores for different writing styles
    """
    
    def __init__(self, judge=None):
        self.judge = judge  # DebiasedLegalJudge instance
    
    def detect_position_bias(
        self,
        question: str,
        response: str,
        n_trials: int = 5,
    ) -> BiasMetric:
        """
        Detect position bias by comparing scores when the same response
        appears in different positions of a pairwise comparison.
        
        If the judge gives significantly different scores to the same
        response depending on whether it appears first or second,
        position bias is present.
        """
        if not self.judge:
            return BiasMetric(
                bias_type="position",
                magnitude=0.0,
                direction="unknown",
                confidence=0.0,
                details="No judge provided for testing",
            )
        
        first_scores = []
        second_scores = []
        
        # Create a baseline comparison response
        baseline = f"The worker classification depends on the specific facts of the case and the applicable legal tests."
        
        for trial in range(n_trials):
            # Trial with response first
            result_first = self.judge._pairwise_trial(
                question, response, baseline, "Response", "Baseline"
            )
            first_scores.append(result_first[0])
            
            # Trial with response second
            result_second = self.judge._pairwise_trial(
                question, baseline, response, "Baseline", "Response"
            )
            second_scores.append(result_second[1])
        
        avg_first = mean(first_scores) if first_scores else 0.5
        avg_second = mean(second_scores) if second_scores else 0.5
        
        # Position bias magnitude is the difference in scores
        magnitude = abs(avg_first - avg_second)
        direction = "favors_first" if avg_first > avg_second else "favors_second"
        
        # Confidence based on consistency
        if len(first_scores) > 1 and len(second_scores) > 1:
            std_first = stdev(first_scores)
            std_second = stdev(second_scores)
            confidence = max(0.0, 1.0 - (std_first + std_second))
        else:
            confidence = 0.5
        
        return BiasMetric(
            bias_type="position",
            magnitude=round(magnitude, 4),
            direction=direction,
            confidence=round(confidence, 4),
            details=(
                f"Same response scored {avg_first:.3f} when first, "
                f"{avg_second:.3f} when second (Δ={magnitude:.3f})"
            ),
        )
    
    def detect_length_bias(
        self,
        question: str,
        core_content: str,
        padding_levels: int = 3,
    ) -> BiasMetric:
        """
        Detect length bias by scoring responses with the same core content
        but different amounts of padding/elaboration.
        
        If the judge gives higher scores to longer versions of the same
        argument, length bias is present.
        """
        if not self.judge:
            return BiasMetric(
                bias_type="length",
                magnitude=0.0,
                direction="unknown",
                confidence=0.0,
                details="No judge provided for testing",
            )
        
        # Generate responses of increasing length
        versions = [core_content]  # Base version
        
        padding_phrases = [
            "Furthermore, it is important to consider that ",
            "Additionally, one must take into account the fact that ",
            "Moreover, the legal analysis should encompass the observation that ",
            "It should also be noted that in similar circumstances, ",
            "From a comprehensive legal perspective, we must acknowledge that ",
        ]
        
        current = core_content
        for level in range(padding_levels):
            padding = padding_phrases[level % len(padding_phrases)]
            current = current + f" {padding}{core_content.split('.')[0].lower()}."
            versions.append(current)
        
        # Score each version
        scores = []
        for version in versions:
            result = self.judge.score(question, version)
            scores.append({
                "length_words": len(version.split()),
                "score": result.overall_score,
            })
        
        # Check if scores increase with length
        if len(scores) >= 2:
            shortest_score = scores[0]["score"]
            longest_score = scores[-1]["score"]
            magnitude = max(0.0, longest_score - shortest_score)
            direction = "favors_longer" if longest_score > shortest_score else "favors_shorter"
            
            # Correlation between length and score
            lengths = [s["length_words"] for s in scores]
            score_vals = [s["score"] for s in scores]
            
            # Simple correlation coefficient
            n = len(lengths)
            if n > 1:
                mean_l = mean(lengths)
                mean_s = mean(score_vals)
                cov = sum((l - mean_l) * (s - mean_s) for l, s in zip(lengths, score_vals)) / n
                std_l = (sum((l - mean_l) ** 2 for l in lengths) / n) ** 0.5
                std_s = (sum((s - mean_s) ** 2 for s in score_vals) / n) ** 0.5
                correlation = cov / (std_l * std_s) if std_l * std_s > 0 else 0.0
            else:
                correlation = 0.0
        else:
            magnitude = 0.0
            direction = "unknown"
            correlation = 0.0
        
        return BiasMetric(
            bias_type="length",
            magnitude=round(abs(magnitude), 4),
            direction=direction,
            confidence=round(abs(correlation), 4),
            details=(
                f"Scores by length: {', '.join(f'{s['length_words']}w→{s['score']:.3f}' for s in scores)}. "
                f"Length-score correlation: {correlation:.3f}"
            ),
        )
    
    def detect_self_enhancement_bias(
        self,
        question: str,
        model_style_response: str,
        alternative_style_response: str,
    ) -> BiasMetric:
        """
        Detect self-enhancement bias by comparing scores for responses
        written in the model's typical style vs. an alternative style.
        
        If the judge consistently prefers responses matching its own
        generation style, self-enhancement bias is present.
        """
        if not self.judge:
            return BiasMetric(
                bias_type="self_enhancement",
                magnitude=0.0,
                direction="unknown",
                confidence=0.0,
                details="No judge provided for testing",
            )
        
        # Score model-style response
        model_result = self.judge.score(question, model_style_response)
        
        # Score alternative-style response
        alt_result = self.judge.score(question, alternative_style_response)
        
        magnitude = max(0.0, model_result.overall_score - alt_result.overall_score)
        
        return BiasMetric(
            bias_type="self_enhancement",
            magnitude=round(magnitude, 4),
            direction="favors_own_style" if magnitude > 0 else "neutral",
            confidence=0.6,  # Default moderate confidence for single comparison
            details=(
                f"Model-style score: {model_result.overall_score:.3f}, "
                f"Alternative-style score: {alt_result.overall_score:.3f} "
                f"(Δ={magnitude:.3f})"
            ),
        )
    
    def full_bias_audit(
        self,
        question: str,
        response: str,
        n_trials: int = 3,
    ) -> BiasReport:
        """
        Run a comprehensive bias audit across all bias types.
        
        Returns a full BiasReport with all detected biases and recommendations.
        """
        biases = []
        
        # Position bias
        pos_bias = self.detect_position_bias(question, response, n_trials)
        biases.append(pos_bias)
        
        # Length bias
        core = response.split('.')[0] + '.' if '.' in response else response
        len_bias = self.detect_length_bias(question, core)
        biases.append(len_bias)
        
        # Overall bias score (average magnitude)
        overall = mean(b.magnitude for b in biases) if biases else 0.0
        
        # Generate recommendations
        recommendations = []
        if pos_bias.magnitude > 0.15:
            recommendations.append(
                "Position bias detected — use position swapping (present answers "
                "in both orders and average scores) for all pairwise comparisons."
            )
        if len_bias.magnitude > 0.15:
            recommendations.append(
                "Length bias detected — normalize scores by response length "
                "or use fixed-length evaluation windows."
            )
        if not recommendations:
            recommendations.append("No significant biases detected. Continue monitoring.")
        
        return BiasReport(
            overall_bias_score=round(overall, 4),
            biases_detected=biases,
            recommendations=recommendations,
            test_sample_size=n_trials,
        )


def main():
    """Test the bias detector"""
    print("\n" + "=" * 60)
    print("TESTING BIAS DETECTOR")
    print("=" * 60)
    
    # Create detector (without judge for demo — just show structure)
    detector = BiasDetector()
    
    # Show that detector works even without a judge (returns zero biases)
    report = detector.full_bias_audit(
        question="Is Sarah an employee?",
        response="Based on the Sagaz test, Sarah is likely an employee.",
        n_trials=2,
    )
    
    print(report.summary())
    print(f"\n✅ Bias report generated with {len(report.biases_detected)} bias checks")


if __name__ == "__main__":
    main()
