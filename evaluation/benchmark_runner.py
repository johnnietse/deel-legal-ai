# Evaluation Framework - Benchmark Runner
"""
CLI tool for running legal AI benchmarks and generating reports.

Orchestrates the full evaluation pipeline:
1. Generate dynamic test suite (or load existing one)
2. Run each test case through the target AI system
3. Score responses using multi-dimensional evaluator
4. Aggregate results and generate report
5. Optionally run bias detection on the judge itself
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EVAL_RESULTS_DIR, EVAL_DEFAULT_N_CASES, EVAL_DIMENSIONS

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class CaseResult:
    """Result of evaluating a single test case"""
    case_id: str
    difficulty: str
    expected_classification: str
    predicted_classification: str
    is_correct: bool
    dimension_scores: Dict[str, float]
    overall_score: float
    response_excerpt: str


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark evaluation report"""
    report_id: str
    generated_at: str
    suite_id: str
    model_name: str
    
    # Aggregate metrics
    total_cases: int
    correct_classifications: int
    accuracy: float
    mean_overall_score: float
    
    # Per-dimension averages
    dimension_averages: Dict[str, float]
    
    # Per-difficulty breakdown
    difficulty_breakdown: Dict[str, Dict[str, float]]
    
    # Individual results
    case_results: List[CaseResult]
    
    # Metadata
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "suite_id": self.suite_id,
            "model_name": self.model_name,
            "total_cases": self.total_cases,
            "correct_classifications": self.correct_classifications,
            "accuracy": self.accuracy,
            "mean_overall_score": self.mean_overall_score,
            "dimension_averages": self.dimension_averages,
            "difficulty_breakdown": self.difficulty_breakdown,
            "case_results": [asdict(r) for r in self.case_results],
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }
    
    def save(self, path: Optional[str] = None):
        """Save report to JSON"""
        if path is None:
            path = EVAL_RESULTS_DIR / f"report_{self.report_id}.json"
        else:
            path = Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Report saved to {path}")
    
    def summary(self) -> str:
        """Generate human-readable report summary."""
        lines = [
            "=" * 60,
            "LEGAL AI BENCHMARK REPORT",
            "=" * 60,
            f"Model: {self.model_name}",
            f"Suite: {self.suite_id}",
            f"Date: {self.generated_at}",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
            "--- OVERALL METRICS ---",
            f"  Accuracy: {self.accuracy:.1%} ({self.correct_classifications}/{self.total_cases})",
            f"  Mean Quality Score: {self.mean_overall_score:.3f}",
            "",
            "--- DIMENSION SCORES ---",
        ]
        
        for dim, score in sorted(self.dimension_averages.items()):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"  {dim:30s} {bar} {score:.3f}")
        
        lines.append("\n--- DIFFICULTY BREAKDOWN ---")
        for diff, metrics in sorted(self.difficulty_breakdown.items()):
            lines.append(f"  {diff:10s}  accuracy={metrics.get('accuracy', 0):.1%}  "
                        f"quality={metrics.get('mean_score', 0):.3f}  "
                        f"n={metrics.get('count', 0)}")
        
        # Worst performing cases
        worst = sorted(self.case_results, key=lambda r: r.overall_score)[:3]
        if worst:
            lines.append("\n--- WEAKEST CASES ---")
            for r in worst:
                lines.append(f"  {r.case_id}: score={r.overall_score:.3f}, "
                           f"correct={'✅' if r.is_correct else '❌'}, "
                           f"difficulty={r.difficulty}")
        
        return "\n".join(lines)


class BenchmarkRunner:
    """
    Orchestrates benchmark execution against a legal AI system.
    
    Takes a target system function (or uses the built-in RAG pipeline)
    and runs it against a dynamically generated test suite, scoring
    results with the multi-dimensional evaluator.
    """
    
    def __init__(
        self,
        model_name: str = "deel-legal-ai",
        target_fn: Optional[Callable] = None,
    ):
        """
        Args:
            model_name: Name of the model being evaluated
            target_fn: Function that takes a question string and returns
                      a response string. If None, uses the built-in RAG pipeline.
        """
        self.model_name = model_name
        self.target_fn = target_fn
    
    def _get_default_target_fn(self) -> Callable:
        """Create default target function using the built-in RAG pipeline."""
        from rag_pipeline.rag_query import LegalRAGQuery
        
        rag = LegalRAGQuery()
        
        def target(question: str) -> str:
            response = rag.query(question, top_k=5)
            return response.answer
        
        return target
    
    def run(
        self,
        n_cases: int = EVAL_DEFAULT_N_CASES,
        seed: int = 42,
        difficulty_distribution: Optional[Dict[str, float]] = None,
        suite_path: Optional[str] = None,
    ) -> BenchmarkReport:
        """
        Run the full benchmark pipeline.
        
        Args:
            n_cases: Number of test cases to generate
            seed: Random seed for test generation
            difficulty_distribution: Custom difficulty distribution
            suite_path: Path to load existing suite (skip generation)
            
        Returns:
            BenchmarkReport with full results
        """
        start_time = datetime.now()
        
        # Get target function
        target = self.target_fn or self._get_default_target_fn()
        
        # Step 1: Generate or load test suite
        from evaluation.dynamic_benchmark import LegalBenchmarkGenerator, MultiDimensionalEvaluator
        
        if suite_path and Path(suite_path).exists():
            logger.info(f"Loading test suite from {suite_path}")
            with open(suite_path) as f:
                suite_data = json.load(f)
            from evaluation.dynamic_benchmark import TestSuite, TestCase
            cases = [TestCase(**c) for c in suite_data["cases"]]
            suite_id = suite_data["suite_id"]
        else:
            logger.info(f"Generating test suite with {n_cases} cases...")
            generator = LegalBenchmarkGenerator(base_seed=seed)
            suite = generator.generate_suite(n_cases, difficulty_distribution)
            cases = suite.cases
            suite_id = suite.suite_id
            suite.save()
        
        # Step 2: Initialize evaluator
        evaluator = MultiDimensionalEvaluator()
        
        # Step 3: Run each case
        logger.info(f"Running {len(cases)} test cases...")
        case_results = []
        
        for i, case in enumerate(cases):
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{len(cases)}")
            
            try:
                # Get model response
                prompt = (
                    f"Analyze the following worker classification scenario and determine "
                    f"whether the worker is an employee or independent contractor. "
                    f"Apply the Sagaz test and cite relevant legal precedents.\n\n"
                    f"SCENARIO:\n{case.scenario}"
                )
                response = target(prompt)
                
                # Evaluate response
                eval_result = evaluator.evaluate(response, case)
                
                # Determine predicted classification
                response_lower = response.lower()
                if "independent contractor" in response_lower:
                    predicted = "Independent Contractor"
                elif "employee" in response_lower:
                    predicted = "Employee"
                else:
                    predicted = "Unknown"
                
                is_correct = predicted == case.expected_classification
                
                case_results.append(CaseResult(
                    case_id=case.case_id,
                    difficulty=case.difficulty,
                    expected_classification=case.expected_classification,
                    predicted_classification=predicted,
                    is_correct=is_correct,
                    dimension_scores=eval_result.dimension_scores,
                    overall_score=eval_result.overall_score,
                    response_excerpt=response[:200],
                ))
                
            except Exception as e:
                logger.error(f"Error on case {case.case_id}: {e}")
                case_results.append(CaseResult(
                    case_id=case.case_id,
                    difficulty=case.difficulty,
                    expected_classification=case.expected_classification,
                    predicted_classification="Error",
                    is_correct=False,
                    dimension_scores={d: 0.0 for d in EVAL_DIMENSIONS},
                    overall_score=0.0,
                    response_excerpt=f"Error: {str(e)[:200]}",
                ))
        
        # Step 4: Aggregate results
        total = len(case_results)
        correct = sum(1 for r in case_results if r.is_correct)
        
        # Dimension averages
        dim_scores = {d: [] for d in EVAL_DIMENSIONS}
        for r in case_results:
            for d, s in r.dimension_scores.items():
                if d in dim_scores:
                    dim_scores[d].append(s)
        
        dim_averages = {
            d: round(sum(scores) / len(scores), 4) if scores else 0.0
            for d, scores in dim_scores.items()
        }
        
        # Difficulty breakdown
        diff_breakdown = {}
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in case_results if r.difficulty == diff]
            if diff_results:
                diff_correct = sum(1 for r in diff_results if r.is_correct)
                diff_breakdown[diff] = {
                    "count": len(diff_results),
                    "accuracy": round(diff_correct / len(diff_results), 4),
                    "mean_score": round(
                        sum(r.overall_score for r in diff_results) / len(diff_results), 4
                    ),
                }
        
        duration = (datetime.now() - start_time).total_seconds()
        
        import hashlib
        report_id = hashlib.md5(
            f"{self.model_name}_{suite_id}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]
        
        report = BenchmarkReport(
            report_id=report_id,
            generated_at=datetime.now().isoformat(),
            suite_id=suite_id,
            model_name=self.model_name,
            total_cases=total,
            correct_classifications=correct,
            accuracy=round(correct / total, 4) if total > 0 else 0.0,
            mean_overall_score=round(
                sum(r.overall_score for r in case_results) / total, 4
            ) if total > 0 else 0.0,
            dimension_averages=dim_averages,
            difficulty_breakdown=diff_breakdown,
            case_results=case_results,
            duration_seconds=round(duration, 1),
        )
        
        report.save()
        return report


def main():
    """CLI entry point for the benchmark runner"""
    parser = argparse.ArgumentParser(description="Legal AI Benchmark Runner")
    parser.add_argument("--n-cases", type=int, default=10, help="Number of test cases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-name", type=str, default="deel-legal-ai", help="Model name")
    parser.add_argument("--suite-path", type=str, default=None, help="Path to existing test suite")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--generate-only", action="store_true", help="Only generate test suite, don't run")
    
    args = parser.parse_args()
    
    if args.generate_only:
        from evaluation.dynamic_benchmark import LegalBenchmarkGenerator
        
        generator = LegalBenchmarkGenerator(base_seed=args.seed)
        suite = generator.generate_suite(n_cases=args.n_cases)
        
        output_path = args.output or str(EVAL_RESULTS_DIR / f"benchmark_{suite.suite_id}.json")
        suite.save(output_path)
        
        print(f"\n✅ Test suite generated: {suite.n_cases} cases")
        print(f"   Distribution: {suite.difficulty_distribution}")
        print(f"   Saved to: {output_path}")
    else:
        runner = BenchmarkRunner(model_name=args.model_name)
        report = runner.run(
            n_cases=args.n_cases,
            seed=args.seed,
            suite_path=args.suite_path,
        )
        
        print(report.summary())


if __name__ == "__main__":
    main()
