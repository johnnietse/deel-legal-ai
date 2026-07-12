# Weekly Evaluation Pipeline (ByteDance §5.4/§6.5)
"""
Scheduled evaluation orchestrator for the Legal AI RAG system.

Implements ByteDance RAG Practice Manual §5.4 (index & retrieval evaluation)
and §6.5 (generation quality evaluation):

  1. Generate a dynamic, anti-contamination test suite
  2. Run each case through the RAG pipeline
  3. Score responses using multi-dimensional evaluator + LLM judge
  4. Produce a comprehensive JSON report
  5. Compare against previous week's results (if available)
  6. Save timestamped report for trend tracking

Usage:
    python scripts/run_weekly_eval.py              # Full eval (default: 10 cases)
    python scripts/run_weekly_eval.py --n-cases 50 # More thorough eval
    python scripts/run_weekly_eval.py --quick       # Quick smoke test (5 cases)
    python scripts/run_weekly_eval.py --compare     # Compare with last report
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EVAL_RESULTS_DIR, LOG_FORMAT, LOG_LEVEL

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


WEEKLY_REPORT_DIR = EVAL_RESULTS_DIR / "weekly"


def generate_suite(n_cases: int, seed: int = 42) -> Dict[str, Any]:
    """Generate a dynamic benchmark suite (ByteDance §5.4)."""
    from evaluation.dynamic_benchmark import LegalBenchmarkGenerator

    logger.info(f"Generating test suite: {n_cases} cases, seed={seed}...")
    generator = LegalBenchmarkGenerator(base_seed=seed)
    suite = generator.generate_suite(n_cases=n_cases)

    # Save to disk
    suite_path = WEEKLY_REPORT_DIR / f"suite_{suite.suite_id}.json"
    WEEKLY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suite.save(str(suite_path))

    logger.info(f"Test suite saved: {suite.n_cases} cases → {suite_path}")
    return {
        "suite_id": suite.suite_id,
        "n_cases": suite.n_cases,
        "difficulty_distribution": suite.difficulty_distribution,
        "path": str(suite_path),
    }


def evaluate_suite(
    suite_path: str,
    model_name: str = "deel-legal-ai",
) -> Dict[str, Any]:
    """
    Run full evaluation pipeline on a test suite (ByteDance §6.5).

    Uses the existing BenchmarkRunner for classification accuracy +
    multi-dimensional quality scoring across 6 rubric dimensions.
    """
    from evaluation.benchmark_runner import BenchmarkRunner

    logger.info(f"Evaluating suite: {suite_path}")
    runner = BenchmarkRunner(model_name=model_name)
    report = runner.run(
        n_cases=0,  # Will load from suite_path instead
        suite_path=suite_path,
    )

    # Build structured summary
    summary = {
        "report_id": report.report_id,
        "generated_at": report.generated_at,
        "suite_id": report.suite_id,
        "model_name": report.model_name,
        "total_cases": report.total_cases,
        "accuracy": report.accuracy,
        "mean_overall_score": report.mean_overall_score,
        "dimension_averages": report.dimension_averages,
        "difficulty_breakdown": report.difficulty_breakdown,
        "duration_seconds": report.duration_seconds,
    }

    logger.info(f"Evaluation complete: accuracy={report.accuracy:.1%}, "
                f"mean_score={report.mean_overall_score:.3f}")
    return summary


def run_llm_judge_evaluation(
    suite_path: str,
    n_samples: int = 5,
) -> List[Dict[str, Any]]:
    """
    Run DebiasedLegalJudge pairwise evaluation on a sample of cases.

    This provides deeper quality analysis than the multi-dim evaluator:
    - Rubric decomposition scoring (ByteDance §6.5 rubric decomposition)
    - Position-bias detection via swap trials
    - Length-bias normalization
    - Self-enhancement bias mitigation

    Compares the RAG response against a reference answer for each case.
    """
    from evaluation.llm_judge import DebiasedLegalJudge
    from evaluation.dynamic_benchmark import TestCase, MultiDimensionalEvaluator
    from rag_pipeline.rag_query import LegalRAGQuery

    # Load suite
    with open(suite_path) as f:
        suite_data = json.load(f)

    cases = [TestCase(**c) for c in suite_data["cases"]]
    sample_cases = cases[:min(n_samples, len(cases))]

    # Initialize
    judge = DebiasedLegalJudge()
    rag = LegalRAGQuery()
    eval_scores = MultiDimensionalEvaluator()

    results = []
    for i, case in enumerate(sample_cases):
        logger.info(f"Judge eval {i + 1}/{len(sample_cases)}: {case.case_id}")

        # Get RAG response
        prompt = (
            f"Analyze the following worker classification scenario and determine "
            f"whether the worker is an employee or independent contractor. "
            f"Apply the Sagaz test and cite relevant legal precedents.\n\n"
            f"SCENARIO:\n{case.scenario}"
        )
        try:
            response = rag.query(prompt, top_k=5)
            response_text = response.answer
        except Exception as e:
            logger.warning(f"RAG query failed for {case.case_id}: {e}")
            response_text = f"Error: {str(e)}"

        # Score using DebiasedLegalJudge (rubric decomposition)
        try:
            judge_result = judge.score(
                question=prompt[:300],
                response=response_text,
                context=case.scenario[:300],
            )
        except Exception as e:
            logger.warning(f"Judge scoring failed for {case.case_id}: {e}")
            judge_result = None

        # Also get multi-dim evaluator score
        try:
            dim_result = eval_scores.evaluate(response_text, case)
        except Exception as e:
            logger.warning(f"Dim evaluator failed for {case.case_id}: {e}")
            dim_result = None

        entry = {
            "case_id": case.case_id,
            "difficulty": case.difficulty,
            "expected_classification": case.expected_classification,
            "response_excerpt": response_text[:300],
            "judge_score": {
                "overall": judge_result.overall_score if judge_result else None,
                "components": judge_result.component_scores if judge_result else {},
                "debiasing": judge_result.debiasing_applied if judge_result else [],
            } if judge_result else None,
            "dimension_scores": dim_result.dimension_scores if dim_result else {},
            "overall_dim_score": dim_result.overall_score if dim_result else None,
        }
        results.append(entry)

    return results


def load_previous_report() -> Optional[Dict[str, Any]]:
    """Load the most recent weekly evaluation report for comparison."""
    if not WEEKLY_REPORT_DIR.exists():
        return None

    report_files = sorted(WEEKLY_REPORT_DIR.glob("weekly_report_*.json"))
    if not report_files:
        return None

    with open(report_files[-1]) as f:
        return json.load(f)


def save_weekly_report(
    suite_info: Dict[str, Any],
    eval_results: Dict[str, Any],
    judge_results: List[Dict[str, Any]],
    previous_report: Optional[Dict[str, Any]],
) -> str:
    """Save comprehensive weekly report with comparison data."""
    WEEKLY_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "report_id": f"weekly_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "generated_at": datetime.now().isoformat(),
        "suite": suite_info,
        "evaluation": eval_results,
        "llm_judge_evaluation": {
            "n_samples": len(judge_results),
            "samples": judge_results,
            "mean_score": (
                round(
                    sum(r["judge_score"]["overall"] for r in judge_results
                        if r.get("judge_score") and r["judge_score"]["overall"] is not None)
                    / max(len([r for r in judge_results
                              if r.get("judge_score") and r["judge_score"]["overall"] is not None]), 1),
                    4
                )
                if judge_results else None
            ),
        },
    }

    # Add comparison with previous week if available
    if previous_report:
        prev_eval = previous_report.get("evaluation", {})
        prev_acc = prev_eval.get("accuracy", 0)
        prev_score = prev_eval.get("mean_overall_score", 0)
        curr_acc = eval_results.get("accuracy", 0)
        curr_score = eval_results.get("mean_overall_score", 0)

        report["comparison"] = {
            "previous_report_id": previous_report.get("report_id"),
            "previous_date": previous_report.get("generated_at"),
            "accuracy_change": round(curr_acc - prev_acc, 4),
            "quality_score_change": round(curr_score - prev_score, 4),
            "accuracy_regression": curr_acc < prev_acc,
            "quality_regression": curr_score < prev_score,
        }

        if report["comparison"]["accuracy_regression"]:
            logger.warning(f"⚠️  Accuracy regression detected: "
                          f"{prev_acc:.1%} → {curr_acc:.1%}")
        if report["comparison"]["quality_regression"]:
            logger.warning(f"⚠️  Quality score regression detected: "
                          f"{prev_score:.3f} → {curr_score:.3f}")

    # Save weekly report
    report_path = WEEKLY_REPORT_DIR / f"weekly_report_{report['report_id']}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Weekly report saved to {report_path}")
    return str(report_path)


def print_summary(report_path: str):
    """Print a human-readable evaluation summary."""
    with open(report_path) as f:
        report = json.load(f)

    eval_data = report.get("evaluation", {})
    print("\n" + "=" * 60)
    print("WEEKLY EVALUATION REPORT")
    print("=" * 60)
    print(f"Report ID:  {report['report_id']}")
    print(f"Date:       {report['generated_at']}")
    print(f"Suite:      {report['suite']['suite_id']}")
    print(f"Cases:      {report['suite']['n_cases']}")

    print("\n--- CLASSIFICATION ACCURACY ---")
    print(f"  Accuracy:            {eval_data.get('accuracy', 0):.1%}")
    print(f"  Mean Quality Score:  {eval_data.get('mean_overall_score', 0):.3f}")

    print("\n--- DIMENSION SCORES ---")
    for dim, score in sorted(eval_data.get("dimension_averages", {}).items()):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {dim:30s} {bar} {score:.3f}")

    print("\n--- DIFFICULTY BREAKDOWN ---")
    for diff, metrics in sorted(eval_data.get("difficulty_breakdown", {}).items()):
        print(f"  {diff:10s}  accuracy={metrics.get('accuracy', 0):.1%}  "
              f"quality={metrics.get('mean_score', 0):.3f}  "
              f"n={metrics.get('count', 0)}")

    judge_data = report.get("llm_judge_evaluation", {})
    if judge_data.get("mean_score") is not None:
        print(f"\n--- LLM JUDGE QUALITY ---")
        print(f"  Judge Score (sampled): {judge_data['mean_score']:.3f}")
        print(f"  Samples evaluated:     {judge_data['n_samples']}")

    comparison = report.get("comparison")
    if comparison:
        print(f"\n--- WEEK-OVER-WEEK ---")
        acc_change = comparison.get("accuracy_change", 0)
        qual_change = comparison.get("quality_score_change", 0)
        acc_arrow = "⬆" if acc_change > 0 else "⬇" if acc_change < 0 else "➡"
        qual_arrow = "⬆" if qual_change > 0 else "⬇" if qual_change < 0 else "➡"
        print(f"  Accuracy:       {acc_arrow} {acc_change:+.1%}")
        print(f"  Quality Score:  {qual_arrow} {qual_change:+.3f}")
        if comparison.get("accuracy_regression"):
            print(f"  ⚠️  ACCURACY REGRESSION DETECTED — review recent changes")
        if comparison.get("quality_regression"):
            print(f"  ⚠️  QUALITY REGRESSION DETECTED — review generation changes")

    print("\n" + "=" * 60)
    print(f"Full report: {report_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Weekly Legal AI RAG Evaluation Pipeline (ByteDance §5.4/§6.5)"
    )
    parser.add_argument("--n-cases", type=int, default=10, help="Number of test cases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-name", type=str, default="deel-legal-ai", help="Model name")
    parser.add_argument("--judge-samples", type=int, default=5, help="LLM judge sample size")
    parser.add_argument("--compare", action="store_true", help="Compare with previous report")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (5 cases)")
    parser.add_argument("--generate-only", action="store_true",
                       help="Only generate test suite, don't evaluate")

    args = parser.parse_args()

    # Quick mode override
    if args.quick:
        args.n_cases = 5
        args.judge_samples = 2

    WEEKLY_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate benchmark suite
    suite_info = generate_suite(n_cases=args.n_cases, seed=args.seed)

    if args.generate_only:
        print(f"\n✅ Suite generated: {suite_info['path']}")
        return

    # Step 2: Load previous report for comparison (optional)
    previous_report = load_previous_report() if args.compare else None
    if previous_report:
        logger.info(f"Loaded previous report: {previous_report['report_id']} "
                    f"({previous_report['generated_at']})")

    # Step 3: Run full evaluation
    eval_results = evaluate_suite(
        suite_path=suite_info["path"],
        model_name=args.model_name,
    )

    # Step 4: Run LLM judge evaluation (ByteDance §6.5 rubric decomposition)
    judge_results = run_llm_judge_evaluation(
        suite_path=suite_info["path"],
        n_samples=args.judge_samples,
    )

    # Step 5: Save and print report
    report_path = save_weekly_report(
        suite_info=suite_info,
        eval_results=eval_results,
        judge_results=judge_results,
        previous_report=previous_report,
    )

    print_summary(report_path)


if __name__ == "__main__":
    main()
