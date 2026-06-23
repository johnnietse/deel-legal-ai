# Evaluation Framework
"""
Dynamic legal AI evaluation framework with anti-contamination
and multi-dimensional scoring capabilities.
"""

from evaluation.dynamic_benchmark import LegalBenchmarkGenerator, TestCase, TestSuite
from evaluation.llm_judge import DebiasedLegalJudge, JudgeResult
from evaluation.bias_detector import BiasDetector, BiasReport
from evaluation.benchmark_runner import BenchmarkRunner, BenchmarkReport

__all__ = [
    "LegalBenchmarkGenerator",
    "TestCase",
    "TestSuite",
    "DebiasedLegalJudge",
    "JudgeResult",
    "BiasDetector",
    "BiasReport",
    "BenchmarkRunner",
    "BenchmarkReport",
]
