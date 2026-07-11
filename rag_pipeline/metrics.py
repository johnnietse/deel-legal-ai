# RAG Pipeline - Full-Pipeline Metrics & Monitoring
"""
Per-layer instrumentation and structured metrics logging,
inspired by ByteDance RAG Guideline §8.1.

Instruments:
  - Retrieval layer: latency, result count, avg similarity, search mode
  - Generation layer: latency, token count (input + output), confidence
  - Verification layer: grounding score, claim count, hallucination count
  - End-to-end: total latency, cost estimate, cache hits
  - Index health: vector count, BM25 index size, staleness

Emits structured JSON logs and provides weekly summary stats.
"""

import time
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

@contextmanager
def timed_stage(stage_name: str, metrics: "QueryMetrics"):
    """Context manager to time a pipeline stage."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics.set_latency(stage_name, elapsed_ms)


# ---------------------------------------------------------------------------
# Per-Query Metrics
# ---------------------------------------------------------------------------

@dataclass
class QueryMetrics:
    """Metrics collected for a single RAG query."""
    query_id: str = ""
    query_text: str = ""
    timestamp: str = ""
    user_id: str = ""  # Track which user made the query

    # Retrieval metrics
    retrieval_latency_ms: float = 0.0
    retrieval_result_count: int = 0
    retrieval_avg_score: float = 0.0
    retrieval_mode: str = ""          # "vector", "bm25", "hybrid"
    retrieval_bm25_weight: float = 0.0
    retrieval_vector_weight: float = 0.0
    retrieval_cache_hit: bool = False

    # Generation metrics
    generation_latency_ms: float = 0.0
    generation_input_tokens: int = 0   # approximate
    generation_output_tokens: int = 0  # approximate
    generation_model: str = ""
    generation_template_used: str = ""
    generation_cache_hit: bool = False

    # Confidence gate metrics
    confidence_score: float = 0.0
    confidence_action: str = ""        # "pass", "hedge", "refuse"

    # Verification metrics (optional; only when verify=True)
    verification_latency_ms: float = 0.0
    verification_grounding_score: float = 0.0
    verification_claim_count: int = 0
    verification_hallucination_count: int = 0

    # End-to-end metrics
    total_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0

    # Cache metrics
    embedding_cache_hit: bool = False

    def set_latency(self, stage: str, ms: float):
        """Set latency for a named stage."""
        attr = f"{stage}_latency_ms"
        if hasattr(self, attr):
            setattr(self, attr, round(ms, 2))

    def estimate_cost(
        self,
        embedding_cost_per_1k: float = 0.00002,   # Gemini embedding
        generation_cost_per_1k: float = 0.000075,  # Gemini Flash
    ):
        """Estimate the USD cost of this query."""
        embed_cost = (self.generation_input_tokens / 1000) * embedding_cost_per_1k
        gen_cost = (
            (self.generation_input_tokens + self.generation_output_tokens) / 1000
        ) * generation_cost_per_1k

        self.estimated_cost_usd = round(embed_cost + gen_cost, 6)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ---------------------------------------------------------------------------
# Metrics Collector (Aggregator)
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Collects and aggregates RAG pipeline metrics.

    ByteDance §8.1:
      - Daily lightweight evaluation (efficiency, user satisfaction)
      - Weekly full evaluation (accuracy, hallucination rate)
      - Monthly optimization review
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        max_in_memory: int = 10000,
    ):
        self.log_dir = log_dir or str(Path(__file__).parent.parent / "logs" / "metrics")
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self._metrics: List[QueryMetrics] = []
        self.max_in_memory = max_in_memory

    def record(self, metrics: QueryMetrics):
        """Record a completed query's metrics."""
        # Fill in timestamp if missing
        if not metrics.timestamp:
            metrics.timestamp = datetime.now().isoformat()

        # Estimate cost
        metrics.estimate_cost()

        # Log as structured JSON
        self._log_metrics(metrics)

        # Store in memory (bounded)
        self._metrics.append(metrics)
        if len(self._metrics) > self.max_in_memory:
            self._metrics = self._metrics[-self.max_in_memory:]

    def _log_metrics(self, metrics: QueryMetrics):
        """Write metrics to structured log file."""
        log_file = os.path.join(
            self.log_dir,
            f"metrics_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        )
        try:
            with open(log_file, "a") as f:
                f.write(metrics.to_json() + "\n")
        except Exception as e:
            logger.warning(f"Failed to write metrics log: {e}")

    # -- Summary statistics ------------------------------------------------

    def summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate summary statistics over recent queries.

        ByteDance §8.1: daily/weekly metric summaries.
        """
        data = self._metrics[-last_n:] if last_n else self._metrics
        if not data:
            return {"total_queries": 0}

        n = len(data)

        def _avg(vals):
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        def _p95(vals):
            if not vals:
                return 0.0
            sorted_vals = sorted(vals)
            idx = int(0.95 * len(sorted_vals))
            return round(sorted_vals[min(idx, len(sorted_vals) - 1)], 2)

        retrieval_latencies = [m.retrieval_latency_ms for m in data if m.retrieval_latency_ms > 0]
        generation_latencies = [m.generation_latency_ms for m in data if m.generation_latency_ms > 0]
        total_latencies = [m.total_latency_ms for m in data if m.total_latency_ms > 0]
        confidence_scores = [m.confidence_score for m in data if m.confidence_score > 0]
        costs = [m.estimated_cost_usd for m in data]

        cache_hits = sum(1 for m in data if m.retrieval_cache_hit or m.generation_cache_hit)
        confidence_passes = sum(1 for m in data if m.confidence_action == "pass")
        hedges = sum(1 for m in data if m.confidence_action == "hedge")
        refusals = sum(1 for m in data if m.confidence_action == "refuse")

        return {
            "total_queries": n,
            "time_range": {
                "from": data[0].timestamp if data else "",
                "to": data[-1].timestamp if data else "",
            },
            "latency": {
                "retrieval_avg_ms": _avg(retrieval_latencies),
                "retrieval_p95_ms": _p95(retrieval_latencies),
                "generation_avg_ms": _avg(generation_latencies),
                "generation_p95_ms": _p95(generation_latencies),
                "total_avg_ms": _avg(total_latencies),
                "total_p95_ms": _p95(total_latencies),
            },
            "quality": {
                "avg_confidence_score": _avg(confidence_scores),
                "confidence_pass_rate": round(confidence_passes / n, 3) if n else 0,
                "hedge_rate": round(hedges / n, 3) if n else 0,
                "refusal_rate": round(refusals / n, 3) if n else 0,
            },
            "cost": {
                "total_usd": round(sum(costs), 4),
                "avg_per_query_usd": _avg(costs),
            },
            "cache": {
                "hit_count": cache_hits,
                "hit_rate": round(cache_hits / n, 3) if n else 0,
            },
            "retrieval_modes": {
                mode: sum(1 for m in data if m.retrieval_mode == mode)
                for mode in set(m.retrieval_mode for m in data if m.retrieval_mode)
            },
        }

    def get_worst_queries(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n queries with lowest confidence scores."""
        sorted_metrics = sorted(
            self._metrics,
            key=lambda m: m.confidence_score
        )
        return [m.to_dict() for m in sorted_metrics[:n]]
