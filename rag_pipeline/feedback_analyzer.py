# RAG Pipeline - User Feedback Loop
"""
Feedback collection and analysis system,
inspired by ByteDance RAG Guideline §6.3.3.

Features:
  - Collect "useful / not useful / wrong" feedback per response
  - Categorize errors (data_error, incomplete, off_topic, hallucination)
  - Periodic analysis: error rate by category, flagging repeat failures
  - Export high-rated Q&A pairs as few-shot examples
  - Weekly summary report
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from collections import Counter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FeedbackEntry:
    """A single user feedback record."""
    query_id: str
    query_text: str
    answer_text: str
    rating: str                  # "useful", "not_useful", "wrong"
    error_type: Optional[str]    # "data_error", "incomplete", "off_topic", "hallucination"
    comment: Optional[str]
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Feedback Store
# ---------------------------------------------------------------------------

class FeedbackStore:
    """
    Append-only JSONL storage for user feedback.

    ByteDance §6.3.3: "有用/无用/错误" three-button feedback collection
    with error categorisation and root cause analysis.
    """

    def __init__(self, store_path: Optional[str] = None):
        self.store_path = store_path or str(
            Path(__file__).parent.parent / "data" / "feedback.jsonl"
        )
        Path(self.store_path).parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: FeedbackEntry):
        """Append a feedback entry to the store."""
        try:
            with open(self.store_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            logger.info(f"Feedback recorded: {entry.rating} for query_id={entry.query_id}")
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")

    def load_all(self) -> List[FeedbackEntry]:
        """Load all feedback entries from disk."""
        entries = []
        if not os.path.exists(self.store_path):
            return entries
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        entries.append(FeedbackEntry(**data))
        except Exception as e:
            logger.error(f"Failed to load feedback: {e}")
        return entries

    def count(self) -> int:
        """Count total feedback entries."""
        if not os.path.exists(self.store_path):
            return 0
        with open(self.store_path, "r") as f:
            return sum(1 for _ in f)


# ---------------------------------------------------------------------------
# Feedback Analyzer
# ---------------------------------------------------------------------------

class FeedbackAnalyzer:
    """
    Analyze feedback to identify patterns and export insights.

    ByteDance §6.3.3:
      - Negative feedback → root cause analysis → optimise system
      - Positive feedback → "excellent answer library" → few-shot examples
    """

    def __init__(self, store: Optional[FeedbackStore] = None):
        self.store = store or FeedbackStore()

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of all feedback."""
        entries = self.store.load_all()
        if not entries:
            return {"total": 0, "message": "No feedback collected yet"}

        total = len(entries)
        rating_counts = Counter(e.rating for e in entries)
        error_counts = Counter(e.error_type for e in entries if e.error_type)

        useful_rate = rating_counts.get("useful", 0) / total
        wrong_rate = rating_counts.get("wrong", 0) / total

        return {
            "total": total,
            "ratings": dict(rating_counts),
            "useful_rate": round(useful_rate, 3),
            "wrong_rate": round(wrong_rate, 3),
            "error_types": dict(error_counts),
            "time_range": {
                "earliest": entries[0].timestamp,
                "latest": entries[-1].timestamp,
            },
        }

    def get_flagged_queries(self, min_wrong_count: int = 3) -> List[Dict[str, Any]]:
        """
        Find queries with ≥N "wrong" ratings → candidates for manual review.

        ByteDance §6.3.3: queries with repeated negative feedback trigger
        root cause analysis.
        """
        entries = self.store.load_all()

        # Group by normalised query text
        wrong_by_query: Dict[str, List[FeedbackEntry]] = {}
        for e in entries:
            if e.rating == "wrong":
                key = e.query_text.strip().lower()
                wrong_by_query.setdefault(key, []).append(e)

        flagged = []
        for query_text, wrong_entries in wrong_by_query.items():
            if len(wrong_entries) >= min_wrong_count:
                error_types = Counter(
                    e.error_type for e in wrong_entries if e.error_type
                )
                flagged.append({
                    "query_text": wrong_entries[0].query_text,
                    "wrong_count": len(wrong_entries),
                    "error_types": dict(error_types),
                    "sample_comments": [
                        e.comment for e in wrong_entries if e.comment
                    ][:5],
                })

        return sorted(flagged, key=lambda x: x["wrong_count"], reverse=True)

    def export_few_shot_examples(
        self, max_examples: int = 20
    ) -> List[Dict[str, str]]:
        """
        Export high-rated Q&A pairs for use as few-shot prompt examples.

        ByteDance §6.3.3: positive feedback answers go into
        "excellent answer library" for prompt enrichment.
        """
        entries = self.store.load_all()

        useful = [
            e for e in entries
            if e.rating == "useful" and e.query_text and e.answer_text
        ]

        # Deduplicate by query text
        seen_queries = set()
        examples = []
        for e in useful:
            key = e.query_text.strip().lower()
            if key not in seen_queries:
                seen_queries.add(key)
                examples.append({
                    "query": e.query_text,
                    "answer": e.answer_text,
                })
            if len(examples) >= max_examples:
                break

        logger.info(f"Exported {len(examples)} few-shot examples from feedback")
        return examples

    def root_cause_breakdown(self) -> Dict[str, Any]:
        """
        Break down negative feedback by root cause category.

        ByteDance §6.3.3: categorise failures as model / prompt / retrieval
        problems and route to the appropriate optimisation.
        """
        entries = self.store.load_all()
        negative = [e for e in entries if e.rating in ("wrong", "not_useful")]

        if not negative:
            return {"total_negative": 0}

        # Map error types to system layers
        layer_mapping = {
            "hallucination": "generation_layer",
            "data_error": "retrieval_layer",
            "incomplete": "retrieval_layer",
            "off_topic": "retrieval_layer",
        }

        by_layer: Dict[str, int] = Counter()
        for e in negative:
            layer = layer_mapping.get(e.error_type, "unknown")
            by_layer[layer] += 1

        return {
            "total_negative": len(negative),
            "by_error_type": dict(Counter(
                e.error_type for e in negative if e.error_type
            )),
            "by_system_layer": dict(by_layer),
            "recommendation": self._generate_recommendation(by_layer),
        }

    def _generate_recommendation(self, by_layer: Counter) -> str:
        """Generate optimisation recommendation based on root cause."""
        total = sum(by_layer.values())
        if total == 0:
            return "No negative feedback to analyse."

        parts = []
        if by_layer.get("generation_layer", 0) / total > 0.4:
            parts.append(
                "High hallucination rate detected. Consider: "
                "(1) tightening fact-anchoring instructions in prompts, "
                "(2) lowering confidence gate threshold, "
                "(3) adding more few-shot examples from positive feedback."
            )
        if by_layer.get("retrieval_layer", 0) / total > 0.4:
            parts.append(
                "High retrieval-layer failure rate. Consider: "
                "(1) improving chunk quality / re-indexing, "
                "(2) adjusting hybrid search weights, "
                "(3) expanding the knowledge base for missing topics."
            )

        return " | ".join(parts) if parts else "No dominant failure pattern detected."
