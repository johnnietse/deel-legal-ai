# RAG Pipeline - Pluggable Reranker (RAGFlow bge-reranker-v2-m3)
"""
Cross-encoder reranker inspired by RAGFlow's bge-reranker-v2-m3.

Backends:
  - "off"   (default): identity passthrough — results returned unchanged
  - "tei"   : HTTP POST to a Text Embeddings Inference server (/rerank)
  - "local" : FlagEmbedding FlagReranker (lazy import, Windows/no-GPU safe)

Every backend degrades gracefully: on any failure the input order is
returned and a warning is logged — reranking never raises.
"""

import logging
from typing import List, Optional

import requests

from config import (
    RERANKER_BACKEND,
    RERANKER_MODEL,
    RERANKER_TEI_URL,
    RERANKER_TIMEOUT,
)
from rag_pipeline.hybrid_retriever import HybridResult

logger = logging.getLogger(__name__)


class Reranker:
    """Re-scores hybrid retrieval results with a cross-encoder."""

    def __init__(self, backend: Optional[str] = None):
        self.backend = (backend or RERANKER_BACKEND or "off").lower()
        self._local_model = None  # lazy-loaded FlagReranker instance

    def rerank(
        self,
        query: str,
        results: List[HybridResult],
        top_k: int,
    ) -> List[HybridResult]:
        """Re-score results and return the top_k reordered (input order on failure)."""
        if not results:
            return results

        if self.backend == "tei":
            return self._rerank_tei(query, results, top_k)
        if self.backend == "local":
            return self._rerank_local(query, results, top_k)

        # "off" (default) — identity passthrough
        return results[:top_k]

    # -- TEI backend -------------------------------------------------------

    def _rerank_tei(
        self,
        query: str,
        results: List[HybridResult],
        top_k: int,
    ) -> List[HybridResult]:
        try:
            payload = {
                "model": RERANKER_MODEL,
                "query": query,
                "texts": [r.content for r in results],
            }
            resp = requests.post(
                f"{RERANKER_TEI_URL.rstrip('/')}/rerank",
                json=payload,
                timeout=RERANKER_TIMEOUT,
            )
            resp.raise_for_status()
            scored = _apply_scores(results, resp.json().get("results", []))
            return scored[:top_k]
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.warning("TEI rerank failed (%s); returning input order", exc)
            return results[:top_k]

    # -- Local backend -----------------------------------------------------

    def _rerank_local(
        self,
        query: str,
        results: List[HybridResult],
        top_k: int,
    ) -> List[HybridResult]:
        try:
            model = self._get_local_model()
            if model is None:
                return results[:top_k]

            pairs = [[query, r.content] for r in results]
            scores = model.compute_score(pairs)
            if isinstance(scores, float):
                scores = [scores]
            scored = _apply_scores(results, [
                {"index": i, "relevance_score": float(score)}
                for i, score in enumerate(scores)
            ])
            return scored[:top_k]
        except Exception as exc:
            logger.warning("Local rerank failed (%s); returning input order", exc)
            return results[:top_k]

    def _get_local_model(self):
        """Lazily import FlagEmbedding; None (→ 'off' behaviour) if unavailable."""
        if self._local_model is not None:
            return self._local_model

        try:
            from FlagEmbedding import FlagReranker  # lazy: avoids hard torch import
            self._local_model = FlagReranker(RERANKER_MODEL)
            return self._local_model
        except ImportError as exc:
            logger.warning(
                "FlagEmbedding/torch unavailable (%s); reranker degrades to 'off'", exc
            )
            return None
        except Exception as exc:
            logger.warning(
                "Failed to load local reranker model (%s); degrades to 'off'", exc
            )
            return None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _apply_scores(
    results: List[HybridResult],
    raw_results: List[dict],
) -> List[HybridResult]:
    """
    Map TEI-style [{index, relevance_score}] onto new HybridResult copies.

    Returns a new list reordered by score descending; never mutates inputs.
    Missing indices keep their original score. Empty input → input order.
    """
    score_by_index = {
        int(item.get("index")): float(item.get("relevance_score", 0.0))
        for item in raw_results
    }
    if not score_by_index:
        return results

    scored: List[HybridResult] = []
    for idx, result in enumerate(results):
        scored.append(HybridResult(
            id=result.id,
            score=score_by_index.get(idx, result.score),
            content=result.content,
            metadata=result.metadata,
            bm25_score=result.bm25_score,
            vector_score=result.vector_score,
            retrieved_by=list(result.retrieved_by),
        ))

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored
