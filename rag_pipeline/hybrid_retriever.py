# RAG Pipeline - Hybrid Retriever (BM25 + Vector Fusion)
"""
Hybrid retrieval combining sparse (BM25) and dense (vector) search,
inspired by ByteDance RAG Guideline §5.2.

Key features:
  - BM25 keyword retrieval for precise legal citation matching
  - Reciprocal Rank Fusion (RRF) to merge sparse + dense results
  - Query-type classifier to dynamically weight BM25 vs. vector
  - Maximum Marginal Relevance (MMR) for diversity in final results
"""

import re
import math
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

from config import KEYWORD_BOOST_ENABLED, KEYWORD_BOOST_MULTIPLIER
from rag_pipeline.keyword_booster import extract_boost_terms, apply_boost

if TYPE_CHECKING:
    from rag_pipeline.reranker import Reranker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    """A single result from hybrid retrieval."""
    id: str
    score: float                       # fused score
    content: str
    metadata: Dict[str, Any]
    bm25_score: float = 0.0            # raw BM25 score (0 if not retrieved)
    vector_score: float = 0.0          # raw vector similarity (0 if not retrieved)
    retrieved_by: List[str] = field(default_factory=list)  # ["bm25", "vector"]


# ---------------------------------------------------------------------------
# Metadata Filter Helpers
# ---------------------------------------------------------------------------

def _matches_filter(metadata: Dict[str, Any], filter: Optional[Dict[str, Any]]) -> bool:
    """
    Return True if a doc's metadata satisfies all filter conditions.

    Supported operators:
      - Equality:      {'court': 'ONSC'}            -> metadata['court'] == 'ONSC'
      - $in:           {'jurisdiction': {'$in': [...]}} -> metadata['jurisdiction'] in [...]
      - $eq:           {'court': {'$eq': 'ONSC'}}   -> metadata['court'] == 'ONSC'

    A doc matches only if ALL filter keys match. A missing metadata field
    excludes the doc, unless the filter value is None (which matches any doc).
    """
    if not filter:
        return True
    for key, value in filter.items():
        if value is None:
            continue
        field_value = metadata.get(key)
        if isinstance(value, dict):
            if "$in" in value:
                if field_value not in value["$in"]:
                    return False
            elif "$eq" in value:
                if field_value != value["$eq"]:
                    return False
            else:
                return False
        else:
            if field_value != value:
                return False
    return True


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    In-memory BM25 index over document chunks.

    Keeps a local inverted index of all chunks so that keyword-heavy
    legal queries (e.g. "ESA s. 56 notice period") can be handled
    without a vector-DB round-trip.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # Corpus storage
        self._docs: List[Dict[str, Any]] = []   # original chunk dicts
        self._tokenized: List[List[str]] = []    # tokenized content
        self._avg_dl: float = 0.0
        self._idf: Dict[str, float] = {}
        self._indexed: bool = False

    # -- Tokeniser (legal-aware) -------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        text = text.lower()
        # Keep legal citation patterns intact (e.g. "2020 ONSC 1234")
        # but split normal prose
        tokens = re.findall(r'\b[\w\-\.]+\b', text)
        return tokens

    # -- Index building ----------------------------------------------------

    def build(self, chunks: List[Dict[str, Any]], content_key: str = "content"):
        """
        Build BM25 index from a list of chunk dicts.

        Each dict must have at least a `content_key` field with text,
        plus an ``id`` (or ``chunk_id``) field.
        """
        self._docs = chunks
        self._tokenized = [self._tokenize(c.get(content_key, "")) for c in chunks]

        # Average document length
        total_tokens = sum(len(t) for t in self._tokenized)
        self._avg_dl = total_tokens / max(len(self._tokenized), 1)

        # IDF calculation
        n = len(self._tokenized)
        df: Dict[str, int] = {}
        for tokens in self._tokenized:
            seen = set(tokens)
            for t in seen:
                df[t] = df.get(t, 0) + 1

        self._idf = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }
        self._indexed = True
        logger.info(f"BM25 index built: {n} documents, {len(self._idf)} terms")

    # -- Search ------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 50,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Return (doc_index, bm25_score) pairs sorted by score descending.

        Args:
            filter: Optional metadata filter; docs whose metadata does not
                    match are excluded (see _matches_filter).
        """
        if not self._indexed:
            logger.warning("BM25 index not built; returning empty results")
            return []

        # Restrict scoring to docs that satisfy the metadata filter
        valid_indices = [
            i for i, doc in enumerate(self._docs)
            if _matches_filter(doc.get("metadata", {}), filter)
        ]
        if not valid_indices:
            return []

        query_tokens = self._tokenize(query)
        scores: List[float] = [0.0] * len(self._tokenized)

        for qt in query_tokens:
            idf = self._idf.get(qt, 0.0)
            if idf == 0.0:
                continue
            for idx in valid_indices:
                doc_tokens = self._tokenized[idx]
                tf = doc_tokens.count(qt)
                dl = len(doc_tokens)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
                scores[idx] += idf * numerator / denominator

        # Sort by score descending, keep top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in ranked[:top_k] if score > 0]

    def search_as_results(
        self,
        query: str,
        top_k: int = 50,
        id_key: str = "chunk_id",
        content_key: str = "content",
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[HybridResult]:
        """Search and return as HybridResult objects."""
        raw = self.search(query, top_k=top_k, filter=filter)
        results = []
        for idx, score in raw:
            doc = self._docs[idx]
            results.append(HybridResult(
                id=doc.get(id_key, doc.get("id", f"bm25_{idx}")),
                score=score,
                content=doc.get(content_key, ""),
                metadata=doc.get("metadata", {}),
                bm25_score=score,
                vector_score=0.0,
                retrieved_by=["bm25"],
            ))
        return results


# ---------------------------------------------------------------------------
# Query Classifier
# ---------------------------------------------------------------------------

class QueryClassifier:
    """
    Rule-based query classifier that determines the optimal retrieval
    strategy (ByteDance §5.2.3 dynamic weight adjustment).

    Categories:
      - "keyword"  → BM25-heavy (legal citations, statute references, exact terms)
      - "semantic"  → vector-heavy (open-ended, conceptual questions)
      - "hybrid"    → balanced
    """

    # Patterns that indicate keyword-heavy queries
    KEYWORD_PATTERNS = [
        r'\d{4}\s+[A-Z]{2,6}\s+\d+',            # Citation: "2020 ONSC 1234"
        r'\b[Ss]\.?\s*\d+',                        # Section reference: "s. 56"
        r'\b[Ss]ection\s+\d+',                     # "Section 56"
        r'\bESA\b|\bOHSA\b|\bCLC\b|\bSCC\b',      # Common legal acronyms
        r'\bAct\b.*\b\d{4}\b',                     # "Employment Standards Act 2000"
        r'\b[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+',    # Case names: "Sagaz v. 671122"
    ]

    SEMANTIC_INDICATORS = [
        "what factors", "how does", "explain", "describe",
        "what is the difference", "compare", "relationship between",
        "what are", "why is", "when should", "in what circumstances",
        "what considerations", "how to determine",
    ]

    @classmethod
    def classify(cls, query: str) -> str:
        """Classify a query as 'keyword', 'semantic', or 'hybrid'."""
        query_lower = query.lower()

        keyword_hits = sum(
            1 for p in cls.KEYWORD_PATTERNS
            if re.search(p, query, re.IGNORECASE)
        )
        semantic_hits = sum(
            1 for ind in cls.SEMANTIC_INDICATORS
            if ind in query_lower
        )

        if keyword_hits >= 2:
            return "keyword"
        if keyword_hits >= 1 and semantic_hits == 0:
            return "keyword"
        if semantic_hits >= 2:
            return "semantic"
        if semantic_hits >= 1 and keyword_hits == 0:
            return "semantic"

        return "hybrid"

    @classmethod
    def get_weights(cls, query: str) -> Tuple[float, float]:
        """
        Return (bm25_weight, vector_weight) based on query type.

        ByteDance §5.2.3:
          keyword → bm25=0.7, vector=0.3
          semantic → bm25=0.2, vector=0.8
          hybrid  → bm25=0.4, vector=0.6
        """
        query_type = cls.classify(query)
        weights = {
            "keyword":  (0.7, 0.3),
            "semantic": (0.2, 0.8),
            "hybrid":   (0.4, 0.6),
        }
        return weights[query_type]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: List[List[HybridResult]],
    k: int = 60,
) -> List[HybridResult]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i) across all lists.
    """
    scores: Dict[str, float] = {}
    result_map: Dict[str, HybridResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            rrf_score = 1.0 / (k + rank + 1)
            scores[result.id] = scores.get(result.id, 0.0) + rrf_score

            if result.id not in result_map:
                result_map[result.id] = result
            else:
                # Merge retrieval sources
                existing = result_map[result.id]
                for source in result.retrieved_by:
                    if source not in existing.retrieved_by:
                        existing.retrieved_by.append(source)
                # Keep max scores
                existing.bm25_score = max(existing.bm25_score, result.bm25_score)
                existing.vector_score = max(existing.vector_score, result.vector_score)

    # Sort by fused score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, fused_score in ranked:
        r = result_map[doc_id]
        r.score = fused_score
        results.append(r)

    return results


def weighted_fusion(
    bm25_results: List[HybridResult],
    vector_results: List[HybridResult],
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
) -> List[HybridResult]:
    """
    Weighted score combination of BM25 and vector results.

    Normalises scores within each list to [0, 1] before weighting.
    """
    def _normalize(results: List[HybridResult]) -> List[HybridResult]:
        if not results:
            return results
        max_score = max(r.score for r in results)
        min_score = min(r.score for r in results)
        score_range = max_score - min_score
        if score_range == 0:
            for r in results:
                r.score = 1.0
        else:
            for r in results:
                r.score = (r.score - min_score) / score_range
        return results

    _normalize(bm25_results)
    _normalize(vector_results)

    combined: Dict[str, HybridResult] = {}

    for r in bm25_results:
        combined[r.id] = HybridResult(
            id=r.id,
            score=r.score * bm25_weight,
            content=r.content,
            metadata=r.metadata,
            bm25_score=r.bm25_score,
            vector_score=0.0,
            retrieved_by=["bm25"],
        )

    for r in vector_results:
        if r.id in combined:
            combined[r.id].score += r.score * vector_weight
            combined[r.id].vector_score = r.vector_score
            combined[r.id].retrieved_by.append("vector")
        else:
            combined[r.id] = HybridResult(
                id=r.id,
                score=r.score * vector_weight,
                content=r.content,
                metadata=r.metadata,
                bm25_score=0.0,
                vector_score=r.vector_score,
                retrieved_by=["vector"],
            )

    ranked = sorted(combined.values(), key=lambda x: x.score, reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# MMR Diversity
# ---------------------------------------------------------------------------

def mmr_rerank(
    results: List[HybridResult],
    query_embedding: Optional[List[float]] = None,
    lambda_param: float = 0.7,
    top_k: int = 5,
) -> List[HybridResult]:
    """
    Maximum Marginal Relevance reranking (ByteDance §5.3.3).

    Balances relevance and diversity:
      MMR = λ * relevance - (1-λ) * max_similarity_to_selected

    Uses simple text overlap as similarity proxy when embeddings
    are not available for inter-result comparison.
    """
    if len(results) <= top_k:
        return results

    def _text_similarity(a: str, b: str) -> float:
        """Jaccard similarity as cheap text overlap proxy."""
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    selected: List[HybridResult] = []
    candidates = list(results)

    # Always pick the top-scoring result first
    selected.append(candidates.pop(0))

    while len(selected) < top_k and candidates:
        best_idx = -1
        best_mmr = -float('inf')

        for i, cand in enumerate(candidates):
            relevance = cand.score

            # Max similarity to any already-selected result
            max_sim = max(
                _text_similarity(cand.content, sel.content)
                for sel in selected
            )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        if best_idx >= 0:
            selected.append(candidates.pop(best_idx))
        else:
            break

    return selected


# ---------------------------------------------------------------------------
# Main Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Hybrid retriever combining BM25 + vector search (Pinecone/Milvus).

    Implements ByteDance RAG Guideline §5.2:
      1. Classify query type (keyword / semantic / hybrid)
      2. Run BM25 + vector search in parallel
      3. Fuse results (RRF or weighted)
      4. Apply MMR for diversity
      5. Return top-K results
    """

    def __init__(
        self,
        vector_store,
        embeddings,
        bm25_index: Optional[BM25Index] = None,
        fusion_method: str = "rrf",       # "rrf" or "weighted"
        mmr_lambda: float = 0.7,
        default_top_k: int = 5,
        reranker: Optional["Reranker"] = None,
        boost_enabled: bool = False,
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.bm25 = bm25_index or BM25Index()
        self.fusion_method = fusion_method
        self.mmr_lambda = mmr_lambda
        self.default_top_k = default_top_k
        self.classifier = QueryClassifier()
        self.reranker = reranker
        self.boost_enabled = boost_enabled

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Build BM25 index from chunks (call during ingestion)."""
        self.bm25.build(chunks)

    def build_bm25_from_vector_store(self, namespace: str = "", top_k: int = 10000):
        """
        Build BM25 index by fetching all vectors from the vector store.
        This should be called after ingestion to populate the BM25 index.
        """
        logger.info(f"Building BM25 index from vector store (namespace={namespace})...")
        
        # Use a dummy query vector to fetch all vectors
        # We'll fetch in batches
        dummy_vector = [0.0] * 3072  # gemini-embedding-001 dimension
        
        try:
            # Fetch vectors from Pinecone
            results = self.vector_store.search(
                query_vector=dummy_vector,
                top_k=top_k,
                namespace=namespace,
                filter=None,
            )
            
            if not results:
                logger.warning("No vectors found in vector store to build BM25 index")
                return
            
            # Convert to BM25 format
            bm25_chunks = []
            for r in results:
                bm25_chunks.append({
                    "id": r.id,
                    "chunk_id": r.id,
                    "content": r.content,
                    "metadata": r.metadata,
                })
            
            self.bm25.build(bm25_chunks)
            logger.info(f"BM25 index built from vector store with {len(bm25_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 from vector store: {e}")
            raise

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        force_mode: Optional[str] = None,  # "bm25", "vector", "hybrid"
    ) -> List[HybridResult]:
        """
        Execute hybrid retrieval.

        Args:
            query: User query
            top_k: Number of results (default: self.default_top_k)
            namespace: Pinecone namespace
            filter: Metadata filter
            force_mode: Override auto-classification

        Returns:
            List of HybridResult objects, ranked and diversified
        """
        top_k = top_k or self.default_top_k
        # Fetch more candidates for fusion, then trim
        fetch_k = min(top_k * 10, 50)

        # Step 1: Classify query
        if force_mode:
            query_type = force_mode
        else:
            query_type = self.classifier.classify(query)

        bm25_weight, vector_weight = self.classifier.get_weights(query)
        logger.info(f"Query classified as '{query_type}' → BM25={bm25_weight}, Vector={vector_weight}")

        # Step 2: Run retrievers
        bm25_results: List[HybridResult] = []
        vector_results: List[HybridResult] = []

        # BM25 search
        if query_type != "vector" and self.bm25._indexed:
            bm25_results = self.bm25.search_as_results(query, top_k=fetch_k, filter=filter)
            logger.info(f"BM25 returned {len(bm25_results)} results")

        # Vector search
        if query_type != "bm25":
            vector_results = self._vector_search(query, fetch_k, namespace, filter)
            logger.info(f"Vector search returned {len(vector_results)} results")

        # Step 3: Fuse results
        if not bm25_results:
            fused = vector_results
        elif not vector_results:
            fused = bm25_results
        elif self.fusion_method == "rrf":
            fused = reciprocal_rank_fusion([bm25_results, vector_results])
        else:
            fused = weighted_fusion(bm25_results, vector_results, bm25_weight, vector_weight)

        # Step 3b: Keyword boost (RAGFlow-inspired) — optional, off by default
        if self.boost_enabled and KEYWORD_BOOST_ENABLED and fused:
            boost_terms = extract_boost_terms(query)
            if boost_terms:
                scores = [r.score for r in fused]
                docs = [{"content": r.content, "metadata": r.metadata} for r in fused]
                boosted = apply_boost(scores, docs, boost_terms, KEYWORD_BOOST_MULTIPLIER)
                for r, s in zip(fused, boosted):
                    r.score = s
                fused.sort(key=lambda r: r.score, reverse=True)
                logger.info(
                    f"Keyword boost applied: {len(boost_terms)} terms, "
                    f"multiplier={KEYWORD_BOOST_MULTIPLIER}"
                )

        # Step 3c: Cross-encoder rerank (RAGFlow-inspired) — optional, off by default
        if self.reranker is not None and self.reranker.backend != "off":
            fused = self.reranker.rerank(query, fused, top_k=top_k)
            logger.info(f"Rerank applied (backend={self.reranker.backend})")

        # Step 4: MMR diversity reranking
        if len(fused) > top_k:
            fused = mmr_rerank(fused, lambda_param=self.mmr_lambda, top_k=top_k)

        return fused[:top_k]

    def _vector_search(
        self,
        query: str,
        top_k: int,
        namespace: str,
        filter: Optional[Dict[str, Any]],
    ) -> List[HybridResult]:
        """Run vector search (Pinecone/Milvus) and convert to HybridResult."""
        embed_result = self.embeddings.embed_text(query)
        if not embed_result.embedding:
            logger.warning("Failed to generate query embedding for vector search")
            return []

        search_results = self.vector_store.search(
            query_vector=embed_result.embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
        )

        return [
            HybridResult(
                id=r.id,
                score=r.score,
                content=r.content,
                metadata=r.metadata,
                bm25_score=0.0,
                vector_score=r.score,
                retrieved_by=["vector"],
            )
            for r in search_results
        ]
