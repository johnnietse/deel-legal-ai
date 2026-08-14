# Tests for metadata filter support in BM25Index and HybridRetriever
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from rag_pipeline.hybrid_retriever import BM25Index, HybridRetriever, HybridResult


class TestBM25Filter:
    """Tests for BM25Index metadata filtering."""

    @pytest.fixture
    def sample_docs(self):
        return [
            {"id": "a1", "chunk_id": "a1", "content": "doc a", "metadata": {"court": "ONSC", "jurisdiction": "ON", "statute": "ESA"}},
            {"id": "a2", "chunk_id": "a2", "content": "doc b", "metadata": {"court": "SCC", "jurisdiction": "ON", "statute": "CLC"}},
            {"id": "a3", "chunk_id": "a3", "content": "doc c", "metadata": {"court": "ONSC", "jurisdiction": "BC", "statute": "OHSA"}},
            {"id": "a4", "chunk_id": "a4", "content": "doc d", "metadata": {"court": "FCA", "jurisdiction": "FED", "statute": "CLC"}},
        ]

    def test_no_filter_returns_all(self, sample_docs):
        """Test unfiltered search returns all matching docs."""
        idx = BM25Index()
        idx.build(sample_docs, content_key="content")
        results = idx.search("doc", top_k=10)
        assert len(results) == 4

    def test_equality_filter_court(self, sample_docs):
        """Test equality filter on court field."""
        idx = BM25Index()
        idx.build(sample_docs, content_key="content")
        results = idx.search("doc", top_k=10, filter={"court": "ONSC"})
        assert len(results) == 2
        for _, score in results:
            assert score > 0

    def test_in_operator_jurisdiction(self, sample_docs):
        """Test $in operator on jurisdiction field."""
        idx = BM25Index()
        idx.build(sample_docs, content_key="content")
        results = idx.search("doc", top_k=10, filter={"jurisdiction": {"$in": ["ON", "BC"]}})
        assert len(results) == 3
        # a1 (ON), a2 (ON), a3 (BC) match; a4 (FED) excluded

    def test_eq_operator_court(self, sample_docs):
        """Test $eq operator on court field."""
        idx = BM25Index()
        idx.build(sample_docs, content_key="content")
        results = idx.search("doc", top_k=10, filter={"court": {"$eq": "SCC"}})
        assert len(results) == 1

    def test_filter_none_value_matches_all(self, sample_docs):
        """Test filter with None value doesn't exclude."""
        idx = BM25Index()
        idx.build(sample_docs, content_key="content")
        results = idx.search("doc", top_k=10, filter={"court": None})
        assert len(results) == 4

    def test_missing_field_excludes_doc(self, sample_docs):
        """Test doc missing filter field is excluded."""
        docs_with_missing = sample_docs + [{"id": "a5", "chunk_id": "a5", "content": "doc e", "metadata": {"jurisdiction": "ON"}}]
        idx = BM25Index()
        idx.build(docs_with_missing, content_key="content")
        results = idx.search("doc", top_k=10, filter={"court": "ONSC"})
        # Only docs with court field set can match
        assert len(results) == 2  # a1, a3 have court=ONSC

    def test_multiple_filter_keys_all_must_match(self, sample_docs):
        """Test multiple filter keys require ALL to match."""
        idx = BM25Index()
        idx.build(sample_docs, content_key="content")
        results = idx.search("doc", top_k=10, filter={"court": "ONSC", "jurisdiction": "ON"})
        assert len(results) == 1  # only a1 has both court=ONSC AND jurisdiction=ON

    def test_empty_filter_results(self, sample_docs):
        """Test filter that matches nothing returns empty."""
        idx = BM25Index()
        idx.build(sample_docs, content_key="content")
        results = idx.search("doc", top_k=10, filter={"court": "NONEXISTENT"})
        assert len(results) == 0


class TestHybridRetrieverFilterPassthrough:
    """Tests that HybridRetriever passes filter to both BM25 and vector."""

    def test_retrieve_accepts_filter_param(self):
        """Test retrieve() accepts filter parameter without error."""
        retriever = HybridRetriever(vector_store=None, embeddings=None)
        # We can't fully test without real vector store, but verify signature
        import inspect
        sig = inspect.signature(retriever.retrieve)
        assert "filter" in sig.parameters
        param = sig.parameters["filter"]
        assert param.default is None
        assert param.annotation is not param.empty  # has Optional[Dict] annotation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])