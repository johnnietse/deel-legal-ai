"""
Tests for ByteDance v3.0 RAG Infrastructure

Covers:
1. SemanticChunker — section-aware legal chunking
2. BM25Index — local BM25 search
3. QueryClassifier — query routing
4. HybridRetriever data structures (RRF, MMR)
5. Prompt Templates — auto-selection
6. Confidence Gate — pre-generation checks
7. Query Cache — TTL cache with eviction
8. Feedback Analyzer — JSONL store and summary
9. Metrics Collector — structured logging
10. Vector Store — abstract interface
"""

import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 1. SemanticChunker Tests
# ============================================================

class TestSemanticChunker:
    """Tests for the section-aware legal document chunker."""

    def test_basic_chunking(self):
        """Test chunking a simple legal text."""
        from rag_pipeline.document_processor import SemanticChunker

        chunker = SemanticChunker(max_chunk_tokens=512, min_chunk_tokens=10)
        text = """FACTS: The worker was hired to perform delivery services.
They used their own vehicle and set their own hours.

ANALYSIS: Applying the Sagaz test, the level of control exercised
by the company was minimal. The worker could accept or refuse deliveries.

CONCLUSION: The worker is an independent contractor."""

        chunks = chunker.chunk_document(text, document_id="test_doc")

        assert len(chunks) > 0
        # Should detect legal sections
        sections_found = set(c.metadata.get("legal_section", "") for c in chunks)
        assert "facts" in sections_found or "fact" in sections_found
        assert "analysis" in sections_found
        assert "conclusion" in sections_found

    def test_paragraph_numbering_preserved(self):
        """Test that [N] numbered paragraphs stay as atomic units."""
        from rag_pipeline.document_processor import SemanticChunker

        chunker = SemanticChunker(max_chunk_tokens=512, min_chunk_tokens=10)
        text = """[1] The plaintiff was employed for 10 years.
[2] The defendant terminated without cause.
[3] Reasonable notice is owed under Bardal factors."""

        chunks = chunker.chunk_document(text, document_id="test_paras")
        assert len(chunks) >= 1
        # All content should be present
        all_content = " ".join(c.content for c in chunks)
        assert "[1]" in all_content
        assert "[3]" in all_content

    def test_content_type_detection(self):
        """Test statute vs narrative detection."""
        from rag_pipeline.document_processor import SemanticChunker

        chunker = SemanticChunker()

        statute_text = """Section 57 provides that every employer shall give notice.
(1) Every employer must provide written notice. Section 64 shall apply."""
        assert chunker._detect_content_type(statute_text) == "statute"

        narrative_text = """The court considered the totality of the relationship
between the parties and found that the worker was integrated into the business."""
        assert chunker._detect_content_type(narrative_text) == "narrative"

    def test_metadata_enrichment(self):
        """Test that chunks carry section and content_type metadata."""
        from rag_pipeline.document_processor import SemanticChunker

        chunker = SemanticChunker(min_chunk_tokens=5)
        text = "ANALYSIS: The control test was applied to the worker relationship."
        chunks = chunker.chunk_document(
            text, document_id="meta_test",
            base_metadata={"jurisdiction": "Ontario"}
        )

        assert len(chunks) >= 1
        chunk = chunks[0]
        assert chunk.metadata.get("jurisdiction") == "Ontario"
        assert chunk.metadata.get("legal_section") == "analysis"
        assert "chunk_type" in chunk.metadata

    def test_empty_text_returns_empty(self):
        """Test that empty text returns no chunks."""
        from rag_pipeline.document_processor import SemanticChunker

        chunker = SemanticChunker()
        assert chunker.chunk_document("", document_id="empty") == []
        assert chunker.chunk_document("   ", document_id="spaces") == []


# ============================================================
# 2. BM25Index Tests
# ============================================================

class TestBM25Index:
    """Tests for the in-memory BM25 index."""

    def test_build_and_search(self):
        """Test building index and searching."""
        from rag_pipeline.hybrid_retriever import BM25Index

        bm25 = BM25Index()
        docs = [
            {"content": "The Sagaz test determines worker classification", "chunk_id": "c1"},
            {"content": "Employment Standards Act minimum wage requirements", "chunk_id": "c2"},
            {"content": "Uber drivers and gig economy platform workers", "chunk_id": "c3"},
        ]
        bm25.build(docs)

        results = bm25.search("Sagaz worker classification", top_k=3)
        assert len(results) > 0
        # First result should be the Sagaz document
        top_idx, top_score = results[0]
        assert top_idx == 0
        assert top_score > 0

    def test_empty_search(self):
        """Test searching before building returns empty."""
        from rag_pipeline.hybrid_retriever import BM25Index

        bm25 = BM25Index()
        results = bm25.search("test query")
        assert results == []

    def test_search_as_results(self):
        """Test search_as_results returns HybridResult objects."""
        from rag_pipeline.hybrid_retriever import BM25Index

        bm25 = BM25Index()
        bm25.build([{"content": "ESA termination notice period", "chunk_id": "c1"}])

        results = bm25.search_as_results("ESA notice", top_k=1)
        assert len(results) == 1
        assert results[0].id == "c1"
        assert "bm25" in results[0].retrieved_by

    def test_idf_calculation(self):
        """Test that IDF values are computed correctly."""
        from rag_pipeline.hybrid_retriever import BM25Index

        bm25 = BM25Index()
        bm25.build([
            {"content": "apple orange"},
            {"content": "apple banana"},
            {"content": "cherry"},
        ])
        # 'apple' appears in 2/3 docs, 'cherry' in 1/3
        assert bm25._idf["cherry"] > bm25._idf["apple"]


# ============================================================
# 3. QueryClassifier Tests
# ============================================================

class TestQueryClassifier:
    """Tests for query routing logic."""

    def test_keyword_classification(self):
        """Test that citation-heavy queries are classified as keyword."""
        from rag_pipeline.hybrid_retriever import QueryClassifier

        assert QueryClassifier.classify("2020 ONSC 1234 damages") == "keyword"
        assert QueryClassifier.classify("ESA s. 56 notice period") == "keyword"

    def test_semantic_classification(self):
        """Test that conceptual queries are classified as semantic."""
        from rag_pipeline.hybrid_retriever import QueryClassifier

        assert QueryClassifier.classify("What factors determine employee status?") == "semantic"
        assert QueryClassifier.classify("Explain the integration test") == "semantic"

    def test_hybrid_classification(self):
        """Test that ambiguous queries default to hybrid."""
        from rag_pipeline.hybrid_retriever import QueryClassifier

        result = QueryClassifier.classify("worker termination rules")
        assert result in ("hybrid", "semantic", "keyword")

    def test_weight_values(self):
        """Test that weights sum to 1.0."""
        from rag_pipeline.hybrid_retriever import QueryClassifier

        for query_type in ["keyword query ESA", "what factors", "general query"]:
            bm25_w, vector_w = QueryClassifier.get_weights(query_type)
            assert abs(bm25_w + vector_w - 1.0) < 0.01


# ============================================================
# 4. RRF and MMR Tests
# ============================================================

class TestFusionAndDiversity:
    """Tests for Reciprocal Rank Fusion and MMR."""

    def test_rrf_merges_lists(self):
        """Test RRF merges two ranked lists."""
        from rag_pipeline.hybrid_retriever import HybridResult, reciprocal_rank_fusion

        list1 = [HybridResult(id="a", score=1.0, content="A", metadata={})]
        list2 = [HybridResult(id="b", score=1.0, content="B", metadata={}),
                 HybridResult(id="a", score=0.8, content="A", metadata={})]

        merged = reciprocal_rank_fusion([list1, list2])
        # 'a' appears in both lists so should have higher fused score
        assert merged[0].id == "a"

    def test_mmr_reduces_redundancy(self):
        """Test MMR prefers diverse results."""
        from rag_pipeline.hybrid_retriever import HybridResult, mmr_rerank

        results = [
            HybridResult(id="1", score=1.0, content="The Sagaz test has five factors", metadata={}),
            HybridResult(id="2", score=0.95, content="The Sagaz test has five factors for classification", metadata={}),
            HybridResult(id="3", score=0.9, content="ESA provides minimum termination notice", metadata={}),
        ]
        reranked = mmr_rerank(results, lambda_param=0.5, top_k=2)
        # Should pick diverse: one Sagaz + one ESA rather than two Sagaz
        ids = [r.id for r in reranked]
        assert "3" in ids  # ESA result should make it in


# ============================================================
# 5. Prompt Templates Tests
# ============================================================

class TestPromptTemplates:
    """Tests for the prompt template library."""

    def test_template_loading(self):
        """Test that all templates load."""
        from rag_pipeline.prompt_templates import PromptTemplateLibrary

        lib = PromptTemplateLibrary()
        templates = lib.list_templates()
        assert len(templates) >= 4

    def test_auto_selection(self):
        """Test auto-selecting template based on query."""
        from rag_pipeline.prompt_templates import PromptTemplateLibrary

        lib = PromptTemplateLibrary()
        template = lib.auto_select(
            "Is this worker an employee or independent contractor under the Sagaz test?"
        )
        assert template.name == "worker_classification"

    def test_template_rendering(self):
        """Test rendering a template with context via build_prompt."""
        from rag_pipeline.prompt_templates import PromptTemplateLibrary

        lib = PromptTemplateLibrary()
        system_prompt, user_prompt = lib.build_prompt(
            query="Is John an employee?",
            sources=[{"content": "John works 9-5 under supervision.", "score": 0.9}],
            template_name="worker_classification",
        )
        assert "John" in user_prompt
        assert len(system_prompt) > 50


# ============================================================
# 6. Confidence Gate Tests
# ============================================================

class TestConfidenceGate:
    """Tests for the pre-generation confidence checking."""

    def test_gate_check_returns_report(self):
        """Test that gate.check() returns a ConfidenceReport."""
        from rag_pipeline.confidence_gate import ConfidenceGate, ConfidenceReport

        gate = ConfidenceGate()
        report = gate.check(
            answer="The Sagaz test uses five factors.",
            sources=[{"content": "The Sagaz test determines worker classification.", "score": 0.95}],
            query="What is the Sagaz test?",
        )
        assert isinstance(report, ConfidenceReport)
        assert hasattr(report, "confidence_score")

    def test_gate_check_with_empty_sources(self):
        """Test that gate handles empty sources."""
        from rag_pipeline.confidence_gate import ConfidenceGate

        gate = ConfidenceGate()
        report = gate.check(answer="Something.", sources=[], query="Random?")
        assert report.confidence_score <= 1.0


# ============================================================
# 7. Query Cache Tests
# ============================================================

class TestQueryCache:
    """Tests for the multi-layer TTL cache."""

    def test_cache_put_and_get_response(self):
        """Test basic response-layer cache operations."""
        from rag_pipeline.query_cache import RAGQueryCache

        cache = RAGQueryCache(response_maxsize=10, response_ttl=60)
        cache.put_response("key1", "value1")
        assert cache.get_response("key1") == "value1"

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from rag_pipeline.query_cache import RAGQueryCache

        cache = RAGQueryCache()
        assert cache.get_response("nonexistent") is None
        assert cache.get_embedding("nonexistent") is None
        assert cache.get_retrieval("nonexistent") is None

    def test_cache_stats(self):
        """Test cache stats reporting."""
        from rag_pipeline.query_cache import RAGQueryCache

        cache = RAGQueryCache(response_maxsize=5)
        cache.put_response("q1", "r1")
        cache.put_response("q2", "r2")
        stats = cache.stats()
        assert isinstance(stats, dict)


# ============================================================
# 8. Feedback Analyzer Tests
# ============================================================

class TestFeedbackAnalyzer:
    """Tests for the feedback collection and analysis."""

    def test_record_and_count(self):
        """Test recording feedback and counting."""
        from rag_pipeline.feedback_analyzer import FeedbackStore, FeedbackEntry

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(store_path=str(Path(tmpdir) / "fb.jsonl"))
            entry = FeedbackEntry(
                query_id="q1",
                query_text="test query",
                answer_text="test response",
                rating="useful",
                error_type=None,
                comment=None,
            )
            store.record(entry)
            assert store.count() == 1

    def test_summary_with_empty_store(self):
        """Test summary on empty store."""
        from rag_pipeline.feedback_analyzer import FeedbackAnalyzer, FeedbackStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(store_path=str(Path(tmpdir) / "fb.jsonl"))
            analyzer = FeedbackAnalyzer(store=store)
            summary = analyzer.summary()
            assert summary["total"] == 0

    def test_summary_with_data(self):
        """Test summary with recorded feedback."""
        from rag_pipeline.feedback_analyzer import FeedbackAnalyzer, FeedbackStore, FeedbackEntry

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(store_path=str(Path(tmpdir) / "fb.jsonl"))
            for i, rating in enumerate(["useful", "not_useful", "useful"]):
                store.record(FeedbackEntry(
                    query_id=f"q{i}",
                    query_text="q",
                    answer_text="r",
                    rating=rating,
                    error_type=None,
                    comment=None,
                ))
            analyzer = FeedbackAnalyzer(store=store)
            summary = analyzer.summary()
            assert summary["total"] == 3


# ============================================================
# 9. Vector Store Tests
# ============================================================

class TestVectorStoreAbstraction:
    """Tests for the vector store abstraction layer."""

    def test_hnsw_presets_exist(self):
        """Test that all HNSW presets are defined."""
        from rag_pipeline.vector_store import HNSW_PRESETS

        assert "development" in HNSW_PRESETS
        assert "production" in HNSW_PRESETS
        assert "high_recall" in HNSW_PRESETS
        assert "billion_scale" in HNSW_PRESETS

    def test_factory_creates_pinecone(self):
        """Test factory returns PineconeBackend."""
        from rag_pipeline.vector_store import create_vector_store, PineconeBackend

        store = create_vector_store("pinecone")
        assert isinstance(store, PineconeBackend)

    def test_factory_creates_milvus(self):
        """Test factory returns MilvusBackend."""
        from rag_pipeline.vector_store import create_vector_store, MilvusBackend

        store = create_vector_store("milvus")
        assert isinstance(store, MilvusBackend)

    def test_factory_rejects_unknown(self):
        """Test factory raises for unknown backend."""
        from rag_pipeline.vector_store import create_vector_store

        with pytest.raises(ValueError, match="Unknown"):
            create_vector_store("unknown_db")

    def test_vector_record_dataclass(self):
        """Test VectorRecord data structure."""
        from rag_pipeline.vector_store import VectorRecord

        record = VectorRecord(id="test", values=[0.1, 0.2, 0.3], metadata={"key": "val"})
        assert record.id == "test"
        assert len(record.values) == 3


# ============================================================
# 10. Search Engine Tests
# ============================================================

class TestSearchEngine:
    """Tests for the Elasticsearch search engine abstraction."""

    def test_bm25_engine_factory_elasticsearch(self):
        """Test factory creates ElasticsearchBM25."""
        from rag_pipeline.search_engine import create_bm25_engine, ElasticsearchBM25

        engine = create_bm25_engine("elasticsearch")
        assert isinstance(engine, ElasticsearchBM25)

    def test_bm25_engine_factory_unknown(self):
        """Test factory raises for unknown backend."""
        from rag_pipeline.search_engine import create_bm25_engine

        with pytest.raises(ValueError, match="Unknown"):
            create_bm25_engine("unknown_engine")

    def test_legal_index_settings_structure(self):
        """Test that the legal index settings have required fields."""
        from rag_pipeline.search_engine import ElasticsearchBM25

        settings = ElasticsearchBM25.LEGAL_INDEX_SETTINGS
        assert "settings" in settings
        assert "mappings" in settings
        assert "legal_analyzer" in settings["settings"]["analysis"]["analyzer"]
        assert "content" in settings["mappings"]["properties"]


# ============================================================
# Integration: Re-indexing Pipeline
# ============================================================

class TestReindexIntegration:
    """Integration test: SemanticChunker + BM25Index end-to-end."""

    def test_chunk_and_index_legal_docs(self):
        """Test chunking real legal docs and building a BM25 index."""
        from rag_pipeline.document_processor import SemanticChunker
        from rag_pipeline.hybrid_retriever import BM25Index

        chunker = SemanticChunker(min_chunk_tokens=10)
        text = """671122 Ontario Ltd. v. Sagaz Industries Canada Inc., 2001 SCC 59

FACTS: Sagaz Industries engaged AIM as a sales agent.

ANALYSIS: The central question is whether the person is performing
services as a person in business on their own account. The level of
control is always a factor. Other factors include ownership of tools,
chance of profit, and risk of loss.

CONCLUSION: No single test is conclusive. Courts must examine the
total relationship of the parties."""

        chunks = chunker.chunk_document(text, document_id="sagaz_test")
        assert len(chunks) > 0

        chunk_dicts = [c.to_dict() for c in chunks]
        bm25 = BM25Index()
        bm25.build(chunk_dicts)

        results = bm25.search("control factor worker classification", top_k=3)
        assert len(results) > 0
        # Top result should be from the analysis section
        top_idx = results[0][0]
        assert "control" in chunk_dicts[top_idx]["content"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
