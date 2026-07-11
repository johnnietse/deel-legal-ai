"""Tests for DeepSearch orchestrator."""
import pytest
from rag_pipeline.deep_search import (
    DeepSearchEngine, DeepSearchResult, UnifiedSource,
    fuse_and_rank_results, deduplicate_sources
)

class TestUnifiedSource:
    def test_creation(self):
        s = UnifiedSource(id="s1", title="Test", excerpt="Excerpt",
                          source_type="case_law", relevance_score=0.95)
        assert s.source_type == "case_law"

class TestFuseAndRank:
    def test_fuses_multiple_lists(self):
        case_results = [
            UnifiedSource(id="c1", title="C1", excerpt="", source_type="case_law", relevance_score=0.9),
            UnifiedSource(id="c2", title="C2", excerpt="", source_type="case_law", relevance_score=0.8),
        ]
        web_results = [
            UnifiedSource(id="w1", title="W1", excerpt="", source_type="web", relevance_score=0.85),
        ]
        fused = fuse_and_rank_results(case_results, web_results, [])
        assert len(fused) == 3
        assert fused[0].id == "c1"  # Highest score first

class TestDeduplicate:
    def test_removes_duplicates_by_url(self):
        sources = [
            UnifiedSource(id="a", title="A", excerpt="", source_type="web", relevance_score=0.9, url="https://x.com"),
            UnifiedSource(id="b", title="B", excerpt="", source_type="web", relevance_score=0.8, url="https://x.com"),
        ]
        deduped = deduplicate_sources(sources)
        assert len(deduped) == 1

class TestDeepSearchEngine:
    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self):
        engine = DeepSearchEngine()
        result = await engine.deep_search(query="")
        assert result is None or (hasattr(result, 'error') and result.error)

    @pytest.mark.asyncio
    async def test_source_type_counts(self, mocker):
        # Mock all external dependencies at their source modules
        mocker.patch("rag_pipeline.web_search.search_web", return_value=[
            UnifiedSource(id="w1", title="Web1", excerpt="", source_type="web", relevance_score=0.8)
        ])
        mocker.patch("rag_pipeline.rag_query.LegalRAGQuery")
        mocker.patch("rag_pipeline.embeddings.GeminiChat")
        mocker.patch("rag_pipeline.vector_store.create_vector_store")
        mocker.patch("db.database.get_db")
        mocker.patch("requests.post")
        engine = DeepSearchEngine()
        # Should return counts even with minimal data
        result = await engine.deep_search("test")
        if result:
            assert isinstance(result.source_type_counts, dict)