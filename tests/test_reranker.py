# Tests for reranker module (pluggable backends)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from rag_pipeline.reranker import Reranker
from rag_pipeline.hybrid_retriever import HybridResult


class TestRerankerOff:
    """Tests for 'off' backend (identity passthrough)."""

    def test_off_backend_returns_input_unchanged(self):
        """Test off backend returns results in same order."""
        reranker = Reranker(backend="off")
        results = [
            HybridResult(id="a", score=0.5, content="a", metadata={}),
            HybridResult(id="b", score=0.3, content="b", metadata={}),
            HybridResult(id="c", score=0.8, content="c", metadata={}),
        ]
        reranked = reranker.rerank("query", results, top_k=3)
        assert [r.id for r in reranked] == ["a", "b", "c"]

    def test_off_backend_respects_top_k(self):
        """Test off backend truncates to top_k."""
        reranker = Reranker(backend="off")
        results = [
            HybridResult(id="a", score=0.5, content="a", metadata={}),
            HybridResult(id="b", score=0.3, content="b", metadata={}),
            HybridResult(id="c", score=0.8, content="c", metadata={}),
        ]
        reranked = reranker.rerank("query", results, top_k=2)
        assert len(reranked) == 2
        assert [r.id for r in reranked] == ["a", "b"]


class TestRerankerTEI:
    """Tests for TEI backend (mocked HTTP)."""

    def test_tei_success_reorders_by_score(self):
        """Test TEI backend reorders results by relevance score."""
        reranker = Reranker(backend="tei")
        results = [
            HybridResult(id="a", score=0.5, content="doc a", metadata={}),
            HybridResult(id="b", score=0.3, content="doc b", metadata={}),
            HybridResult(id="c", score=0.8, content="doc c", metadata={}),
        ]
        # TEI returns: b=0.9, a=0.4, c=0.2 → order b, a, c
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.9},  # doc b
                {"index": 0, "relevance_score": 0.4},  # doc a
                {"index": 2, "relevance_score": 0.2},  # doc c
            ]
        }
        mock_response.raise_for_status.return_value = None

        with patch("rag_pipeline.reranker.requests.post", return_value=mock_response):
            reranked = reranker.rerank("query", results, top_k=3)
        
        assert [r.id for r in reranked] == ["b", "a", "c"]
        assert reranked[0].score == 0.9
        assert reranked[1].score == 0.4
        assert reranked[2].score == 0.2

    def test_tei_timeout_falls_back_to_input_order(self):
        """Test TEI timeout falls back to original order."""
        import requests
        reranker = Reranker(backend="tei")
        results = [
            HybridResult(id="a", score=0.5, content="a", metadata={}),
            HybridResult(id="b", score=0.3, content="b", metadata={}),
        ]
        with patch("rag_pipeline.reranker.requests.post", side_effect=requests.Timeout("timeout")):
            reranked = reranker.rerank("query", results, top_k=2)
        assert [r.id for r in reranked] == ["a", "b"]

    def test_tei_http_error_falls_back_to_input_order(self):
        """Test TEI HTTP error falls back to original order."""
        import requests
        reranker = Reranker(backend="tei")
        results = [
            HybridResult(id="a", score=0.5, content="a", metadata={}),
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500")
        with patch("rag_pipeline.reranker.requests.post", return_value=mock_response):
            reranked = reranker.rerank("query", results, top_k=1)
        assert [r.id for r in reranked] == ["a"]

    def test_tei_invalid_response_falls_back(self):
        """Test TEI invalid JSON falls back to original order."""
        reranker = Reranker(backend="tei")
        results = [HybridResult(id="a", score=0.5, content="a", metadata={})]
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("invalid json")
        mock_response.raise_for_status.return_value = None
        with patch("rag_pipeline.reranker.requests.post", return_value=mock_response):
            reranked = reranker.rerank("query", results, top_k=1)
        assert [r.id for r in reranked] == ["a"]


class TestRerankerLocal:
    """Tests for local FlagEmbedding backend."""

    def test_local_import_error_degrades_to_off(self, capsys):
        """Test local backend degrades gracefully when FlagEmbedding missing."""
        reranker = Reranker(backend="local")
        results = [
            HybridResult(id="a", score=0.5, content="a", metadata={}),
            HybridResult(id="b", score=0.3, content="b", metadata={}),
        ]
        # Simulate ImportError on lazy import by patching where it's used
        with patch.object(reranker, "_get_local_model", side_effect=ImportError("no FlagEmbedding")):
            reranked = reranker.rerank("query", results, top_k=2)
        # Should return original order with warning logged
        assert [r.id for r in reranked] == ["a", "b"]
        # Verify no exception raised


if __name__ == "__main__":
    pytest.main([__file__, "-v"])