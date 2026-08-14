# Tests for graphrag module (GraphRAG/LightRAG-style retrieval)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock
import networkx as nx
from rag_pipeline.graphrag import compute_pagerank, extract_entities, query_graphrag
from rag_pipeline.rag_query import RAGResponse
import config


class TestComputePageRank:
    """Tests for compute_pagerank function."""

    def test_empty_graph_returns_empty_dict(self):
        """Test empty graph returns empty dict."""
        kg = MagicMock()
        kg.graph = nx.DiGraph()
        result = compute_pagerank(kg)
        assert result == {}

    def test_pagerank_scores_nodes(self):
        """Test PageRank computes scores for nodes."""
        kg = MagicMock()
        g = nx.DiGraph()
        g.add_edges_from([
            ("Sagaz Test", "Worker Classification"),
            ("Worker Classification", "ESA"),
            ("ESA", "Notice Period"),
            ("Notice Period", "Sagaz Test"),  # cycle
        ])
        kg.graph = g
        result = compute_pagerank(kg)
        assert len(result) == 4
        # All scores should be positive
        assert all(v > 0 for v in result.values())
        # Sum should be ~1.0
        assert abs(sum(result.values()) - 1.0) < 0.01


class TestExtractEntities:
    """Tests for extract_entities function."""

    def test_reuses_kg_extract_entities(self):
        """Test extract_entities calls kg.extract_entities_from_query."""
        kg = MagicMock()
        kg.extract_entities_from_query.return_value = ["Sagaz Test", "ESA"]
        result = extract_entities(kg, "How does Sagaz test apply under ESA?")
        assert result == ["Sagaz Test", "ESA"]
        kg.extract_entities_from_query.assert_called_once_with("How does Sagaz test apply under ESA?")


class TestQueryGraphRAG:
    """Tests for query_graphrag function (mocked)."""

    def test_disabled_raises_runtime_error(self):
        """Test query_graphrag raises when GRAPHRAG_ENABLED=False."""
        with patch.object(config, "GRAPHRAG_ENABLED", False):
            rag = MagicMock()
            with pytest.raises(RuntimeError, match="GraphRAG disabled"):
                query_graphrag(rag, "test query")

    def test_no_entities_falls_back_to_plain_query(self):
        """Test no entities found falls back to plain rag.query."""
        rag = MagicMock()
        kg = MagicMock()
        kg.extract_entities_from_query.return_value = []
        rag.hybrid_retriever = MagicMock()
        
        with patch.object(config, "GRAPHRAG_ENABLED", True):
            with patch("rag_pipeline.graphrag.LegalKnowledgeGraph", return_value=kg):
                mock_response = RAGResponse(query="test", answer="fallback", sources=[], confidence="medium")
                rag.query.return_value = mock_response
                
                result = query_graphrag(rag, "test query")
                
                assert isinstance(result, RAGResponse)
                assert result.answer == "fallback"
                rag.query.assert_called_once()

    def test_entities_found_merges_graph_context(self):
        """Test entities found merges graph context with hybrid retrieval."""
        rag = MagicMock()
        kg = MagicMock()
        kg.extract_entities_from_query.return_value = ["Sagaz Test"]
        
        # Mock subgraph result - nodes must be dicts with "id" per implementation
        subgraph = MagicMock()
        subgraph.linearized_text = "Sagaz Test -> applies_test -> Worker Classification"
        subgraph.nodes = [
            {"id": "Sagaz Test", "type": "LegalTest"},
            {"id": "Worker Classification", "type": "Factor"}
        ]
        kg.query_subgraph.return_value = subgraph
        
        # Mock hybrid retrieval
        mock_hybrid_results = [MagicMock(id="doc1", score=0.9, content="hybrid content", metadata={})]
        rag.hybrid_retriever.retrieve.return_value = mock_hybrid_results
        
        # Mock chat generation
        rag.chat.generate.return_value = "GraphRAG answer"
        
        # Mock _format_sources
        rag._format_sources.return_value = [{"index": 1, "case_name": "Test Case"}]
        
        with patch.object(config, "GRAPHRAG_ENABLED", True):
            with patch("rag_pipeline.graphrag.LegalKnowledgeGraph", return_value=kg):
                with patch("rag_pipeline.graphrag.compute_pagerank", return_value={"Sagaz Test": 0.5, "Worker Classification": 0.3}):
                    result = query_graphrag(rag, "How does Sagaz test apply?")
        
        assert isinstance(result, RAGResponse)
        assert result.retrieval_mode == "graphrag"
        assert "pagerank_nodes" in result.metrics
        assert "entities" in result.metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])