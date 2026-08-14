# Tests for parent_store module (ES-backed parent document store)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from rag_pipeline.parent_store import ParentStore


class TestParentStore:
    """Tests for ParentStore with mocked Elasticsearch."""

    @pytest.fixture
    def mock_es(self):
        """Mock Elasticsearch client."""
        with patch("rag_pipeline.parent_store.Elasticsearch") as mock_es_class:
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client
            yield mock_client

    def test_put_and_get_roundtrip(self, mock_es):
        """Test put_parent and get_parent roundtrip."""
        mock_es.get.return_value = {"_source": {"content": "Hello parent content"}}
        store = ParentStore()
        store.put_parent("parent1", "Hello parent content")
        result = store.get_parent("parent1")
        assert result == "Hello parent content"
        mock_es.index.assert_called()
        mock_es.get.assert_called_with(index="deel-legal-parents", id="parent1")

    def test_get_missing_parent_returns_none(self, mock_es):
        """Test get_parent returns None for missing parent."""
        from elasticsearch import NotFoundError
        mock_es.get.side_effect = NotFoundError("Not found", "", 404)
        store = ParentStore()
        result = store.get_parent("nonexistent")
        assert result is None

    def test_get_parent_es_exception_returns_none(self, mock_es):
        """Test get_parent returns None on any ES exception."""
        mock_es.get.side_effect = Exception("Connection refused")
        store = ParentStore()
        result = store.get_parent("parent1")
        assert result is None

    def test_put_parent_es_exception_no_raise(self, mock_es):
        """Test put_parent doesn't raise on ES exception."""
        mock_es.index.side_effect = Exception("Connection refused")
        store = ParentStore()
        # Should not raise
        store.put_parent("parent1", "content")

    def test_delete_parent(self, mock_es):
        """Test delete_parent calls ES delete."""
        store = ParentStore()
        store.delete_parent("parent1")
        mock_es.delete.assert_called_with(index="deel-legal-parents", id="parent1")

    def test_delete_missing_parent_no_raise(self, mock_es):
        """Test delete_parent doesn't raise on missing."""
        from elasticsearch import NotFoundError
        mock_es.delete.side_effect = NotFoundError("Not found", "", 404)
        store = ParentStore()
        store.delete_parent("parent1")  # Should not raise

    def test_content_truncation(self, mock_es):
        """Test content truncated to PARENT_CHUNK_MAX_SIZE."""
        store = ParentStore()
        long_content = "x" * 5000  # PARENT_CHUNK_MAX_SIZE = 4096
        store.put_parent("parent1", long_content)
        # Check call was made with truncated content
        call_args = mock_es.index.call_args
        doc = call_args.kwargs.get("document", {})
        assert len(doc.get("content", "")) == 4096

    def test_lazy_client_init(self):
        """Test client not created until first use."""
        with patch("rag_pipeline.parent_store.Elasticsearch") as mock_es_class:
            store = ParentStore()
            # Client not created yet
            assert store._client is None
            # First use triggers creation
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client
            mock_client.get.return_value = {"_source": {"content": "test"}}
            result = store.get_parent("parent1")
            mock_es_class.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])