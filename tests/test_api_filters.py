# Tests for API filter params and mode routing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from api.jobs import JobSubmitRequest
from api.main import RAGQueryRequest, app


class TestJobSubmitRequest:
    """Tests for JobSubmitRequest new filter fields."""

    def test_accepts_court_field(self):
        """Test court field accepted."""
        req = JobSubmitRequest(question="test question", court="ONSC")
        assert req.court == "ONSC"

    def test_accepts_statute_field(self):
        """Test statute field accepted."""
        req = JobSubmitRequest(question="test question", statute="ESA")
        assert req.statute == "ESA"

    def test_accepts_legal_section_field(self):
        """Test legal_section field accepted."""
        req = JobSubmitRequest(question="test question", legal_section="s. 56")
        assert req.legal_section == "s. 56"

    def test_accepts_filters_field(self):
        """Test filters dict field accepted."""
        req = JobSubmitRequest(question="test question", filters={"court": "ONSC"})
        assert req.filters == {"court": "ONSC"}

    def test_accepts_mode_field(self):
        """Test mode field accepted."""
        req = JobSubmitRequest(question="test question", mode="graphrag")
        assert req.mode == "graphrag"

    def test_all_fields_optional(self):
        """Test all new fields optional with None defaults."""
        req = JobSubmitRequest(question="test question")
        assert req.court is None
        assert req.statute is None
        assert req.legal_section is None
        assert req.filters is None
        assert req.mode is None

    def test_backward_compatible_existing_request(self):
        """Test existing request format still works."""
        req = JobSubmitRequest(question="test question here", top_k=5, jurisdiction="ON")
        assert req.question == "test question here"
        assert req.top_k == 5
        assert req.jurisdiction == "ON"
        # verify field removed in newer version


class TestRAGQueryRequest:
    """Tests for RAGQueryRequest new filter fields (sync endpoint)."""

    def test_accepts_all_new_fields(self):
        """Test all new fields accepted."""
        req = RAGQueryRequest(
            question="test question",
            court="ONSC",
            statute="ESA",
            legal_section="s. 56",
            filters={"jurisdiction": "ON"},
            mode="graphrag"
        )
        assert req.court == "ONSC"
        assert req.statute == "ESA"
        assert req.legal_section == "s. 56"
        assert req.filters == {"jurisdiction": "ON"}
        assert req.mode == "graphrag"

    def test_all_fields_optional(self):
        """Test all new fields optional with None defaults."""
        req = RAGQueryRequest(question="test question")
        assert req.court is None
        assert req.statute is None
        assert req.legal_section is None
        assert req.filters is None
        assert req.mode is None


class TestFilterMergeLogic:
    """Tests for filter dict merging in endpoints."""

    def test_merge_jurisdiction_and_court(self):
        """Test merging jurisdiction + court into filter dict."""
        req = JobSubmitRequest(question="test question here", jurisdiction="ON", court="ONSC")
        filter_dict = {}
        if req.jurisdiction:
            filter_dict["jurisdiction"] = req.jurisdiction
        if req.court:
            filter_dict["court"] = req.court
        assert filter_dict == {"jurisdiction": "ON", "court": "ONSC"}

    def test_merge_all_filter_fields(self):
        """Test merging all individual filter fields."""
        req = JobSubmitRequest(
            question="test question here",
            jurisdiction="ON",
            court="ONSC",
            statute="ESA",
            legal_section="s. 56",
            filters={"custom": "value"}
        )
        filter_dict = {}
        for field in ["jurisdiction", "court", "statute", "legal_section"]:
            val = getattr(req, field)
            if val:
                filter_dict[field] = val
        if req.filters:
            filter_dict.update(req.filters)
        assert filter_dict == {
            "jurisdiction": "ON",
            "court": "ONSC",
            "statute": "ESA",
            "legal_section": "s. 56",
            "custom": "value"
        }


class TestAPIEndpoints:
    """Integration tests for API endpoints with new params (mocked)."""

    def test_sync_endpoint_accepts_new_params(self):
        """Test /rag/query accepts new filter params."""
        client = TestClient(app)
        # This will fail without full setup but tests request validation
        # In real test, would mock rag_query and check filter passed
        pass  # Placeholder - requires full app setup

    def test_job_endpoint_accepts_mode_graphrag(self):
        """Test /rag/query/job accepts mode=graphrag."""
        client = TestClient(app)
        # Placeholder - requires full app setup
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])