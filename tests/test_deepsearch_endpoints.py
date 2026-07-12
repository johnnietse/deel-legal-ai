# Tests for DeepSearch, TTS, and Search API endpoints (Task 8)
import sys
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with the full app (includes api_router)."""
    from api.main import app
    return TestClient(app)


@pytest.fixture
def mock_user():
    return {
        "user_id": "test-user-id",
        "email": "test@example.com",
        "tier": "free",
    }


def _authenticate(app, mock_user):
    """Override auth dependencies so endpoints see an authenticated user."""
    from api.auth import get_current_user, get_api_key_user
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_api_key_user] = lambda: None


def _clear(app):
    app.dependency_overrides.clear()


# =====================
# Endpoint Registration / Auth
# =====================

class TestEndpointRegistration:
    def test_deepsearch_requires_auth(self, client):
        response = client.post("/rag/deepsearch", json={"query": "What is wrongful dismissal?"})
        assert response.status_code == 401

    def test_deepsearch_followup_requires_auth(self, client):
        response = client.post(
            "/rag/deepsearch/followup",
            params={"original_query": "x", "follow_up": "y"},
        )
        assert response.status_code == 401

    def test_tts_requires_auth(self, client):
        response = client.post("/api/chat/tts", json={"text": "hello"})
        assert response.status_code == 401

    def test_tts_voices_requires_no_auth(self, client):
        # GET endpoint has no auth dependency
        response = client.get("/api/chat/tts/voices")
        assert response.status_code == 200
        assert "voices" in response.json()

    def test_search_requires_auth(self, client):
        response = client.post("/rag/search", params={"query": "dismissal"})
        assert response.status_code == 401


# =====================
# Request Validation
# =====================

class TestValidation:
    def test_deepsearch_short_query(self, client, mock_user):
        _authenticate(client.app, mock_user)
        try:
            response = client.post("/rag/deepsearch", json={"query": "ab"})
            assert response.status_code == 422
        finally:
            _clear(client.app)

    def test_deepsearch_max_sources_bounds(self, client, mock_user):
        _authenticate(client.app, mock_user)
        try:
            response = client.post(
                "/rag/deepsearch",
                json={"query": "valid query here", "max_sources": 999},
            )
            assert response.status_code == 422
        finally:
            _clear(client.app)

    def test_tts_invalid_voice(self, client, mock_user):
        _authenticate(client.app, mock_user)
        try:
            response = client.post(
                "/api/chat/tts", json={"text": "hello", "voice": "nonexistent-voice"}
            )
            assert response.status_code == 400
        finally:
            _clear(client.app)

    def test_search_invalid_source_type(self, client, mock_user):
        _authenticate(client.app, mock_user)
        try:
            response = client.post(
                "/rag/search", params={"query": "dismissal", "source_type": "bogus"}
            )
            assert response.status_code == 422
        finally:
            _clear(client.app)

    def test_search_short_query(self, client, mock_user):
        _authenticate(client.app, mock_user)
        try:
            response = client.post("/rag/search", params={"query": ""})
            assert response.status_code == 422
        finally:
            _clear(client.app)


# =====================
# Happy Path
# =====================

class TestDeepSearchHappyPath:
    def test_deepsearch_success(self, client, mock_user):
        from rag_pipeline.deep_search import DeepSearchResult, UnifiedSource

        fake_result = DeepSearchResult(
            answer="A synthesized answer.",
            sources=[
                UnifiedSource(
                    id="src-1",
                    title="Sample Case",
                    excerpt="Some excerpt text.",
                    url="https://example.com/case",
                    source_type="case_law",
                    relevance_score=0.9123,
                )
            ],
            source_type_counts={"case_law": 1},
            suggested_follow_ups=["What about remedies?"],
            processing_time_ms=1234,
        )

        mock_engine = MagicMock()
        mock_engine.deep_search = AsyncMock(return_value=fake_result)

        _authenticate(client.app, mock_user)
        try:
            with patch("api.deepsearch.DeepSearchEngine", return_value=mock_engine):
                response = client.post(
                    "/rag/deepsearch",
                    json={"query": "What is wrongful dismissal?", "max_sources": 10},
                )
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "A synthesized answer."
            assert len(data["sources"]) == 1
            assert data["sources"][0]["id"] == "src-1"
            # relevance_score rounded to 3 decimals
            assert data["sources"][0]["relevance_score"] == 0.912
            assert data["source_type_counts"] == {"case_law": 1}
            assert data["suggested_follow_ups"] == ["What about remedies?"]
            assert data["processing_time_ms"] == 1234
        finally:
            _clear(client.app)

    def test_deepsearch_engine_error(self, client, mock_user):
        from rag_pipeline.deep_search import DeepSearchResult

        fake_result = DeepSearchResult(answer="", error="Query must be at least 3 characters.")

        mock_engine = MagicMock()
        mock_engine.deep_search = AsyncMock(return_value=fake_result)

        _authenticate(client.app, mock_user)
        try:
            with patch("api.deepsearch.DeepSearchEngine", return_value=mock_engine):
                response = client.post(
                    "/rag/deepsearch", json={"query": "valid query"}
                )
            assert response.status_code == 400
            assert "Query must be at least" in response.json()["detail"]
        finally:
            _clear(client.app)

    def test_deepsearch_followup_success(self, client, mock_user):
        from rag_pipeline.deep_search import DeepSearchResult, UnifiedSource

        fake_result = DeepSearchResult(
            answer="Follow-up answer.",
            sources=[
                UnifiedSource(
                    id="src-2", title="Another Case", excerpt="excerpt",
                    url="", source_type="web", relevance_score=0.5,
                )
            ],
            suggested_follow_ups=[],
        )

        mock_engine = MagicMock()
        mock_engine.deep_search = AsyncMock(return_value=fake_result)

        _authenticate(client.app, mock_user)
        try:
            with patch("api.deepsearch.DeepSearchEngine", return_value=mock_engine):
                response = client.post(
                    "/rag/deepsearch/followup",
                    params={"original_query": "What is dismissal?", "follow_up": "And remedies?"},
                )
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Follow-up answer."
            assert data["sources"][0]["id"] == "src-2"
        finally:
            _clear(client.app)


class TestTTSHappyPath:
    def test_tts_success(self, client, mock_user):
        _authenticate(client.app, mock_user)
        try:
            with patch("rag_pipeline.tts.generate_audio", new=AsyncMock(return_value=b"FAKE_MP3_BYTES")):
                response = client.post(
                    "/api/chat/tts", json={"text": "hello world", "voice": "en-CA-LiamNeural"}
                )
            assert response.status_code == 200
            assert response.content == b"FAKE_MP3_BYTES"
            assert response.headers["content-type"] == "audio/mpeg"
            assert "attachment" in response.headers["content-disposition"]
        finally:
            _clear(client.app)

    def test_tts_audio_unavailable(self, client, mock_user):
        _authenticate(client.app, mock_user)
        try:
            with patch("rag_pipeline.tts.generate_audio", new=AsyncMock(return_value=None)):
                response = client.post(
                    "/api/chat/tts", json={"text": "hello world"}
                )
            assert response.status_code == 503
        finally:
            _clear(client.app)

    def test_tts_voices_list(self, client):
        response = client.get("/api/chat/tts/voices")
        assert response.status_code == 200
        voices = response.json()["voices"]
        assert any(v["id"] == "en-CA-LiamNeural" for v in voices)


class TestSearchHappyPath:
    def test_search_success(self, client, mock_user):
        fake_store = MagicMock()
        fake_store.search.return_value = [
            {
                "id": "doc-1",
                "score": 0.87,
                "metadata": {
                    "title": "Dismissal Case",
                    "content": "Detailed content about dismissal.",
                    "jurisdiction": "ON",
                    "court": "ONSC",
                    "year": "2021",
                    "citation": "2021 ONSC 123",
                },
            },
            {
                "id": "doc-2",
                "score": 0.65,
                "metadata": {
                    "title": "Another Case",
                    "content": "More content.",
                },
            },
        ]

        _authenticate(client.app, mock_user)
        try:
            with patch("rag_pipeline.vector_store.create_vector_store", return_value=fake_store):
                response = client.post(
                    "/rag/search",
                    params={"query": "wrongful dismissal", "page": 1, "page_size": 10},
                )
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 2
            assert len(data["results"]) == 2
            assert data["results"][0]["id"] == "doc-1"
            assert data["results"][0]["title"] == "Dismissal Case"
            assert data["results"][0]["jurisdiction"] == "ON"
            assert data["results"][0]["citation"] == "2021 ONSC 123"
            assert data["results"][0]["relevance_score"] == 0.87
            assert data["results"][0]["source_type"] == "case_law"
            assert data["page"] == 1
            assert data["page_size"] == 10
        finally:
            _clear(client.app)

    def test_search_pagination(self, client, mock_user):
        fake_store = MagicMock()
        # 3 results, page_size 2, page 2 -> 1 result (index 2)
        fake_store.search.return_value = [
            {"id": f"doc-{i}", "score": 0.9 - i * 0.1, "metadata": {"title": f"Case {i}"}}
            for i in range(3)
        ]

        _authenticate(client.app, mock_user)
        try:
            with patch("rag_pipeline.vector_store.create_vector_store", return_value=fake_store):
                response = client.post(
                    "/rag/search",
                    params={"query": "dismissal", "page": 2, "page_size": 2},
                )
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 3
            assert len(data["results"]) == 1
            assert data["results"][0]["id"] == "doc-2"
        finally:
            _clear(client.app)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
