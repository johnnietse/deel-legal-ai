"""Tests for web search client."""
import pytest
import requests
from rag_pipeline.web_search import WebResult, search_web, SearxngClient


class TestWebResult:
    def test_web_result_creation(self):
        r = WebResult(title="Test", snippet="Snippet", url="https://example.com",
                      content="Full content", engine="duckduckgo", score=0.95)
        assert r.title == "Test"
        assert r.score == 0.95

    def test_web_result_defaults(self):
        r = WebResult(title="Test", snippet="S", url="https://ex.com")
        assert r.engine == ""
        assert r.score == 0.0
        assert r.content == ""


class TestSearxngClient:
    def test_build_url(self):
        client = SearxngClient(base_url="http://localhost:8888")
        url = client._build_url("test query", top_k=5)
        assert "test%20query" in url
        assert "format=json" in url
        assert "language=en" in url
        assert "categories=general" in url
        assert "pageno=1" in url

    @pytest.mark.asyncio
    async def test_parse_response(self, mocker):
        client = SearxngClient()
        mock_response = {
            "results": [
                {"title": "R1", "content": "Content 1", "url": "https://a.com",
                 "engine": "ddg", "score": 0.9},
                {"title": "R2", "content": "Content 2", "url": "https://b.com",
                 "engine": "google", "score": 0.8},
            ]
        }
        results = client._parse_response(mock_response)
        assert len(results) == 2
        assert results[0].title == "R1"
        assert results[0].engine == "ddg"

    @pytest.mark.asyncio
    async def test_parse_response_with_infobox(self, mocker):
        client = SearxngClient()
        mock_response = {
            "results": [],
            "infoboxes": [
                {"infobox": "Person", "content": "John Doe is a lawyer",
                 "urls": [{"url": "https://example.com/john"}]}
            ]
        }
        results = client._parse_response(mock_response)
        assert len(results) == 1
        assert results[0].engine == "infobox"
        assert results[0].score == 0.7

    @pytest.mark.asyncio
    async def test_search_web_timeout(self, mocker):
        # Patch the session's get method since we use requests.Session()
        mocker.patch("requests.Session.get", side_effect=requests.Timeout)
        results = await search_web("test query")
        assert results == []  # Graceful degradation

    @pytest.mark.asyncio
    async def test_search_web_connection_error(self, mocker):
        mocker.patch("requests.Session.get", side_effect=requests.ConnectionError)
        results = await search_web("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_web_generic_error(self, mocker):
        mocker.patch("requests.Session.get", side_effect=Exception("Unexpected error"))
        results = await search_web("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_sorts_by_score(self, mocker):
        """Test that results are sorted by score descending."""
        mock_response = {
            "results": [
                {"title": "Low", "content": "C", "url": "https://c.com", "engine": "e", "score": 0.3},
                {"title": "High", "content": "H", "url": "https://h.com", "engine": "e", "score": 0.9},
                {"title": "Med", "content": "M", "url": "https://m.com", "engine": "e", "score": 0.6},
            ]
        }
        mocker.patch("requests.Session.get")
        # Create a mock response object
        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status.return_value = None
        mocker.patch("requests.Session.get", return_value=mock_resp)

        client = SearxngClient()
        results = client.search("test", top_k=3)
        assert len(results) == 3
        assert results[0].title == "High"
        assert results[1].title == "Med"
        assert results[2].title == "Low"

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, mocker):
        """Test that search respects top_k limit."""
        mock_response = {
            "results": [
                {"title": f"R{i}", "content": f"C{i}", "url": f"https://{i}.com", "engine": "e", "score": 0.5}
                for i in range(10)
            ]
        }
        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status.return_value = None
        mocker.patch("requests.Session.get", return_value=mock_resp)

        client = SearxngClient()
        results = client.search("test", top_k=3)
        assert len(results) == 3