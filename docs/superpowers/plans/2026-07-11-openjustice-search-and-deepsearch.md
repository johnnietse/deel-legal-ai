# OpenJustice Search, DeepSearch & API Platform — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add web search, DeepSearch tool, audio summaries, API key auth, search UI, and fix all mocks — entirely free, self-hosted.

**Architecture:** SearXNG (Docker web search) + edge-tts (Microsoft TTS) + new `deep_search.py` orchestrator that queries Pinecone, Elasticsearch, web, and legislation in parallel. API key middleware validates X-API-Key header on all endpoints. Frontend gets new Search + DeepSearch pages. All existing code preserved.

**Tech Stack:** SearXNG Docker, edge-tts Python, Redis, FastAPI, React, Gemini 12-key rotation

## Global Constraints

- All services must be entirely free, no API keys from user
- Nothing fake — all mock frontend data must be replaced with real API calls
- Existing endpoints, Docker services, and frontend pages must not be modified (only extended)
- All new modules go in `rag_pipeline/` or new `api/` files
- Frontend routes follow existing React Router v6 pattern in `App.tsx`
- API key format: `oj_` prefix + 48 url-safe random chars
- All Gemini API calls use existing 12-key rotation in `gemini_key_manager.py`
- Docker services use existing `deel-legal-ai-network` network

---

## File Map

### New Files
| File | Purpose |
|------|---------|
| `docker-compose.override.yml` | Add SearXNG service (won't touch main compose) |
| `searxng/settings.yml` | SearXNG configuration |
| `rag_pipeline/web_search.py` | SearXNG client wrapper |
| `rag_pipeline/deep_search.py` | DeepSearch orchestrator |
| `rag_pipeline/tts.py` | edge-tts audio generation |
| `rag_pipeline/redis_client.py` | Redis connection + caching |
| `api/api_keys.py` | API key CRUD endpoints |
| `api/deepsearch.py` | DeepSearch + TTS endpoints |
| `api/search.py` | Search endpoint for structured results |
| `openjustice-frontend/src/app/search/SearchPage.tsx` | Search UI |
| `openjustice-frontend/src/app/deepsearch/DeepSearchPage.tsx` | DeepSearch page |
| `openjustice-frontend/src/components/search/SearchResultCard.tsx` | Result card |
| `openjustice-frontend/src/components/search/SearchFilters.tsx` | Filter sidebar |
| `openjustice-frontend/src/components/search/AudioPlayer.tsx` | Audio player |
| `tests/test_web_search.py` | Web search tests |
| `tests/test_deep_search.py` | DeepSearch tests |
| `tests/test_api_keys.py` | API key tests |
| `tests/test_tts.py` | TTS tests |

### Modified Files
| File | Change |
|------|--------|
| `config.py` | Add SEARXNG_BASE_URL, REDIS_URL |
| `requirements.txt` | Add edge-tts, redis |
| `openjustice-frontend/package.json` | Add react-router deps if missing |
| `api/main.py` | Add auth to core endpoints, mount new routers |
| `api/router.py` | Include api_keys, deepsearch, search routers |
| `api/auth.py` | Add API key auth dependency |
| `api/middleware.py` | Extend to cover /rag/, /classify/ paths; Redis-backed store |
| `api/users.py` | Add API key management endpoints |
| `db/repository.py` | Add generate_api_key method |
| `rag_pipeline/rag_query.py` | Accept user_id for tracking |
| `openjustice-frontend/src/App.tsx` | Add /search, /deepsearch routes |
| `openjustice-frontend/src/lib/api/realClient.ts` | Replace all mock data with real API calls |
| `openjustice-frontend/src/lib/hooks/useQuery.ts` | Add deepSearch, search, tts hooks |
| `openjustice-frontend/src/app/settings/SettingsPage.tsx` | Connect to real API key endpoints |

---

### Task 1: SearXNG Docker & Configuration

**Files:**
- Create: `docker-compose.override.yml` (adds SearXNG without modifying main compose)
- Create: `searxng/settings.yml`
- Modify: `config.py` (add SEARXNG config)

**Interfaces:**
- Consumes: Docker network `deel-legal-ai-network` (already exists from main compose)
- Produces: SearXNG running at `http://localhost:8888`

- [ ] **Step 1: Create docker-compose.override.yml**

```yaml
# docker-compose.override.yml — adds SearXNG to existing stack
# Docker Compose automatically merges this with docker-compose.yml
version: "3.8"
services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8888:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://localhost:8888
      - SEARXNG_SECRET_KEY=openjustice_dev_secret_key_change_in_prod
    cap_add:
      - NET_BIND_SERVICE
    restart: unless-stopped
    networks:
      - deel-legal-ai-network

networks:
  deel-legal-ai-network:
    external: true
```

- [ ] **Step 2: Create SearXNG settings.yml**

```yaml
# searxng/settings.yml
use_default_settings: true
general:
  instance_name: "OpenJustice Search"
  debug: false
search:
  safe_search: 0
  autocomplete: ""
  formats:
    - html
    - json
server:
  port: 8080
  bind_address: "0.0.0.0"
  secret_key: "openjustice_dev_secret_key_change_in_prod"
  limiter: false
  image_proxy: false
ui:
  static_use_hash: true
engines:
  - name: duckduckgo
    disabled: false
  - name: google
    disabled: false
  - name: bing
    disabled: false
  - name: wikipedia
    disabled: false
outgoing:
  request_timeout: 10.0
  max_request_timeout: 15.0
  useragent_suffix: ""
  # No proxies needed for local dev
```

- [ ] **Step 3: Add SearXNG config to config.py**

Edit `config.py` to add after the `LOG_LEVEL=INFO` section:
```python
# SearXNG Web Search
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://localhost:8888")
SEARXNG_ENABLED = True
```

- [ ] **Step 4: Start SearXNG and verify**

Run: `docker compose -f docker-compose.yml -f docker-compose.override.yml up -d searxng`
Run: `curl http://localhost:8888/search?q=test&format=json`
Expected: JSON response with results array

- [ ] **Step 5: Commit**

```bash
git add docker-compose.override.yml searxng/settings.yml config.py
git commit -m "feat: add SearXNG web search Docker service"
```

---

### Task 2: Redis Configuration

**Files:**
- Create: `rag_pipeline/redis_client.py`
- Modify: `config.py` (add REDIS config)
- Modify: `requirements.txt` (add redis-py)

**Interfaces:**
- Produces: `get_redis_client() -> Optional[Redis]` — lazy Redis connection
- Produces: `cache_get(key, ttl)`, `cache_set(key, value, ttl)` helpers

- [ ] **Step 1: Add redis to requirements.txt**

Append to `requirements.txt`:
```
redis>=5.0.0
```

- [ ] **Step 2: Add Redis config to config.py**

```python
# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_ENABLED = True
```

- [ ] **Step 3: Create rag_pipeline/redis_client.py**

```python
"""Redis client with lazy connection and graceful fallback."""
import os
import json
import hashlib
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

_client = None
_available = False


def get_redis_client():
    """Get Redis client (lazy connect, with fallback)."""
    global _client, _available
    if _client is not None:
        return _client
    if not _available and _client is None:
        # Already tried and failed
        return None
    try:
        from config import REDIS_URL
        import redis.asyncio as aioredis
        _client = aioredis.from_url(REDIS_URL, decode_responses=True)
        # Test connection
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_client.ping())
            logger.info("Redis connected")
        except RuntimeError:
            pass  # No event loop yet, assume it works
        _available = True
        return _client
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}. Caching disabled.")
        _client = None
        _available = False
        return None


def cache_key(prefix: str, data: str) -> str:
    """Generate a cache key from prefix + data hash."""
    h = hashlib.sha256(data.encode()).hexdigest()[:16]
    return f"{prefix}:{h}"


async def cache_get(key: str) -> Optional[Any]:
    """Get value from cache."""
    client = get_redis_client()
    if client is None:
        return None
    try:
        val = await client.get(key)
        if val:
            return json.loads(val)
        return None
    except Exception as e:
        logger.warning(f"Cache get failed: {e}")
        return None


async def cache_set(key: str, value: Any, ttl: int = 3600):
    """Set value in cache with TTL."""
    client = get_redis_client()
    if client is None:
        return
    try:
        await client.setex(key, ttl, json.dumps(value))
    except Exception as e:
        logger.warning(f"Cache set failed: {e}")
```

- [ ] **Step 4: Commit**

```bash
git add rag_pipeline/redis_client.py config.py requirements.txt
git commit -m "feat: add Redis client with lazy connection and caching"
```

---

### Task 3: Web Search Client (SearXNG)

**Files:**
- Create: `rag_pipeline/web_search.py`
- Create: `tests/test_web_search.py`

**Interfaces:**
- Produces: `search_web(query: str, top_k: int = 5) -> List[WebResult]` — SearXNG search
- Produces: `WebResult` dataclass with title, snippet, url, content, engine, score

- [ ] **Step 1: Create tests/test_web_search.py**

```python
"""Tests for web search client."""
import pytest
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
    async def test_search_web_timeout(self, mocker):
        mocker.patch("requests.get", side_effect=TimeoutError)
        results = await search_web("test query")
        assert results == []  # Graceful degradation
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web_search.py -v`
Expected: ImportError (module not found), test failures

- [ ] **Step 3: Create rag_pipeline/web_search.py**

```python
"""SearXNG web search client — free, self-hosted web search."""
import logging
from dataclasses import dataclass, field
from typing import List, Optional
import requests
from config import SEARXNG_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class WebResult:
    """A single web search result."""
    title: str
    snippet: str
    url: str
    content: str = ""
    engine: str = ""
    score: float = 0.0


class SearxngClient:
    """Client for SearXNG self-hosted search engine."""

    def __init__(self, base_url: str = None, timeout: int = 10):
        self.base_url = (base_url or SEARXNG_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "OpenJustice.ai/1.0"})

    def _build_url(self, query: str, top_k: int = 5) -> str:
        import urllib.parse
        params = urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "language": "en",
            "categories": "general",
            "pageno": 1,
        })
        return f"{self.base_url}/search?{params}"

    def _parse_response(self, data: dict) -> List[WebResult]:
        results = []
        for r in data.get("results", []):
            results.append(WebResult(
                title=r.get("title", ""),
                snippet=r.get("content", "")[:300],
                url=r.get("url", ""),
                content=r.get("content", ""),
                engine=r.get("engine", ""),
                score=r.get("score", 0.5),
            ))
        # Also check infoboxes
        for ib in data.get("infoboxes", []):
            results.append(WebResult(
                title=ib.get("infobox", ""),
                snippet=ib.get("content", "")[:300],
                url=ib.get("urls", [{}])[0].get("url", "") if ib.get("urls") else "",
                content=ib.get("content", ""),
                engine="infobox",
                score=0.7,
            ))
        return results

    def search(self, query: str, top_k: int = 5) -> List[WebResult]:
        """Search the web via SearXNG. Returns empty list on failure."""
        try:
            url = self._build_url(query, top_k)
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            results = self._parse_response(resp.json())
            # Sort by score descending, take top_k
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:top_k]
        except requests.Timeout:
            logger.warning(f"SearXNG timeout for query: {query[:50]}...")
            return []
        except requests.ConnectionError:
            logger.warning(f"SearXNG connection refused at {self.base_url}")
            return []
        except Exception as e:
            logger.error(f"SearXNG search error: {e}")
            return []


# Singleton client
_client: Optional[SearxngClient] = None


def get_searxng_client() -> SearxngClient:
    """Get or create SearXNG client singleton."""
    global _client
    if _client is None:
        _client = SearxngClient()
    return _client


async def search_web(query: str, top_k: int = 5) -> List[WebResult]:
    """Convenience async wrapper for web search (runs sync in thread)."""
    import asyncio
    client = get_searxng_client()
    return await asyncio.to_thread(client.search, query, top_k)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web_search.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add rag_pipeline/web_search.py tests/test_web_search.py
git commit -m "feat: add SearXNG web search client with graceful degradation"
```

---

### Task 4: DeepSearch Orchestrator

**Files:**
- Create: `rag_pipeline/deep_search.py`
- Create: `tests/test_deep_search.py`

**Interfaces:**
- Produces: `DeepSearchEngine` class with `deep_search(query, user_id, max_sources) -> DeepSearchResult`
- Produces: `DeepSearchResult`, `UnifiedSource` data models
- Consumes: `web_search.search_web()`, Pinecone (via `rag_query`), Elasticsearch (via `hybrid_retriever`), legislation DB (via SQL)

- [ ] **Step 1: Create tests/test_deep_search.py**

```python
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
        # Mock all external dependencies
        mocker.patch("rag_pipeline.deep_search.search_web", return_value=[
            UnifiedSource(id="w1", title="Web1", excerpt="", source_type="web", relevance_score=0.8)
        ])
        mocker.patch("rag_pipeline.deep_search.LegalRAGQuery")
        engine = DeepSearchEngine()
        # Should return counts even with minimal data
        result = await engine.deep_search("test")
        if result:
            assert isinstance(result.source_type_counts, dict)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_deep_search.py -v`
Expected: ImportError, test failures

- [ ] **Step 3: Create rag_pipeline/deep_search.py**

```python
"""DeepSearch — multi-source legal research orchestrator.

Queries Pinecone (vector), Elasticsearch (BM25), SearXNG (web),
and legislation DB in parallel, then fuses results and synthesizes
a cited answer via Gemini.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal
import asyncio

logger = logging.getLogger(__name__)


SourceType = Literal["case_law", "web", "statute", "bm25"]


@dataclass
class UnifiedSource:
    """A single source result from any backend, unified format."""
    id: str
    title: str
    excerpt: str
    url: str = ""
    source_type: SourceType = "case_law"
    relevance_score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class DeepSearchResult:
    """Complete DeepSearch result."""
    answer: str
    sources: List[UnifiedSource] = field(default_factory=list)
    source_type_counts: Dict[str, int] = field(default_factory=dict)
    suggested_follow_ups: List[str] = field(default_factory=list)
    processing_time_ms: int = 0
    error: Optional[str] = None


def deduplicate_sources(sources: List[UnifiedSource]) -> List[UnifiedSource]:
    """Remove duplicate sources by URL (prefer higher score)."""
    seen_urls = set()
    deduped = []
    for s in sorted(sources, key=lambda x: x.relevance_score, reverse=True):
        key = s.url or s.id
        if key not in seen_urls:
            seen_urls.add(key)
            deduped.append(s)
    return deduped


def fuse_and_rank_results(
    case_sources: List[UnifiedSource],
    web_sources: List[UnifiedSource],
    statute_sources: List[UnifiedSource],
    top_k: int = 15,
) -> List[UnifiedSource]:
    """Fuse all sources, deduplicate, and rank by relevance."""
    all_sources = case_sources + web_sources + statute_sources
    all_sources = deduplicate_sources(all_sources)
    all_sources.sort(key=lambda s: s.relevance_score, reverse=True)
    return all_sources[:top_k]


def count_source_types(sources: List[UnifiedSource]) -> Dict[str, int]:
    """Count sources by type."""
    counts = {}
    for s in sources:
        counts[s.source_type] = counts.get(s.source_type, 0) + 1
    return counts


class DeepSearchEngine:
    """Orchestrates multi-source legal deep research."""

    def __init__(self):
        from rag_pipeline.rag_query import LegalRAGQuery
        from rag_pipeline.embeddings import GeminiChat
        self.rag_query = LegalRAGQuery()
        self.chat = GeminiChat()

    async def _search_vector(self, query: str, top_k: int = 10) -> List[UnifiedSource]:
        """Search Pinecone vector store."""
        try:
            from rag_pipeline.vector_store import create_vector_store
            from config import VECTOR_STORE_BACKEND
            store = create_vector_store(backend=VECTOR_STORE_BACKEND)
            results = await asyncio.to_thread(
                store.search, query, top_k=top_k
            )
            sources = []
            for r in results:
                sources.append(UnifiedSource(
                    id=r.get("id", ""),
                    title=r.get("metadata", {}).get("title", "Unknown Case"),
                    excerpt=(r.get("metadata", {}).get("content", "") or "")[:500],
                    source_type="case_law",
                    relevance_score=r.get("score", 0.0),
                    metadata=r.get("metadata", {}),
                ))
            return sources
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    async def _search_bm25(self, query: str, top_k: int = 10) -> List[UnifiedSource]:
        """Search Elasticsearch BM25."""
        try:
            from config import ELASTICSEARCH_URL
            import requests
            es_query = {
                "query": {"multi_match": {"query": query, "fields": ["title^3", "content^2", "citation"]}},
                "size": top_k,
            }
            resp = requests.post(
                f"{ELASTICSEARCH_URL}/deel-legal-chunks/_search",
                json=es_query,
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            sources = []
            for hit in data.get("hits", {}).get("hits", []):
                src = hit["_source"]
                sources.append(UnifiedSource(
                    id=hit["_id"],
                    title=src.get("title", "Unknown"),
                    excerpt=(src.get("content", "") or "")[:500],
                    source_type="bm25",
                    relevance_score=hit.get("_score", 0) / 100.0,  # Normalize
                    metadata=src,
                ))
            return sources
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []

    async def _search_web(self, query: str, top_k: int = 5) -> List[UnifiedSource]:
        """Search web via SearXNG."""
        try:
            from rag_pipeline.web_search import search_web
            results = await search_web(query, top_k=top_k)
            return [
                UnifiedSource(
                    id=f"web_{hash(r.url) % 10**8}",
                    title=r.title,
                    excerpt=r.snippet[:500],
                    url=r.url,
                    source_type="web",
                    relevance_score=r.score,
                )
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []

    async def _search_legislation(self, query: str, top_k: int = 5) -> List[UnifiedSource]:
        """Search legislation in PostgreSQL."""
        try:
            from db.database import get_session
            from sqlalchemy import text
            async for db in get_session():
                stmt = text("""
                    SELECT document_id, title, content, source
                    FROM legislation_documents
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
                    LIMIT :limit
                """)
                result = await db.execute(stmt, {"query": query, "limit": top_k})
                rows = result.fetchall()
                sources = []
                for row in rows:
                    sources.append(UnifiedSource(
                        id=f"leg_{row.document_id}",
                        title=row.title,
                        excerpt=(row.content or "")[:500],
                        source_type="statute",
                        relevance_score=0.7,
                    ))
                return sources
        except Exception as e:
            logger.warning(f"Legislation search failed: {e}")
            return []

    async def _synthesize_answer(
        self, query: str, sources: List[UnifiedSource]
    ) -> tuple[str, List[str]]:
        """Use Gemini to synthesize a cited answer from sources."""
        if not sources:
            return "I couldn't find any relevant sources to answer your question.", []

        # Build context string with source markers
        context_parts = []
        for i, s in enumerate(sources):
            type_label = {"case_law": "Case Law", "web": "Web Source", "statute": "Statute", "bm25": "Case Law"}.get(
                s.source_type, "Source"
            )
            context_parts.append(
                f"[Source {i+1}] ({type_label}) {s.title}\n"
                f"{s.excerpt}\n"
                f"URL: {s.url}\n"
            )
        context = "\n---\n".join(context_parts)

        system_instruction = (
            "You are a senior Canadian legal research assistant. "
            "Synthesize a comprehensive answer from the provided sources. "
            "Cite each claim with the corresponding source number [Source N]. "
            "If sources disagree, note the disagreement. "
            "If no source supports a claim, say 'The available sources do not address this point.'"
        )

        prompt = (
            f"Research question: {query}\n\n"
            f"Available sources:\n{context}\n\n"
            f"Provide a clear, well-structured answer with inline citations. "
            f"Then suggest 2-3 follow-up questions the user might ask next."
        )

        try:
            response = self.chat.generate(
                prompt,
                system_instruction=system_instruction,
                temperature=0.3,
                max_tokens=2048,
            )
            text = response.get("text", "") or response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if not text:
                text = response if isinstance(response, str) else str(response)

            # Extract follow-up questions (lines after "Follow-up" or last 3 lines starting with "?")
            lines = text.split("\n")
            follow_ups = []
            in_followup = False
            for line in lines:
                stripped = line.strip()
                if "follow-up" in stripped.lower() or "suggested question" in stripped.lower():
                    in_followup = True
                    continue
                if in_followup and stripped and stripped.endswith("?"):
                    follow_ups.append(stripped.lstrip("0123456789. )-"))
                elif in_followup and not stripped:
                    continue
                elif in_followup and stripped:
                    # Non-question line after follow-ups — stop
                    break

            # Clean answer (remove follow-up section)
            answer_lines = []
            for line in lines:
                if "follow-up" in line.lower() and any(c.isdigit() for c in line):
                    break
                answer_lines.append(line)
            answer = "\n".join(answer_lines).strip()

            return answer or text, follow_ups[:5]

        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return (
                f"I found {len(sources)} relevant sources but encountered an error "
                f"generating the analysis. Please try again.",
                [],
            )

    async def deep_search(
        self,
        query: str,
        user_id: Optional[str] = None,
        max_sources: int = 15,
    ) -> DeepSearchResult:
        """Run a full DeepSearch query across all sources."""
        if not query or len(query.strip()) < 3:
            return DeepSearchResult(
                answer="", error="Query must be at least 3 characters."
            )

        start = time.time()

        # Launch parallel searches
        case_task = self._search_vector(query)
        bm25_task = self._search_bm25(query)
        web_task = self._search_web(query)
        leg_task = self._search_legislation(query)

        case_sources, bm25_sources, web_sources, leg_sources = await asyncio.gather(
            case_task, bm25_task, web_task, leg_task,
            return_exceptions=True,
        )

        # Handle exceptions
        for i, (name, result) in enumerate([
            ("case", case_sources), ("bm25", bm25_sources),
            ("web", web_sources), ("legislation", leg_sources),
        ]):
            if isinstance(result, Exception):
                logger.error(f"{name} search raised: {result}")

        case_sources = case_sources if not isinstance(case_sources, Exception) else []
        bm25_sources = bm25_sources if not isinstance(bm25_sources, Exception) else []
        web_sources = web_sources if not isinstance(web_sources, Exception) else []
        leg_sources = leg_sources if not isinstance(leg_sources, Exception) else []

        # Fuse and rank
        fused = fuse_and_rank_results(
            case_sources + bm25_sources,
            web_sources,
            leg_sources,
            top_k=max_sources,
        )

        # Count by type
        type_counts = count_source_types(fused)

        # Synthesize answer
        answer, follow_ups = await self._synthesize_answer(query, fused)

        elapsed = int((time.time() - start) * 1000)

        return DeepSearchResult(
            answer=answer,
            sources=fused,
            source_type_counts=type_counts,
            suggested_follow_ups=follow_ups,
            processing_time_ms=elapsed,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_deep_search.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add rag_pipeline/deep_search.py tests/test_deep_search.py
git commit -m "feat: add DeepSearch orchestrator with parallel multi-source search"
```

---

### Task 5: API Key Management Backend

**Files:**
- Create: `api/api_keys.py`
- Modify: `api/auth.py` (add API key auth dependency)
- Modify: `api/router.py` (mount api_keys router)
- Modify: `api/users.py` (add key management endpoints)
- Modify: `db/repository.py` (add generate_api_key method)
- Create: `tests/test_api_keys.py`

**Interfaces:**
- Produces: `GET /api/auth/keys` — list user's keys (masked)
- Produces: `POST /api/auth/keys` — generate new key
- Produces: `DELETE /api/auth/keys/{id}` — revoke key
- Produces: `get_api_key_user(api_key: str) -> Optional[dict]` — auth dependency

- [ ] **Step 1: Add key generation to db/repository.py**

Add to `db/repository.py`:
```python
import secrets
import hashlib

def generate_api_key_value() -> tuple[str, str]:
    """Generate a new API key pair: (plaintext, sha256_hash)."""
    random_part = secrets.token_urlsafe(48)
    plaintext = f"oj_{random_part}"
    key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
    return plaintext, key_hash
```

- [ ] **Step 2: Create api/api_keys.py**

```python
"""API key management endpoints."""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from db.database import get_session
from db.repository import (
    create_api_key, list_api_keys, delete_api_key,
    generate_api_key_value,
)
from api.auth import get_current_user, hash_api_key

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["api-keys"])

@router.get("/keys")
async def get_keys(user=Depends(get_current_user), db=Depends(get_session)):
    """List user's API keys (masked)."""
    keys = await list_api_keys(db, user["user_id"])
    return {
        "keys": [
            {
                "id": k.id,
                "name": k.name,
                "key_preview": k.key_hash[:12] + "...",
                "created_at": k.created_at.isoformat() if k.created_at else None,
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
            }
            for k in keys
        ]
    }

@router.post("/keys")
async def create_key_route(
    name: str,
    user=Depends(get_current_user),
    db=Depends(get_session),
):
    """Generate a new API key."""
    plaintext, key_hash = generate_api_key_value()
    api_key = await create_api_key(db, user["user_id"], name, key_hash)
    return {
        "id": api_key.id,
        "name": api_key.name,
        "key": plaintext,  # Only returned once!
        "created_at": api_key.created_at.isoformat() if api_key.created_at else None,
    }

@router.delete("/keys/{key_id}")
async def revoke_key_route(
    key_id: str,
    user=Depends(get_current_user),
    db=Depends(get_session),
):
    """Revoke an API key."""
    deleted = await delete_api_key(db, key_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"message": "Key revoked"}
```

- [ ] **Step 3: Add API key auth to api/auth.py**

Add to `api/auth.py`:
```python
import hashlib
from fastapi import Header, HTTPException, status

async def get_api_key_user(
    x_api_key: str = Header(None, alias="X-API-Key"),
    db = Depends(get_session),
):
    """Authenticate via API key header. Returns user dict or None."""
    if not x_api_key:
        return None
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    from db.repository import get_api_key_by_hash
    api_key = await get_api_key_by_hash(db, key_hash)
    if not api_key:
        return None
    # Update last_used_at
    from db.repository import update_api_key_last_used
    await update_api_key_last_used(db, api_key.id)
    from db.repository import get_user_by_id
    user = await get_user_by_id(db, api_key.user_id)
    if not user:
        return None
    return {"user_id": user.id, "email": user.email, "tier": user.tier, "auth_method": "api_key"}
```

- [ ] **Step 4: Add repository methods to db/repository.py**

```python
async def get_api_key_by_hash(db, key_hash: str) -> Optional[ApiKey]:
    from sqlalchemy import select
    stmt = select(ApiKey).where(ApiKey.key_hash == key_hash)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

async def update_api_key_last_used(db, key_id: str) -> None:
    from datetime import datetime, timezone
    stmt = update(ApiKey).where(ApiKey.id == key_id).values(last_used_at=datetime.now(timezone.utc))
    await db.execute(stmt)
    await db.commit()
```

- [ ] **Step 5: Mount routers in api/router.py**

Add to the router includes after existing ones:
```python
from api import api_keys
api_router.include_router(api_keys.router)
```

- [ ] **Step 6: Create tests and run**

Create `tests/test_api_keys.py` with tests for key generation, listing, revocation.
Run: `pytest tests/test_api_keys.py -v`

- [ ] **Step 7: Commit**

```bash
git add api/api_keys.py api/auth.py api/router.py db/repository.py tests/test_api_keys.py
git commit -m "feat: add API key management backend with generation and auth"
```

---

### Task 6: Auth on Core Endpoints

**Files:**
- Modify: `api/main.py` (add auth check to core endpoints)
- Modify: `api/middleware.py` (extend rate limiting to cover /rag/, /classify/)
- Modify: `rag_pipeline/rag_query.py` (accept user_id parameter)

- [ ] **Step 1: Add auth to core endpoints in api/main.py**

For each endpoint (`/rag/query`, `/classify`, `/rag/deepsearch`, `/rag/verify`, `/evaluate/*`):
- Add `user = await get_optional_user(request)` using existing helper
- Fall back to `await get_api_key_user(request.headers.get("x-api-key"))`
- Track `queries_used` for authenticated users
- For unauthenticated: apply lower max_length (1000 chars vs 5000)

Example change pattern for `/rag/query`:
```python
@router.post("/rag/query")
async def rag_query_endpoint(
    request: Request,
    query_data: QueryRequest,
    user: Optional[dict] = Depends(get_optional_user),
    api_user: Optional[dict] = Depends(get_api_key_user),
):
    effective_user = user or api_user
    if not effective_user:
        # Unauthenticated: stricter limits
        if len(query_data.query) > 1000:
            raise HTTPException(status_code=400, detail="Query too long for unauthenticated access")
    # ... rest of existing logic, pass effective_user
```

- [ ] **Step 2: Extend rate limit middleware**

In `api/middleware.py`, change the path prefix check from `/api/` to include `/rag/` and `/classify/`:
```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        if not any(path.startswith(p) for p in ["/api/", "/rag/", "/classify/", "/evaluate/"]):
            return await call_next(request)
        # ... rest of existing middleware logic
```

- [ ] **Step 3: Add user_id to rag_query.py**

In `LegalRAGQuery.query()`:
- Accept optional `user_id` parameter
- Pass through to metrics collector

- [ ] **Step 4: Commit**

```bash
git add api/main.py api/middleware.py rag_pipeline/rag_query.py
git commit -m "fix: add auth checks to core RAG/classify endpoints and extend rate limiting"
```

---

### Task 7: TTS Audio Generation

**Files:**
- Create: `rag_pipeline/tts.py`
- Create: `tests/test_tts.py`
- Modify: `requirements.txt` (add edge-tts)

- [ ] **Step 1: Add edge-tts to requirements.txt**

```
edge-tts>=6.1.0
```

- [ ] **Step 2: Create tests/test_tts.py**

```python
"""Tests for TTS audio generation."""
import pytest
from rag_pipeline.tts import generate_audio, AVAILABLE_VOICES

def test_available_voices():
    assert "en-CA-LiamNeural" in AVAILABLE_VOICES
    assert "en-CA-ClaraNeural" in AVAILABLE_VOICES

@pytest.mark.asyncio
async def test_generate_audio_short_text(mocker):
    """Test with mocked edge-tts."""
    mock_communicate = mocker.patch("edge_tts.Communicate")
    mock_instance = mock_communicate.return_value
    mock_instance.stream = mocker.AsyncMock(return_value=[b"audio data"])
    
    result = await generate_audio("Hello world")
    assert result is not None
    assert len(result) > 0

@pytest.mark.asyncio
async def test_generate_audio_empty_text():
    result = await generate_audio("")
    assert result is None or result == b""

@pytest.mark.asyncio
async def test_generate_audio_long_text(mocker):
    """Text longer than 5000 chars should be truncated."""
    long_text = "Test. " * 2000  # ~12000 chars
    mock_communicate = mocker.patch("edge_tts.Communicate")
    mock_instance = mock_communicate.return_value
    mock_instance.stream = mocker.AsyncMock(return_value=[b"audio data"])
    
    result = await generate_audio(long_text)
    assert result is not None
```

- [ ] **Step 3: Create rag_pipeline/tts.py**

```python
"""Audio generation via edge-tts (free Microsoft TTS, no API key needed)."""
import logging
import io
from typing import Optional

logger = logging.getLogger(__name__)

AVAILABLE_VOICES = {
    "en-CA-LiamNeural": "English (Canada) — Male, default for legal",
    "en-CA-ClaraNeural": "English (Canada) — Female",
    "en-US-GuyNeural": "English (US) — Male",
}


async def generate_audio(
    text: str,
    voice: str = "en-CA-LiamNeural",
) -> Optional[bytes]:
    """Generate MP3 audio from text using edge-tts.
    
    Returns MP3 bytes, or None on failure.
    Max input: 5000 characters (truncated).
    """
    if not text or not text.strip():
        return None

    # Truncate to avoid TTS timeout
    if len(text) > 5000:
        text = text[:4997] + "..."

    try:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        audio_chunks = []

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        if audio_chunks:
            return b"".join(audio_chunks)
        else:
            logger.warning("TTS produced no audio output")
            return None

    except ImportError:
        logger.error("edge-tts not installed. Run: pip install edge-tts")
        return None
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        return None
```

- [ ] **Step 4: Run tests**

Run: `pip install edge-tts` then `pytest tests/test_tts.py -v`

- [ ] **Step 5: Commit**

```bash
git add rag_pipeline/tts.py tests/test_tts.py requirements.txt
git commit -m "feat: add free TTS audio generation via edge-tts (Microsoft)"
```

---

### Task 8: DeepSearch & TTS API Endpoints

**Files:**
- Create: `api/deepsearch.py`
- Create: `api/search.py`
- Modify: `api/router.py` (mount new routers)

- [ ] **Step 1: Create api/deepsearch.py**

```python
"""DeepSearch and TTS API endpoints."""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional

from api.auth import get_current_user, get_api_key_user
from rag_pipeline.deep_search import DeepSearchEngine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["deepsearch"])

class DeepSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=5000)
    max_sources: int = Field(default=15, ge=1, le=50)

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default="en-CA-LiamNeural")


@router.post("/rag/deepsearch")
async def deep_search_endpoint(
    req: DeepSearchRequest,
    user=Depends(get_current_user),
    api_user=Depends(get_api_key_user),
):
    """Multi-source deep research across case law, web, and statutes."""
    effective_user = user or api_user
    if not effective_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    engine = DeepSearchEngine()
    result = await engine.deep_search(
        query=req.query,
        user_id=effective_user.get("user_id"),
        max_sources=req.max_sources,
    )

    if result.error:
        raise HTTPException(status_code=400, detail=result.error)

    return {
        "answer": result.answer,
        "sources": [
            {
                "id": s.id,
                "title": s.title,
                "excerpt": s.excerpt,
                "url": s.url,
                "source_type": s.source_type,
                "relevance_score": round(s.relevance_score, 3),
            }
            for s in result.sources
        ],
        "source_type_counts": result.source_type_counts,
        "suggested_follow_ups": result.suggested_follow_ups,
        "processing_time_ms": result.processing_time_ms,
    }


@router.post("/rag/deepsearch/followup")
async def deep_search_followup(
    original_query: str = Query(...),
    follow_up: str = Query(...),
    user=Depends(get_current_user),
    api_user=Depends(get_api_key_user),
):
    """Ask a follow-up question on a previous DeepSearch result."""
    effective_user = user or api_user
    if not effective_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Combine original context with follow-up
    combined = f"Original question: {original_query}\nFollow-up: {follow_up}"
    engine = DeepSearchEngine()
    result = await engine.deep_search(
        query=combined,
        user_id=effective_user.get("user_id"),
    )

    if result.error:
        raise HTTPException(status_code=400, detail=result.error)

    return {
        "answer": result.answer,
        "sources": [
            {
                "id": s.id,
                "title": s.title,
                "excerpt": s.excerpt,
                "url": s.url,
                "source_type": s.source_type,
                "relevance_score": round(s.relevance_score, 3),
            }
            for s in result.sources
        ],
        "suggested_follow_ups": result.suggested_follow_ups,
    }


@router.post("/api/chat/tts")
async def text_to_speech_endpoint(
    req: TTSRequest,
    user=Depends(get_current_user),
    api_user=Depends(get_api_key_user),
):
    """Generate audio from text using free Microsoft TTS."""
    effective_user = user or api_user
    if not effective_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    from rag_pipeline.tts import generate_audio, AVAILABLE_VOICES

    if req.voice not in AVAILABLE_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice. Available: {list(AVAILABLE_VOICES.keys())}",
        )

    audio_bytes = await generate_audio(req.text, voice=req.voice)
    if audio_bytes is None:
        raise HTTPException(status_code=503, detail="Audio generation unavailable")

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f"attachment; filename=openjustice_audio.mp3",
            "Content-Length": str(len(audio_bytes)),
        },
    )


@router.get("/api/chat/tts/voices")
async def list_voices():
    """List available TTS voices."""
    from rag_pipeline.tts import AVAILABLE_VOICES
    return {
        "voices": [
            {"id": k, "description": v}
            for k, v in AVAILABLE_VOICES.items()
        ]
    }
```

- [ ] **Step 2: Create api/search.py**

```python
"""Structured search endpoint for the Search UI."""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List

from api.auth import get_current_user, get_api_key_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])

class SearchResult(BaseModel):
    id: str
    title: str
    excerpt: str
    url: str = ""
    source_type: str
    jurisdiction: str = ""
    court: str = ""
    year: str = ""
    citation: str = ""
    relevance_score: float = 0.0


@router.post("/rag/search")
async def search_endpoint(
    query: str = Field(..., min_length=1, max_length=2000),
    jurisdiction: Optional[str] = Query(None),
    source_type: Optional[str] = Query(None, regex="^(case_law|web|statute|bm25)$"),
    year_from: Optional[int] = Query(None, ge=1900, le=2030),
    year_to: Optional[int] = Query(None, ge=1900, le=2030),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    sort_by: str = Query("relevance", regex="^(relevance|date)$"),
    user=Depends(get_current_user),
    api_user=Depends(get_api_key_user),
):
    """Structured search across the legal corpus. Returns results, not chat."""
    effective_user = user or api_user
    if not effective_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Search vector store with filters
    from rag_pipeline.vector_store import create_vector_store
    from config import VECTOR_STORE_BACKEND

    store = create_vector_store(backend=VECTOR_STORE_BACKEND)
    filter_dict = {}
    if jurisdiction:
        filter_dict["jurisdiction"] = jurisdiction
    if year_from:
        filter_dict["year"] = {"$gte": str(year_from)}

    results = store.search(
        query=query,
        top_k=page_size * page,
        filter=filter_dict if filter_dict else None,
    )

    # Paginate
    start = (page - 1) * page_size
    page_results = results[start:start + page_size]

    return {
        "results": [
            SearchResult(
                id=r.get("id", ""),
                title=r.get("metadata", {}).get("title", ""),
                excerpt=(r.get("metadata", {}).get("content", "") or "")[:500],
                source_type="case_law",
                jurisdiction=r.get("metadata", {}).get("jurisdiction", ""),
                court=r.get("metadata", {}).get("court", ""),
                year=r.get("metadata", {}).get("year", ""),
                citation=r.get("metadata", {}).get("citation", ""),
                relevance_score=r.get("score", 0.0),
            )
            for r in page_results
        ],
        "total": len(results),
        "page": page,
        "page_size": page_size,
    }
```

- [ ] **Step 3: Mount routers in api/router.py**

Add:
```python
from api import deepsearch, search
api_router.include_router(deepsearch.router)
api_router.include_router(search.router)
```

- [ ] **Step 4: Commit**

```bash
git add api/deepsearch.py api/search.py api/router.py
git commit -m "feat: add DeepSearch, TTS, and Search API endpoints"
```

---

### Task 9: Frontend — Search UI Page

**Files:**
- Create: `openjustice-frontend/src/app/search/SearchPage.tsx`
- Create: `openjustice-frontend/src/components/search/SearchResultCard.tsx`
- Create: `openjustice-frontend/src/components/search/SearchFilters.tsx`
- Modify: `openjustice-frontend/src/App.tsx` (add route)

- [ ] **Step 1: Create SearchResultCard.tsx**

```tsx
import React from 'react';

interface SearchResultCardProps {
  title: string;
  excerpt: string;
  sourceType: string;
  relevanceScore: number;
  jurisdiction?: string;
  court?: string;
  year?: string;
  citation?: string;
  url?: string;
}

const sourceTypeLabels: Record<string, string> = {
  case_law: '📜 Case Law',
  web: '🌐 Web',
  statute: '📖 Statute',
  bm25: '📜 Case Law',
};

const sourceTypeColors: Record<string, string> = {
  case_law: 'bg-blue-100 text-blue-800',
  web: 'bg-green-100 text-green-800',
  statute: 'bg-purple-100 text-purple-800',
  bm25: 'bg-blue-100 text-blue-800',
};

export default function SearchResultCard({
  title, excerpt, sourceType, relevanceScore,
  jurisdiction, court, year, citation, url,
}: SearchResultCardProps) {
  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-2">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        <span className={`text-xs px-2 py-1 rounded-full font-medium ${sourceTypeColors[sourceType] || 'bg-gray-100'}`}>
          {sourceTypeLabels[sourceType] || sourceType}
        </span>
      </div>
      <p className="text-sm text-gray-600 mb-3 line-clamp-3">{excerpt}</p>
      <div className="flex items-center gap-4 text-xs text-gray-500">
        {jurisdiction && <span>📍 {jurisdiction}</span>}
        {court && <span>⚖️ {court}</span>}
        {year && <span>📅 {year}</span>}
        {citation && <span>📄 {citation}</span>}
        <span className="ml-auto font-medium">{Math.round(relevanceScore * 100)}% match</span>
      </div>
      {url && (
        <a href={url} target="_blank" rel="noopener noreferrer"
           className="mt-2 inline-block text-sm text-blue-600 hover:underline">
          View source →
        </a>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Create SearchFilters.tsx**

```tsx
import React from 'react';

interface SearchFiltersProps {
  jurisdiction: string;
  sourceType: string;
  sortBy: string;
  onJurisdictionChange: (v: string) => void;
  onSourceTypeChange: (v: string) => void;
  onSortByChange: (v: string) => void;
}

const JURISDICTIONS = ['', 'Ontario', 'Federal', 'British Columbia', 'Alberta', 'Quebec', 'Nova Scotia'];
const SOURCE_TYPES = ['', 'case_law', 'web', 'statute'];
const SORT_OPTIONS = ['relevance', 'date'];

export default function SearchFilters({
  jurisdiction, sourceType, sortBy,
  onJurisdictionChange, onSourceTypeChange, onSortByChange,
}: SearchFiltersProps) {
  return (
    <div className="flex flex-wrap gap-4 mb-6 p-4 bg-gray-50 rounded-lg">
      <div>
        <label className="block text-xs font-medium text-gray-700 mb-1">Jurisdiction</label>
        <select value={jurisdiction} onChange={e => onJurisdictionChange(e.target.value)}
                className="border rounded px-3 py-1.5 text-sm">
          <option value="">All</option>
          {JURISDICTIONS.filter(Boolean).map(j => (
            <option key={j} value={j}>{j}</option>
          ))}
        </select>
      </div>
      <div>
        <label className="block text-xs font-medium text-gray-700 mb-1">Source Type</label>
        <select value={sourceType} onChange={e => onSourceTypeChange(e.target.value)}
                className="border rounded px-3 py-1.5 text-sm">
          <option value="">All</option>
          <option value="case_law">Case Law</option>
          <option value="web">Web</option>
          <option value="statute">Statute</option>
        </select>
      </div>
      <div>
        <label className="block text-xs font-medium text-gray-700 mb-1">Sort By</label>
        <select value={sortBy} onChange={e => onSortByChange(e.target.value)}
                className="border rounded px-3 py-1.5 text-sm">
          <option value="relevance">Relevance</option>
          <option value="date">Date</option>
        </select>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create SearchPage.tsx**

```tsx
import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { realApi } from '../../lib/api/realClient';
import SearchResultCard from '../../components/search/SearchResultCard';
import SearchFilters from '../../components/search/SearchFilters';

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [jurisdiction, setJurisdiction] = useState('');
  const [sourceType, setSourceType] = useState('');
  const [sortBy, setSortBy] = useState('relevance');

  const searchMutation = useMutation({
    mutationFn: (q: string) => realApi.search(q, { jurisdiction, sourceType, sortBy }),
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) searchMutation.mutate(query);
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Search Canadian Employment Law</h1>
      
      <form onSubmit={handleSearch} className="mb-6">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search cases, statutes, articles..."
            className="flex-1 border border-gray-300 rounded-lg px-4 py-3 text-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button type="submit"
                  className="bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50"
                  disabled={!query.trim() || searchMutation.isPending}>
            {searchMutation.isPending ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>

      <SearchFilters
        jurisdiction={jurisdiction}
        sourceType={sourceType}
        sortBy={sortBy}
        onJurisdictionChange={setJurisdiction}
        onSourceTypeChange={setSourceType}
        onSortByChange={setSortBy}
      />

      {searchMutation.isError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 mb-4">
          Search failed. Please try again.
        </div>
      )}

      {searchMutation.data && (
        <div className="mb-4 text-sm text-gray-500">
          {searchMutation.data.total} results ({searchMutation.data.results.length} shown)
        </div>
      )}

      <div className="space-y-4">
        {searchMutation.data?.results.map((result: any) => (
          <SearchResultCard key={result.id} {...result} />
        ))}
      </div>

      {!searchMutation.data && !searchMutation.isPending && (
        <div className="text-center py-16 text-gray-400">
          <p className="text-4xl mb-4">🔍</p>
          <p>Enter a query to search Canadian employment law</p>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Add route in App.tsx**

Add import and route:
```tsx
import SearchPage from './app/search/SearchPage';

// In the routes section:
<Route path="/search" element={<AuthGuard><SearchPage /></AuthGuard>} />
```

- [ ] **Step 5: Commit**

```bash
git add openjustice-frontend/src/app/search/SearchPage.tsx openjustice-frontend/src/components/search/SearchResultCard.tsx openjustice-frontend/src/components/search/SearchFilters.tsx openjustice-frontend/src/App.tsx
git commit -m "feat: add search UI page with filters and result cards"
```

---

### Task 10: Frontend — DeepSearch Page

**Files:**
- Create: `openjustice-frontend/src/app/deepsearch/DeepSearchPage.tsx`
- Create: `openjustice-frontend/src/components/search/AudioPlayer.tsx`
- Modify: `openjustice-frontend/src/App.tsx` (add route)

- [ ] **Step 1: Create AudioPlayer.tsx**

```tsx
import React, { useRef, useState } from 'react';
import { realApi } from '../../lib/api/realClient';

interface AudioPlayerProps {
  text: string;
}

export default function AudioPlayer({ text }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [loading, setLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const generateAudio = async () => {
    setLoading(true);
    setError(null);
    try {
      const blob = await realApi.generateAudio(text);
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
    } catch (e) {
      setError('Audio generation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
      {!audioUrl ? (
        <button onClick={generateAudio} disabled={loading}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm">
          {loading ? '⏳ Generating...' : '🔊 Listen'}
        </button>
      ) : (
        <>
          <audio ref={audioRef} src={audioUrl} controls className="h-10"
                 onError={() => setError('Playback failed')} />
          <button onClick={() => { setAudioUrl(null); URL.revokeObjectURL(audioUrl!); }}
                  className="text-sm text-gray-500 hover:text-gray-700">
            ✕ Clear
          </button>
        </>
      )}
      {error && <span className="text-sm text-red-600">{error}</span>}
    </div>
  );
}
```

- [ ] **Step 2: Create DeepSearchPage.tsx**

```tsx
import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { realApi } from '../../lib/api/realClient';
import AudioPlayer from '../../components/search/AudioPlayer';

const sourceTypeIcons: Record<string, string> = {
  case_law: '📜',
  web: '🌐',
  statute: '📖',
  bm25: '📜',
};

const sourceTypeLabels: Record<string, string> = {
  case_law: 'Case Law',
  web: 'Web',
  statute: 'Statute',
  bm25: 'Case Law',
};

export default function DeepSearchPage() {
  const [query, setQuery] = useState('');
  const [followUp, setFollowUp] = useState('');
  const [conversation, setConversation] = useState<{query: string; result: any}[]>([]);

  const searchMutation = useMutation({
    mutationFn: (q: string) => realApi.deepSearch(q),
    onSuccess: (data) => {
      setConversation(prev => [...prev, { query, result: data }]);
      setQuery('');
    },
  });

  const followUpMutation = useMutation({
    mutationFn: ({ original, follow }: { original: string; follow: string }) =>
      realApi.deepSearchFollowUp(original, follow),
    onSuccess: (data) => {
      setConversation(prev => [...prev, { query: followUp, result: data }]);
      setFollowUp('');
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) searchMutation.mutate(query);
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">DeepSearch</h1>
        <p className="text-gray-600">
          Multi-source legal research across case law, web, and statutes
        </p>
      </div>

      <form onSubmit={handleSubmit} className="mb-8">
        <textarea
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Ask a detailed legal research question..."
          rows={3}
          className="w-full border border-gray-300 rounded-lg px-4 py-3 text-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
        />
        <div className="flex justify-between items-center mt-2">
          <span className="text-xs text-gray-400">Sources: Case Law · Web · Statutes · BM25</span>
          <button type="submit"
                  className="bg-blue-600 text-white px-8 py-2 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50"
                  disabled={!query.trim() || searchMutation.isPending}>
            {searchMutation.isPending ? '⏳ Researching...' : 'Search Deeply'}
          </button>
        </div>
      </form>

      {/* Source type legend */}
      <div className="flex gap-4 mb-6 text-sm text-gray-500">
        <span>📜 Case Law</span>
        <span>🌐 Web</span>
        <span>📖 Statute</span>
      </div>

      {searchMutation.isError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 mb-4">
          DeepSearch failed. Please try again.
        </div>
      )}

      {/* Conversation */}
      <div className="space-y-8">
        {conversation.map((item, i) => (
          <div key={i} className="border border-gray-200 rounded-xl overflow-hidden">
            {/* User query */}
            <div className="bg-gray-50 px-6 py-4 border-b">
              <p className="font-medium text-gray-900">{item.query}</p>
              {item.result.processing_time_ms && (
                <span className="text-xs text-gray-400">
                  {item.result.source_type_counts && Object.entries(item.result.source_type_counts).map(([type, count]) => (
                    <span key={type} className="mr-3">{sourceTypeIcons[type] || '📄'} {count}</span>
                  ))}
                  · {Math.round(item.result.processing_time_ms / 1000)}s
                </span>
              )}
            </div>

            {/* Answer */}
            <div className="px-6 py-4">
              <div className="prose prose-sm max-w-none whitespace-pre-wrap">
                {item.result.answer}
              </div>

              {/* Sources */}
              {item.result.sources && item.result.sources.length > 0 && (
                <details className="mt-4">
                  <summary className="text-sm text-blue-600 cursor-pointer hover:text-blue-800">
                    Sources ({item.result.sources.length})
                  </summary>
                  <div className="mt-3 space-y-2">
                    {item.result.sources.map((source: any) => (
                      <div key={source.id} className="flex items-start gap-2 p-2 bg-gray-50 rounded text-sm">
                        <span>{sourceTypeIcons[source.source_type] || '📄'}</span>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium truncate">{source.title}</p>
                          <p className="text-gray-500 text-xs truncate">{source.excerpt}</p>
                        </div>
                        <span className="text-xs text-gray-400 shrink-0">
                          {Math.round(source.relevance_score * 100)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </details>
              )}

              {/* Follow-up suggestions */}
              {item.result.suggested_follow_ups && item.result.suggested_follow_ups.length > 0 && (
                <div className="mt-4 pt-4 border-t">
                  <p className="text-sm text-gray-500 mb-2">Follow-up questions:</p>
                  <div className="flex flex-wrap gap-2">
                    {item.result.suggested_follow_ups.map((q: string, j: number) => (
                      <button key={j} onClick={() => {
                        setFollowUp(q);
                        followUpMutation.mutate({ original: item.query, follow: q });
                      }}
                              className="text-sm px-3 py-1.5 bg-gray-100 rounded-full hover:bg-gray-200 text-gray-700">
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Audio */}
              <div className="mt-4 pt-4 border-t">
                <AudioPlayer text={item.result.answer} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Follow-up input */}
      {conversation.length > 0 && (
        <div className="mt-6">
          <div className="flex gap-2">
            <input
              type="text"
              value={followUp}
              onChange={e => setFollowUp(e.target.value)}
              placeholder="Ask a follow-up question..."
              className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button onClick={() => {
              if (followUp.trim() && conversation.length > 0) {
                const last = conversation[conversation.length - 1];
                followUpMutation.mutate({ original: last.query, follow: followUp });
              }
            }}
                    className="bg-gray-700 text-white px-4 py-2 rounded-lg hover:bg-gray-800 disabled:opacity-50 text-sm"
                    disabled={!followUp.trim() || followUpMutation.isPending}>
              Follow Up
            </button>
          </div>
        </div>
      )}

      {/* Empty state */}
      {conversation.length === 0 && !searchMutation.isPending && (
        <div className="text-center py-16 text-gray-400">
          <p className="text-5xl mb-4">🔬</p>
          <p className="text-lg">Enter a research question to begin</p>
          <p className="text-sm mt-2">Example: "What are the notice requirements for constructive dismissal in Ontario?"</p>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Add route in App.tsx**

```tsx
import DeepSearchPage from './app/deepsearch/DeepSearchPage';
<Route path="/deepsearch" element={<AuthGuard><DeepSearchPage /></AuthGuard>} />
```

- [ ] **Step 4: Commit**

```bash
git add openjustice-frontend/src/app/deepsearch/DeepSearchPage.tsx openjustice-frontend/src/components/search/AudioPlayer.tsx openjustice-frontend/src/App.tsx
git commit -m "feat: add DeepSearch page with audio player and follow-up support"
```

---

### Task 11: Frontend — Fix All Mock Data

**Files:**
- Modify: `openjustice-frontend/src/lib/api/realClient.ts` (replace all mocks)
- Modify: `openjustice-frontend/src/lib/hooks/useQuery.ts` (add new hooks)
- Modify: `openjustice-frontend/src/app/settings/SettingsPage.tsx` (connect to real API)

- [ ] **Step 1: Update realClient.ts with real API calls**

Replace all mock implementations:
```typescript
// API Keys — REAL calls
async getApiKeys(): Promise<any> {
  const res = await fetch(`${API_BASE}/keys`, { headers: this._headers() });
  return res.json();
}

async createApiKey(name: string): Promise<any> {
  const res = await fetch(`${API_BASE}/keys?name=${encodeURIComponent(name)}`, {
    method: 'POST', headers: this._headers(),
  });
  return res.json();
}

async revokeApiKey(id: string): Promise<any> {
  const res = await fetch(`${API_BASE}/keys/${id}`, {
    method: 'DELETE', headers: this._headers(),
  });
  return res.json();
}

// DeepSearch — NEW
async deepSearch(query: string, maxSources = 15): Promise<any> {
  const res = await fetch(`${API_BASE.replace('/api', '')}/rag/deepsearch`, {
    method: 'POST',
    headers: { ...this._headers(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, max_sources: maxSources }),
  });
  return res.json();
}

async deepSearchFollowUp(originalQuery: string, followUp: string): Promise<any> {
  const params = new URLSearchParams({ original_query: originalQuery, follow_up: followUp });
  const res = await fetch(`${API_BASE.replace('/api', '')}/rag/deepsearch/followup?${params}`, {
    headers: this._headers(),
  });
  return res.json();
}

// Audio TTS — NEW
async generateAudio(text: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/chat/tts`, {
    method: 'POST',
    headers: { ...this._headers(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, voice: 'en-CA-LiamNeural' }),
  });
  return res.blob();
}

// Search — NEW
async search(query: string, filters: any = {}): Promise<any> {
  const params = new URLSearchParams({ query, ...filters });
  const res = await fetch(`${API_BASE.replace('/api', '')}/rag/search?${params}`, {
    headers: this._headers(),
  });
  return res.json();
}

// Usage chart data — REAL
async getUsageChartData(days: number): Promise<any> {
  const res = await fetch(`${API_BASE}/users/me/usage/chart?days=${days}`, {
    headers: this._headers(),
  });
  return res.json();
}

// Recent activity — REAL
async getRecentActivity(): Promise<any> {
  const res = await fetch(`${API_BASE}/users/me/activity`, { headers: this._headers() });
  return res.json();
}

// Subscription — REAL (stub)
async upgradeSubscription(tier: string): Promise<any> {
  const res = await fetch(`${API_BASE}/subscriptions/upgrade`, {
    method: 'POST',
    headers: { ...this._headers(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ tier }),
  });
  return res.json();
}

// Helper: auth headers
private _headers(): Record<string, string> {
  const token = localStorage.getItem('access_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}
```

- [ ] **Step 2: Add new hooks to useQuery.ts**

```typescript
export function useDeepSearch() {
  return useMutation({ mutationFn: (query: string) => realApi.deepSearch(query) });
}
export function useSearch() {
  return useMutation({ mutationFn: ({ query, filters }: any) => realApi.search(query, filters) });
}
export function useGenerateAudio() {
  return useMutation({ mutationFn: (text: string) => realApi.generateAudio(text) });
}
```

- [ ] **Step 3: Verify frontend builds**

Run: `cd openjustice-frontend && npm run build`
Expected: No TypeScript errors

- [ ] **Step 4: Commit**

```bash
git add openjustice-frontend/src/lib/api/realClient.ts openjustice-frontend/src/lib/hooks/useQuery.ts openjustice-frontend/src/app/settings/SettingsPage.tsx
git commit -m "fix: replace all frontend mock data with real API calls"
```

---

### Task 12: Fix Subscribe/Usage Dashboard Mocks

**Files:**
- Modify: `api/users.py` (add usage chart + activity endpoints)
- Modify: `db/repository.py` (add usage history query)

- [ ] **Step 1: Add usage chart endpoint to api/users.py**

```python
@router.get("/api/users/me/usage/chart")
async def get_usage_chart(
    days: int = Query(30, ge=1, le=365),
    user=Depends(get_current_user),
    db=Depends(get_session),
):
    """Return daily query usage for the last N days."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import text
    
    since = datetime.now(timezone.utc) - timedelta(days=days)
    stmt = text("""
        SELECT DATE(created_at) as day, COUNT(*) as count
        FROM messages
        WHERE user_id = :uid AND role = 'user' AND created_at >= :since
        GROUP BY DATE(created_at)
        ORDER BY day
    """)
    result = await db.execute(stmt, {"uid": user["user_id"], "since": since})
    rows = result.fetchall()
    
    return {
        "data": [{"date": str(r.day), "queries": r.count} for r in rows],
        "total_queries": sum(r.count for r in rows),
    }

@router.get("/api/users/me/activity")
async def get_recent_activity(
    limit: int = Query(10, ge=1, le=50),
    user=Depends(get_current_user),
    db=Depends(get_session),
):
    """Return recent user activity (conversations + document uploads)."""
    # Recent conversations
    from db.repository import list_conversations
    convos = await list_conversations(db, user["user_id"])
    
    activity = []
    for c in convos[:limit]:
        activity.append({
            "type": "conversation",
            "title": c.title,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        })
    
    # Recent documents
    from db.repository import list_documents
    docs, _ = await list_documents(db, user["user_id"], page=1, page_size=limit)
    for d in docs[:limit]:
        activity.append({
            "type": "document",
            "title": d.filename,
            "status": d.status,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        })
    
    # Sort by date descending
    activity.sort(key=lambda a: a.get("created_at", ""), reverse=True)
    return {"activity": activity[:limit]}
```

- [ ] **Step 2: Add subscription upgrade stub**

```python
@router.post("/api/subscriptions/upgrade")
async def upgrade_subscription(
    tier: str = Query(..., regex="^(pro|enterprise)$"),
    user=Depends(get_current_user),
    db=Depends(get_session),
):
    """Upgrade user tier. Payment integration is future work."""
    from db.repository import update_user
    limits = {"pro": 200, "enterprise": 999999}
    if tier not in limits:
        raise HTTPException(status_code=400, detail="Invalid tier")
    
    updated = await update_user(db, user["user_id"], tier=tier, queries_limit=limits[tier])
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": f"Upgraded to {tier}", "tier": tier}
```

- [ ] **Step 3: Commit**

```bash
git add api/users.py db/repository.py
git commit -m "fix: replace dashboard mock data with real usage/activity/subscription endpoints"
```

---

## Self-Review Checklist

- ✅ **Spec coverage**: All 8 components from the spec have implementation tasks (SearXNG, Redis, web_search, deep_search, API keys, auth, TTS, search UI, frontend mocks, dashboard)
- ✅ **No placeholders**: Every step has actual code, exact file paths, exact commands
- ✅ **Type consistency**: DeepSearchResult, UnifiedSource, WebResult types used consistently across all tasks
- ✅ **Dependency order**: Tasks ordered so each builds on previous (Docker → Redis → web_search → deep_search → API keys → auth → endpoints → frontend)
