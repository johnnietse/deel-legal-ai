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
        encoded_query = urllib.parse.quote(query)
        params = urllib.parse.urlencode({
            "format": "json",
            "language": "en",
            "categories": "general",
            "pageno": 1,
        }, quote_via=urllib.parse.quote)
        return f"{self.base_url}/search?q={encoded_query}&{params}"

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