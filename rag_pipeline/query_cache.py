# RAG Pipeline - Query & Embedding Cache
"""
Multi-layer caching for RAG pipeline,
inspired by ByteDance RAG Guideline §4.2.1, §6.4.2.

Cache layers:
  1. Embedding cache  — same query text → same vector (save API $)
  2. Retrieval cache  — same query → same search results (save DB calls)
  3. Response cache   — exact same query → cached LLM response (save LLM $)

ByteDance reports 30% reduction in generation requests via FAQ caching
in their e-commerce RAG system.

Uses TTL-based eviction (default 10 minutes) with configurable size.
Disk-backed serialisation for persistence across restarts.
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from functools import wraps

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache Entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cache entry with TTL."""
    key: str
    value: Any
    created_at: float
    ttl_seconds: float
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


# ---------------------------------------------------------------------------
# TTL Cache Implementation
# ---------------------------------------------------------------------------

class TTLCache:
    """
    Time-To-Live cache with LRU eviction and optional disk persistence.

    ByteDance §6.4.2:
      - High-frequency query results cached with 10-min TTL
      - Cache hit rate target ≥60%
    """

    def __init__(
        self,
        maxsize: int = 1000,
        ttl_seconds: float = 600,  # 10 minutes default
        persist_path: Optional[str] = None,
    ):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.persist_path = persist_path

        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # LRU tracking

        # Stats
        self._hits = 0
        self._misses = 0

        # Load from disk if path specified
        if persist_path and os.path.exists(persist_path):
            self._load_from_disk()

    # -- Core operations ---------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache, or None if missing/expired."""
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_expired:
            self._evict(key)
            self._misses += 1
            return None

        entry.hit_count += 1
        self._hits += 1
        self._touch(key)
        return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Store a value in cache."""
        if len(self._cache) >= self.maxsize:
            self._evict_lru()

        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl_seconds=ttl or self.ttl_seconds,
        )
        self._touch(key)

    def invalidate(self, key: str):
        """Remove a specific key."""
        self._evict(key)

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()

    # -- Stats -------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "maxsize": self.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
            "ttl_seconds": self.ttl_seconds,
        }

    # -- LRU internals -----------------------------------------------------

    def _touch(self, key: str):
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict(self, key: str):
        self._cache.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)

    def _evict_lru(self):
        """Evict the least recently used entry."""
        # First try to evict expired entries
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired:
            self._evict(k)
        if len(self._cache) < self.maxsize:
            return

        # Otherwise evict LRU
        if self._access_order:
            lru_key = self._access_order[0]
            self._evict(lru_key)

    # -- Disk persistence --------------------------------------------------

    def save_to_disk(self):
        """Persist cache to disk (JSON-serializable entries only)."""
        if not self.persist_path:
            return
        try:
            data = {}
            for key, entry in self._cache.items():
                if not entry.is_expired:
                    try:
                        # Only persist JSON-serializable values
                        json.dumps(entry.value)
                        data[key] = {
                            "value": entry.value,
                            "created_at": entry.created_at,
                            "ttl_seconds": entry.ttl_seconds,
                        }
                    except (TypeError, ValueError):
                        continue  # Skip non-serializable entries

            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump(data, f)
            logger.debug(f"Cache saved to {self.persist_path}: {len(data)} entries")
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")

    def _load_from_disk(self):
        """Load cache from disk."""
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            loaded = 0
            for key, entry_data in data.items():
                entry = CacheEntry(
                    key=key,
                    value=entry_data["value"],
                    created_at=entry_data["created_at"],
                    ttl_seconds=entry_data["ttl_seconds"],
                )
                if not entry.is_expired:
                    self._cache[key] = entry
                    self._access_order.append(key)
                    loaded += 1
            logger.info(f"Cache loaded from {self.persist_path}: {loaded} entries")
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")


# ---------------------------------------------------------------------------
# RAG Query Cache (Multi-Layer)
# ---------------------------------------------------------------------------

class RAGQueryCache:
    """
    Multi-layer cache for the RAG pipeline.

    Layers:
      1. embedding_cache — query text → embedding vector
      2. retrieval_cache — query text → search results
      3. response_cache — query text + params → full LLM response

    ByteDance §6.4.2: FAQ caching reduced generation requests by 30%.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        embedding_maxsize: int = 2000,
        embedding_ttl: float = 3600,       # 1 hour (embeddings are stable)
        retrieval_maxsize: int = 500,
        retrieval_ttl: float = 600,        # 10 minutes
        response_maxsize: int = 500,
        response_ttl: float = 300,         # 5 minutes
    ):
        cache_dir = cache_dir or str(Path(__file__).parent.parent / "data" / "cache")

        self.embedding_cache = TTLCache(
            maxsize=embedding_maxsize,
            ttl_seconds=embedding_ttl,
            persist_path=os.path.join(cache_dir, "embedding_cache.json"),
        )
        self.retrieval_cache = TTLCache(
            maxsize=retrieval_maxsize,
            ttl_seconds=retrieval_ttl,
            persist_path=os.path.join(cache_dir, "retrieval_cache.json"),
        )
        self.response_cache = TTLCache(
            maxsize=response_maxsize,
            ttl_seconds=response_ttl,
            persist_path=os.path.join(cache_dir, "response_cache.json"),
        )

    @staticmethod
    def _make_key(text: str, **kwargs) -> str:
        """Generate a deterministic cache key."""
        payload = text + json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    # -- Embedding layer ---------------------------------------------------

    def get_embedding(self, text: str) -> Optional[List[float]]:
        key = self._make_key(text, layer="embedding")
        return self.embedding_cache.get(key)

    def put_embedding(self, text: str, embedding: List[float]):
        key = self._make_key(text, layer="embedding")
        self.embedding_cache.put(key, embedding)

    # -- Retrieval layer ---------------------------------------------------

    def get_retrieval(self, query: str, **params) -> Optional[List[Dict]]:
        key = self._make_key(query, layer="retrieval", **params)
        return self.retrieval_cache.get(key)

    def put_retrieval(self, query: str, results: List[Dict], **params):
        key = self._make_key(query, layer="retrieval", **params)
        # Convert results to serializable format
        serializable = []
        for r in results:
            if hasattr(r, "__dict__"):
                serializable.append(r.__dict__)
            elif isinstance(r, dict):
                serializable.append(r)
        self.retrieval_cache.put(key, serializable)

    # -- Response layer ----------------------------------------------------

    def get_response(self, query: str, **params) -> Optional[Dict]:
        key = self._make_key(query, layer="response", **params)
        return self.response_cache.get(key)

    def put_response(self, query: str, response: Dict, **params):
        key = self._make_key(query, layer="response", **params)
        self.response_cache.put(key, response)

    # -- Stats & maintenance -----------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "embedding_cache": self.embedding_cache.stats(),
            "retrieval_cache": self.retrieval_cache.stats(),
            "response_cache": self.response_cache.stats(),
        }

    def save_all(self):
        """Persist all cache layers to disk."""
        self.embedding_cache.save_to_disk()
        self.retrieval_cache.save_to_disk()
        self.response_cache.save_to_disk()

    def clear_all(self):
        """Clear all cache layers."""
        self.embedding_cache.clear()
        self.retrieval_cache.clear()
        self.response_cache.clear()
