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