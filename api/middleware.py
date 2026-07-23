# OpenJustice.ai - Rate Limiting Middleware
"""
In-memory rate limiting middleware based on user subscription tiers.

Provides:
- Tier-based rate limits (Free: 10/min, 20/day; Pro: 60/min, 200/day; Enterprise: 300/min)
- Per-user sliding window counters (minute and daily)
- 429 response with Retry-After header when exceeded
- Fallback for unauthenticated requests (Free tier limits)
- In-memory store (Redis-ready for production)
"""

import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config import (
    RATE_LIMIT_FREE_RPM,
    RATE_LIMIT_FREE_RPD,
    RATE_LIMIT_PRO_RPM,
    RATE_LIMIT_PRO_RPD,
    RATE_LIMIT_ENTERPRISE_RPM,
    RATE_LIMIT_ENTERPRISE_RPD,
    DEV_MODE,
)

# Setup logging
logger = logging.getLogger(__name__)


# =====================
# Rate Limit Configuration
# =====================

# Tier -> (requests_per_minute, requests_per_day)
TIER_LIMITS: Dict[str, Tuple[int, int]] = {
    "free": (RATE_LIMIT_FREE_RPM, RATE_LIMIT_FREE_RPD),
    "pro": (RATE_LIMIT_PRO_RPM, RATE_LIMIT_PRO_RPD),
    "enterprise": (RATE_LIMIT_ENTERPRISE_RPM, RATE_LIMIT_ENTERPRISE_RPD),
}

DEFAULT_TIER = "free"

# Window sizes in seconds
MINUTE_WINDOW = 60
DAY_WINDOW = 86400  # 24 hours


# =====================
# In-Memory Rate Store
# =====================

class RateStore:
    """
    Thread-safe in-memory rate limit store.
    
    Uses sliding window counters per user:
        - minute_window: {timestamp: count} entries for the last 60 seconds
        - day_window: {timestamp: count} entries for the last 24 hours
    
    In production, replace with Redis for:
        - Distributed rate limiting across multiple workers
        - Persistent counters across restarts
        - Atomic increment operations
    """
    
    def __init__(self):
        # Structure: {user_id: {"minute": [(ts, count), ...], "day": [(ts, count), ...]}}
        self._store: Dict[str, Dict[str, list]] = defaultdict(
            lambda: {"minute": [], "day": []}
        )
    
    def _purge_old(self, user_id: str, window: str, window_size: int) -> None:
        """Remove entries outside the window for a given user and window type."""
        now = time.time()
        cutoff = now - window_size
        self._store[user_id][window] = [
            (ts, count) for ts, count in self._store[user_id][window]
            if ts > cutoff
        ]
    
    def _window_count(self, user_id: str, window: str) -> int:
        """Get total count in the current window for a user."""
        return sum(count for _, count in self._store[user_id][window])
    
    def record_request(self, user_id: str) -> None:
        """Record a request for the given user."""
        now = time.time()
        
        # Prune old entries first
        self._purge_old(user_id, "minute", MINUTE_WINDOW)
        self._purge_old(user_id, "day", DAY_WINDOW)
        
        # Add current request
        minute_entries = self._store[user_id]["minute"]
        day_entries = self._store[user_id]["day"]
        
        # Coalesce: if last entry is within 1 second, increment it
        if minute_entries and (now - minute_entries[-1][0]) < 1:
            ts, count = minute_entries[-1]
            minute_entries[-1] = (ts, count + 1)
        else:
            minute_entries.append((now, 1))
        
        if day_entries and (now - day_entries[-1][0]) < 1:
            ts, count = day_entries[-1]
            day_entries[-1] = (ts, count + 1)
        else:
            day_entries.append((now, 1))
    
    def check_limits(self, user_id: str, tier: str) -> Tuple[bool, Optional[int], str]:
        """
        Check if a user has exceeded their rate limits.
        
        Args:
            user_id: Unique user identifier
            tier: Subscription tier (free/pro/enterprise)
            
        Returns:
            Tuple of:
                - allowed: True if request is within limits
                - retry_after: Seconds to wait before retrying (None if allowed)
                - reason: Description of which limit was hit ("", "minute", or "day")
        """
        limits = TIER_LIMITS.get(tier, TIER_LIMITS[DEFAULT_TIER])
        rpm_limit, rpd_limit = limits
        
        self._purge_old(user_id, "minute", MINUTE_WINDOW)
        self._purge_old(user_id, "day", DAY_WINDOW)
        
        minute_count = self._window_count(user_id, "minute")
        day_count = self._window_count(user_id, "day")
        
        # Check minute limit first (more restrictive)
        if minute_count >= rpm_limit:
            # Calculate retry-after: time until oldest entry falls out
            oldest = self._store[user_id]["minute"][0][0] if self._store[user_id]["minute"] else time.time()
            retry_after = max(1, int(MINUTE_WINDOW - (time.time() - oldest)))
            return False, retry_after, "minute"
        
        # Check daily limit
        if day_count >= rpd_limit:
            oldest = self._store[user_id]["day"][0][0] if self._store[user_id]["day"] else time.time()
            retry_after = max(1, int(DAY_WINDOW - (time.time() - oldest)))
            return False, retry_after, "day"
        
        return True, None, ""
    
    def get_stats(self, user_id: str) -> Dict[str, int]:
        """Get current usage stats for a user (for debugging/headers)."""
        self._purge_old(user_id, "minute", MINUTE_WINDOW)
        self._purge_old(user_id, "day", DAY_WINDOW)
        return {
            "minute_count": self._window_count(user_id, "minute"),
            "day_count": self._window_count(user_id, "day"),
        }


# Singleton rate store
_rate_store = RateStore()


# =====================
# Rate Limit Header Helpers
# =====================

def _get_user_id(request: Request) -> Tuple[str, str]:
    """
    Extract user ID and tier from the request.
    
    Tries JWT token first, then falls back to client IP.
    
    Returns:
        Tuple of (user_id, tier)
    """
    # Try to extract from auth header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            from jose import jwt
            from config import JWT_SECRET_KEY, JWT_ALGORITHM
            
            token = auth_header[7:]
            payload = jwt.decode(
                token,
                JWT_SECRET_KEY,
                algorithms=[JWT_ALGORITHM],
                options={"verify_exp": False},  # Allow rate limiting even for expired tokens
            )
            user_id = payload.get("sub", "anonymous")
            tier = payload.get("tier", DEFAULT_TIER)
            return user_id, tier
        except Exception:
            pass
    
    # Fallback: use client IP (respect X-Forwarded-For for proxied requests)
    client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not client_ip:
        client_ip = request.client.host if request.client else "unknown"
    return f"ip_{client_ip}", DEFAULT_TIER


def _get_user_tier_from_token(request: Request) -> str:
    """Extract just the tier from a request for header information."""
    _, tier = _get_user_id(request)
    return tier


# =====================
# Rate Limit Headers
# =====================

def add_rate_limit_headers(response: Response, user_id: str, tier: str) -> None:
    """Add X-RateLimit-* headers to the response."""
    limits = TIER_LIMITS.get(tier, TIER_LIMITS[DEFAULT_TIER])
    rpm_limit, rpd_limit = limits
    stats = _rate_store.get_stats(user_id)
    
    response.headers["X-RateLimit-Limit-Minute"] = str(rpm_limit)
    response.headers["X-RateLimit-Limit-Day"] = str(rpd_limit)
    response.headers["X-RateLimit-Remaining-Minute"] = str(max(0, rpm_limit - stats["minute_count"]))
    response.headers["X-RateLimit-Remaining-Day"] = str(max(0, rpd_limit - stats["day_count"]))


# =====================
# FastAPI Middleware
# =====================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for tier-based rate limiting.
    
    Applies to /api/, /rag/, /classify/, /evaluate/ routes. Returns 429 with Retry-After header
    when limits are exceeded.
    
    Add to the app:
        app.add_middleware(RateLimitMiddleware)
    """
    
    # Paths exempt from rate limiting
    EXEMPT_PATHS = {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/",
    }
    
    # Path prefixes that should be rate limited
    RATE_LIMITED_PREFIXES = ("/api/", "/rag/", "/classify/", "/evaluate/")

    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process the request through rate limiting."""
        
        # Skip rate limiting entirely in development mode
        if DEV_MODE:
            return await call_next(request)
        
        # Skip rate limiting for exempt paths and non-rate-limited paths
        path = request.url.path
        if path in self.EXEMPT_PATHS or not any(path.startswith(p) for p in self.RATE_LIMITED_PREFIXES):
            return await call_next(request)
        
        # Get user info
        user_id, tier = _get_user_id(request)
        
        # Check limits
        allowed, retry_after, reason = _rate_store.check_limits(user_id, tier)
        
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {user_id} (tier={tier}, "
                f"limit_type={reason}, retry_after={retry_after}s)"
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests",
                    "retry_after_seconds": retry_after,
                    "limit_type": f"{reason}_limit_exceeded",
                    "tier": tier,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit-Exceeded": reason,
                    "X-RateLimit-User-Tier": tier,
                },
            )
        
        # Record the request
        _rate_store.record_request(user_id)
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers
        add_rate_limit_headers(response, user_id, tier)
        
        return response


def get_rate_store() -> RateStore:
    """Get the singleton rate store (for testing/inspection)."""
    return _rate_store
