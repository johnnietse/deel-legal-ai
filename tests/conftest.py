"""Pytest configuration for the OpenJustice.ai backend test suite.

The rate-limiting middleware uses a module-level singleton in-memory store.
Because the store is shared across the entire test session and the free tier
allows only 10 requests/minute, running the full suite exhausts the budget and
causes unrelated tests to receive 429 responses. Resetting the store around
each test keeps every test isolated with a fresh budget.
"""

import pytest

from api.middleware import _rate_store


@pytest.fixture(autouse=True)
def _reset_rate_limit_store():
    """Clear the in-memory rate limit store before and after each test."""
    _rate_store._store.clear()
    yield
    _rate_store._store.clear()
