"""
Gemini API Key Manager — auto-rotates keys on rate limits.

Loads keys from GEMINI_API_KEY (primary) and GEMINI_BACKUP_KEYS
(comma-separated backups). Keys are assumed to be from independent
Google Cloud projects, each with its own quota.

When a 429 is detected on a key, we rotate to the next key.
Instead of a global cooldown, each key tracks its own last-used
timestamp. By the time we cycle through all keys, the first ones
have had time to cool down (no global pause needed).

Usage:
    from rag_pipeline.gemini_key_manager import key_manager
    
    api_key = key_manager.get_key()
    # ... call Gemini, if 429:
    key_manager.report_rate_limit()
    api_key = key_manager.get_key()
"""

import os
import time as _time
import threading
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class GeminiKeyManager:
    """Thread-safe Gemini API key rotation manager for independent keys."""

    def __init__(self, primary_env="GEMINI_API_KEY", backup_env="GEMINI_BACKUP_KEYS"):
        self._lock = threading.Lock()
        self._keys: List[str] = []
        self._current_index = 0
        self._last_used: List[float] = []  # timestamps per key
        self._rate_limits_hit: List[int] = []
        self._primary_env = primary_env
        self._backup_env = backup_env
        self._load_keys()

    def _load_keys(self):
        """Load keys from environment variables."""
        primary = os.environ.get(self._primary_env, "")
        if primary:
            self._keys.append(primary)

        backup_str = os.environ.get(self._backup_env, "")
        if backup_str:
            backup_keys = [k.strip() for k in backup_str.split(",") if k.strip()]
            self._keys.extend(backup_keys)

        n = len(self._keys)
        self._last_used = [0.0] * n
        self._rate_limits_hit = [0] * n

        logger.info(f"Loaded {len(self._keys)} Gemini API keys")
        if self._keys:
            masked = [f"{k[:12]}...{k[-4:]}" for k in self._keys]
            logger.info(f"Keys: {masked}")

    def get_key(self) -> str:
        """Get the current active API key (marks it as used)."""
        with self._lock:
            if not self._keys:
                raise ValueError("No Gemini API keys configured")
            self._last_used[self._current_index] = _time.time()
            return self._keys[self._current_index]

    def get_key_masked(self) -> str:
        """Get masked key for logging."""
        key = self.get_key()
        return f"{key[:12]}...{key[-4:]}"

    def report_rate_limit(self) -> str:
        """
        Report a 429 rate limit and rotate to the next key.
        No global cooldown — each key is assumed independent,
        so by the time we cycle back the first keys have reset.
        """
        with self._lock:
            if not self._keys:
                raise ValueError("No Gemini API keys configured")

            old_idx = self._current_index
            self._rate_limits_hit[old_idx] += 1
            old_masked = f"{self._keys[old_idx][:12]}...{self._keys[old_idx][-4:]}"

            # Rotate to next key
            self._current_index = (self._current_index + 1) % len(self._keys)

            new_masked = f"{self._keys[self._current_index][:12]}...{self._keys[self._current_index][-4:]}"

            logger.warning(
                f"Rate limit on key {old_idx + 1}/{len(self._keys)} ({old_masked}). "
                f"Rotated to key {self._current_index + 1}/{len(self._keys)} ({new_masked}). "
                f"Key {old_idx + 1} hit {self._rate_limits_hit[old_idx]}x rate limited."
            )

            return self._keys[self._current_index]

    def check_cooldown(self):
        """No-op: keys are independent, rotation handles cooldown naturally."""
        pass

    def report_success(self):
        """Report a successful API call."""
        with self._lock:
            if self._keys and self._current_index < len(self._rate_limits_hit):
                if self._rate_limits_hit[self._current_index] > 0:
                    self._rate_limits_hit[self._current_index] -= 1

    @property
    def key_count(self) -> int:
        return len(self._keys)

    @property
    def current_index(self) -> int:
        return self._current_index + 1


# Singleton instance (used by the embedder / batch pipeline — the 12-key pool)
key_manager = GeminiKeyManager()

# Dedicated key manager for the user-facing API (search + deepsearch), isolated
# from the embedder's key pool so search is never starved by background ingestion.
search_key_manager = GeminiKeyManager(
    primary_env="SEARCH_GEMINI_API_KEY",
    backup_env="SEARCH_GEMINI_BACKUP_KEYS",
)
