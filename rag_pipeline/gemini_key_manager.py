"""
Gemini API Key Manager — auto-rotates keys on rate limits.

Loads keys from GEMINI_API_KEY (primary) and GEMINI_BACKUP_KEYS
(comma-separated backups). Keys are assumed to be from independent
Google Cloud projects, each with its own quota.

When a 429 is detected on a key, we rotate to the next key.
Instead of a global cooldown, each key tracks its own last-used
timestamp. By the time we cycle through all keys, the first ones
have had time to cool down (no global pause needed).

Key Features:
  - Jittered exponential backoff: random 0-25% jitter on delays
  - Circuit breaker: 5 consecutive failures = 60s cooldown per key
  - Token-awareness: longer prompts = longer backoff
  - Key isolation: search key manager uses its own dedicated pool

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
import random
from typing import List, Optional

logger = logging.getLogger(__name__)


class GeminiKeyManager:
    """Thread-safe Gemini API key rotation manager with circuit breaker and jitter."""

    # Circuit breaker thresholds
    CIRCUIT_BREAKER_THRESHOLD = 5       # consecutive failures before cooldown
    CIRCUIT_BREAKER_COOLDOWN = 60       # seconds to cool down a tripped key
    MAX_JITTER_FRACTION = 0.25          # random jitter up to 25% of base delay
    BASE_BACKOFF_DELAY = 2.0            # base seconds for exponential backoff
    MAX_BACKOFF_DELAY = 60.0            # cap on backoff delay
    TOKEN_BACKOFF_FACTOR = 0.001        # extra seconds per input token

    def __init__(self, primary_env="GEMINI_API_KEY", backup_env="GEMINI_BACKUP_KEYS"):
        self._lock = threading.Lock()
        self._keys: List[str] = []
        self._current_index = 0
        self._last_used: List[float] = []            # timestamps per key
        self._rate_limits_hit: List[int] = []         # total rate limits per key
        self._consecutive_failures: List[int] = []    # consecutive failures for circuit breaker
        self._cool_until: List[float] = []            # timestamp when key can be used again
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
        self._consecutive_failures = [0] * n
        self._cool_until = [0.0] * n

        logger.info(f"Loaded {len(self._keys)} Gemini API keys (from {self._primary_env})")

    def get_key(self, input_token_count: int = 0) -> str:
        """
        Get the current active API key (marks it as used).
        
        Args:
            input_token_count: Number of input tokens for token-aware backoff.
                               Longer prompts add proportionally more backoff.
        
        Returns:
            The API key string.
        
        Raises:
            ValueError: If no keys are configured.
        """
        with self._lock:
            if not self._keys:
                raise ValueError("No Gemini API keys configured")

            now = _time.time()
            
            # Check if current key is in cooldown; if so, rotate
            attempts = 0
            while self._cool_until[self._current_index] > now and attempts < len(self._keys):
                self._current_index = (self._current_index + 1) % len(self._keys)
                attempts += 1

            # If all keys are in cooldown, wait for the shortest cooldown
            if self._cool_until[self._current_index] > now:
                shortest_wait = min(max(0.0, c - now) for c in self._cool_until)
                logger.warning(
                    f"All {len(self._keys)} keys in cooldown. Waiting {shortest_wait:.1f}s."
                )
                # Sleep outside the lock to avoid blocking other threads
                _time.sleep(shortest_wait)

            self._last_used[self._current_index] = _time.time()
            return self._keys[self._current_index]

    def get_key_masked(self) -> str:
        """Get masked key for logging (only shows count, never key material)."""
        return f"key-{self._current_index + 1}-of-{len(self._keys)}"

    def _compute_backoff(self, old_idx: int, input_token_count: int = 0) -> float:
        """
        Compute jittered exponential backoff delay.
        
        Base: 2^consecutive_failures seconds (capped at MAX_BACKOFF_DELAY)
        Jitter: random 0-25% of base delay
        Token factor: extra delay proportional to input tokens
        """
        failures = self._consecutive_failures[old_idx]
        base_delay = min(
            self.MAX_BACKOFF_DELAY,
            self.BASE_BACKOFF_DELAY * (2 ** failures)
        )
        jitter = base_delay * self.MAX_JITTER_FRACTION * random.random()
        token_delay = input_token_count * self.TOKEN_BACKOFF_FACTOR
        total = base_delay + jitter + token_delay
        return min(total, self.MAX_BACKOFF_DELAY * 2)

    def report_rate_limit(self, input_token_count: int = 0) -> str:
        """
        Report a 429 rate limit, apply circuit breaker, and rotate to next key.
        
        Args:
            input_token_count: Input token count for token-aware backoff.
        
        Returns:
            The new active API key after rotation.
        """
        with self._lock:
            if not self._keys:
                raise ValueError("No Gemini API keys configured")

            old_idx = self._current_index
            self._rate_limits_hit[old_idx] += 1
            self._consecutive_failures[old_idx] += 1

            # Circuit breaker: if consecutive failures >= threshold, cool down
            if self._consecutive_failures[old_idx] >= self.CIRCUIT_BREAKER_THRESHOLD:
                backoff = self.CIRCUIT_BREAKER_COOLDOWN
                logger.warning(
                    f"Circuit breaker tripped for key {old_idx + 1}/{len(self._keys)} "
                    f"({self._consecutive_failures[old_idx]} consecutive failures). "
                    f"Cooling down for {backoff}s."
                )
                self._cool_until[old_idx] = _time.time() + backoff
                self._consecutive_failures[old_idx] = 0  # Reset counter after tripping
            else:
                # Apply jittered exponential backoff
                backoff = self._compute_backoff(old_idx, input_token_count)
                logger.warning(
                    f"Rate limit on key {old_idx + 1}/{len(self._keys)}. "
                    f"Applying {backoff:.1f}s backoff (failure #{self._consecutive_failures[old_idx]})."
                )
                self._cool_until[old_idx] = _time.time() + backoff

            # Rotate to next key that is not in cooldown
            next_idx = (self._current_index + 1) % len(self._keys)
            attempts = 0
            while self._cool_until[next_idx] > _time.time() and attempts < len(self._keys):
                next_idx = (next_idx + 1) % len(self._keys)
                attempts += 1
            self._current_index = next_idx

            logger.warning(
                f"Key {old_idx + 1}/{len(self._keys)} rate limited. "
                f"Rotated to key {self._current_index + 1}/{len(self._keys)}. "
                f"Key {old_idx + 1}: {self._rate_limits_hit[old_idx]} total rate limits."
            )

            return self._keys[self._current_index]

    def check_cooldown(self) -> float:
        """
        Check if the current key is in cooldown.
        
        Returns:
            Seconds remaining in cooldown (0 if not in cooldown).
        """
        with self._lock:
            remaining = self._cool_until[self._current_index] - _time.time()
            return max(0.0, remaining)

    def report_success(self):
        """Report a successful API call — resets consecutive failure counter."""
        with self._lock:
            if self._keys and self._current_index < len(self._consecutive_failures):
                self._consecutive_failures[self._current_index] = 0
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
