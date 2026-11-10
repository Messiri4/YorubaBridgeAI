"""Redis-backed translation cache with graceful degradation.

Design rules:
- A cache failure must NEVER take translation down: every Redis operation
  is wrapped and falls back to "cache miss" behaviour.
- Keys are content-addressed: SHA-256 of normalized text + language pair.
"""

import hashlib
import json
import logging
import unicodedata
from typing import Any, Protocol

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class RedisLike(Protocol):
    """Minimal typing surface of a redis client (sync)."""

    def get(self, key: str) -> Any: ...

    def set(self, key: str, value: str, ex: int = ...) -> Any: ...

    def ping(self) -> bool: ...


def normalize_for_cache(text: str) -> str:
    """Normalize text so trivial differences share one cache entry.

    NFC keeps composed/decomposed Yoruba diacritics identical
    (e.g. "ọ" as one codepoint vs "o" + combining dot).
    """
    return unicodedata.normalize("NFC", text).strip().lower()


def cache_key(text: str, source_language: str, target_language: str,
              decoding_policy: str | None = None) -> str:
    """Build the Redis key for a translation request.

    The active decoding policy is part of the key (when known): cached
    hypotheses from an experimental policy must never be served as though
    they were produced by the production policy (and vice versa). The
    default profile is treated as the empty suffix so existing keys stay
    valid and no migration is needed.
    """
    digest = hashlib.sha256(normalize_for_cache(text).encode("utf-8")).hexdigest()[:24]
    suffix = (
        f":{decoding_policy}"
        if decoding_policy and decoding_policy != "production_default"
        else ""
    )
    return f"translate:{source_language}:{target_language}:{digest}{suffix}"


class TranslationCache:
    """Read-through cache around TranslationService results.

    Attributes:
        client: a redis-like client, or None to disable caching entirely
            (unit tests, or no-Redis deployments).
    """

    def __init__(self, client: RedisLike | None) -> None:
        self._client = client
        self._settings = get_settings()
        self._enabled = False
        if client is not None:
            try:
                client.ping()
                self._enabled = True
            except Exception:  # noqa: BLE001 - any Redis error means "no cache"
                logger.warning("Redis unreachable; translation cache disabled")

    def get(self, text: str, source_language: str, target_language: str,
            decoding_policy: str | None = None) -> dict[str, Any] | None:
        """Return a cached translation payload, or None on miss/disabled/error."""
        if not self._enabled or self._client is None:
            return None
        try:
            raw = self._client.get(
                cache_key(text, source_language, target_language, decoding_policy))
            if raw is None:
                return None
            logger.info("Cache hit")
            data: dict[str, Any] = json.loads(raw)
            return data
        except Exception:  # noqa: BLE001 - cache errors must not break requests
            logger.warning("Cache read failed; treating as miss", exc_info=True)
            return None

    def set(
        self,
        text: str,
        source_language: str,
        target_language: str,
        payload: dict[str, Any],
        decoding_policy: str | None = None,
    ) -> None:
        """Store a translation payload; silently ignores Redis failures."""
        if not self._enabled or self._client is None:
            return
        try:
            self._client.set(
                cache_key(text, source_language, target_language, decoding_policy),
                json.dumps(payload, ensure_ascii=False),
                ex=self._settings.cache_ttl_seconds,
            )
        except Exception:  # noqa: BLE001
            logger.warning("Cache write failed", exc_info=True)
