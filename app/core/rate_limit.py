"""Per-client rate limiting using Redis counters with in-memory fallback.

Uses a fixed-window counter: key `ratelimit:{ip}:{window_start}` incremented
with INCR and given a TTL. In multi-worker deployments the counters live in
Redis so the limit is shared; without Redis a process-local dict is used
(fine for development, not for production).
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Fixed-window rate limiter.

    Args:
        redis_getter: callable returning a redis-like client or None. Kept as
            a callable so the limiter can reconnect after a Redis restart.
    """

    def __init__(self, redis_getter: Callable[[], Any]) -> None:
        self._redis_getter = redis_getter
        self._local_counts: dict[str, int] = {}

    def check(self, identifier: str) -> None:
        """Raise HTTP 429 if `identifier` exceeded the configured rate."""
        settings = get_settings()
        window = settings.rate_limit_window_seconds
        bucket = int(time.time()) // window
        key = f"ratelimit:{identifier}:{bucket}"

        count = self._incr(key, window)
        if count > settings.rate_limit_max_requests:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please slow down.",
            )

    def _incr(self, key: str, window: int) -> int:
        """Increment the counter in Redis if possible, else in memory."""
        try:
            client = self._redis_getter()
            if client is not None:
                pipe = client.pipeline()
                pipe.incr(key)
                pipe.expire(key, window)
                return int(pipe.execute()[0])
        except Exception:  # noqa: BLE001 - limiter must not crash the app
            logger.warning("Redis rate-limit failed; using in-memory counter")
        self._local_counts[key] = self._local_counts.get(key, 0) + 1
        return self._local_counts[key]


def client_identifier(request: Request) -> str:
    """Identify the caller: X-Forwarded-For if behind a proxy, else client IP."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
