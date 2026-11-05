"""Unit tests for RateLimiter (in-memory mode and Redis mode)."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.core.rate_limit import RateLimiter


def test_blocks_after_limit_in_memory() -> None:
    limiter = RateLimiter(lambda: None)  # no redis -> local counters
    for _ in range(10):
        limiter.check("1.2.3.4")  # default max_requests = 10
    with pytest.raises(HTTPException) as exc:
        limiter.check("1.2.3.4")
    assert exc.value.status_code == 429


def test_identifiers_counted_independently() -> None:
    limiter = RateLimiter(lambda: None)
    for _ in range(10):
        limiter.check("1.2.3.4")
    limiter.check("5.6.7.8")  # different IP: not blocked


def test_redis_mode_shared_counter() -> None:
    client = MagicMock()
    pipe = MagicMock()
    client.pipeline.return_value = pipe
    pipe.execute.return_value = [11]  # counter already at limit
    limiter = RateLimiter(lambda: client)

    with pytest.raises(HTTPException) as exc:
        limiter.check("1.2.3.4")
    assert exc.value.status_code == 429
    pipe.incr.assert_called_once()


def test_redis_failure_falls_back_to_memory() -> None:
    def broken_redis() -> None:
        raise ConnectionError("down")

    limiter = RateLimiter(broken_redis)
    limiter.check("1.2.3.4")  # must not raise; counted locally
