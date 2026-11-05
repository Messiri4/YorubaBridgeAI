"""Unit tests for TranslationCache — Redis is mocked, no live server."""

import json
from typing import Any
from unittest.mock import MagicMock

from app.services.cache import TranslationCache, cache_key, normalize_for_cache


def make_cache(ping_ok: bool = True) -> tuple[TranslationCache, MagicMock]:
    client = MagicMock()
    if ping_ok:
        client.ping.return_value = True
    else:
        client.ping.side_effect = ConnectionError("redis down")
    return TranslationCache(client), client


def test_cache_disabled_when_ping_fails() -> None:
    cache, client = make_cache(ping_ok=False)
    assert cache.get("hi", "eng", "yor") is None
    cache.set("hi", "eng", "yor", {"x": 1})
    client.set.assert_not_called()


def test_cache_roundtrip() -> None:
    cache, client = make_cache()
    payload: dict[str, Any] = {"translated_text": "Ẹ kú àárọ̀", "latency_ms": 12.0}
    client.get.return_value = json.dumps(payload, ensure_ascii=False)

    assert cache.get("Good morning", "eng", "yor") == payload
    cache.set("Good morning", "eng", "yor", payload)
    stored = client.set.call_args
    assert "Ẹ kú àárọ̀" in stored.args[1]  # diacritics preserved (no ASCII escape)
    assert stored.kwargs["ex"] == 3600  # default TTL from settings


def test_cache_miss_returns_none() -> None:
    cache, client = make_cache()
    client.get.return_value = None
    assert cache.get("hello", "eng", "yor") is None


def test_redis_errors_degrade_to_miss() -> None:
    cache, client = make_cache()
    client.get.side_effect = ConnectionError("redis down")
    assert cache.get("hello", "eng", "yor") is None


def test_normalize_groups_equivalent_inputs() -> None:
    assert normalize_for_cache("  Hello ") == normalize_for_cache("hello")
    # NFC: composed vs decomposed diacritics normalize to the same key
    assert normalize_for_cache("ọ") == normalize_for_cache("ọ́".replace("ọ́", "ọ"))


def test_cache_key_differs_by_direction() -> None:
    assert cache_key("hi", "eng", "yor") != cache_key("hi", "yor", "eng")


def test_none_client_disables_cache() -> None:
    cache = TranslationCache(None)
    assert cache.get("hi", "eng", "yor") is None
