"""End-to-end API tests with mocked model/cache — no Redis, no downloads."""

from typing import Any

from fastapi.testclient import TestClient


def test_translate_endpoint_ok(api_client: TestClient) -> None:
    resp = api_client.post(
        "/api/v1/translate",
        json={"text": "The mountain is tall", "source_language": "eng", "target_language": "yor"},
    )
    assert resp.status_code == 200
    body: dict[str, Any] = resp.json()
    assert body["translated_text"] == "Ẹ kú àárọ̀"  # mocked model's canned output
    assert body["cached"] is False
    assert body["latency_ms"] >= 0


def test_rate_limit_returns_429(api_client: TestClient) -> None:
    # Default limit is 10/60s from settings; send 11 requests from one IP.
    for _ in range(10):
        assert (
            api_client.post(
                "/api/v1/translate",
                json={"text": "hi", "source_language": "eng", "target_language": "yor"},
            ).status_code
            == 200
        )
    resp = api_client.post(
        "/api/v1/translate",
        json={"text": "hi", "source_language": "eng", "target_language": "yor"},
    )
    assert resp.status_code == 429


def test_invalid_language_code_is_422(api_client: TestClient) -> None:
    resp = api_client.post(
        "/api/v1/translate",
        json={"text": "hi", "source_language": "french", "target_language": "yor"},
    )
    assert resp.status_code == 422


def test_same_languages_rejected(api_client: TestClient) -> None:
    resp = api_client.post(
        "/api/v1/translate",
        json={"text": "hi", "source_language": "eng", "target_language": "eng"},
    )
    assert resp.status_code == 422


def test_empty_text_rejected(api_client: TestClient) -> None:
    resp = api_client.post(
        "/api/v1/translate",
        json={"text": "", "source_language": "eng", "target_language": "yor"},
    )
    assert resp.status_code == 422
