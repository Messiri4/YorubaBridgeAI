"""API tests using FastAPI's TestClient (no server or model needed)."""

from fastapi.testclient import TestClient

from app.main import create_app


def _client() -> TestClient:
    # load_model=False: never touch the real model in unit tests.
    return TestClient(create_app(load_model=False))


def test_health_ok() -> None:
    resp = _client().get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["app"] == "YorubaBridgeAI"


def test_openapi_docs_available() -> None:
    client = _client()
    assert client.get("/docs").status_code == 200
