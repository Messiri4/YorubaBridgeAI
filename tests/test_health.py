"""API tests using FastAPI's TestClient (no server or model needed)."""
from fastapi.testclient import TestClient

from app.main import create_app


def test_health_ok() -> None:
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["app"] == "YorubaBridgeAI"


def test_openapi_docs_available() -> None:
    client = TestClient(create_app())
    assert client.get("/docs").status_code == 200
