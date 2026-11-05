"""Tests for the typed settings object."""
from app.core.config import Settings


def test_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.app_name == "YorubaBridgeAI"
    assert s.app_env == "development"
    assert s.port == 8000


def test_env_override(monkeypatch: "object") -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("PORT", "9001")
    s = Settings(_env_file=None)
    assert s.app_env == "test"
    assert s.port == 9001


def test_secrets_default_empty() -> None:
    s = Settings(_env_file=None)
    assert s.whatsapp_token == ""
