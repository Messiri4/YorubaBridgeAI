"""Shared pytest fixtures."""
import pytest

from app.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Reset the settings singleton before each test for isolation."""
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Fresh Settings instance per test."""
    return get_settings()
