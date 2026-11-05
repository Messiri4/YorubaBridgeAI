"""Shared pytest fixtures."""

from typing import Any

import pytest

from app.core.config import Settings, get_settings
from app.core.rate_limit import RateLimiter
from app.services.cache import TranslationCache


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Reset the settings singleton before each test for isolation."""
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Fresh Settings instance per test."""
    return get_settings()


@pytest.fixture
def api_client() -> Any:
    """TestClient for the real app with mocked internals (no model, no Redis).

    Monkeypatches app.state so the translate route can run end-to-end with
    a fake translation service — a true integration test of routing,
    rate limiting and cache plumbing.
    """
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.services.model_loader import LoadedModel
    from app.services.translation import TranslationService

    tokenizer = MagicMock()
    tokenizer.src_lang = "eng_Latn"
    tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
    tokenizer.convert_tokens_to_ids.return_value = 999
    tokenizer.batch_decode.return_value = ["Ẹ kú àárọ̀"]

    model = MagicMock()
    model.generate.return_value = [[7, 8, 9]]

    app = create_app(load_model=False)
    app.state.translation_service = TranslationService(
        LoadedModel(model=model, tokenizer=tokenizer, device="cpu")
    )
    app.state.translation_cache = TranslationCache(None)  # disabled
    app.state.rate_limiter = RateLimiter(lambda: None)  # in-memory
    return TestClient(app)
