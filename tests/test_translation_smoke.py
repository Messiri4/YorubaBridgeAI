"""Real-model smoke tests — downloads ~2.4 GB on first run.

Opt-in: run with `make test-smoke` (or
`.venv/bin/pytest -m smoke`). Excluded from the default test suite so
`make test` stays fast and offline-friendly.
"""

import pytest

from app.services.model_loader import SUPPORTED_LANGUAGES, load_nllb
from app.services.translation import TranslationService

pytestmark = pytest.mark.smoke


def test_supported_language_codes_are_flores200() -> None:
    assert SUPPORTED_LANGUAGES == {"eng": "eng_Latn", "yor": "yor_Latn"}


def test_real_model_loads_and_translates() -> None:
    loaded = load_nllb()
    service = TranslationService(loaded)

    result = service.translate("Good morning, how are you?", "eng", "yor")
    assert result.translated_text.strip()
    assert result.latency_ms > 0

    back = service.translate(result.translated_text, "yor", "eng")
    assert back.translated_text.strip()
