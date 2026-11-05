"""Tests for the phrase-override table and its integration in the service."""

from unittest.mock import MagicMock

import pytest

from app.services.model_loader import LoadedModel
from app.services.translation import PhraseOverrides, TranslationService


@pytest.fixture
def overrides(tmp_path: "object") -> PhraseOverrides:
    import json

    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps(
            {
                "eng_yor": {"good morning.": "Ẹ kú àárọ̀."},
                "yor_eng": {"báwo ni?": "How are you?"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return PhraseOverrides(str(path))


def test_lookup_exact_match(overrides: PhraseOverrides) -> None:
    assert overrides.lookup("Good morning.", "eng", "yor") == "Ẹ kú àárọ̀."


def test_lookup_normalizes_case_and_whitespace(overrides: PhraseOverrides) -> None:
    assert overrides.lookup("  GOOD MORNING.  ", "eng", "yor") == "Ẹ kú àárọ̀."


def test_lookup_miss_returns_none(overrides: PhraseOverrides) -> None:
    assert overrides.lookup("Where is the bank?", "eng", "yor") is None


def test_lookup_direction_specific(overrides: PhraseOverrides) -> None:
    # Same normalized text, opposite direction: no match.
    assert overrides.lookup("Good morning.", "yor", "eng") is None


def test_missing_file_degrades_to_empty(tmp_path: "object") -> None:
    table = PhraseOverrides(str(tmp_path / "nope.json"))
    assert table.lookup("anything", "eng", "yor") is None


def test_invalid_json_degrades_to_empty(tmp_path: "object") -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    table = PhraseOverrides(str(path))
    assert table.lookup("anything", "eng", "yor") is None


def test_service_uses_override_before_model(overrides: PhraseOverrides) -> None:
    tokenizer = MagicMock()
    tokenizer.src_lang = "eng_Latn"
    tokenizer.return_value = {"input_ids": [[1]]}
    tokenizer.convert_tokens_to_ids.return_value = 1
    tokenizer.batch_decode.return_value = ["Àárọ̀ àárọ̀."]  # the bad model output
    model = MagicMock()
    model.generate.return_value = [[1]]

    service = TranslationService(LoadedModel(model=model, tokenizer=tokenizer, device="cpu"))
    service._overrides = overrides  # inject table directly

    result = service.translate("Good morning.", "eng", "yor")
    assert result.translated_text == "Ẹ kú àárọ̀."
    model.generate.assert_not_called()  # the model was bypassed

    # Non-table text still goes through the model.
    service.translate("What time is it?", "eng", "yor")
    model.generate.assert_called_once()


def test_nfc_composed_and_decomposed_keys_match(tmp_path: "object") -> None:
    """'ọ' composed vs decomposed must hit the same table entry."""
    import json
    import unicodedata

    path = tmp_path / "o.json"
    decomposed = unicodedata.normalize("NFD", "báwo ni?")  # á as a + combining
    path.write_text(
        json.dumps({"yor_eng": {decomposed: "How are you?"}}, ensure_ascii=False), "utf-8"
    )
    table = PhraseOverrides(str(path))
    assert table.lookup("Báwo ni?", "yor", "eng") == "How are you?"
