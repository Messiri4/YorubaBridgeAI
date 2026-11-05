"""Unit tests for TranslationService — the model is fully mocked.

No downloads, runs in milliseconds. Run with: make test
"""

from unittest.mock import MagicMock

import pytest

from app.services.model_loader import LoadedModel
from app.services.translation import TranslationError, TranslationResult, TranslationService


def make_service(
    fake_output_ids: list[list[int]],
) -> tuple[TranslationService, MagicMock, MagicMock]:
    """Build a TranslationService whose tokenizer and model are mocks.

    Returns (service, tokenizer_mock, model_mock).
    """
    tokenizer = MagicMock()
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "yor_Latn"
    tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
    tokenizer.convert_tokens_to_ids.return_value = 999

    model = MagicMock()
    model.generate.return_value = fake_output_ids

    loaded = LoadedModel(model=model, tokenizer=tokenizer, device="cpu")
    return TranslationService(loaded), tokenizer, model


def test_translate_happy_path_eng_to_yor() -> None:
    # Inputs chosen OFF the phrase-override table so the MODEL path runs.
    service, tokenizer, model = make_service([[7, 8, 9]])
    tokenizer.batch_decode.return_value = ["Ó dàgbà sokè"]

    result = service.translate("The mountain is tall", "eng", "yor")

    assert isinstance(result, TranslationResult)
    assert result.translated_text == "Ó dàgbà sokè"
    assert result.source_language == "eng"
    assert result.target_language == "yor"
    assert result.latency_ms >= 0.0

    # Source prefix set on tokenizer, target token forced in generation.
    assert tokenizer.src_lang == "eng_Latn"
    model.generate.assert_called_once()
    kwargs = model.generate.call_args.kwargs
    assert kwargs["forced_bos_token_id"] == 999
    assert kwargs["num_beams"] == 4  # from default settings


def test_translate_yor_to_eng_sets_correct_langs() -> None:
    service, tokenizer, model = make_service([[1]])
    tokenizer.batch_decode.return_value = ["The mountain is tall"]

    service.translate("Òkè náà ga", "yor", "eng")

    assert tokenizer.src_lang == "yor_Latn"
    assert tokenizer.convert_tokens_to_ids.call_args.args == ("eng_Latn",)
    assert model.generate.call_args.kwargs["forced_bos_token_id"] == 999


@pytest.mark.parametrize(
    ("text", "src", "tgt"),
    [
        ("", "eng", "yor"),  # empty input
        ("   ", "eng", "yor"),  # whitespace only
        ("Hi", "eng", "eng"),  # same languages
        ("Hi", "french", "yor"),  # unsupported source
        ("Hi", "eng", "igbo"),  # unsupported target
    ],
)
def test_translate_rejects_invalid_input(text: str, src: str, tgt: str) -> None:
    service, _, model = make_service([[1]])
    with pytest.raises(TranslationError):
        service.translate(text, src, tgt)
    model.generate.assert_not_called()


def test_diacritics_preserved_in_output() -> None:
    service, tokenizer, _ = make_service([[1]])
    # batch_decode returns a LIST of strings; simulate the diacritics surviving.
    tokenizer.batch_decode.return_value = ["Ilé àti ọjà, Ọmọ ṣeun"]

    result = service.translate("hello", "eng", "yor")
    assert result.translated_text == "Ilé àti ọjà, Ọmọ ṣeun"
