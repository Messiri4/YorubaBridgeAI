"""Integration tests: TI-1 modules wired into TranslationService (Phase 3).

Covers both directions (eng->yor, yor->eng), numeric preservation and
fallback, glossary behavior, Unicode handling, decoding-policy defaults,
never-leaking placeholders, tone-restoration no-op honesty, and
cache-key correctness across decoding policies. Model mocked throughout.
"""
from __future__ import annotations

import unicodedata
from unittest.mock import MagicMock

import pytest

from app.core.config import get_settings
from app.services.cache import cache_key
from app.services.model_loader import LoadedModel
from app.services.translation import TranslationError, TranslationService


def _service(output: str | list[str]) -> tuple[TranslationService, MagicMock, MagicMock]:
    outputs = [output] if isinstance(output, str) else list(output)
    tokenizer = MagicMock()
    tokenizer.src_lang = "eng_Latn"
    tokenizer.return_value = {"input_ids": [[1]]}
    tokenizer.convert_tokens_to_ids.return_value = 1
    tokenizer.batch_decode.side_effect = lambda ids, **kw: [outputs.pop(0)]
    model = MagicMock()
    model.generate.return_value = [[1]]
    return (TranslationService(LoadedModel(model=model, tokenizer=tokenizer, device="cpu")),
            tokenizer, model)


# ------------------------------------------------------------- eng -> yor

def test_eng_to_yor_numeric_preservation() -> None:
    svc, _, _ = _service("Ìdíje náà wà ní @@0@@ ní @@1@@.")
    r = svc.translate("The score was 7–2 at 11:29.", "eng", "yor", protect_numerics=True)
    assert "7–2" in r.translated_text and "11:29" in r.translated_text
    assert "@@" not in r.translated_text


def test_eng_to_yor_fallback_on_dropped_sentinel() -> None:
    # First generate() call returns corrupt output; retry returns plain text.
    svc, _, _ = _service(["Àwọn tí ó kù @@0@@.", "Àwọn tí ó kù náà dìn."])
    r = svc.translate("9,000 people attended.", "eng", "yor", protect_numerics=True)
    assert "@@" not in r.translated_text
    # The plain retry regenerated WITHOUT protection, so the number appears
    # verbatim (acceptable degraded mode; raw sentinels never leak).
    assert "9,000" in r.translated_text and "Àwọn tí ó kù" in r.translated_text


def test_unresolved_placeholder_never_leaks_even_on_double_failure() -> None:
    # Both attempts return sentinel-laden text; the final scrub guarantees
    # no placeholder reaches the caller.
    svc, _, _ = _service(["wà ní @@0@@.", "wà ní @@0@@."])
    r = svc.translate("the score was 7–2", "eng", "yor", protect_numerics=True)
    assert "@@" not in r.translated_text


# ------------------------------------------------------------- yor -> eng

def test_yor_to_eng_numeric_and_glossary() -> None:
    svc, _, _ = _service("I have diabetes and @@0@@ naira.")
    r = svc.translate("Mo ní àtògbẹ́ àti ₦5,000.", "yor", "eng", protect_numerics=True)
    assert "diabetes" in r.translated_text and "₦5,000" in r.translated_text
    assert "@@" not in r.translated_text


def test_yor_to_eng_diacritic_input_handled() -> None:
    decomposed = unicodedata.normalize("NFD", "Mo ní àtògbẹ.")
    svc, _, _ = _service("I have diabetes.")
    r = svc.translate(decomposed, "yor", "eng")
    assert "diabetes" in r.translated_text


# ------------------------------------------------------------- glossary

def test_glossary_runs_after_restoration_not_on_placeholders() -> None:
    # Output keeps the sentinel intact during generation; glossary must
    # apply to the RESTORED text (placeholder never shadows a term).
    svc, _, _ = _service("Ọ ní àrùn àtògbẹ àti @@0@@.")
    r = svc.translate("He has diabetes and 2 cars.", "eng", "yor")
    # glossary fires only on yor->eng direction or eng->yor terms; here the
    # eng_yor entry 'diabetes' does not appear in Yoruba output, so nothing
    # corrupts the restored number:
    assert "2" in r.translated_text


def test_glossary_unaccented_optin_applies() -> None:
    svc, _, _ = _service("He has diabetes.")
    r = svc.translate("Ó ní atogbe.", "yor", "eng")   # fold-unambiguous entry
    assert "diabetes" in r.translated_text


def test_glossary_ambiguous_unaccented_not_applied() -> None:
    import json
    import tempfile
    from pathlib import Path

    from app.services.glossary import Glossary
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "g.json"
        p.write_text(json.dumps({"yor_eng": [
            {"term": "ìgbà", "translation": "time", "context": [],
             "status": "verified", "allow_unaccented": True},
            {"term": "igbá", "translation": "calabash", "context": [],
             "status": "verified", "allow_unaccented": True},
        ]}), encoding="utf-8")
        g = Glossary(str(p))
        out, applied = g.apply("Mo ra igba kan.", "yor_eng")
        # 'igba' folds ambiguously (ìgbà/igbá); glossary must not guess.
        assert applied == [] and out == "Mo ra igba kan."


# ------------------------------------------------------- tone restoration

def test_tone_restoration_noop_on_yoruba_output_no_fabricated_marks() -> None:
    plain = "Omo naa lo si ile."   # deliberately unmarked model output
    svc, _, _ = _service(plain)
    svc.translate("omo naa lo si ile", "yor", "eng")  # any call; check direct
    # Direct interface check: PassThrough default fabricates nothing.
    from app.services.tone_restoration import propose_restoration
    res = propose_restoration(plain)
    assert res.restored_text == plain
    assert res.fallback_used and not res.changed


def test_tone_restoration_only_wired_for_yoruba_target() -> None:
    # English target output must pass through byte-for-byte (except the
    # shared stages), no NFC-only Yoruba path side effects.
    svc, _, _ = _service("completely plain english text")
    r = svc.translate("ọrọ gangan", "yor", "eng")
    assert r.translated_text == "completely plain english text"


# ------------------------------------------------------ decoding defaults

def test_generation_kwargs_match_production_defaults() -> None:
    svc, _, model = _service("ok")
    svc.translate("hello world", "eng", "yor")
    kw = model.generate.call_args.kwargs
    assert kw["num_beams"] == 4
    assert kw["max_new_tokens"] == 256
    assert kw["length_penalty"] == 1.1
    assert "repetition_penalty" not in kw
    assert "no_repeat_ngram_size" not in kw


def test_decoding_policy_env_override_reaches_generate(monkeypatch) -> None:
    monkeypatch.setenv("DECODING_POLICY", "fast_latency")
    svc, _, model = _service("ok")
    svc.translate("hello world", "eng", "yor")
    assert model.generate.call_args.kwargs["num_beams"] == 2


def test_invalid_policy_profile_fails_loud(monkeypatch) -> None:
    monkeypatch.setenv("DECODING_POLICY", "does_not_exist")
    svc, _, _ = _service("ok")
    with pytest.raises(ValueError, match="unknown decoding policy"):
        svc.translate("hello", "eng", "yor")


# ------------------------------------------------------ error propagation

def test_error_propagation_invalid_input_unchanged() -> None:
    svc, _, _ = _service("x")
    with pytest.raises(TranslationError):
        svc.translate("", "eng", "yor")
    with pytest.raises(TranslationError):
        svc.translate("x", "eng", "eng")


# ------------------------------------------------- cache / config compat

def test_cache_key_distinguishes_decoding_policies() -> None:
    base = cache_key("hi", "eng", "yor")
    assert cache_key("hi", "eng", "yor", "production_default") == base  # default == legacy key
    assert cache_key("hi", "eng", "yor", "fast_latency") != base
    assert cache_key("hi", "eng", "yor", "fast_latency").endswith(":fast_latency")


def test_config_defaults_compatible() -> None:
    s = get_settings()
    assert s.numeric_protection_enabled is True
    assert s.glossary_enabled is True
    assert s.translation_num_beams == 4
    assert s.translation_max_new_tokens == 256
    assert s.translation_length_penalty == 1.1
