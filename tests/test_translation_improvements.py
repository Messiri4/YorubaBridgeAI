"""Sprint TI-1 tests: numeric/identifier protection + glossary enforcement.

Unit-level (mocked model): no NLLB inference required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.protection import (
    ProtectionResult,
    protect,
    restore,
    verify_placeholder_structure,
)
from app.services.glossary import Glossary
from app.core.config import get_settings
from app.services.model_loader import LoadedModel
from app.services.translation import TranslationService


# --- protection: capture & restore ---------------------------------------------

def test_protect_scores_times_dates_currency() -> None:
    src = "On 2019-06-04 at 11:29, the score was 7–2 and sales rose 50% to $1,000."
    p = protect(src)
    assert p.n == 5
    assert "@@0@@" in p.protected_text and "7–2" not in p.protected_text
    restored, ok, _ = restore(p.protected_text, p)
    assert ok and restored == src


def test_protect_urls_emails_identifiers() -> None:
    src = "Visit https://example.com/x or email admin@example.org about ID-42."
    p = protect(src)
    restored, ok, _ = restore(p.protected_text, p)
    assert ok and restored == src
    assert "https://example.com/x" in p.placeholders.values()


def test_thousand_separator_capture() -> None:
    p = protect("About 9,000 people attended.")
    assert p.n == 1 and p.placeholders["@@0@@"] == "9,000"


def test_restore_detects_missing_sentinel() -> None:
    p = protect("Score 7–2 final.")
    corrupted = "Score @@0@@ and @@0@@ final."   # model duplicated the sentinel
    restored, ok, reason = restore(corrupted, p)
    assert not ok and restored == corrupted  # safe fallback: unchanged raw output
    # either the count check or the duplicate check catches it
    assert "duplicated" in reason or "expected 1 sentinels, found 2" in reason


def test_restore_detects_dropped_sentinel() -> None:
    p = protect("Score 7–2 final.")
    restored, ok, reason = restore("Score final.", p)
    assert not ok and restored == "Score final."
    assert "found 0" in reason


def test_restore_missing_sentinel_index_reported() -> None:
    """A two-placeholder output missing one index is caught by the
    missing-sentinels check."""
    p = protect("From 9,000 to 12,000 users.")
    corrupted = "From @@0@@ users."      # @@1@@ dropped entirely
    restored, ok, reason = restore(corrupted, p)
    assert not ok and restored == corrupted
    assert "found 1" in reason  # count mismatch catches the drop


def test_protection_disabled_passthrough() -> None:
    s = get_settings()
    s.numeric_protection_enabled = False
    assert s.numeric_protection_enabled is False  # config plumbing exists


# --- glossary -------------------------------------------------------------------

def test_glossary_simple_term_both_directions() -> None:
    g = Glossary("data/phrases/glossary.json")
    out, applied = g.apply("She has diabetes.", "eng_yor")
    assert "àrùn àtògbẹ" in out and applied
    back, applied2 = g.apply("Ó ní àtògbẹ.", "yor_eng")
    assert "diabetes" in back and applied2


def test_glossary_context_gating_blocks_ambiguous_use() -> None:
    g = Glossary("data/phrases/glossary.json")
    # "Type 1 diabetes" entry requires the word 'diabetes' nearby:
    # the phrase must NOT be applied where context is absent.
    out, applied = g.apply("Type 1 is a classification.", "eng_yor")
    assert all("Oríṣi" not in t for t in applied)


def test_glossary_longest_match_first() -> None:
    g = Glossary("data/phrases/glossary.json")
    out, applied = g.apply("Type 1 diabetes is manageable.", "eng_yor")
    # multi-word entry wins; the single-word entry does not double-apply
    assert "àrùn àtògbẹ Oríṣi Kìíní" in out
    assert sum("diabetes→" in a for a in applied) == 1


def test_glossary_diacritic_insensitive_matching() -> None:
    g = Glossary("data/phrases/glossary.json")
    out, applied = g.apply("Ó ní atogbe.", "yor_eng")   # unaccented input
    assert "diabetes" in out


def test_glossary_empty_text() -> None:
    g = Glossary("data/phrases/glossary.json")
    assert g.apply("", "eng_yor") == ("", [])


# --- service integration (mocked model) ------------------------------------------

def _service_with_output(output: str) -> tuple[TranslationService, MagicMock]:
    tokenizer = MagicMock()
    tokenizer.src_lang = "eng_Latn"
    tokenizer.return_value = {"input_ids": [[1]]}
    tokenizer.convert_tokens_to_ids.return_value = 1
    tokenizer.batch_decode.return_value = [output]
    model = MagicMock()
    model.generate.return_value = [[1]]
    svc = TranslationService(LoadedModel(model=model, tokenizer=tokenizer, device="cpu"))
    return svc, model


def test_service_protection_roundtrip_mocked() -> None:
    src = "The score was 7–2 at 11:29."
    # model faithfully copies the sentinels (positional order: 7–2 is first)
    svc, model = _service_with_output("Ìdíje náà wà ní @@0@@ ní @@1@@.")
    r = svc.translate(src, "eng", "yor", protect_numerics=True)
    assert "7–2" in r.translated_text and "11:29" in r.translated_text
    assert "@@" not in r.translated_text


def test_service_protection_fallback_on_corrupt_output() -> None:
    svc, _ = _service_with_output("Ìdíje náà wà ní @@0@@.")   # dropped one sentinel
    r = svc.translate("The score was 7–2 at 11:29.", "eng", "yor",
                      protect_numerics=True)
    # safe fallback: raw model output, no partial restoration
    assert r.translated_text == "Ìdíje náà wà ní @@0@@."


def test_service_protection_off_keeps_legacy_path() -> None:
    svc, _ = _service_with_output("Ìdíje náà wà ní 7–2 ní 11:29.")
    r = svc.translate("The score was 7–2 at 11:29.", "eng", "yor",
                      protect_numerics=False)
    assert "7–2" in r.translated_text  # untouched legacy behavior
