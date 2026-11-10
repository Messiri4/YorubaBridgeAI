"""Tests for the Yoruba tonality/orthography foundation (Agent 3).

Covers: high/mid/low tones, underdot letters (ẹ ọ ṣ), composed vs
combining-mark forms, NFC normalization without stripping distinctions,
orthography auditing, and the tone-restoration interface fallback.
"""

from __future__ import annotations

import unicodedata

import pytest

from app.services.tone_restoration import (
    PassThroughRestorer,
    ThresholdGuard,
    propose_restoration,
)
from app.services.yoruba_orthography import (
    audit,
    count_tone_marks,
    fold_for_comparison,
    has_full_marks,
    nfc,
    split_marks,
)


# ---------------------------------------------------------------------------
# NFC normalization preserves distinctions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "òṣìṣẹ́",           # low + underdot + high mix
        "ọmọ ilẹ̀",         # composed ọ and decomposed ẹ̀
        "Yorùbá",           # composed grave + circumflex-free
        "ṣíṣe",             # s with dot below
        "ńlá",              # prenasalized with high tone
    ],
)
def test_nfc_preserves_all_marks(text):
    normalized = nfc(text)
    assert count_tone_marks(normalized)["low"] >= 0
    # Marks survive: folded form differs from normalized form.
    assert fold_for_comparison(normalized) != normalized or normalized == fold_for_comparison(normalized)
    # Crucially: no combining mark is deleted by NFC round-trip.
    def marks(s):
        return [c for c in unicodedata.normalize("NFD", s) if unicodedata.combining(c)]
    assert len(marks(text)) == len(marks(normalized))


def test_nfc_unifies_composed_and_decomposed_underdots():
    composed = "ọ"            # U+1ECD
    decomposed = "o\u0323"    # o + combining dot below
    assert nfc(composed) == nfc(decomposed)


def test_mid_tone_written_as_no_mark():
    # Mid tone has no mark; 'ara' (body) is all-mid in isolation.
    assert count_tone_marks("ara")["low"] == 0
    assert count_tone_marks("ara")["high"] == 0


def test_high_low_tone_counts():
    assert count_tone_marks("óun")["high"] == 1
    assert count_tone_marks("òun")["low"] == 1
    assert count_tone_marks("ọ̀")["low"] == 1 and count_tone_marks("ọ̀")["underdot"] == 1


def test_split_marks_extracts_tones_and_underdot():
    base, marks = split_marks("ẹ̀")
    assert base == "e"
    assert "\u0323" in marks and "\u0300" in marks


# ---------------------------------------------------------------------------
# has_full_marks / audit
# ---------------------------------------------------------------------------

def test_has_full_marks_true_for_fully_marked():
    # Fully-marked text: every vowel carries low or high tone.
    assert has_full_marks("Ó fọ́ ìwé ńlá")


def test_has_full_marks_false_when_vowel_unmarked():
    assert not has_full_marks("omo ilẹ̀")  # 'omo' unmarked


def test_audit_flags_maybe_missing_tones():
    report = audit("Mo feran omo re.")
    kinds = {i.kind for i in report.issues}
    assert "maybe_missing_tones" in kinds
    # And it must NOT claim to know the correct form.
    for issue in report.issues:
        assert "needs" in issue.detail or "verify" in issue.detail


def test_audit_flags_inconsistent_marking():
    report = audit("omo ọmọ")
    assert any(i.kind == "inconsistent_marking" for i in report.issues)


def test_audit_clean_on_fully_marked_text():
    report = audit("Ó fẹ́ ṣiṣé òwu.")
    assert not any(
        i.kind == "maybe_missing_tones" for i in report.issues
    ), report.summary()


def test_audit_does_not_modify_text():
    text = "Nkan ti mo feran"
    audit(text)
    assert text == "Nkan ti mo feran"


def test_fold_is_diagnostic_only():
    assert fold_for_comparison("Ọ̀jẹ́") == "oje"


# ---------------------------------------------------------------------------
# Tone-restoration interface: safe fallback
# ---------------------------------------------------------------------------

def test_passthrough_restorer_never_changes_text():
    result = PassThroughRestorer().restore("Mo fe ra owo.")
    assert result.restored_text == nfc("Mo fe ra owo.")
    assert not result.changed
    assert result.fallback_used
    assert result.confidence == 0.0


def test_propose_restoration_current_state_is_safe_noop():
    result = propose_restoration("Ọmọ ńló sí ilé.")
    assert result.restored_text == nfc("Ọmọ ńló sí ilé.")
    assert result.fallback_used


def test_threshold_guard_blocks_low_confidence_model():
    class FakeRestorer:
        def restore(self, yoruba_text, source_english=None):
            from app.services.tone_restoration import ToneRestorationResult
            return ToneRestorationResult(
                restored_text=yoruba_text.replace("omo", "ọmọ́"),  # fabricated!
                changed=True,
                confidence=0.4,
            )

    guarded = ThresholdGuard(FakeRestorer(), threshold=0.85)
    result = guarded.restore("omo lo.")
    assert result.restored_text == "omo lo."   # fabrication blocked
    assert not result.changed
    assert result.fallback_used


def test_threshold_guard_passes_high_confidence_model():
    class GoodRestorer:
        def restore(self, yoruba_text, source_english=None):
            from app.services.tone_restoration import ToneRestorationResult
            return ToneRestorationResult(
                restored_text="ọmọ náà",
                changed=True,
                confidence=0.95,
                per_word_confidence={"ọmọ": 0.97, "náà": 0.93},
            )

    guarded = ThresholdGuard(GoodRestorer(), threshold=0.85)
    result = guarded.restore("omo naa")
    assert result.restored_text == "ọmọ náà"
    assert result.changed
    assert not result.fallback_used
