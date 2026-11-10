"""Sprint 2C-Q regression tests: numerical-equivalence classification.

Each test pins one adversarial requirement from the sprint brief. All
cases use numeric_consistency(preds, srcs) — note the argument order.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.scorecard import (
    _NUMBER_WORD_VALUES,
    _fold_tones,
    _pred_number_word_values,
    _strip_combining,
    numeric_consistency,
)

CLS_KEYS = ("exact_digits_preserved", "verified_number_word_equivalent",
            "numeric_mismatch_warning", "numeric_review_required")


def cls_of(pred: str, src: str) -> str:
    r = numeric_consistency([pred], [src])
    hits = [k for k in CLS_KEYS if r[k]]
    assert len(hits) == 1, f"ambiguous classification for {pred!r}: {r}"
    return hits[0]


# --- required adversarial cases -----------------------------------------------

def test_source_11_pred_eleven_verified() -> None:
    assert cls_of("The score was eleven.", "The score was 11.") == \
        "verified_number_word_equivalent"


def test_source_11_pred_twelve_is_mismatch() -> None:
    assert cls_of("The score was twelve.", "The score was 11.") == \
        "numeric_mismatch_warning"


def test_source_11_unrelated_number_word_not_verified() -> None:
    # 'twenty' is a supported word but of the wrong value with full coverage:
    # this is a verified mismatch, never "preserved".
    assert cls_of("He said twenty.", "The score was 11.") == \
        "numeric_mismatch_warning"
    # an unrelated NON-number word must not verify either
    assert cls_of("He smiled.", "The score was 11.") == "numeric_review_required"


def test_source_time_11_29_partial_word_review_required() -> None:
    assert cls_of("Ìtò náà bẹ̀rẹ̀ ní aago mọ́kànlá.", "It starts at 11:29.") == \
        "numeric_review_required"


def test_score_7_2_to_72_is_mismatch() -> None:
    assert cls_of("Akọsilẹ náà jẹ 72.", "The record is 7–2.") == \
        "numeric_mismatch_warning"


def test_9000_to_900_is_mismatch() -> None:
    assert cls_of("Ní ipà 900.", "About 9,000 people.") == \
        "numeric_mismatch_warning"


def test_percent_fifty_word_verified() -> None:
    assert cls_of("fifty percent of users", "50% of users") == \
        "verified_number_word_equivalent"


def test_yoruba_tone_variants_resolve_to_same_value() -> None:
    # tone/underdot variants fold to one key; original text never altered
    for form in ("ọ̀kan", "òkan", "okan"):
        assert _fold_tones(form) == "okan"
        assert _NUMBER_WORD_VALUES["okan"] == 1
    assert _pred_number_word_values("ọ̀kànlá") == {11}


def test_compound_yoruba_expression_review_required() -> None:
    # 41 as 'ogójì ọ̀kànlá' is a compound the table cannot verify
    assert cls_of("Ó kọ ọ́ nípa ogójì ọ̀kànlá.", "Almost 41 people.") == \
        "numeric_review_required"


def test_source_41_compound_rendering_never_verified() -> None:
    # even exact-token compound ('ogoji' + 'okan') must NOT verify as 41
    r = numeric_consistency(["ogójì ọ̀kànlá."], ["Almost 41 people."])
    assert r["verified_number_word_equivalent"] == 0


# --- duplicate/contradiction dictionary cleanup --------------------------------

def test_no_duplicate_conflicting_values_in_table() -> None:
    # every folded key maps to exactly one value; no key collides
    assert len({k for k in _NUMBER_WORD_VALUES}) == len(_NUMBER_WORD_VALUES)
    assert _NUMBER_WORD_VALUES["aadota"] == 50
    assert _NUMBER_WORD_VALUES["ogoji"] == 40
    assert _NUMBER_WORD_VALUES["ogorin"] == 80


# --- edge cases -----------------------------------------------------------------

def test_empty_inputs() -> None:
    r = numeric_consistency([], [])
    assert r["sentences_with_source_digits"] == 0
    assert r["consistency_rate"] is None and r["verified_total_rate"] is None


def test_no_digits_in_source_skips_sentence() -> None:
    r = numeric_consistency(["Mo fẹ́ ra ìwé."], ["I want to buy a book."])
    assert r["sentences_with_source_digits"] == 0


def test_multiple_and_duplicate_numbers() -> None:
    assert cls_of("He has 5 cats and 7 dogs.", "He has 5 cats and 7 dogs.") == \
        "exact_digits_preserved"
    assert cls_of("3, 3 cats", "3, 3 cats") == "exact_digits_preserved"
    r = numeric_consistency(["3, 3 cats"], ["3, 4 cats"])
    assert r["numeric_mismatch_warning"] == 1


def test_unicode_normalization_equivalence() -> None:
    # NFC vs NFD spellings of the same Yoruba number word classify identically
    nfc = "ọ̀kan"
    nfd = "o\u0323\u0300kan"
    assert _fold_tones(nfc) == _fold_tones(nfd) == "okan"


def test_tokenizer_no_longer_splits_tone_marked_words() -> None:
    # regression for the discovered \w-boundary bug: 'àádọ́ta' was split
    assert _strip_combining("àádọ́ta") == "aadota"
    assert _pred_number_word_values("àádọ́ta") == {50}


# --- backward compatibility of exported keys ------------------------------------

def test_backward_compatible_keys_present() -> None:
    r = numeric_consistency(["72"], ["record 7–2."])
    assert r["digits_changed"] == 1          # legacy alias kept
    assert r["numeric_mismatch_warning"] == 1  # new explicit name
    r2 = numeric_consistency(["eleven."], ["score 11."])
    assert r2["digits_spelled_out"] == 1     # legacy bucket now strictly verified
    assert r2["digits_missing"] == 0
    assert r2["numeric_review_required"] == 0


def test_heuristic_counts_are_not_error_rates() -> None:
    """Documentation-by-test: warning-class keys must never be exposed as a
    'verified error' measure."""
    r = numeric_consistency(["Ó gbà ipò kan."], ["He scored 72."])
    assert "numeric_mismatch_warning" in r and "verified_error_rate" not in r
