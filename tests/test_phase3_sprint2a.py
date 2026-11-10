"""Sprint 2A regression tests: detector bug fixes + metric-discrepancy guard.

Each test pins one behavior that was wrong or ambiguous before Sprint 2A.
"""
from __future__ import annotations

import csv
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.scorecard import (
    REDUPLICATION_WHITELIST,
    _fold_tones,
    numeric_consistency,
    repetition_omission_warnings,
)


# --- Bug 2A-1: legitimate reduplication must not be flagged -------------------

def test_yoruba_emphatic_reduplication_not_flagged() -> None:
    res = repetition_omission_warnings(
        ["Wọ́n wá kí ọjọ ọjọ kiri."], ["They greeted day after day."])
    assert res["adjacent_repeat_warnings"] == 0


def test_genuine_adjacent_duplicate_still_flagged() -> None:
    res = repetition_omission_warnings(["Mo ra ìwé ìwé."], ["I bought books."])
    assert res["adjacent_repeat_warnings"] == 1


def test_fold_tones_maps_underdot_words_to_whitelist_keys() -> None:
    assert _fold_tones("ọjọ") == "ojo"
    assert _fold_tones("ọjọ") in REDUPLICATION_WHITELIST
    assert _fold_tones("ìwé") == "iwe"
    assert _fold_tones("ìwé") not in REDUPLICATION_WHITELIST


# --- Sprint-1 fix regression: underdot-split adjacent pair repeat --------------

def test_underdot_split_pair_repeat_still_caught() -> None:
    res = repetition_omission_warnings(
        ["Mo fẹ́ ìtòlé ṣẹẹsẹ ìtòlé ṣẹẹsẹ"], ["A sequel movie premiere."])
    assert res["adjacent_repeat_warnings"] == 1


# --- Bug 2A-2: spelled-out numbers are not "missing" ---------------------------

def test_spelled_out_time_not_counted_missing() -> None:
    res = numeric_consistency(
        ["Ètò náà bẹ̀rẹ̀ ní aago mọ́kànlá."],       # 11 o'clock (prediction)
        ["The show starts at 11:29."])               # (source)
    assert res["digits_missing"] == 0
    assert res["digits_spelled_out"] == 1


def test_genuinely_missing_digits_still_flagged() -> None:
    res = numeric_consistency(["Ó gba ipò kan."], ["He scored 72."])
    assert res["digits_missing"] == 1 and res["digits_spelled_out"] == 0


def test_spelled_bucket_does_not_count_into_rate() -> None:
    res = numeric_consistency(
        ["Ètò náà bẹ̀rẹ̀ ní aago mọ́kànlá."], ["The show starts at 11:29."])
    assert res["consistency_rate"] == 0.0  # preserved-digits rate is honest


# --- numeric-format edge cases pinned (were ambiguous) --------------------------

def test_dash_variant_digits_compare_equal() -> None:
    res = numeric_consistency(["score 7-2"], ["score 7–2"])  # hyphen vs en-dash
    assert res["digits_changed"] == 0 and res["digits_preserved"] == 1


def test_percent_word_vs_symbol_counts_preserved() -> None:
    # digits ('50') match regardless of %-vs-word rendering; the % SYMBOL is
    # not a digit, so this compares as preserved. Documented: unit detection
    # ('%' vs 'percent') is out of scope for the digit-consistency heuristic.
    res = numeric_consistency(["50% of users"], ["50 percent of users"])
    assert res["digits_preserved"] == 1 and res["digits_changed"] == 0


# --- metric discrepancy guard: plain chrF == historical numbers -----------------

def test_plain_chrf_reproduces_historical_scores() -> None:
    """Historical beam-A/B summary used sacrebleu defaults (plain chrF,
    word_order=0). This guards the documented explanation of the
    28.12-vs-26.07 discrepancy."""
    import sacrebleu
    rows = [r for r in csv.DictReader(
        Path("data/eval/beams_ab_20261109T231413Z.csv").open(encoding="utf-8"))
        if r["direction"] == "eng_yor" and r["config"] == "beams=2"]
    h = [r["prediction"] for r in rows]
    refs = [r["reference"] for r in rows]
    assert round(sacrebleu.corpus_chrf(h, [refs]).score, 2) == 28.12
    assert round(sacrebleu.corpus_chrf(
        h, [refs], char_order=6, word_order=2, beta=2.0).score, 2) == 26.07


def test_nfd_marker_ngram_inflation_is_real_not_harmonization() -> None:
    """Pin the mechanism: NFD scores HIGHER than NFC on a mixed-encoding
    sentence pair where ONE vowel+tone was substituted. If this stops holding
    after a sacrebleu upgrade, the Sprint-2A Unicode interpretation must be
    re-derived."""
    import sacrebleu
    a = "ẹ kú ìrọ̀lẹ́, ọ̀rẹ́ mi"
    b = "ọ kù ìrọ̀lẹ́, ọ̀rẹ́ mi"  # two vowels substituted, marks identical
    nfd = sacrebleu.corpus_chrf(
        [unicodedata.normalize("NFD", b)],
        [[unicodedata.normalize("NFD", a)]], char_order=6, word_order=2, beta=2.0).score
    nfc = sacrebleu.corpus_chrf(
        [unicodedata.normalize("NFC", b)],
        [[unicodedata.normalize("NFC", a)]], char_order=6, word_order=2, beta=2.0).score
    assert nfd > nfc


def test_identical_strings_score_identically_across_regimes() -> None:
    """No regime gives free points on identical text (rules out 'NFD is better'
    as an encoding-fix claim)."""
    import sacrebleu
    for s in ("ẹ kú ìrọ̀lẹ́, ọ̀rẹ́ mi.", "The rain fell in Kano overnight."):
        for form in (unicodedata.normalize("NFC", s), unicodedata.normalize("NFD", s)):
            assert sacrebleu.corpus_chrf([form], [[form]],
                                         char_order=6, word_order=2, beta=2.0).score == 100.0


def test_priority_review_subset_shape_and_blanks() -> None:
    rows = list(csv.DictReader(
        Path("data/eval/flores_priority_review_20.csv").open(encoding="utf-8")))
    assert len(rows) == 20
    reasons = ";".join(r["selection_reasons"] for r in rows)
    assert "meaning_change" in reasons and "control" in reasons
    for r in rows:
        assert r["reference_accuracy"] == ""      # human fields untouched
        assert r["final_review_status"] == ""
        assert r["yoruba_reference"].strip() and r["english_source"].strip()
        assert r["pair_idx"].isdigit()


def test_priority_subset_deterministic() -> None:
    import subprocess
    p1 = [r["pair_idx"] for r in csv.DictReader(
        Path("data/eval/flores_priority_review_20.csv").open(encoding="utf-8"))]
    subprocess.run([str(Path(sys.executable)), "scripts/build_priority_review.py"],
                   cwd=Path(__file__).resolve().parents[1], check=True,
                   capture_output=True)
    p2 = [r["pair_idx"] for r in csv.DictReader(
        Path("data/eval/flores_priority_review_20.csv").open(encoding="utf-8"))]
    assert p1 == p2
