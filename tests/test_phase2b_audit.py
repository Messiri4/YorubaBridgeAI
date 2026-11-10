"""Tests for the Phase-2B reference audit and chrF normalization experiment."""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_flores_references import (
    audit_entities,
    audit_numbers,
    audit_orthography,
    audit_structure,
    load_pairs,
)
from scripts.phase2b_outputs import fold_accents, fold_underdots_only


def cats(findings) -> set[str]:
    return {f.category for f in findings}


# --- audit: orthography ----------------------------------------------------

def test_mixed_unicode_forms_detected() -> None:
    yo = "Ẹ kú àárọ̀ e\u0323 máa bá o lọ́jà."  # precomposed Ẹ + combining e̲
    f = audit_orthography(0, "Good morning.", yo)
    assert "MIXED_UNICODE_FORMS" in cats(f)


def test_pure_nfc_not_flagged_as_mixed() -> None:
    yo = "Ẹ kú àárọ̀, ọ̀rẹ́ mi."  # fully precomposed NFC
    f = audit_orthography(0, "Good morning, my friend.", yo)
    assert "MIXED_UNICODE_FORMS" not in cats(f)


def test_adjacent_duplicate_detected_but_reduplication_exempt() -> None:
    dup = audit_orthography(0, "The the meeting.", "Awa nigeria nigeria wa.")
    assert "ADJACENT_DUPLICATE" in cats(dup)
    redup = audit_orthography(0, "Slowly.", "Ó fi fini fini kó oró náà.")
    assert "ADJACENT_DUPLICATE" not in cats(redup)


def test_missing_underdot_candidate() -> None:
    f = audit_orthography(0, "He does work.", "O n se ise daadaa.")  # ṣe/iṣẹ́ unaccented
    assert "MISSING_UNDERDOT_CANDIDATE" in cats(f)


def test_unaccented_reference_flagged() -> None:
    yo = "awon omo ilu naa Bere lowo lowo ipin owo oori ti ile ise naa"  # no tone marks
    f = audit_orthography(0, "The citizens asked about the funds.", yo)
    assert "REFERENCE_UNACCENTED" in cats(f)


def test_accented_reference_not_flagged() -> None:
    yo = "Àwọn ọmọ ìlú náà bère lọ́wọ́ lọ́wọ́ ipin owó oorí ti ilé iṣẹ́ náà."
    f = audit_orthography(0, "The citizens asked about the funds.", yo)
    assert "REFERENCE_UNACCENTED" not in cats(f)


# --- audit: numbers ----------------------------------------------------------

def test_digits_dropped_flagged() -> None:
    f = audit_numbers(0, "He scored 2 goals in 5 games.", "Ó gba goolu ninu ere.")
    assert "NUM_DIGITS_DROPPED" in cats(f)


def test_word_vs_digit_convention_not_error() -> None:
    # EN words, YO words: legitimate rendering, must NOT flag
    f = audit_numbers(0, "Fourteen banks were listed.",
                      "Ilé-ifowopamọ́ mẹ́rìnlá ni a mẹ̀nùmọ̀.")
    assert not [x for x in f if x.category.startswith("NUM_")]


def test_same_digits_not_flagged() -> None:
    f = audit_numbers(0, "The year 2016 was notable.", "Ọdún 2016 kún fọ́gbajẹ́.")
    assert not [x for x in f if x.category.startswith("NUM_")]


# --- audit: entities ---------------------------------------------------------

def test_transliterated_entity_not_flagged() -> None:
    # Nigeria -> Naijiria shares consonant skeleton; must count as carried
    f = audit_entities(0, "Nigeria joined the union.", "Orílẹ̀-èdè Naijiria darapọ̀ mọ́ ìgbìmọ̀.")
    assert "ENTITY_NOT_CARRIED" not in cats(f)


def test_dropped_entity_flagged() -> None:
    f = audit_entities(0, "Costello met Trudeau.", "Òṣìṣẹ́ àgbà pàdé alága.")
    assert "ENTITY_NOT_CARRIED" in cats(f)


# --- audit: structure --------------------------------------------------------

def test_empty_and_short_references() -> None:
    assert "EMPTY_REFERENCE" in cats(audit_structure(0, "Hello there.", ""))
    assert "SUSPICIOUSLY_SHORT" in cats(audit_structure(0, "A "*30, "Ẹ kú ìrọ̀lẹ́."))


def test_unbalanced_quotes() -> None:
    f = audit_structure(0, "He said: hello.", "Ó sọ pé: ó dàárọ̀.")
    assert "UNBALANCED_QUOTES" not in cats(f)


# --- dataset integrity -------------------------------------------------------

def test_dev_pairs_count_and_alignment() -> None:
    pairs = load_pairs()
    assert len(pairs) == 997
    assert all(y.strip() for _, y in pairs)
    # a clear majority of dev YO references carry Yoruba diacritics (pre-
    # composed underdot letters or combining tone marks); a minority are
    # accent-dropped — this is itself a documented audit finding, so we
    # assert the aggregate property rather than per-line diacritics.
    n_marked = sum(1 for _, y in pairs if any(ord(c) > 127 for c in y))
    assert n_marked >= 800  # measured: 862/997; ~14% accent-dropped (audit finding)


# --- chrF normalization helpers ----------------------------------------------

def test_fold_accents_is_lossy_and_deterministic() -> None:
    s = "Ẹ kú ìrọ̀lẹ́, ọ̀rẹ́ mi."
    f1, f2 = fold_accents(s), fold_accents(s)
    assert f1 == f2
    assert "ẹ" not in f1 and "\u0300" not in unicodedata.normalize("NFD", f1)
    assert fold_accents("ṣe") == "se"


def test_fold_underdots_preserves_tones() -> None:
    s = "Ẹ kú ìrọ̀lẹ́."
    folded = fold_underdots_only(s)
    # underdot letters mapped to plain letters ( decomposition folds ọ -> o )
    assert folded.startswith("E kú")
    # tone marks survive folding (low tone on ì retained)
    assert "\u0300" in unicodedata.normalize("NFD", folded)


def test_nfc_nfd_equivalence_of_score_inputs() -> None:
    a = "ọjọ́"  # fully precomposed
    b = "o\u0323jọ́"  # first ọ as base+U+0323 (decomposed underdot)
    assert unicodedata.normalize("NFD", a) == unicodedata.normalize("NFD", b)
    # NFC keeps them distinct: NFC of b restores ọ precomposed, but the two
    # forms ARE canonically equivalent — NFC equality fails only when one
    # sequence is not canonically ordered. Verify normalization stabilizes b:
    assert unicodedata.normalize("NFC", b) == a


def test_review_sample_deterministic() -> None:
    from scripts.phase2b_outputs import build_review_sample
    a = build_review_sample(target=20, seed=7)
    b = build_review_sample(target=20, seed=7)
    assert [r["pair_idx"] for r in a] == [r["pair_idx"] for r in b]
    assert all(r["reference_accuracy"] == "" for r in a)  # human fields blank
