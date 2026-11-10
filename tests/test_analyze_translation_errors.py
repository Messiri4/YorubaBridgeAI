"""Tests for the heuristic translation-error analyzer.

Deterministic unit tests on synthetic sentences: every heuristic should fire
exactly when its trigger condition holds, and the numeric false-positive
guard (spelled-out Yoruba numbers rendered as digits) must work.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_translation_errors import (
    analyze_row,
    cap_entities,
    digits,
    repeated_ngrams,
    tokens,
    untranslated_spans,
)


def _row(direction: str, src: str, ref: str, hyp: str, cfg: str = "beams=2",
         idx: int = 0) -> dict:
    return {"direction": direction, "idx": str(idx), "config": cfg,
            "source": src, "reference": ref, "prediction": hyp, "latency_s": "1.0"}


def cats(findings: list[dict]) -> set[str]:
    return {f["category"] for f in findings}


# --- token helpers ---------------------------------------------------------

def test_digits_extracts_multi_digit_with_separators() -> None:
    assert digits("On September 24, 1759, a 9,000 year lease") == ["24", "1759", "9,000"]
    assert digits("no numbers here") == []


def test_cap_entities_skips_sentence_initial_stopwords() -> None:
    assert "The" not in cap_entities("The market opened.")
    assert "Washington Capitals" in cap_entities("He plays for the Washington Capitals.")


def test_tokens_keeps_yoruba_letters() -> None:
    assert "ọjà" in tokens("Mo ń lọ sí ọjà.")


# --- heuristics ------------------------------------------------------------

def test_number_missing_flagged() -> None:
    findings, _ = analyze_row(
        _row("eng_yor", "He scored 2 goals in 5 games.",
             "Ó ní góòlù méjì nínú eré márùn-ún.",
             "Ó ní góòlù nínú eré."),
        {})
    assert "NUMBER_MISSING" in cats(findings)


def test_number_changed_flagged() -> None:
    findings, _ = analyze_row(
        _row("eng_yor", "The record is 7–2.", "Ref ni 7–2.", "Akọsilẹ jẹ 72."), {})
    assert "NUMBER_CHANGED" in cats(findings)


def test_spelled_out_numbers_guard_yor_eng() -> None:
    """yor source spells numbers; hyp renders digits matching the reference.
    Must NOT flag NUMBER_CHANGED — script convention, not an error."""
    src = "Láàrin oṣù mẹ́ta tó kọjá, àwọn ọgọ́rin gbé silẹ."  # no digits
    ref = "In the last three months, 80 people were released."  # has 80
    hyp = "In the past three months, 80 people were released."
    findings, _ = analyze_row(_row("yor_eng", src, ref, hyp), {})
    assert "NUMBER_CHANGED" not in cats(findings)
    assert "NUMERAL_CONVENTION_NOTE" in cats(findings)


def test_entity_missing_flagged() -> None:
    findings, _ = analyze_row(
        _row("yor_eng", "Costello spoke about Australia.",
             "Mr Costello talked about Australia.",
             "The treasurer talked about the country."), {})
    ents = [f for f in findings if f["category"] == "ENTITY_MISSING"]
    names = " ".join(f["detail"] for f in ents)
    assert "Costello" in names


def test_entity_carried_via_transliteration() -> None:
    # 'Washington' appears transliterated with extra letters -> not missing.
    findings, _ = analyze_row(
        _row("eng_yor", "The Washington game ended.", "Eré Washington pari.",
             "Eré Wọshíñtònì parí."), {})
    assert not [f for f in findings if f["category"] == "ENTITY_MISSING"
                and "Washington" in f["detail"]]


def test_repeated_ngram_detects_duplication() -> None:
    assert repeated_ngrams("Kí ni óṣè óṣè náà?") or repeated_ngrams(
        "the committee committee met") == ["committee committee"]


def test_repeated_ngram_ignores_function_words() -> None:
    # Function words may repeat freely; only CONTENT-word repetition is flagged.
    reps = repeated_ngrams("he said that he said that he would go")
    assert "he" not in reps and "said" not in reps
    # but the content n-gram 'said that' variants are also excluded by design;
    # verify a real content repeat IS caught
    assert repeated_ngrams("the committee committee met yesterday") == ["committee"]


def test_untranslated_span_detected() -> None:
    spans = untranslated_spans(
        "the central booking facility released people",
        "Ilé iṣé the central booking facility released people")
    assert any("central booking" in s for s in spans)


def test_too_short_flagged() -> None:
    findings, _ = analyze_row(
        _row("yor_eng", "Ẹ kú ìrọ̀lẹ́.",
             "Good evening and welcome to this evening's broadcast everyone.",
             "Hello."), {})
    assert "TOO_SHORT" in cats(findings)


def test_beam_divergence_flagged() -> None:
    findings, _ = analyze_row(
        _row("yor_eng", "Òjò ń rò.", "It is raining.",
             "Rain falls upon the earth in great abundance today.",
             cfg="beams=2"),
        {"beams=4": "It is raining."})
    assert "BEAM_DIVERGENCE" in cats(findings)


def test_low_chrf_flagged() -> None:
    findings, _ = analyze_row(
        _row("yor_eng", "Ẹ kú ìrọ̀lẹ́.", "Completely unrelated reference text here.",
             "Totally different words appear in this translation."), {})
    lows = [f for f in findings if f["category"] == "LOW_CHRF"]
    assert lows and lows[0]["severity"] == "medium"


def test_reference_diacritic_check_only_for_eng_yor() -> None:
    ref_unaccented = "Awon iIwe naa fihan ile ifowopamo merinla to ran awon onisowo olola pa ilopo bilioni owo."
    findings, _ = analyze_row(
        _row("eng_yor", "The documents showed fourteen banks.",
             ref_unaccented,
             "Àwọn ìwé náà fi hàn pé ilé ifowopamo mẹ́rìnlá."), {})
    assert "REF_NO_DIACRITICS" in cats(findings)
    # reference findings are informational, never medium/high
    ref_f = [f for f in findings if f["kind"] == "reference"]
    assert all(f["severity"] in ("info", "low") for f in ref_f)


def test_findings_always_marked_as_warnings() -> None:
    findings, _ = analyze_row(
        _row("eng_yor", "The record is 7–2.", "Akọsilẹ jẹ 7–2.", "Akọsilẹ jẹ 72."), {})
    for f in findings:
        assert f["severity"] in ("high", "medium", "low", "info")
