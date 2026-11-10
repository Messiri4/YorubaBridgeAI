"""Sprint 2 regression tests: failure-analysis + semantic-diagnostics scripts.

Offline only — no model, no network. Guards the analysis tooling itself:
- number-word/vigesimal credit must not flag legitimate renderings
- negation detection is bilingual and non-literal (kò/cannot/refused...)
- complementizer 'kí' must NOT count as negation (verified false positive)
- degeneration-loop detector
- diagnostics flag tiers and dimension coverage
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fa = _load("sprint2_failure_analysis", "scripts/sprint2_failure_analysis.py")
sd = _load("semantic_diagnostics", "scripts/semantic_diagnostics.py")


# ------------------------------------------------------- numeric classifier

def test_digits_verbatim_is_preserved():
    st, dropped = fa.classify_row("He ran 41 balls", "Ó ṣe 41 papọ̀.")
    assert st == "preserved" and not dropped


def test_number_word_rendering_not_failure():
    # "4" rendered as mẹ́rin must NOT be a drop
    st, dropped = fa.classify_row("I bought 4 books.", "Mo ra àwọn ìwé mẹ́rin.")
    assert st in ("rendered_as_words", "preserved") and not dropped


def test_vigesimal_rendering_not_failure():
    # 16 -> kẹrìndínlógún (20 minus 4); attested in devtest #68
    st, dropped = fa.classify_row("on November 16", "ójọ́ kẹrìndínlógún oṣù November")
    assert st == "rendered_as_words" and not dropped


def test_genuine_drop_detected():
    st, dropped = fa.classify_row("made 36 runs", "ó ṣe ìsọfúnni púpọ̀.")
    assert st == "dropped" and "36" in dropped


def test_degeneration_detector():
    assert fa._has_degeneration("ìkọ́ " * 8)
    assert not fa._has_degeneration("ọjọ́ ọjọ́ kẹta ni ó ṣe")


# ------------------------------------------------------------ negation cues

def test_negation_eng_variants():
    assert sd.has_negation("He refused to go.", "eng")
    assert sd.has_negation("They cannot win.", "eng")
    assert sd.has_negation("It was not possible.", "eng")
    assert sd.has_negation("The FBI prevented it.", "eng")


def test_negation_yor_particles():
    assert sd.has_negation("Kò ní dayabẹ́tìsì.", "yor")
    assert sd.has_negation("kì í ṣe e", "yor") or sd.has_negation("kìí ṣe", "yor")
    assert sd.has_negation("àwọn tí kò sí nílé.", "yor")


def test_complementizer_ki_is_not_negation():
    # verified false positive from devtest #69: 'ló kí o dẹ̀ sọ̀rọ̀' = greeted
    assert not sd.has_negation("ó ló kí i dẹ̀ sọ̀rọ̀", "yor")


def test_positive_text_has_no_negation():
    assert not sd.has_negation("They won the match.", "eng")
    assert not sd.has_negation("Ó fẹ́ ṣiṣé òwu.", "yor")


# --------------------------------------------------------------- evaluator

def test_evaluator_flags_negation_loss_as_pattern():
    flags = sd.evaluate("Kò ní dayabẹ́tìsì.", "does not have diabetes.",
                        "He has diabetes.", "yor_eng")
    dims = {f.dimension: f.tier for f in flags}
    assert dims.get("negation") == "pattern"


def test_evaluator_credits_nonliteral_negation():
    # 'not possible' -> 'kò ṣeéṣe' is preserved negation, no flag
    flags = sd.evaluate("It is not possible.", "", "Kò ṣeéṣe.", "eng_yor")
    assert all(f.dimension != "negation" for f in flags)


def test_evaluator_credits_yor_negation_in_english_target():
    flags = sd.evaluate("Kò ṣeéṣe.", "not possible", "It is not possible.", "yor_eng")
    assert all(f.dimension != "negation" for f in flags)


def test_evaluator_flags_placeholder_leak():
    flags = sd.evaluate("I have 3.", "", "Mo ní @@0@@.", "eng_yor")
    assert any(f.dimension == "placeholder_leak" for f in flags)


def test_evaluator_flags_extreme_omission():
    flags = sd.evaluate("long source", "a " * 40, "short.", "yor_eng")
    assert any(f.dimension == "omission" for f in flags)


def test_flags_have_tiers():
    flags = sd.evaluate("Ó lè wá.", "He may come.", "He came.", "yor_eng")
    assert flags and all(f.tier in ("pattern", "heuristic") for f in flags)
