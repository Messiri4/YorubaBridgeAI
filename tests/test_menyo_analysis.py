"""Lightweight tests for MENYO baseline-evaluation helpers and error analysis.

These run in seconds: no model load, no network, synthetic sentences only.
"""
from __future__ import annotations

from scripts.analyze_menyo_errors import flag_pair
from scripts.evaluate_menyo import (
    numeric_preserved,
    placeholder_failures,
    repeated_output,
    stratified_sample,
)


# ------------------------------------------------------------ metrics helpers

def test_numeric_preserved_true_and_false():
    assert numeric_preserved("I have 3 goats and 12 yams", "Mo ní 3 ewuré àti 12 isu")
    assert not numeric_preserved("I have 3 goats", "Mo ní 4 ewuré")


def test_placeholder_failure_detected():
    assert placeholder_failures("send <num> naira", "fi <num> náírà owó") == 0
    assert placeholder_failures("send <num> naira", "fi owó náírà") == 1


def test_repeated_output_detected():
    assert repeated_output("wá wá wá wá wá wá wá wá wá wá wá wá wá sí ibi")
    assert not repeated_output("This is a perfectly normal short sentence.")


# ------------------------------------------------------- stratified sampling

def test_stratified_sample_deterministic():
    from research.menyo20k.loader import Pair
    pairs = [Pair(idx=i, split="dev", en=f"e{i}", yo=f"y{i}", domain=["news", "book"][i % 2])
             for i in range(20)]
    a = stratified_sample(pairs, 6, 13)
    b = stratified_sample(pairs, 6, 13)
    c = stratified_sample(pairs, 6, 7)
    assert [p.idx for p in a] == [p.idx for p in b]  # same seed => same sample
    assert [p.idx for p in a] != [p.idx for p in c]  # different seed => different sample
    assert len(a) == 6


# ---------------------------------------------------------- error-flag logic

def test_flags_number_mismatch():
    flags = flag_pair("He bought 5 books", "Ó ra ìwé mẹ̀ta", "Ó ra 5 ìwé")
    assert flags["numbers_mismatch"] is False  # digit preserved in prediction
    flags = flag_pair("He bought 5 books", "ó ra 5 iwé", "ó ra 6 iwé")
    assert flags["numbers_mismatch"] is True


def test_flags_negation_missing():
    flags = flag_pair("I do not like it", "Èmi kò fẹ́ ẹ́", "Èmi fẹ́ ẹ́")
    assert flags["negation_mismatch"] is True


def test_flags_repetition():
    text = " ".join(["sísì sílẹ̀"] * 7)  # 14 words: crosses the 4-gram threshold
    flags = flag_pair("source", "ref", text)
    assert flags["repetition"] is True
