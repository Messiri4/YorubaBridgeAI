"""Validation tests for the MENYO-20k research loader (fast, no model load)."""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from research.menyo20k import loader
from research.menyo20k.loader import (
    DatasetStats,
    cross_split_overlap,
    load_split,
    normalize,
    parse_line,
)

RAW = Path(loader.DATA_DIR)


def _stats() -> DatasetStats:
    return DatasetStats()


# ---------------------------------------------------------------- parse_line

def test_parse_line_keeps_tone_marks_and_underdots():
    stats = _stats()
    pair = parse_line(2, "Hello\tẸ kú àálẹ́, ọjọ́ tí mo ṣe é", "dev", stats)
    assert pair is not None
    assert "ẹ" in pair.yo and "ọ" in pair.yo and "ṣ" in pair.yo
    assert "\u0301" in unicodedata.normalize("NFD", pair.yo)  # acute tone mark present
    assert stats.rejected == 0


def test_parse_line_normalizes_to_nfc():
    stats = _stats()
    decomposed = unicodedata.normalize("NFD", "ẹ dá")
    pair = parse_line(1, f"A\t{decomposed}", "dev", stats)
    assert pair is not None
    assert pair.yo == normalize("ẹ dá")
    assert unicodedata.is_normalized("NFC", pair.yo)


def test_parse_line_rejects_wrong_column_count():
    stats = _stats()
    assert parse_line(1, "only-one-column", "dev", stats) is None
    assert parse_line(1, "a\tb\tc", "dev", stats) is None
    assert stats.rejected == 2


def test_parse_line_rejects_empty_sides():
    stats = _stats()
    assert parse_line(1, "\tsomething", "dev", stats) is None
    assert parse_line(1, "something\t   ", "dev", stats) is None
    assert stats.rejected == 2


def test_parse_line_rejects_non_alpha_target():
    stats = _stats()
    assert parse_line(1, "hello\t12345", "dev", stats) is None
    assert stats.rejected == 1


def test_parse_line_rejects_leading_combining_mark():
    stats = _stats()
    bad = unicodedata.normalize("NFD", "ẹ")[1:] + "bad"  # bare accent + text
    assert parse_line(1, f"a\t{bad}", "dev", stats) is None
    assert stats.rejected == 1


# --------------------------------------------------------------- load_split

@pytest.mark.skipif(not (RAW / "dev.tsv").exists(), reason="MENYO raw files not downloaded")
def test_load_dev_split_shape():
    pairs, stats = load_split("dev")
    assert stats.pairs == len(pairs) > 0
    assert stats.rejected >= 0
    # every pair aligned, non-empty
    assert all(p.en and p.yo for p in pairs)
    # tone marks survived the whole pipeline
    sample = next(p for p in pairs if any(c in p.yo for c in "ẹọṣ"))
    assert unicodedata.is_normalized("NFC", sample.yo)


@pytest.mark.skipif(not (RAW / "dev.tsv").exists(), reason="MENYO raw files not downloaded")
def test_dev_domain_labels_present():
    pairs, stats = load_split("dev")
    labeled = [p for p in pairs if p.domain]
    assert labeled, "expected some dev pairs to carry a domain label"
    assert set(p.domain for p in labeled) <= {"book", "digital", "news", "proverbs", "tedTalks"}
    assert sum(stats.domains.values()) == len(labeled)


def test_test_split_blocked_by_default():
    with pytest.raises(ValueError, match="test split"):
        load_split("test")


# ------------------------------------------------------------ duplicates etc.

@pytest.mark.skipif(not all((RAW / f).exists() for f in ("train.tsv", "dev.tsv")),
                    reason="MENYO raw files not downloaded")
def test_cross_split_overlap_between_train_and_dev():
    train, _ = load_split("train", with_domains=False)
    dev, _ = load_split("dev", with_domains=False)
    overlap = cross_split_overlap(train, dev)
    # Diagnostic only: report it, do not mutate the authors' splits.
    assert overlap >= 0
    print(f"[info] train∩dev exact-pair overlap: {overlap}")


def test_official_splits_not_altered_by_loader():
    """The loader must not rewrite raw files (mtime/hash stability)."""
    import hashlib
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in RAW.glob("*.tsv")}
    if RAW.joinpath("dev.tsv").exists():
        load_split("dev")
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in RAW.glob("*.tsv")}
    assert before == after
