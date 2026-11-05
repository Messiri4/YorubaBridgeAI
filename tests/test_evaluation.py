"""Tests for the evaluation module — metric math only, no model needed."""

import pytest

from app.services.evaluation import EntryResult, score_direction


def _res(direction: str, ref: str, hyp: str, verified: bool = True) -> EntryResult:
    return EntryResult(
        id="x",
        direction=direction,
        source="s",
        reference=ref,
        hypothesis=hyp,
        verified=verified,
        latency_ms=1.0,
    )


def test_perfect_translation_scores_100() -> None:
    results = [_res("eng_yor", "Mo ń lọ sí ọjà.", "Mo ń lọ sí ọjà.")]
    scores = score_direction(results)
    assert scores.chrf == pytest.approx(100.0)
    assert scores.bleu == pytest.approx(100.0)


def test_identical_texts_score_higher_than_different() -> None:
    same = score_direction([_res("eng_yor", "Báwo ni?", "Báwo ni?")])
    diff = score_direction([_res("eng_yor", "Báwo ni?", "Ó dàárọ̀.")])
    assert same.chrf > diff.chrf
    # NOTE: BLEU is NOT compared here — on a single short sentence BLEU's
    # 4-gram precision is 0 even for a perfect match. This is exactly why
    # chrF is the primary metric for our tiny eval set.


def test_corpus_level_bleu_rewards_matches() -> None:
    # With a larger corpus BLEU works normally: perfect corpus -> 100.
    results = [_res("eng_yor", r, r) for r in ("Báwo ni?", "Ó dàárọ̀.", "Mo ń lọ sí ọjà.")]
    assert score_direction(results).bleu == pytest.approx(100.0)


def test_directions_scored_separately() -> None:
    results = [
        _res("eng_yor", "a", "a"),
        _res("yor_eng", "b", "b"),
    ]
    scores = score_direction([r for r in results if r.direction == "eng_yor"])
    assert scores.direction == "eng_yor"
    assert scores.n_entries == 1


def test_verified_count() -> None:
    results = [_res("eng_yor", "a", "a", True), _res("eng_yor", "b", "b", False)]
    scores = score_direction(results)
    assert scores.n_verified == 1


def test_empty_hypothesis_scores_zero_chrf() -> None:
    scores = score_direction([_res("eng_yor", "Báwo ni?", "")])
    assert scores.chrf == 0.0
