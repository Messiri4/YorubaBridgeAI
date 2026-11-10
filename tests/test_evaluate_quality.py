"""Tests for the unified evaluation CLI + scorecard (Phase 3 Sprint 1)."""
from __future__ import annotations

import csv
import hashlib
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.evaluate_quality as eq
from app.services.scorecard import (
    METRIC_PROVENANCE,
    build_scorecard_row,
    fold_accents,
    fold_underdots_only,
    numeric_consistency,
    repetition_omission_warnings,
    _looks_like_english,
)
import sacrebleu


def _rows_csv(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


# --- Unicode / normalization ------------------------------------------------

def test_fold_accents_strips_tones_and_underdots() -> None:
    assert fold_accents("Ẹ kú ìrọ̀lẹ́, ọ̀rẹ́ mi.") == "E ku irole, ore mi."
    assert fold_underdots_only("ṣugá") == "sugá"  # tone kept, dot gone


def test_nfc_nfd_canonical_equivalence_scoring() -> None:
    a = "ọjọ́"
    b = "o\u0323jọ́"
    assert unicodedata.normalize("NFD", a) == unicodedata.normalize("NFD", b)
    assert unicodedata.normalize("NFC", b) == unicodedata.normalize("NFC", a) == "ọjọ́"


def test_metric_reproducibility_deterministic() -> None:
    preds = ["Ẹ kú ìrọ̀lẹ́.", "Mo ń lọ sí ọjà."]
    refs = ["Ẹ kú ìrọ̀lẹ́.", "Mo ń lọ sí ọjà."]
    s1 = sacrebleu.corpus_chrf(preds, [refs], char_order=6, word_order=2, beta=2.0).score
    s2 = sacrebleu.corpus_chrf(preds, [refs], char_order=6, word_order=2, beta=2.0).score
    assert s1 == s2


def test_perfect_translation_scores_100() -> None:
    r = build_scorecard_row("t|b", ["Ẹ kú ìrọ̀lẹ́."], ["Ẹ kú ìrọ̀lẹ́."],
                            ["Good evening."], latencies=[100.0])
    assert r.chrf_standard == 100.0 and r.bleu == 100.0


# --- detectors ----------------------------------------------------------------

def test_numeric_consistency_rates_and_missing_handling() -> None:
    res = numeric_consistency(
        ["He scored 72", "year 2016", "no digits here"],
        ["He scored 7-2", "year 2016", "no digits"])
    assert res["sentences_with_source_digits"] == 2
    assert res["digits_changed"] == 1 and res["digits_preserved"] == 1
    assert res["consistency_rate"] == 0.5


def test_numeric_rate_is_none_when_no_numeric_sources() -> None:
    assert numeric_consistency(["a"], ["a"])["consistency_rate"] is None


def test_entity_detector_symmetric_script_routing() -> None:
    # English prediction from Yoruba source must route YO->EN properly
    assert _looks_like_english("The governor spoke.") is True
    assert _looks_like_english("Ó sọ fún àwọn olówó.") is False


def test_repetition_detection() -> None:
    res = repetition_omission_warnings(
        ["Mo ń fẹ́ ìtòlẹ́sẹẹsẹ ìtòlẹ́sẹẹsẹ"], ["A sequel movie premiere announcement."])
    assert res["adjacent_repeat_warnings"] == 1


# --- CLI: config parsing, governance, outputs ---------------------------------

def test_yaml_parser_reads_full_config() -> None:
    cfg = eq._yaml_load(Path("configs/evaluation.yaml"))
    preds = eq._norm_predictions(cfg["predictions"])
    assert len(preds) == 1 and preds[0]["path"].endswith(".csv")
    assert cfg["metrics"]["bleu"] is True
    assert cfg["report"]["out_json"].endswith(".json")


def test_cli_produces_outputs_and_governance_pass(tmp_path: Path) -> None:
    # run the CLI against the real config (outputs are new phase3 files)
    rc = eq.main(["--config", "configs/evaluation.yaml"])
    assert rc == 0
    payload = __import__("json").loads(
        (Path("data/eval/phase3_scorecard.json")).read_text(encoding="utf-8"))
    arms = {a["arm"]: a for a in payload["arms"]}
    assert len(arms) == 4
    # standard chrF++ for eng_yor beams=4 must equal the direct computation
    direct = sacrebleu.corpus_chrf(
        [r["prediction"] for r in _rows_csv(Path("data/eval/beams_ab_20261109T231413Z.csv"))
         if r["direction"] == "eng_yor" and r["config"] == "beams=4"],
        [[r["reference"] for r in _rows_csv(Path("data/eval/beams_ab_20261109T231413Z.csv"))
          if r["direction"] == "eng_yor" and r["config"] == "beams=4"]],
        char_order=6, word_order=2, beta=2.0).score
    assert abs(arms["eng_yor|beams=4"]["chrfpp_standard"] - round(direct, 2)) < 0.01


def test_manifest_contains_hashes_and_version(tmp_path: Path) -> None:
    eq.main(["--config", "configs/evaluation.yaml"])
    man = __import__("json").loads(
        Path("data/eval/phase3_manifest.json").read_text(encoding="utf-8"))
    assert man["experiment_id"].startswith("phase3")
    real = hashlib.sha256(
        Path("data/eval/beams_ab_20261109T231413Z.csv").read_bytes()).hexdigest()
    for info in man["inputs"].values():
        assert info["sha256"] == real
    assert man["metric_versions"]["sacrebleu"] == sacrebleu.__version__
    assert man["acceptance_decision"] == "pending — foundation sprint, no candidate change"


def test_leakage_guard_blocks_unknown_source(tmp_path: Path) -> None:
    # craft a prediction file whose source is NOT in the dev split
    rogue = tmp_path / "rogue.csv"
    with rogue.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["direction", "idx", "config", "source",
                                          "reference", "prediction", "latency_s"])
        w.writeheader()
        w.writerow({"direction": "eng_yor", "idx": "0", "config": "beams=2",
                    "source": "TOTALLY UNKNOWN SENTENCE xyzzy",
                    "reference": "x", "prediction": "x", "latency_s": "1"})
    cfg = eq._yaml_load(Path("configs/evaluation.yaml"))
    try:
        eq.enforce_governance(cfg, [rogue])
        leaked = False
    except SystemExit as e:
        leaked = "LEAKAGE GUARD" in str(e)
    assert leaked


def test_protected_artifacts_never_overwritten(tmp_path: Path) -> None:
    assert "beams_ab_20261109T231413Z.csv" in eq.PROTECTED
    assert "flores_reference_review.csv" in eq.PROTECTED
    assert "translation_error_analysis.csv" in eq.PROTECTED


def test_human_labels_absent_reported_unavailable_not_zero() -> None:
    r = build_scorecard_row("a|b", ["x"], ["x"], ["s"])
    d = r.to_dict()
    assert d["human_labels"] == {} and "human_review_labels" in d["unavailable"]
    # and latency absent likewise
    assert "latency" in d["unavailable"] and d["latency"] == {}


def test_scorecard_heuristic_counts_distinct_from_verified() -> None:
    r = build_scorecard_row("a|b", ["He scored 72 today."],
                            ["Ó gba 7–2 lónìí."], ["He scored 7–2 today."],
                            latencies=[1.0])
    d = r.to_dict()
    # heuristic warning counts exist...
    assert d["numeric_consistency"]["digits_changed"] >= 1
    # ...but the only "verified" rate field is verified_total_rate — the sum
    # of exact preservation + *verified-equivalent* word renderings, neither
    # of which is a heuristic-warning count posing as ground truth
    nc = d["numeric_consistency"]
    assert "verified_error_rate" not in nc
    assert nc["verified_total_rate"] == 0.0  # 7–2->72: not preserved, not equivalent
    assert nc["numeric_mismatch_warning"] == 1  # stays a warning, not an error rate


def test_dataset_alignment_997() -> None:
    en = Path("data/flores_eval/dev.997.en.txt").read_text(encoding="utf-8").splitlines()
    yo = Path("data/eval/dev.997.yo.txt")
    yo = Path("data/flores_eval/dev.997.yo.txt").read_text(encoding="utf-8").splitlines()
    assert len(en) == len(yo) == 997
    # index 0 aligns: mirror == archived excerpt line 1
    assert en[0] == Path("data/flores/dev.eng_Latn").read_text(encoding="utf-8").splitlines()[0]


def test_rubric_fields_blank_in_review_worksheet() -> None:
    rows = _rows_csv(Path("data/eval/flores_reference_review.csv"))
    assert len(rows) == 100
    for r in rows:
        assert r["reference_accuracy"] == ""
        assert r["final_review_status"] == ""
        assert r["yoruba_reference"].strip() and r["english_source"].strip()
        # predictions are in SEPARATE fields, never in judgment columns
        assert "nllb_beams2_eng_yor" in r


def test_regressions_vs_phase2b_numbers(tmp_path: Path) -> None:
    """Regression guard: the unified scorecard must reproduce the phase-2B
    CSV values for standard chrF++ (documenting the two corrections made in
    Sprint 1: the mislabeled NFC/NFD rows in the old prose table)."""
    eq.main(["--config", "configs/evaluation.yaml"])
    sc = {(r["direction"], r["config"], r["regime"]): r["chrf"]
          for r in _rows_csv(Path("data/eval/chrf_normalization_comparison.csv"))}
    assert sc[("eng_yor", "beams=2", "nfd")] == "35.72"
    assert sc[("eng_yor", "beams=2", "nfc")] == "29.3"
