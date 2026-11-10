"""Phase 2B driver: chrF normalization experiment + human-review sample.

Reads ONLY existing artifacts (beam A/B CSV, audit findings); runs no
inference. Also validates Phase-2 detector findings against reference
ground-truth to separate false positives from plausible errors.
"""
from __future__ import annotations

import csv
import json
import random
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval"
AB_CSV = OUT / "beams_ab_20261109T231413Z.csv"
AUDIT_CSV = OUT / "flores_reference_audit.csv"

# sacrebleu version recorded for reproducibility
SACREBLEU_VERSION = sacrebleu.__version__
CHRF_KWARGS = dict(char_order=6, word_order=2, beta=2.0)  # chrF++ defaults


# --------------------------------------------------------------------------
# normalization regimes
# --------------------------------------------------------------------------

def fold_accents(text: str) -> str:
    """Diagnostic lossy fold: strip tones AND dot-below; also map the three
    precomposed underdot letters to plain latin letters. NOT for production —
    this intentionally destroys Yoruba tonal information."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", text)


def fold_underdots_only(text: str) -> str:
    """Strip ONLY dot-below (ẹ→e, ọ→o, ṣ→s); tones preserved."""
    out = []
    for c in unicodedata.normalize("NFD", text):
        if c == "\u0323":
            continue
        if c in "ẹọṣ":
            out.append({"ẹ": "e", "ọ": "o", "ṣ": "s"}[c])
        else:
            out.append(c)
    return unicodedata.normalize("NFC", "".join(out))


REGIMES = {
    "original": lambda t: t,
    "nfc": lambda t: unicodedata.normalize("NFC", t),
    "nfd": lambda t: unicodedata.normalize("NFD", t),
    "fold_underdots": fold_underdots_only,
    "fold_all_accents": fold_accents,
}


def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
    return sacrebleu.corpus_chrf(hyps, [refs], **CHRF_KWARGS).score


def run_normalization_experiment() -> list[dict]:
    rows = list(csv.DictReader(AB_CSV.open(encoding="utf-8")))
    groups: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    for r in rows:
        key = (r["direction"], r["config"])
        hyps, refs = groups.get(key, ([], []))
        hyps.append(r["prediction"])
        refs.append(r["reference"])
        groups[key] = (hyps, refs)

    out_rows = []
    for (direction, config), (hyps, refs) in sorted(groups.items()):
        for regime, fn in REGIMES.items():
            h = [fn(x) for x in hyps]
            rf = [fn(x) for x in refs]
            score = corpus_chrf(h, rf)
            out_rows.append({
                "direction": direction, "config": config, "regime": regime,
                "chrf": round(score, 2),
            })
        orig = next(float(o["chrf"]) for o in out_rows
                    if o["direction"] == direction and o["config"] == config
                    and o["regime"] == "original")
        for regime in ("nfc", "nfd", "fold_underdots", "fold_all_accents"):
            val = next(float(o["chrf"]) for o in out_rows
                       if o["direction"] == direction and o["config"] == config
                       and o["regime"] == regime)
            out_rows.append({
                "direction": direction, "config": config,
                "regime": f"delta_vs_original:{regime}",
                "chrf": round(val - orig, 2),
            })
    return out_rows


# --------------------------------------------------------------------------
# Phase-2 findings vs reference ground truth
# --------------------------------------------------------------------------

def validate_phase2_findings() -> dict:
    """Cross-check Phase-2 model-error findings against what the AUDIT knows
    about the same references. A finding is 'likely-false-positive' when the
    flagged property actually originates in the (unaccented/odd) reference,
    or when the pattern is a legitimate convention."""
    ab = {(r["direction"], int(r["idx"]), r["config"]): r
          for r in csv.DictReader(AB_CSV.open(encoding="utf-8"))}
    audit_pairs = {int(r["pair_idx"]): r for r in csv.DictReader(AUDIT_CSV.open(encoding="utf-8"))}
    phase2 = list(csv.DictReader((OUT / "translation_error_analysis.csv").open(encoding="utf-8")))

    # Map A/B (direction, idx) to dev.997 indices: the A/B 30 pairs are a
    # seeded sample of dev; recompute the mapping the same way the sampler did.
    en_full = (ROOT / "data" / "flores_eval" / "dev.997.en.txt").read_text("utf-8").splitlines()
    idx_list = list(range(len(en_full)))
    random.Random(42).shuffle(idx_list)
    sel = idx_list[:30]

    verdict: Counter = Counter()
    fp_examples: list[dict] = []
    plausible: list[dict] = []
    for f in phase2:
        if f["kind"] == "reference":
            continue
        if f["direction"] != "eng_yor":
            verdict[f["category"] + "|n/a-reverse"] += 1
            continue
        dev_idx = sel[int(f["idx"])]
        aref = audit_pairs.get(dev_idx)
        low_chrf_ref_issue = aref and aref["category"] in (
            "REFERENCE_UNACCENTED", "LOW_TONE_MARK_COVERAGE", "MIXED_UNICODE_FORMS")

        if f["category"] == "LOW_CHRF" and low_chrf_ref_issue:
            verdict["LOW_CHRF|ref-attributable"] += 1
            fp_examples.append({"category": f["category"], "idx": f["idx"],
                                "reason": "reference itself unaccented/mixed-form"})
        elif f["category"] in ("UNTRANSLATED",):
            verdict[f["category"] + "|plausible"] += 1
            plausible.append({"category": f["category"], "idx": f["idx"],
                              "detail": f["detail"][:80]})
        elif f["category"] in ("NUMBER_CHANGED", "NUMBER_MISSING"):
            # if the reference also lost/differs the digits, model followed ref
            if any(aref and aref["category"].startswith("NUM_") for aref in [audit_pairs.get(dev_idx)]):
                verdict[f["category"] + "|ref-followed"] += 1
            else:
                verdict[f["category"] + "|plausible"] += 1
                plausible.append({"category": f["category"], "idx": f["idx"],
                                  "detail": f["detail"][:80]})
        else:
            verdict[f["category"] + "|plausible"] += 1

    return {
        "verdict_counts": dict(verdict),
        "false_positive_examples": fp_examples[:20],
        "plausible_examples": plausible[:20],
    }


# --------------------------------------------------------------------------
# human review sample
# --------------------------------------------------------------------------

REVIEW_FIELDS = [
    "pair_idx", "sampling_reasons", "english_source", "yoruba_reference",
    "nllb_beams2_eng_yor", "nllb_beams4_eng_yor",
    "nllb_beams2_yor_eng", "nllb_beams4_yor_eng",
    "reference_accuracy", "meaning_preservation", "grammar_naturalness",
    "diacritic_correctness", "numerical_consistency", "named_entity_consistency",
    "reviewer_comments", "confidence", "final_review_status",
]


def build_review_sample(target: int = 100, seed: int = 20261109) -> list[dict]:
    pairs = [
        (i, en, yo)
        for i, (en, yo) in enumerate(zip(
            (ROOT / "data" / "flores_eval" / "dev.997.en.txt").read_text("utf-8").splitlines(),
            (ROOT / "data" / "flores_eval" / "dev.997.yo.txt").read_text("utf-8").splitlines(),
            strict=True))
    ]
    audit = list(csv.DictReader(AUDIT_CSV.open(encoding="utf-8")))

    # A/B predictions mapped onto dev indices (same seeded selection as A/B)
    en_full = [p[1] for p in pairs]
    idx_list = list(range(len(pairs)))
    random.Random(42).shuffle(idx_list)
    sel = idx_list[:30]
    ab_rows = list(csv.DictReader(AB_CSV.open(encoding="utf-8")))
    pred_for = {}  # (dev_idx, direction, config) -> prediction
    for pos, dev_idx in enumerate(sel):
        for r in ab_rows:
            if int(r["idx"]) == pos:
                pred_for[(dev_idx, r["direction"], r["config"])] = r["prediction"]

    reasons: dict[int, list[str]] = {}
    chosen: set[int] = set()

    def take(idx: int, reason: str) -> None:
        if len(chosen) >= target:
            return
        chosen.add(idx)
        reasons.setdefault(idx, [])
        if reason not in reasons[idx]:
            reasons[idx].append(reason)

    by_cat: dict[str, list[int]] = {}
    for r in audit:
        by_cat.setdefault(r["category"], []).append(int(r["pair_idx"]))
    for cat, cap in (("NUM_DIGIT_DIFF", 15), ("NUM_DIGITS_DROPPED", 15),
                     ("ENTITY_NOT_CARRIED", 20), ("REFERENCE_UNACCENTED", 10),
                     ("LOW_TONE_MARK_COVERAGE", 10), ("MISSING_UNDERDOT_CANDIDATE", 10),
                     ("SUSPICIOUSLY_SHORT", 5), ("ADJACENT_DUPLICATE", 5)):
        for i in sorted(set(by_cat.get(cat, [])))[:cap]:
            take(i, f"audit:{cat}")
    rng = random.Random(seed)
    for i in rng.sample(range(len(pairs)), target * 2):
        take(i, "random-unflagged")
    for i in list(chosen):
        if len(chosen) < target and not reasons.get(i):
            pass
    # fill remainder randomly
    while len(chosen) < target:
        i = rng.randrange(len(pairs))
        take(i, "random-fill")

    out = []
    for i in sorted(chosen):
        _, en, yo = pairs[i]
        out.append({
            "pair_idx": i,
            "sampling_reasons": ";".join(reasons.get(i, ["random-fill"])),
            "english_source": en,
            "yoruba_reference": yo,
            "nllb_beams2_eng_yor": pred_for.get((i, "eng_yor", "beams=2"), ""),
            "nllb_beams4_eng_yor": pred_for.get((i, "eng_yor", "beams=4"), ""),
            "nllb_beams2_yor_eng": "",   # reverse uses the EN sentence as source;
            "nllb_beams4_yor_eng": "",   # yor_eng predictions belong to A/B yor sources
            "reference_accuracy": "", "meaning_preservation": "",
            "grammar_naturalness": "", "diacritic_correctness": "",
            "numerical_consistency": "", "named_entity_consistency": "",
            "reviewer_comments": "", "confidence": "", "final_review_status": "",
        })
    # attach reverse-direction predictions where this pair's YO was used as an
    # A/B yor_eng source
    for row in out:
        i = row["pair_idx"]
        row["nllb_beams2_yor_eng"] = pred_for.get((i, "yor_eng", "beams=2"), "")
        row["nllb_beams4_yor_eng"] = pred_for.get((i, "yor_eng", "beams=4"), "")
    return out


def main() -> None:
    # 1. normalization experiment
    chrf_rows = run_normalization_experiment()
    chrf_path = OUT / "chrf_normalization_comparison.csv"
    with chrf_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["direction", "config", "regime", "chrf"])
        w.writeheader()
        w.writerows(chrf_rows)

    # 2. phase-2 validation
    validation = validate_phase2_findings()

    # 3. review sample
    review = build_review_sample()
    review_path = OUT / "flores_reference_review.csv"
    with review_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        w.writerows(review)

    meta = {
        "sacrebleu_version": SACREBLEU_VERSION,
        "chrf_kwargs": {k: v for k, v in CHRF_KWARGS.items()},
        "review_pairs": len(review),
        "validation_verdicts": validation["verdict_counts"],
        "seed": 20261109,
    }
    (OUT / "phase2b_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("chrf regimes written;", len(review), "review pairs;",
          "validation:", dict(validation["verdict_counts"]))


if __name__ == "__main__":
    main()
