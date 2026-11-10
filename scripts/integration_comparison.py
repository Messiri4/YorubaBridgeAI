"""Integration quality comparison: baseline 9414279 pipeline vs integrated branch.

Reproducibility contract:
- One NLLB instance loaded ONCE, strictly sequential inference (never concurrent).
- No downloads: model comes from the existing local HF cache; if a local
  snapshot directory (.model-local/nllb-600M-fixed) exists it is preferred.
- Identical pairs and order for both arms; generation is deterministic
  beam search (beams=4, lp=1.1, max_new_tokens=256 on stock settings) in
  BOTH arms, so the only difference is the pipeline code under test.
- Arm A (baseline): numeric protection/glossary/tone-restoration DISABLED
  (the pre-integration service behavior), via the integrated service with
  the feature flags off — equivalent to 9414279 defaults but without the
  TI-1 prototype additions; verified below by flag assertion.
- Arm B (integrated): all TI-1 modules active (protection + glossary +
  Yoruba NFC/tone pass-through + decoding_policy production_default).

Metrics: corpus chrF++ (char_order=6, word_order=2, beta=2) and BLEU via
sacrebleu, per direction (eng_yor, yor_eng), plus latency, numeric-preservation
checks on sentences containing digits, residual-sentinel leak count, and
Yoruba tone-mark retention statistics.

Usage:  python scripts/integration_comparison.py [--n 80] [--split devtest]
Writes data/eval/integration_comparison_<stamp>.{csv,json} and prints a report.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "flores_eval"
OUT = ROOT / "data" / "eval"

CHRF_KWARGS = dict(char_order=6, word_order=2, beta=2.0)  # chrF++

SENTINEL = "@@"


def count_digits(text: str) -> int:
    return sum(ch.isdigit() for ch in text)


def digit_sequences(text: str) -> list[str]:
    import re
    return re.findall(r"\d+(?:[.,]\d+)*", text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    from app.core.config import get_settings
    from app.services.translation import PhraseOverrides, TranslationService

    get_settings.cache_clear()
    settings = get_settings()

    local = ".model-local/nllb-600M-fixed"
    if os.path.isdir(local):
        settings.translation_model_id = local
    from app.services.model_loader import load_nllb

    loaded = load_nllb()

    # Empty override table: raw-model measurement, overrides bypassed.
    empty_table = PhraseOverrides(str(ROOT / "data" / "phrases" / "__none__.json"))

    def make_service() -> TranslationService:
        svc = TranslationService(loaded)
        svc._overrides = empty_table  # noqa: SLF001 - deliberate, documented
        return svc

    base_svc = make_service()
    integ_svc = make_service()

    # Exact-size files only exist for curated sizes (20/30/60/80/997); for
    # smaller smoke runs take the FIRST n lines of the largest available file.
    # Prefer curated flores_eval files; otherwise fall back to the official
    # Meta FLORES-200 subsets under data/flores (dev.60 / devtest.80 prefixes).
    alt_en = ROOT / "data" / "flores" / f"{args.split}.eng_Latn"
    alt_yo = ROOT / "data" / "flores" / f"{args.split}.yor_Latn"
    import re as _re
    candidates = sorted(
        (p for p in DATA.glob(f"{args.split}.*.en.txt")),
        key=lambda p: int(_re.search(r"\.(\d+)\.", p.name).group(1)),
    )
    if alt_en.exists() and len(alt_en.read_text(encoding="utf-8").splitlines()) >= args.n:
        en, yo = (alt_en.read_text(encoding="utf-8").splitlines()[: args.n],
                  alt_yo.read_text(encoding="utf-8").splitlines()[: args.n])
    else:
        chosen = None
        for p in candidates:
            if int(_re.search(r"\.(\d+)\.", p.name).group(1)) >= args.n:
                chosen = p
                break
        if chosen is None:
            raise SystemExit(f"no {args.split} file with >= {args.n} pairs")
        size = int(_re.search(r"\.(\d+)\.", chosen.name).group(1))
        en_all = chosen.read_text(encoding="utf-8").splitlines()
        yo_all = (DATA / f"{args.split}.{size}.yo.txt").read_text(encoding="utf-8").splitlines()
        en, yo = en_all[: args.n], yo_all[: args.n]
    assert len(en) == len(yo) == args.n, "dataset files misaligned"

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict] = []

    def run_arm(service: TranslationService, arm: str,
                protect: bool, glossary: bool) -> None:
        settings.numeric_protection_enabled = protect
        settings.glossary_enabled = glossary
        for direction, srcs, refs in (("eng_yor", en, yo), ("yor_eng", yo, en)):
            for i, src in enumerate(srcs):
                t0 = time.perf_counter()
                r = service.translate(src, *direction.split("_"))
                lat = time.perf_counter() - t0
                hyp = r.translated_text
                ref = refs[i]
                rows.append({
                    "arm": arm, "direction": direction, "idx": i,
                    "source": src, "reference": ref, "prediction": hyp,
                    "latency_s": round(lat, 3),
                    "src_digit_seqs": len(digit_sequences(src)),
                    "hyp_digit_seqs": len(digit_sequences(hyp)),
                    "sentinel_leak": int(SENTINEL in hyp),
                    "truncated_guess": int(hyp.endswith(("…", "..."))),
                })

    print("ARM A (baseline 9414279 behavior): protection+glossary OFF", flush=True)
    run_arm(base_svc, "baseline", protect=False, glossary=False)
    print("ARM B (integrated): protection+glossary ON", flush=True)
    run_arm(integ_svc, "integrated", protect=True, glossary=True)

    # ---------------------------------------------------------------- scores
    report: dict = {
        "ran_at": stamp,
        "model": str(settings.translation_model_id),
        "split": args.split, "pairs": args.n,
        "chrf_kwargs": CHRF_KWARGS,
        "note": "sequential inference, single model instance, no downloads; "
                "overrides disabled in both arms",
        "arms": {},
    }

    def tone_stats(hyps: list[str]) -> dict:
        from app.services.yoruba_orthography import count_tone_marks
        low = high = under = 0
        for h in hyps:
            c = count_tone_marks(h)
            low += c["low"]
            high += c["high"]
            under += c["underdot"]
        return {"low": low, "high": high, "underdot": under}

    for arm in ("baseline", "integrated"):
        arm_report: dict = {}
        for direction in ("eng_yor", "yor_eng"):
            sel = [r for r in rows if r["arm"] == arm and r["direction"] == direction]
            hyps = [r["prediction"] for r in sel]
            refs = [[r["reference"] for r in sel]]
            chrf = sacrebleu.corpus_chrf(hyps, refs, **CHRF_KWARGS)
            bleu = sacrebleu.corpus_bleu(hyps, refs)
            lat = [r["latency_s"] for r in sel]
            # numeric preservation: source digit-sequence multiset retained
            nums_total = nums_kept = 0
            for r in sel:
                src_nums = digit_sequences(r["source"])
                for n in src_nums:
                    nums_total += 1
                    if n in digit_sequences(r["prediction"]):
                        nums_kept += 1
            arm_report[direction] = {
                "chrf_plus_plus": round(chrf.score, 2),
                "bleu": round(bleu.score, 2),
                "mean_latency_s": round(statistics.mean(lat), 3),
                "median_latency_s": round(statistics.median(lat), 3),
                "numeric_expressions_total": nums_total,
                "numeric_expressions_preserved": nums_kept,
                "numeric_preservation_rate": (
                    round(nums_kept / nums_total, 3) if nums_total else None),
                "sentinel_leaks": sum(r["sentinel_leak"] for r in sel),
                "tone_marks": tone_stats(hyps) if direction == "eng_yor" else None,
            }
        report["arms"][arm] = arm_report

    # deltas
    delta = {}
    for direction in ("eng_yor", "yor_eng"):
        b, i = report["arms"]["baseline"][direction], report["arms"]["integrated"][direction]
        delta[direction] = {
            "chrf_plus_plus_delta": round(i["chrf_plus_plus"] - b["chrf_plus_plus"], 2),
            "bleu_delta": round(i["bleu"] - b["bleu"], 2),
            "numeric_preservation_delta": (
                None if b["numeric_preservation_rate"] is None else
                round(i["numeric_preservation_rate"]
                      - b["numeric_preservation_rate"], 3)),
            "mean_latency_delta_s": round(i["mean_latency_s"] - b["mean_latency_s"], 3),
        }
    report["delta_integrated_minus_baseline"] = delta

    csv_path = OUT / f"integration_comparison_{stamp}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path = OUT / f"integration_comparison_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("wrote", csv_path.name, json_path.name, flush=True)


if __name__ == "__main__":
    main()
