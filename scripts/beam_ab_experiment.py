"""Decoding A/B experiment: num_beams=4 vs num_beams=2 on identical pairs.

Same 30 dev-split pairs, both directions, all other parameters identical
(loaded model instance reused; overrides disabled; greedy else default).
Reports corpus chrF/BLEU, total time, mean and MEDIAN latency per config.

Outputs (timestamped, prior results preserved):
  data/eval/beams_ab_<stamp>.csv          per-sentence predictions + timings
  data/eval/beams_ab_<stamp>.summary.json config + scores + latency stats
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from app.services.evaluation import score_direction
from app.services.translation import PhraseOverrides, TranslationService

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "flores_eval"
OUT = ROOT / "data" / "eval"


def run_config(
    service: TranslationService,
    direction: str,
    srcs: list[str],
    refs: list[str],
    beams: int,
    rows_out: list[dict],
) -> dict:
    """Translate with a temporary beam setting, restore production setting after."""
    from app.core.config import get_settings

    settings = get_settings()
    original = settings.translation_num_beams
    settings.translation_num_beams = beams  # instance attr; restored below
    try:
        src_code, tgt_code = direction.split("_")
        hyps, lats = [], []
        for src in srcs:
            t0 = time.perf_counter()
            r = service.translate(src, src_code, tgt_code)
            lats.append(round(time.perf_counter() - t0, 3))
            hyps.append(r.translated_text)
        entries = [
            type("E", (), {
                "id": f"{direction}-{i}", "direction": direction, "source": s,
                "reference": ref, "hypothesis": h, "verified": True, "latency_ms": l * 1000,
            })
            for i, (s, ref, h, l) in enumerate(zip(srcs, refs, hyps, lats, strict=True))
        ]
        scores = score_direction(entries)
        for i, (s, ref, h, l) in enumerate(zip(srcs, refs, hyps, lats, strict=True)):
            rows_out.append({
                "config": f"beams={beams}", "direction": direction, "idx": i,
                "source": s, "reference": ref, "prediction": h, "latency_s": l,
            })
        return {
            "chrf": round(scores.chrf, 2),
            "bleu": round(scores.bleu, 2),
            "total_s": round(sum(lats), 1),
            "mean_s": round(statistics.mean(lats), 2),
            "median_s": round(statistics.median(lats), 2),
            "max_s": round(max(lats), 2),
        }
    finally:
        settings.translation_num_beams = original  # production value untouched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--beams", type=int, nargs="+", default=[4, 2])
    args = ap.parse_args()

    from app.services.model_loader import load_nllb

    en = (DATA / f"{args.split}.{args.n}.en.txt").read_text(encoding="utf-8").splitlines()
    yo = (DATA / f"{args.split}.{args.n}.yo.txt").read_text(encoding="utf-8").splitlines()
    assert len(en) == len(yo) == args.n

    loaded = load_nllb()
    service = TranslationService(loaded)
    service._overrides = PhraseOverrides(  # noqa: SLF001 - raw-model mode
        str(ROOT / "data" / "phrases" / "__none__.json")
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict] = []
    results: dict = {
        "split": args.split, "pairs": args.n, "ran_at": stamp,
        "model": "nllb-200-distilled-600M (local safetensors)",
        "configs": {},
    }

    for beams in args.beams:
        for direction, srcs, refs in (("eng_yor", en, yo), ("yor_eng", yo, en)):
            r = run_config(service, direction, srcs, refs, beams, rows)
            results["configs"].setdefault(f"beams={beams}", {})[direction] = r
            print(f"beams={beams} {direction}: chrF={r['chrf']} BLEU={r['bleu']} "
                  f"total={r['total_s']}s mean={r['mean_s']}s median={r['median_s']}s", flush=True)

    csv_path = OUT / f"beams_ab_{stamp}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    results["csv"] = csv_path.name
    (OUT / f"beams_ab_{stamp}.summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("wrote", csv_path.name, flush=True)


if __name__ == "__main__":
    main()
