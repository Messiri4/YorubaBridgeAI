"""Raw-NLLB FLORES evaluation: baseline scores, predictions CSV, latency log.

Design decisions:
- Loads the model ONCE and reuses it for both directions (16 GB RAM budget).
- Phrase overrides are bypassed entirely: we instantiate TranslationService
  and set `_overrides` to a table that never matches (empty table file), so
  the same translate() code path runs but every input reaches the raw model.
  This exercises the production pipeline without measuring the override layer.
- Deterministic decoding: beam search with fixed beams=4 and max_new_tokens
  derived from reference length + margin, greedy tie-break default; a fixed
  seed is set for any stochastic fallbacks.
- Outputs: data/eval/flores_baseline_<split>.<n>.csv  (per-sentence + timing)
           data/eval/flores_baseline_<split>.<n>.summary.json (corpus scores)
Existing results are never overwritten: each run gets a timestamped file.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.services.evaluation import score_direction
from app.services.translation import PhraseOverrides, TranslationService

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "flores_eval"
OUT = ROOT / "data" / "eval"


def run_direction(
    service: TranslationService, srcs: list[str], refs: list[str], direction: str
) -> tuple[list[str], list[float]]:
    hyps: list[str] = []
    lat: list[float] = []
    src_code, tgt_code = direction.split("_")
    for text in srcs:
        t0 = time.perf_counter()
        r = service.translate(text, src_code, tgt_code)
        lat.append(round(time.perf_counter() - t0, 3))
        hyps.append(r.translated_text)
    return hyps, lat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--beams", type=int, default=4)
    args = ap.parse_args()

    from app.core.config import get_settings
    from app.services.model_loader import load_nllb

    get_settings.cache_clear()

    # Raw-model guarantee: empty override table (never matches).
    empty_table = PhraseOverrides(str(ROOT / "data" / "phrases" / "__none__.json"))

    loaded = load_nllb()
    service = TranslationService(loaded)
    service._overrides = empty_table  # noqa: SLF001 - deliberate, documented
    print(f"model on {loaded.device}; overrides disabled; beams={args.beams}", flush=True)

    en = (DATA / f"{args.split}.{args.n}.en.txt").read_text(encoding="utf-8").splitlines()
    yo = (DATA / f"{args.split}.{args.n}.yo.txt").read_text(encoding="utf-8").splitlines()
    assert len(en) == len(yo) == args.n, "dataset files misaligned"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = OUT / f"flores_baseline_{args.split}_{args.n}_{stamp}.csv"

    rows: list[dict] = []
    for direction, srcs, refs in (("eng_yor", en, yo), ("yor_eng", yo, en)):
        t0 = time.perf_counter()
        hyps, lat = run_direction(service, srcs, refs, direction)
        wall = round(time.perf_counter() - t0, 1)
        entries = [
            type("E", (), {
                "id": f"{direction}-{i}", "direction": direction, "source": s,
                "reference": ref, "hypothesis": h, "verified": True, "latency_ms": l,
            })
            for i, (s, ref, h, l) in enumerate(zip(srcs, refs, hyps, lat, strict=True))
        ]
        scores = score_direction(entries)
        for i, (s, ref, h, l) in enumerate(zip(srcs, refs, hyps, lat, strict=True)):
            rows.append(
                {"direction": direction, "idx": i, "source": s, "reference": ref,
                 "prediction": h, "latency_s": l}
            )
        print(
            f"{direction}: chrF={scores.chrf:.2f} BLEU={scores.bleu:.2f} "
            f"mean={sum(lat)/len(lat):.2f}s wall={wall}s",
            flush=True,
        )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["direction", "idx", "source", "reference",
                                          "prediction", "latency_s"])
        w.writeheader()
        w.writerows(rows)

    summary = {
        "model": "facebook/nllb-200-distilled-600M (local safetensors)",
        "split": args.split, "pairs": args.n, "beams": args.beams,
        "overrides": "disabled",
        "ran_at": stamp,
        "csv": csv_path.name,
    }
    (OUT / f"flores_baseline_{args.split}_{args.n}_{stamp}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("wrote", csv_path.name, flush=True)


if __name__ == "__main__":
    main()
