"""MENYO-20k baseline evaluation for NLLB-200-distilled-600M (research only).

What it does:
- Loads the LOCAL MENYO-20k dev split via research.menyo20k.loader (official
  split preserved; test split is never read).
- Selects a DETERMINISTIC stratified sample (seeded by dataset order, not a
  random shuffle of the file): round-robin across labeled domains, then
  unlabeled pairs — stable across runs because the underlying file order is
  fixed by the authors.
- Translates both directions (eng->yor, yor->eng) with the production
  TranslationService on the existing NLLB checkpoint, overrides disabled,
  PRODUCTION CACHING DISABLED (every call hits the model).
- Metrics per direction: SacreBLEU, chrF++, numeric preservation,
  placeholder-restoration failures, repeated-output failures, and
  mean/median/P95 latency.
- Writes: data/eval/menyo_baseline_<stamp>.csv (sentence level),
  .summary.json (metrics + full provenance), and nothing else.
Does NOT touch data/flores*, the production service config, or the test split.

Run:  python -m scripts.evaluate_menyo --n 60 [--seed 13] [--dry-run]
Before a full run, the script prints a duration estimate and requires
--yes to proceed (cost approval gate).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import sacrebleu

from research.menyo20k.loader import load_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval"

# Optional throughput constant from the repo's own prior CPU benchmarks
# (measured on this machine with the same checkpoint): ~2.5 s / sentence
# for eng->yor at beams=4. Used only for the cost-approval estimate.
SECONDS_PER_SENTENCE_CPU = 2.5


def stratified_sample(pairs, n: int, seed: int):
    """Deterministic stratified sample: interleave by domain, then fixed order.

    Same (n, seed) + same dataset file => same ids. Seed currently selects the
    rotation offset so different seeds give different but reproducible sets.
    """
    labeled: dict[str, list] = {}
    unlabeled: list = []
    for p in pairs:
        (labeled.setdefault(p.domain or "unlabeled", [])).append(p)
    buckets = [labeled[k] for k in sorted(labeled) if k != "unlabeled"]
    buckets.append(labeled.get("unlabeled", []))
    for b in buckets:
        if b:
            b[:] = b[seed % len(b):] + b[: seed % len(b)]
    # Round-robin drain. NOTE: iterators stay truthy after exhaustion, so we
    # track emptiness with a sentinel, not bool(iter).
    _EMPTY = object()
    pooled: list = []
    iters = [iter(b) for b in buckets]
    while iters:
        for it in list(iters):
            nxt = next(it, _EMPTY)
            if nxt is _EMPTY:
                iters.remove(it)
            else:
                pooled.append(nxt)
        if len(pooled) >= n:
            break
    return pooled[:n]


def numeric_preserved(src: str, hyp: str) -> bool:
    import re
    src_nums = re.findall(r"\d+(?:[.,]\d+)?", src)
    hyp_nums = re.findall(r"\d+(?:[.,]\d+)?", hyp)
    return src_nums == hyp_nums


PLACEHOLDER_TOKENS = ("<num>", "<url>", "<email>", "<cur>")


def placeholder_failures(src: str, hyp: str) -> int:
    """Count protection sentinels that vanished or mutated (1 = failure)."""
    import re
    for tok in PLACEHOLDER_TOKENS:
        src_n, hyp_n = len(re.findall(re.escape(tok), src)), len(re.findall(re.escape(tok), hyp))
        if src_n and hyp_n != src_n:
            return 1
    return 0


def repeated_output(hyp: str) -> bool:
    """Heuristic repetition failure: a 4+-gram repeated 3+ times consecutively."""
    words = hyp.split()
    if len(words) < 12:
        return False
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    for i in range(len(grams) - 2):
        if grams[i] == grams[i + 1] == grams[i + 2]:
            return True
    return False


def latency_stats(lat: list[float]) -> dict:
    s = sorted(lat)
    p95 = s[max(0, int(round(0.95 * len(s))) - 1)]
    return {
        "mean_s": round(statistics.fmean(lat), 3),
        "median_s": round(statistics.median(lat), 3),
        "p95_s": round(p95, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60, help="pairs per direction")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--beams", type=int, default=None, help="override decoding; default = production settings")
    ap.add_argument("--yes", action="store_true", help="approve the estimated runtime")
    ap.add_argument("--dry-run", action="store_true", help="show plan and exit without loading the model")
    args = ap.parse_args()

    dev_pairs, dev_stats = load_split("dev")
    sample = stratified_sample(dev_pairs, args.n, args.seed)
    est_minutes = args.n * 2 * SECONDS_PER_SENTENCE_CPU / 60
    plan = {
        "split": "dev (official MENYO-20k dev)",
        "pairs_per_direction": len(sample),
        "directions": ["eng->yor", "yor->eng"],
        "estimated_minutes_cpu": round(est_minutes, 1),
        "seed": args.seed,
        "caching": "disabled",
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        print("dry-run: exiting before model load")
        return
    if not args.yes:
        print(f"Estimated ~{est_minutes:.0f} min on CPU. Re-run with --yes to approve.")
        return

    from app.core.config import get_settings
    from app.services.model_loader import load_nllb
    from app.services.translation import PhraseOverrides, TranslationService

    get_settings.cache_clear()
    settings = get_settings()
    if args.beams is not None:
        settings.translation_num_beams = args.beams

    empty_table = PhraseOverrides(str(ROOT / "data" / "phrases" / "__none__.json"))
    loaded = load_nllb()
    service = TranslationService(loaded)
    service._overrides = empty_table  # noqa: SLF001 - raw-model evaluation
    if hasattr(service, "_cache_enabled"):
        service._cache_enabled = False  # noqa: SLF001
    print(f"model on {loaded.device}; overrides+cache disabled", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = OUT / f"menyo_baseline_dev{args.n}_seed{args.seed}_{stamp}.csv"
    rows: list[dict] = []
    summaries: dict[str, dict] = {}

    for direction, srcs, refs in (
        ("eng_yor", [p.en for p in sample], [p.yo for p in sample]),
        ("yor_eng", [p.yo for p in sample], [p.en for p in sample]),
    ):
        src_code, tgt_code = direction.split("_")
        hyps, lat = [], []
        for src in srcs:
            t0 = time.perf_counter()
            r = service.translate(src, src_code, tgt_code)
            lat.append(round(time.perf_counter() - t0, 3))
            hyps.append(r.translated_text)
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2, char_order=6)  # chrF++
        num_ok = [numeric_preserved(s, h) for s, h in zip(srcs, hyps, strict=True)]
        ph_fail = [placeholder_failures(s, h) for s, h in zip(srcs, hyps, strict=True)]
        rep = [repeated_output(h) for h in hyps]
        summaries[direction] = {
            "sacrebleu": round(bleu.score, 2),
            "chrfpp": round(chrf.score, 2),
            "numeric_preservation_rate": round(sum(num_ok) / len(num_ok), 4),
            "placeholder_failure_rate": round(sum(ph_fail) / len(ph_fail), 4),
            "repeated_output_rate": round(sum(rep) / len(rep), 4),
            "latency": latency_stats(lat),
        }
        print(direction, summaries[direction], flush=True)
        for i, (s, rf, h, l) in enumerate(zip(srcs, refs, hyps, lat, strict=True)):
            rows.append({
                "direction": direction, "idx": i, "pair_id": f"{sample[i].split}:{sample[i].idx}",
                "domain": sample[i].domain or "", "source": s, "reference": rf,
                "prediction": h, "latency_s": l,
                "numeric_ok": int(num_ok[i]), "placeholder_fail": int(ph_fail[i]),
                "repeated": int(rep[i]),
            })

    import subprocess
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=ROOT).stdout.strip()
    summary = {
        "ran_at": stamp,
        "dataset": {
            "name": "MENYO-20k", "version": "master @ download date 2026-11-10",
            "source": "https://github.com/uds-lsv/menyo-20k_MT",
            "license": "CC BY-NC 4.0 (non-commercial)",
            "split": "dev", "loader_rejections": dev_stats.rejected,
            "sha256": (ROOT / "data/menyo20k/SHA256SUMS").read_text().splitlines()[0][:16],
        },
        "git_commit": git_commit,
        "model_checkpoint": settings.translation_model_id,
        "decoding": {
            "num_beams": settings.translation_num_beams,
            "length_penalty": settings.translation_length_penalty,
            "max_new_tokens": settings.translation_max_new_tokens,
            "sampling": "greedy default (deterministic)",
        },
        "random_seed": args.seed,
        "sample_ids": [f"{p.split}:{p.idx}" for p in sample],
        "metric_config": {"bleu": "sacrebleu corpus_bleu (13a tokenizer)",
                          "chrfpp": "chrF++ (char_order=6, word_order=2)"},
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform()},
        "results": summaries,
        "csv": csv_path.name,
    }
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / f"menyo_baseline_dev{args.n}_seed{args.seed}_{stamp}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", csv_path.name)


if __name__ == "__main__":
    main()
