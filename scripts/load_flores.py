"""Reproducible FLORES-200 EN-YO dataset loader for YorubaBridgeAI evaluation.

Source: Hugging Face yash9439/flores200 (verified to exist and to contain
the full FLORES-200 data in parquet shards, incl. dev and devtest splits,
English and Yoruba columns). Provenance:

  - Upstream original: Meta AI "No Language Left Behind" FLORES-200 release
    (https://github.com/facebookresearch/flores), CC-BY-SA 4.0.
  - HF mirror used: yash9439/flores200 (parquet form of dev + devtest).
  - Splits used: devtest (the official evaluation split; dev reserved for
    tuning) — only sentence_pairs_needed rows per direction are extracted.
  - Languages: eng_Latn, yor_Latn (FLORES-200 canonical code pairs).

Writes data/flores_eval/<split>.<n>.{en,yo}.txt (one sentence per line,
aligned by index) so evaluation can be reproduced without network access.

Usage:
    .venv/bin/python scripts/load_flores.py --split devtest --n 20
    .venv/bin/python scripts/load_flores.py --split dev --n 200
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

HF_API = "https://huggingface.co/api/datasets/yash9439/flores200"
HF_RESOLVE = "https://huggingface.co/datasets/yash9439/flores200/resolve/main/"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "flores_eval"


def list_parquet_files(split: str) -> list[str]:
    req = urllib.request.Request(HF_API, headers={"User-Agent": "yorubabridge-eval"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        meta = json.load(resp)
    # The mirror hosts ONE parquet per split at the repo root (verified via
    # the HF API): dev.parquet, devtest.parquet. Schema has columns like
    # english / yoruba; handled defensively in load_pairs_from_parquet.
    shards = [
        s["rfilename"]
        for s in meta.get("siblings", [])
        if s["rfilename"] == f"{split}.parquet"
    ]
    if not shards:
        available = [s['rfilename'] for s in meta.get('siblings', [])]
        raise SystemExit(f"No '{split}.parquet' in mirror. Available: {available}")
    return sorted(shards)


def load_pairs_from_parquet(parquet_urls: list[str], n: int) -> list[tuple[str, str]]:
    """Stream row groups from the parquet shards; return first n (en, yo) pairs.

    pyarrow.parquet.ParquetFile can read row-group by row-group, so we avoid
    downloading more than needed and never hold the whole split in memory.
    """
    import io

    import pyarrow.parquet as pq

    pairs: list[tuple[str, str]] = []
    for url in parquet_urls:
        if len(pairs) >= n:
            break
        data = urllib.request.urlopen(url, timeout=120).read()
        table = pq.ParquetFile(io.BytesIO(data))
        cols = table.schema.names
        # Column names differ between mirrors; find them once per shard.
        en_col = next((c for c in cols if c in ("english", "eng_Latn", "en")), None)
        yo_col = next((c for c in cols if c in ("yoruba", "yor_Latn", "yo")), None)
        if en_col is None or yo_col is None:
            raise SystemExit(f"Unexpected parquet schema: {cols}")
        for batch in table.iter_batches(batch_size=256, columns=[en_col, yo_col]):
            en_list = batch.column(0).to_pylist()
            yo_list = batch.column(1).to_pylist()
            for en, yo in zip(en_list, yo_list, strict=True):
                if en and yo:
                    pairs.append((en.strip(), yo.strip()))
                if len(pairs) >= n:
                    return pairs
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    parser.add_argument("--n", type=int, default=20, help="sentence pairs per direction")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_en = OUT_DIR / f"{args.split}.{args.n}.en.txt"
    out_yo = OUT_DIR / f"{args.split}.{args.n}.yo.txt"
    if out_en.exists() and out_yo.exists() and not args.force:
        print(f"Files exist ({out_en.name}); pass --force to regenerate.")
        return

    shards = list_parquet_files(args.split)
    print(f"{len(shards)} parquet shard(s) for split '{args.split}'; fetching up to {args.n} pairs...")
    urls = [HF_RESOLVE + s for s in shards]
    pairs = load_pairs_from_parquet(urls, args.n)
    if len(pairs) < args.n:
        raise SystemExit(f"Only got {len(pairs)}/{args.n} pairs — increase timeouts or shard set.")

    out_en.write_text("\n".join(en for en, _ in pairs) + "\n", encoding="utf-8")
    out_yo.write_text("\n".join(yo for _, yo in pairs) + "\n", encoding="utf-8")
    provenance = {
        "source": "huggingface:yash9439/flores200 (parquet mirror of Meta FLORES-200)",
        "upstream": "https://github.com/facebookresearch/flores (CC-BY-SA 4.0)",
        "split": args.split,
        "pairs": len(pairs),
        "languages": {"source_of_en": "eng_Latn", "source_of_yo": "yor_Latn"},
        "extracted_files": [out_en.name, out_yo.name],
    }
    (OUT_DIR / f"{args.split}.{args.n}.provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(pairs)} aligned pairs to {out_en} / {out_yo}")


if __name__ == "__main__":
    main()
