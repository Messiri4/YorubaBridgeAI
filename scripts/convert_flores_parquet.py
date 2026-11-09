"""One-off converter: FLORES parquet (from the HF mirror) -> aligned plain text.

Runs in a THROWAWAY venv (scripts/README.md documents the two commands),
because the project venv's pyarrow requires numpy>=2 while the main app
pins numpy<2 for torch 2.2.2 on Intel Mac. Keeping the converter isolated
means the app's dependency graph stays untouched.

Input : single parquet per split from HF yash9439/flores200
Output: data/flores_eval/<split>.<n>.en.txt / .yo.txt (aligned, UTF-8)
        + a provenance JSON alongside.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "flores_eval"
SRC = "https://huggingface.co/datasets/yash9439/flores200/resolve/main/{split}.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq  # available only in the converter venv

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_en = OUT_DIR / f"{args.split}.{args.n}.en.txt"
    out_yo = OUT_DIR / f"{args.split}.{args.n}.yo.txt"
    if out_en.exists() and not args.force:
        print("exists:", out_en.name)
        return

    url = SRC.format(split=args.split)
    print("downloading", url, flush=True)
    data = urllib.request.urlopen(url, timeout=180).read()
    table = pq.ParquetFile(__import__("io").BytesIO(data))
    cols = table.schema.names
    en_col = next((c for c in cols if c in ("english", "eng_Latn", "en")), None)
    yo_col = next((c for c in cols if c in ("yoruba", "yor_Latn", "yo")), None)
    if not (en_col and yo_col):
        sys.exit(f"unexpected schema: {cols}")

    pairs: list[tuple[str, str]] = []
    for batch in table.iter_batches(batch_size=512, columns=[en_col, yo_col]):
        for en, yo in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
            if en and yo:
                pairs.append((en.strip(), yo.strip()))
            if len(pairs) >= args.n:
                break
        if len(pairs) >= args.n:
            break

    out_en.write_text("\n".join(e for e, _ in pairs) + "\n", encoding="utf-8")
    out_yo.write_text("\n".join(y for _, y in pairs) + "\n", encoding="utf-8")
    prov = {
        "source": "huggingface:yash9439/flores200",
        "upstream": "Meta FLORES-200 (https://github.com/facebookresearch/flores), CC-BY-SA 4.0",
        "split": args.split,
        "pairs": len(pairs),
        "columns": [en_col, yo_col],
        "license": "CC-BY-SA 4.0",
    }
    (OUT_DIR / f"{args.split}.{args.n}.provenance.json").write_text(json.dumps(prov, indent=2), "utf-8")
    print(f"wrote {len(pairs)} aligned pairs -> {out_en.name}, {out_yo.name}")


if __name__ == "__main__":
    main()
