from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import sacrebleu
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.model_loader import load_nllb


MODEL_ID = str(ROOT / "models" / "nllb-200-distilled-600M-safetensors")
OFFICIAL_DIR = Path(
    r"C:\Users\User\.copilot\session-state\f878f44d-c9e3-47e7-b91a-91547d3f095c"
    r"\files\flores200\flores200_dataset\devtest"
)
ENG = OFFICIAL_DIR / "eng_Latn.devtest"
YOR = OFFICIAL_DIR / "yor_Latn.devtest"
OUTPUT_DIR = ROOT / "data" / "eval" / "full_flores_devtest"


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\r\n") for line in handle]


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def translate(loaded, text: str, src: str, tgt: str, beams: int,
              max_new_tokens: int, length_penalty: float) -> tuple[str, float, bool]:
    tokenizer = loaded.tokenizer
    tokenizer.src_lang = src
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2000)
    forced_bos = tokenizer.convert_tokens_to_ids(tgt)
    start = time.perf_counter()
    with torch.inference_mode():
        generated = loaded.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_new_tokens=max_new_tokens,
            num_beams=beams,
            length_penalty=length_penalty,
        )
    latency = (time.perf_counter() - start) * 1000
    output = tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True,
    )[0]
    truncated = bool(
        generated.shape[-1] >= max_new_tokens
        or (generated.shape[-1] >= 1024)
    )
    return unicodedata.normalize("NFC", output), latency, truncated


def load_done(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    done: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = f"{row['direction']}:{row['sentence_id']}"
            done[key] = row
    return done


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beams", type=int, choices=(2, 4), required=True)
    parser.add_argument("--direction", choices=("both", "eng_yor", "yor_eng"), default="both")
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--length-penalty", dest="length_penalty", type=float, default=1.1)
    args = parser.parse_args()

    english = read_lines(ENG)
    yoruba = read_lines(YOR)
    if len(english) != 1012 or len(yoruba) != 1012:
        raise RuntimeError(f"devtest misaligned: eng={len(english)}, yor={len(yoruba)}")

    loaded = load_nllb(MODEL_ID)
    warmup = translate(
        loaded, "Hello.", "eng_Latn", "yor_Latn",
        args.beams, args.max_new_tokens, args.length_penalty,
    )
    print("warm-up:", warmup[0])

    if args.smoke:
        started = time.perf_counter()
        for index in range(args.smoke):
            translate(
                loaded, english[index], "eng_Latn", "yor_Latn",
                args.beams, args.max_new_tokens, args.length_penalty,
            )
        elapsed = time.perf_counter() - started
        print(json.dumps({"smoke_pairs": args.smoke, "elapsed_s": elapsed,
                          "estimated_full_four_runs_s": elapsed / args.smoke * 1012 * 4 * 2}))
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"beams{args.beams}_max{args.max_new_tokens}_lp{str(args.length_penalty).replace('.', '')}"
    csv_path = OUTPUT_DIR / f"flores_devtest_full_{tag}.csv"
    json_path = OUTPUT_DIR / f"flores_devtest_full_{tag}.json"
    if not csv_path.exists():
        csv_path.write_text(
            "sentence_id,direction,source,reference,prediction,latency_ms,truncated\n",
            encoding="utf-8",
        )
    done = load_done(csv_path)

    configurations = (
        ("eng_yor", english, yoruba, "eng_Latn", "yor_Latn"),
        ("yor_eng", yoruba, english, "yor_Latn", "eng_Latn"),
    )
    all_rows: list[dict[str, object]] = []
    scores: dict[str, dict[str, object]] = {}

    for direction, sources, references, src_id, tgt_id in configurations:
        if args.direction != "both" and args.direction != direction:
            continue
        predictions = []
        latencies = []
        truncated_count = 0
        existing = [row for key, row in done.items() if key.startswith(direction + ":")]
        existing_by_id = {row["sentence_id"]: row for row in existing}
        pending_rows = []

        for index, (source, reference) in enumerate(zip(sources, references), start=1):
            sentence_id = f"devtest-{index:04d}"
            if sentence_id in existing_by_id:
                row = existing_by_id[sentence_id]
                predictions.append(str(row["prediction"]))
                latencies.append(float(row["latency_ms"]))
                truncated_count += str(row["truncated"]).lower() == "true"
                continue

            prediction, latency, truncated = translate(
                loaded, source, src_id, tgt_id,
                args.beams, args.max_new_tokens, args.length_penalty,
            )
            predictions.append(prediction)
            latencies.append(latency)
            truncated_count += int(truncated)
            row = {
                "sentence_id": sentence_id,
                "direction": direction,
                "source": unicodedata.normalize("NFC", source),
                "reference": unicodedata.normalize("NFC", reference),
                "prediction": prediction,
                "latency_ms": round(latency, 3),
                "truncated": str(truncated).lower(),
            }
            pending_rows.append(row)
            all_rows.append(row)
            done[f"{direction}:{sentence_id}"] = row

            if len(pending_rows) >= 25:
                append_rows(csv_path, pending_rows)
                pending_rows.clear()

        if pending_rows:
            append_rows(csv_path, pending_rows)

        refs = [[unicodedata.normalize("NFC", ref) for ref in references]]
        hyps = [unicodedata.normalize("NFC", hyp) for hyp in predictions]
        if len(hyps) == 1012:
            scores[direction] = {
                "chrfpp": sacrebleu.corpus_chrf(hyps, refs, word_order=2).score,
                "bleu": sacrebleu.corpus_bleu(hyps, refs).score,
                "median_latency_ms": percentile(latencies, 0.50),
                "p95_latency_ms": percentile(latencies, 0.95),
                "completed_pairs": len(hyps),
                "truncated_pairs": truncated_count,
            }
        else:
            scores[direction] = {"completed_pairs": len(hyps), "partial": True,
                                 "truncated_pairs": truncated_count}

    metadata = {
        "commit": git_commit(),
        "model": "facebook/nllb-200-distilled-600M",
        "model_local_path": MODEL_ID,
        "dataset": "FLORES-200",
        "dataset_version": "official archive dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
        "split": "devtest",
        "n": 1012,
        "aligned_pairs": len(english),
        "num_beams": args.beams,
        "max_new_tokens": args.max_new_tokens,
        "length_penalty": args.length_penalty,
        "phrase_overrides": "disabled",
        "cache": "disabled",
        "span_protection": "disabled",
        "metric_settings": {
            "bleu": "sacrebleu.corpus_bleu default 13a",
            "chrF++": "sacrebleu.corpus_chrf(hyps, refs, word_order=2)",
            "unicode": "NFC",
        },
        "hardware": {
            "platform": platform.platform(),
            "device": loaded.device,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        },
        "latency_method": "time.perf_counter around model.generate only, after load and warm-up",
        "source_truncation": "input max_length=2000 characters; output heuristic generated length >= max_new_tokens",
        "scores": scores,
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
