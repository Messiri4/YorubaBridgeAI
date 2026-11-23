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
ENG = ROOT / "data" / "flores" / "dev.eng_Latn"
YOR = ROOT / "data" / "flores" / "dev.yor_Latn"
OUTPUT_DIR = ROOT / "data" / "eval"


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\r\n") for line in handle.readlines()[:30]]


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
              max_new_tokens: int, length_penalty: float) -> tuple[str, float]:
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
    return unicodedata.normalize("NFC", output), latency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beams", type=int, choices=(2, 4), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--length-penalty", type=float, default=1.1)
    args = parser.parse_args()

    loaded = load_nllb(MODEL_ID)
    warmup = translate(
        loaded, "Hello.", "eng_Latn", "yor_Latn",
        args.beams, args.max_new_tokens, args.length_penalty,
    )
    print("warm-up:", warmup[0])

    english = read_lines(ENG)
    yoruba = read_lines(YOR)
    rows: list[dict[str, object]] = []
    scores: dict[str, dict[str, float]] = {}

    for direction, sources, references, src_id, tgt_id in (
        ("eng_yor", english, yoruba, "eng_Latn", "yor_Latn"),
        ("yor_eng", yoruba, english, "yor_Latn", "eng_Latn"),
    ):
        predictions = []
        latencies = []
        for index, (source, reference) in enumerate(zip(sources, references), start=1):
            prediction, latency = translate(
                loaded, source, src_id, tgt_id,
                args.beams, args.max_new_tokens, args.length_penalty,
            )
            predictions.append(prediction)
            latencies.append(latency)
            rows.append({
                "sentence_id": f"dev-{index:04d}",
                "direction": direction,
                "source": unicodedata.normalize("NFC", source),
                "reference": unicodedata.normalize("NFC", reference),
                "prediction": prediction,
                "latency_ms": round(latency, 3),
            })

        refs = [[unicodedata.normalize("NFC", ref) for ref in references]]
        hyps = [unicodedata.normalize("NFC", hyp) for hyp in predictions]
        scores[direction] = {
            "chrfpp": sacrebleu.corpus_chrf(hyps, refs, word_order=2).score,
            "bleu": sacrebleu.corpus_bleu(hyps, refs).score,
            "median_latency_ms": percentile(latencies, 0.50),
            "p95_latency_ms": percentile(latencies, 0.95),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"dev_beams{args.beams}_max{args.max_new_tokens}_lp{args.length_penalty}".replace(".", "")
    csv_path = OUTPUT_DIR / f"beam_eval_{tag}.csv"
    json_path = OUTPUT_DIR / f"beam_eval_{tag}.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "commit": git_commit(),
        "model": "facebook/nllb-200-distilled-600M",
        "model_local_path": MODEL_ID,
        "dataset": "FLORES-200",
        "split": "dev",
        "sampling_rule": "first 30 aligned lines",
        "n": 30,
        "source_ids": {"eng": "eng_Latn", "yor": "yor_Latn"},
        "num_beams": args.beams,
        "max_new_tokens": args.max_new_tokens,
        "length_penalty": args.length_penalty,
        "phrase_overrides": "disabled",
        "number_entity_protection": "disabled",
        "cache": "disabled",
        "hardware": {
            "platform": platform.platform(),
            "device": loaded.device,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        },
        "metric_settings": {
            "bleu": "sacrebleu.corpus_bleu default 13a",
            "chrF++": "sacrebleu.corpus_chrf(hyps, refs, word_order=2)",
            "unicode": "NFC",
        },
        "latency_method": "time.perf_counter around model.generate only, after model load and warm-up",
        "scores": scores,
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print(csv_path)


if __name__ == "__main__":
    main()
