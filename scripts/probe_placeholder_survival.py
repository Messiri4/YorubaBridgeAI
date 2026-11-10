"""Sprint 2, Phase 2: OFFLINE investigation of placeholder reliability.

No model inference: this probes what the TOKENIZER does to `@@i@@` sentinels
and adjacent real text, because restoration failures begin at tokenization.
Everything here runs against the local checkpoint tokenizer in
.model-local/nllb-600M-fixed (or the HF-cached one) with no downloads.

Findings feed docs/sprint2_failure_analysis.md §Mechanisms.

Usage:  python scripts/probe_placeholder_survival.py
Writes: data/eval/placeholder_probe_<stamp>.json
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval"

SENTINELS = ["@@0@@", "@@1@@", "@@2@@", "@@7@@", "@@12@@"]

# (label, sentence-with-sentinels) probe cases drawn from the observed failure
# classes: multiple placeholders, adjacency to punctuation, Yoruba context,
# repeated numbers, decimals/negatives, dates/times, currency.
CASES = [
    ("two_sentinels", "The score was @@0@@ at @@1@@."),
    ("sentinel_then_comma", "It costs @@0@@, and ships today."),
    ("sentinel_full_stop", "Attendance was @@0@@."),
    ("yoruba_context", "Ìdíje náà wà ní @@0@@ ní @@1@@."),
    ("yoruba_sentinel_period", "Ó ní @@0@@."),
    ("repeated_number", "He ran @@0@@ laps and @@1@@ laps, @@2@@ in total."),
    ("decimal", "The value is @@0@@ units (was @@1@@)."),
    ("negative", "Temperatures fell to @@0@@ overnight."),
    ("date_time", "On @@0@@ at @@1@@ the score was @@2@@."),
    ("currency", "Revenue reached @@0@@ while costs hit @@1@@."),
    ("sentinel_adjacent_yoruba_word", "Wọn ní @@0@@ ọmọ."),
    ("sentinel_in_quotes", "The sign read \"@@0@@\"."),
    ("sentinel_percentage", "About @@0@@ of voters agree."),
    ("three_sentinels", "From @@0@@ to @@1@@ between @@2@@ and dusk."),
]


def main() -> None:
    from transformers import NllbTokenizerFast

    local = ROOT / ".model-local/nllb-600M-fixed"
    tok = NllbTokenizerFast.from_pretrained(str(local if local.is_dir()
                                              else "facebook/nllb-200-distilled-600M"))

    report: dict = {"model_local": str(local.is_dir()), "cases": []}
    for label, text in CASES:
        for lang in ("eng_Latn", "yor_Latn"):
            tok.src_lang = lang
            ids = tok(text, return_tensors=None, add_special_tokens=True)["input_ids"]
            back = tok.decode(ids, skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)
            # per-token view of the sentinel region
            pieces = tok.convert_ids_to_tokens(ids)
            sent_tokens = [p for p in pieces if "@@" in p or p.strip("@") .isdigit()]
            report["cases"].append({
                "case": label, "src_lang": lang,
                "n_tokens": len(ids),
                "roundtrip_identical": back == text,
                "roundtrip": back,
                "sentinel_related_tokens": sent_tokens[:12],
            })

    # Aggregate: which cases round-trip losslessly in BOTH languages?
    by_case: dict[str, list] = {}
    for c in report["cases"]:
        by_case.setdefault(c["case"], []).append(c["roundtrip_identical"])
    report["summary"] = {k: all(v) for k, v in by_case.items()}

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = OUT / f"placeholder_probe_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", path.name)
    for k, v in report["summary"].items():
        print(f"  {k}: lossless={v}")
    # print interesting losses
    for c in report["cases"]:
        if not c["roundtrip_identical"]:
            print("LOSS:", c["case"], c["src_lang"], "->", repr(c["roundtrip"]))


if __name__ == "__main__":
    main()
