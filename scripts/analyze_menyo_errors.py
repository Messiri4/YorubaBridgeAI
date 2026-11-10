"""Semantic-failure analysis over a MENYO baseline prediction CSV (research).

Groups sentence-level errors into diagnostic categories and writes a
bilingual-human-review CSV with blank verdict columns (human judgement is
NEVER fabricated — those columns stay empty until a reviewer fills them).

Usage:
  python -m scripts.analyze_menyo_errors data/eval/menyo_baseline_dev60_seed13_*.csv \
      [--out data/eval/menyo_error_review.csv]
"""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

# Yoruba-specific cue words for crude negation / tense-aspect diagnostics.
NEGATION_CUES_YO = {"kò", "kì", "má", "mà", "kí", "ìkọ̀"}  # negation particles
NEGATION_CUES_EN = {"not", "no", "never", "n't", "don't", "doesn't", "didn't", "won't"}
TENSE_ASPECT_YO = {"ń", "ti", "yóò", "á", "ìgbà", "tí"}  # progressive/perfect/future markers
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.lower(), re.UNICODE) if t}


def flag_pair(source: str, reference: str, prediction: str) -> dict[str, bool]:
    """Diagnostic flags. These are HEURISTICS — never proof of semantics."""
    flags: dict[str, bool] = {}
    hyp_toks = _tokens(prediction)
    ref_toks = _tokens(reference)

    # Negation: negation cue present in source/reference but missing in prediction
    neg_src = {t for t in _tokens(source) if t in NEGATION_CUES_EN or t.endswith("n't")}
    neg_ref = ref_toks & NEGATION_CUES_YO
    neg_hyp = hyp_toks & (neg_ref | neg_src)
    flags["negation_mismatch"] = bool((neg_ref or neg_src) and not neg_hyp)

    # Numbers: digit sequences differ between source and prediction
    flags["numbers_mismatch"] = (NUM_RE.findall(source) != NUM_RE.findall(prediction))

    # Named entities: capitalized tokens in source absent from prediction
    ents = {w for w in re.findall(r"\b[A-Z][a-zA-Zéẹọṣ]+\b", source)} - _caps_at_start(source)
    flags["entity_possible_loss"] = bool(ents and not any(w.lower() in prediction.lower() for w in ents))

    # Word-order / roles proxy: content-token overlap too low (crude SOV proxy)
    content = ref_toks - {"the", "a", "an", "of", "and", "to", "is", "was", "ni"}
    overlap = len(hyp_toks & content) / max(1, len(content))
    flags["low_reference_overlap"] = overlap < 0.2

    # Tense/aspect: known Yoruba aspect markers missing when reference has them
    flags["tense_aspect_flag"] = bool(ref_toks & TENSE_ASPECT_YO
                                      and not (hyp_toks & TENSE_ASPECT_YO))

    # Omission: prediction much shorter than reference
    flags["omission_suspected"] = len(hyp_toks) < 0.5 * len(ref_toks)

    # Context-sensitive meaning: prediction identical to a different domain's
    # stock phrase or reference used elsewhere — flagged only via overlap here.
    flags["context_check_needed"] = 0.0 < overlap < 0.5 and len(hyp_toks) >= 3

    # Diacritics: prediction has Yoruba-ish letters but zero tone marks, or
    # underdot letters replaced by plain ones vs reference
    if re.search(r"[àáèéẹéíìóòọṣúùẸỌṢ]", prediction) or prediction.isascii():
        tone_ratio_ref = _tone_char_ratio(reference)
        tone_ratio_hyp = _tone_char_ratio(prediction)
        flags["diacritics_suspect"] = tone_ratio_ref > 0.1 and tone_ratio_hyp < 0.3 * tone_ratio_ref

    # Repetition: any 4-gram occurring 3+ times (catches loops and run-on cycles)
    words = prediction.split()
    if len(words) >= 12:
        from collections import Counter as _C
        counts = _C(" ".join(words[i:i + 4]) for i in range(len(words) - 3))
        flags["repetition"] = any(c >= 3 for c in counts.values())

    return flags


def _caps_at_start(text: str) -> set[str]:
    first = text.split()
    return {first[0]} if first else set()


def _tone_char_ratio(text: str) -> float:
    """Fraction of characters carrying a tone mark or underdot (NFC-aware)."""
    text = unicodedata.normalize("NFC", text)
    if not text:
        return 0.0
    marked = sum(1 for ch in text if ch in "àáèéẹéíìóòọṣúùǹẸỌṢÀÁÈÉÍÌÓÒÚÙǸ")
    return marked / len(text)


REVIEW_COLUMNS = [
    "direction", "idx", "domain", "source", "reference", "prediction",
    "error_flags",
    # Blank human-review columns — filled by bilingual reviewers, never by code:
    "verdict",           # ok | minor | major | failure
    "category_verdict",  # reviewer's own category (free text)
    "notes",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prediction_csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/eval/menyo_error_review.csv"))
    args = ap.parse_args()

    rows = list(csv.DictReader(args.prediction_csv.open(encoding="utf-8")))
    flag_counts: Counter[str] = Counter()
    out_rows: list[dict] = []
    for r in rows:
        flags = flag_pair(r["source"], r["reference"], r["prediction"])
        active = sorted(k for k, v in flags.items() if v)
        flag_counts.update(active)
        out_rows.append({
            **{c: r[c] for c in REVIEW_COLUMNS[:6]},
            "error_flags": ";".join(active),
            "verdict": "", "category_verdict": "", "notes": "",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        w.writerows(out_rows)

    print("flag counts (diagnostic heuristics only, NOT ground truth):")
    for name, n in flag_counts.most_common():
        print(f"  {name}: {n} ({n / len(rows):.1%})")
    print(f"wrote {args.out} ({len(out_rows)} rows; verdict columns blank)")


if __name__ == "__main__":
    main()
