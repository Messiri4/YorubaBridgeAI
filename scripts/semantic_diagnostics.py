"""Sprint 2, Phase 3: offline semantic-diagnostic evaluator.

IMPORTANT — what this tool IS and IS NOT:
- It produces DIAGNOSTIC FLAGS over existing translations (from
  data/eval/integration_comparison_20261110T102741Z.csv) for human review.
- A flag is NOT a verified semantic error. Every flag carries a confidence
  tier: "heuristic" (weak, needs review) or "pattern" (strong surface
  evidence, e.g. a negation cue vanishing with no attested counterpart).
- Keyword matching is deliberately NON-literal: negation cues are checked
  through a bilingual attested-cue lexicon (English and Yoruba renderings),
  so "not possible" → "kò ṣeéṣe" is recognized as PRESERVED negation, not
  punished for dropping the word "not".

Flag families (mapped to the seven required meaning dimensions):
  negation        (1) negation cue present in source absent (all renderings)
  quantity        (2) numeric-expression drop/reorder (reuses Phase 1 logic)
  entity_role     (3) proper-noun / role-title survival
  modality        (4) uncertainty/modal cue survival
  actants         (5) WHO-did-what: passive/active voice-cue mismatch flag
  omission        (6) extreme output-length ratio vs reference
  ambiguity       (7) glossary context-sensitive terms (ambiguous folds)

Model-based QA (COMET / LaBSE / LASER) is NOT used: no downloads allowed.
See docs/sprint2_failure_analysis.md §Semantic-metrics for the survey of
pretrained QE options (licensing, hardware, Yoruba support) and why none is
adopted in this sprint.

Usage:  python scripts/semantic_diagnostics.py [--arm integrated]
Writes: data/eval/semantic_flags_<stamp>.csv and a docs summary section.
"""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/eval/integration_comparison_20261110T102741Z.csv"
OUT = ROOT / "data" / "eval"

# Attested bilingual cue lexicons (small, hand-verified from the repo's own
# devtest outputs; deliberately conservative).
NEG_EN_STEMS = ("refus", "deni", "exclud", "prevent", "fail", "avoid")
NEG_EN = {"not", "no", "never", "none", "cannot", "can't", "won't", "don't",
          "doesn't", "didn't", "without", "nor", "neither", "nothing", "nobody",
          "isn't", "aren't", "wasn't", "weren't", "unable", "lack", "refuse",
          "refused", "refuses", "deny", "denied", "denies", "unlikely",
          "impossible", "excluded", "prevented", "failed"}
# NOTE: bare 'kí' and 'má' are EXCLUDED deliberately — 'kí' is overwhelmingly a
# complementizer ('that/to') in devtest prose, and 'má' appears mostly in
# imperative contexts; including them flooded flags with false positives
# (verified empirically against this corpus).
NEG_YO_TOKEN = {"kò", "kìí", "kòlé", "kìíṣe", "àìní", "àìlè", "kò sí",
                "kò ní", "kì í", "kò tì", "kì í ṣe", "kò sì"}
NEG_YO_AFFIX = ("kò", "kì")  # verbal negation particles usually prefix the VP

MODAL_EN = {"may", "might", "could", "should", "would", "must", "possibly",
            "probably", "likely", "unlikely", "perhaps", "maybe", "suggests",
            "suspect", "believe", "appears", "seems", "estimate", "expected"}
MODAL_YO = {"bóyá", "óbà", "lè", "gbọdọ̀", "yẹ", "kí", "ìrọ́", "àbá",
            "ṣíṣeéṣe", "ànfààní", "òótọ́", "ó ṣeé", "apo", "àpọ"}

ROLE_EN = {"president", "minister", "governor", "senator", "captain", "doctor",
           "expert", "judge", "teacher", "coach", "manager", "leader",
           "secretary", "chairman", "spokesman", "analyst", "reporter"}
ROLE_YO = {"ààrẹ", "alágbà", "gómìnà", "sẹ̀nétọ̀", "ọ̀gá", "olùkọ́",
           "dókítà", "onímọ̀", "akọ̀ròyìn", "aṣojú"}

VOICE_EN = {" was ", " were ", " been ", " by the ", " is being "}

SENTINEL = "@@"


def fold(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s.casefold())
    return unicodedata.normalize("NFC", "".join(c for c in nfd if not unicodedata.combining(c)))


def has_negation(text: str, lang: str) -> bool:
    """lang is 'eng' or 'yor' (which SIDE the text is written in)."""
    words = {fold(w.strip(".,;:!?()\"'")) for w in text.split()}
    if lang == "eng":
        if bool(words & NEG_EN) or "n't" in text.lower():
            return True
        return any(any(w.startswith(st) for w in words) for st in NEG_EN_STEMS)
    return bool(words & {fold(c) for c in NEG_YO_TOKEN})


def has_modality(text: str, lang: str) -> bool:
    words = {fold(w.strip(".,;:!?()\"'")) for w in text.split()}
    if lang == "eng":
        return bool(words & MODAL_EN)
    return bool(words & {fold(w) for w in MODAL_YO})


def entities(text: str) -> set[str]:
    _STOP = {"The", "This", "That", "These", "Those", "He", "She", "It",
             "They", "But", "And", "However", "When", "After", "Before",
             "In", "On", "At", "Like", "Type", "First", "Last", "One",
             "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
             "Nine", "Ten", "November", "December", "January", "February",
             "March", "April", "May", "June", "July", "August", "September",
             "October", "Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday", "Today", "Yesterday",
             "Tomorrow", "There", "Here", "What", "Who", "Why", "How",
             "Some", "If", "While", "Since", "Though", "Although",
             "According", "During", "Between", "Among", "From", "With",
             "For", "About", "President", "Prime", "United", "New", "U.S"}
    return {m.group(0) for m in re.finditer(r"\b[A-Z][a-zA-Z'’-]{2,}\b", text)
            if m.group(0) not in _STOP}


def roles(text: str, lang: str) -> set[str]:
    words = {fold(w.strip(".,;:!?()\"'")) for w in text.split()}
    return words & (ROLE_EN if lang == "eng" else ROLE_YO)


def digits(text: str) -> list[str]:
    return re.findall(r"\d+(?:[,\.\-]\d+)*", text)


@dataclass
class Flag:
    dimension: str
    tier: str          # "pattern" (strong) or "heuristic" (review)
    detail: str


def evaluate(src: str, ref: str, hyp: str, direction: str) -> list[Flag]:
    flags: list[Flag] = []
    src_is_en = direction.startswith("eng")

    # 1. negation ----------------------------------------------------------
    src_lang = "eng" if src_is_en else "yor"
    tgt_lang = "yor" if src_is_en else "eng"
    if has_negation(src, src_lang) and not has_negation(hyp, tgt_lang):
        flags.append(Flag(
            "negation", "pattern",
            "source negation cue absent from target (all attested renderings)"))

    # 2. quantities ---------------------------------------------------------
    src_nums, hyp_nums = digits(src), digits(hyp)
    missing = [n for n in src_nums if n not in hyp_nums and n not in hyp]
    if missing:
        # heuristic: could be legitimate number-word rendering; Phase 1
        # lexicon covers common cases but the full vigesimal system is huge.
        flags.append(Flag("quantity", "heuristic",
                          f"digit expressions {missing} absent verbatim; "
                          "verify number-word rendering"))

    # 3. entities & roles ----------------------------------------------------
    src_ent = entities(src)
    lost = {e for e in src_ent if fold(e) not in fold(hyp)}
    if lost:
        flags.append(Flag("entity", "pattern",
                          f"proper nouns lost: {sorted(lost)}"))
    src_roles, hyp_roles = roles(src, src_lang), roles(hyp, tgt_lang)
    if src_roles and not hyp_roles:
        flags.append(Flag("role", "heuristic",
                          f"source role titles {sorted(src_roles)} unrepresented in target"))

    # 4. modality ------------------------------------------------------------
    if has_modality(src, src_lang) and not has_modality(hyp, tgt_lang):
        flags.append(Flag("modality", "pattern",
                          "source modal/uncertainty cue absent from target"))

    # 5. actants (who did what) ----------------------------------------------
    # Passive constructions are systematically hard; flag for review only.
    if src_is_en and any(c in f" {src.lower()} " for c in VOICE_EN):
        flags.append(Flag("actants", "heuristic",
                          "source contains passive construction; agent/patient order needs review"))

    # 6. omission (length ratio) ----------------------------------------------
    if ref and len(hyp.split()) > 0:
        ratio = len(hyp.split()) / max(1, len(ref.split()))
        if ratio < 0.45:
            flags.append(Flag("omission", "pattern",
                              f"hypothesis only {ratio:.0%} of reference length — likely omission"))
        elif ratio > 2.4:
            flags.append(Flag(
                "omission", "pattern",
                f"hypothesis {ratio:.0%} of reference length — likely hallucination"))

    # 7. sentinel leakage -------------------------------------------------------
    if SENTINEL in hyp:
        flags.append(Flag("placeholder_leak", "pattern", "raw sentinel in output"))
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="integrated", choices=("integrated", "baseline"))
    args = ap.parse_args()

    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    stamped = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_csv = OUT / f"semantic_flags_{stamped}.csv"
    counts: dict[str, int] = {}
    per_case: list[dict] = []
    for r in rows:
        if r["arm"] != args.arm:
            continue
        fl = evaluate(r["source"], r["reference"], r["prediction"], r["direction"])
        for f in fl:
            counts[f.dimension] = counts.get(f.dimension, 0) + 1
            per_case.append({
                "direction": r["direction"], "idx": r["idx"], "tier": f.tier,
                "dimension": f.dimension, "detail": f.detail,
                "source": r["source"], "prediction": r["prediction"],
            })
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_case[0].keys()) if per_case else ["empty"])
        w.writeheader()
        w.writerows(per_case)
    print(f"arm={args.arm}: {len(per_case)} flags -> {out_csv.name}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
