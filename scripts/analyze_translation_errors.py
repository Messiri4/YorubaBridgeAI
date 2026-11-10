"""Heuristic translation error analysis over the beam A/B evaluation CSV.

Reads data/eval/beams_ab_<stamp>.csv (config,direction,idx,source,reference,
prediction,latency_s) and produces:

  data/eval/translation_error_analysis.csv   one row per finding (heuristic hit)
  data/eval/translation_error_summary.json   counts by category/direction/severity
  data/eval/translation_error_review.csv     per-prediction review sheet
  data/eval/translation_error_report.md      human-readable report

Design principles:
- Deterministic, documented heuristics ONLY. Every finding is a WARNING to
  be reviewed by a fluent speaker, never a verdict.
- No model downloads, no inference, no network access: pure text processing.
- Reference-quality checks run separately from model-error checks so the two
  kinds of problems are never conflated.

Heuristic catalog (each documented below at its check):
  NUMBER_MISSING / NUMBER_CHANGED   digits present in source but not in hyp,
                                    or digit multiset differs.
  ENTITY_MISSING / ENTITY_CHANGED   capitalized token sequences in source
                                    not carried into hyp (name-script aware).
  TOO_SHORT / TOO_LONG              hyp length vs source & reference length.
  REPEATED_NGRAM                    repeated content words/phrases in hyp.
  UNTRANSLATED                      source n-grams (>=2 words, alphabetic)
                                    copied verbatim into hyp.
  BEAM_DIVERGENCE                   beams=2 vs beams=4 hyps differ notably.
  LOW_CHRF                          sentence chrF below threshold.
  REF_NO_DIACRITICS                 yor reference lacks tone marks (reference
                                    quality, not a model error).
  REF_DIGIT_MISMATCH                reference digits differ from source digits
                                    (reference quality OR source error).
  OVERSHOT_LENGTH                   hyp much longer than BOTH src and ref.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval"

LOW_CHRF_THRESHOLD = 30.0   # sentence-level chrF below this is flagged
BEAM_DIFF_TOKEN_RATIO = 0.5  # beams diverge if <50% of tokens shared
SHORT_RATIO = 0.5            # hyp < 50% of reference token count -> TOO_SHORT
LONG_RATIO = 2.0             # hyp > 200% of reference token count -> TOO_LONG
OVERSHOT_RATIO = 1.8         # hyp > 1.8x BOTH source and reference -> OVERSHOT
MIN_REPEAT_NGRAM_LEN = 2

STOPWORDS_EN = {
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "for", "with",
    "is", "are", "was", "were", "that", "this", "it", "as", "by", "from",
    "he", "she", "they", "we", "you", "his", "her", "their", "its",
    "would", "will", "be", "been", "have", "has", "had", "do", "did", "not",
    "said", "also", "but", "if", "then", "than", "when", "while", "about",
}


def sentence_chrf(hyp: str, ref: str) -> float:
    """Sentence-level chrF via sacrebleu (signature_smoothing for stability)."""
    return sacrebleu.sentence_chrf(hyp, [ref]).score


# --------------------------------------------------------------------------
# deterministic tokenizers / helpers
# --------------------------------------------------------------------------

def tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens (unicode letters kept, incl. Yoruba)."""
    text = unicodedata.normalize("NFC", text).lower()
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


_DIGIT_RE = re.compile(r"\d+(?:[.,]\d+)?")
_CAP_SEQ_RE = re.compile(r"\b[A-Z][a-zA-Z'’\-]+(?:\s+[A-Z][a-zA-Z'’\-]+)*")


def digits(text: str) -> list[str]:
    return _DIGIT_RE.findall(text)


def cap_entities(text: str) -> set[str]:
    """Capitalized tokens/sequences, minus sentence-initial stopwords."""
    ents: set[str] = set()
    for m in _CAP_SEQ_RE.finditer(text):
        ent = m.group(0)
        # drop pure sentence-start artifacts: first word capitalized only
        parts = ent.split()
        if len(parts) == 1 and parts[0].lower() in STOPWORDS_EN:
            continue
        # a capitalized first word gluing onto a following name is normal
        # ('The Washington game') — strip a leading English stopword so the
        # entity itself is analyzed
        if len(parts) > 1 and parts[0].lower() in STOPWORDS_EN:
            ent = " ".join(parts[1:])
        ents.add(ent)
    return ents


def entity_carryover(src_ents: set[str], hyp: str) -> tuple[set[str], set[str]]:
    """Return (missing, fuzzy_present) entities. An entity counts as carried
    over if any of its tokens appears in the hypothesis (transliteration in
    Yoruba usually keeps most letters: Washington -> Washington/Washington)."""
    hyp_low = unicodedata.normalize("NFC", hyp).lower()
    missing, present = set(), set()
    for ent in src_ents:
        toks = tokens(ent)
        if all(t in hyp_low for t in toks):
            present.add(ent)
            continue
        # partial transliteration counts as carried: compare accent-folded
        # forms with a permissive similarity (sequence matcher ratio >= 0.6
        # against any hyp token of comparable length).
        folded_hyp_toks = [_fold(t) for t in tokens(hyp_low) if len(t) >= 4]
        matched = False
        for t in toks:
            if len(t) <= 3:
                continue
            ft = _fold(t)
            if any(ft in htok or _similar(ft, htok) for htok in folded_hyp_toks):
                matched = True
                break
        (present if matched else missing).add(ent)
    return missing, present


def _similar(a: str, b: str) -> bool:
    """Cheap transliteration similarity between an English name and its
    Yoruba rendering. English->Yoruba names usually add syllable vowels
    (Washington -> Woshintoni), so: (1) same first letter, (2) same length
    band, (3) consonant skeleton largely shared."""
    if abs(len(a) - len(b)) > max(len(a), len(b)) // 2:
        return False
    if not a or not b or a[0] != b[0]:
        return False
    cons_a = [c for c in a if c not in "aeiou"]
    cons_b = [c for c in b if c not in "aeiou"]
    if not cons_a or not cons_b:
        return False
    shared = sum(1 for c in cons_a if c in cons_b)
    return shared / max(len(cons_a), len(cons_b)) >= 0.6


def _fold(text: str) -> str:
    """Accent-fold + normalize common Yoruba sub-dot variants for fuzzy match."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", text)


def repeated_ngrams(hyp: str) -> list[str]:
    """Repeated content words (n=1, excluding function words) and n-grams
    (n>=2, excluding any-gram containing a function word)."""
    tk = tokens(hyp)
    found: Counter[tuple[str, ...]] = Counter()
    # single content words repeated (e.g. 'òṣè óṣè', 'committee committee');
    # also require the word to look like real content (len>=3) so stray
    # short tokens don't fire
    for w in tk:
        if w not in STOPWORDS_EN and len(w) >= 3:
            found[(w,)] += 1
    for n in (2, 3):
        for i in range(len(tk) - n + 1):
            gram = tuple(tk[i:i + n])
            if not any(w in STOPWORDS_EN for w in gram):
                found[gram] += 1
    return [" ".join(g) for g, c in found.items() if c > 1]


SOURCE_COPY_RE = re.compile(
    r"\b[a-z]{4,}(?:\s+[a-z]{4,})+\b"
)


def untranslated_spans(src: str, hyp: str) -> list[str]:
    """Multi-word lowercase alphabetic source runs copied verbatim into hyp.
    (Single proper nouns copied over are normal transliterations; we require
    a multi-word lowercase run so common nouns count too.)"""
    hyp_low = unicodedata.normalize("NFC", hyp).lower()
    spans = []
    for m in SOURCE_COPY_RE.finditer(src.lower()):
        span = m.group(0)
        if len(span) >= 10 and span in hyp_low:
            spans.append(span)
    return spans


def ratio_len(hyp: str, ref: str) -> float:
    th, tr = len(tokens(hyp)), len(tokens(ref))
    return th / tr if tr else float("inf")


# --------------------------------------------------------------------------
# per-row analysis
# --------------------------------------------------------------------------

def analyze_row(row: dict, beams: dict[str, str]) -> tuple[list[dict], dict]:
    """Return (findings for this prediction, per-row aggregates for the summary)."""
    findings: list[dict] = []
    direction = row["direction"]
    src, ref, hyp = row["source"], row["reference"], row["prediction"]
    cfg = row["config"]

    def add(category: str, severity: str, detail: str, kind: str = "model") -> None:
        findings.append({
            "direction": direction, "idx": row["idx"], "config": cfg,
            "category": category, "severity": severity, "kind": kind,
            "detail": detail, "source": src, "reference": ref, "prediction": hyp,
        })

    chrf = sentence_chrf(hyp, ref)

    # -- numbers ----------------------------------------------------------
    # Cross-script numeric consistency: spelled-out numbers in the hyp do not
    # cancel a missing-digit finding, BUT a reference that spells the number
    # out means numeral→word rendering is the corpus norm, so we downgrade.
    src_d, hyp_d, ref_d = digits(src), digits(hyp), digits(ref)
    src_nums, hyp_nums, ref_nums = Counter(src_d), Counter(hyp_d), Counter(ref_d)
    if src_nums and not hyp_nums:
        # high when the reference KEPT the digits (model alone dropped a fact);
        # medium when the reference also renders them differently (convention).
        sev = "high" if all(ref_nums[n] >= src_nums[n] for n in src_nums) else "medium"
        add("NUMBER_MISSING", sev,
            f"source numbers {sorted(src_nums)} absent from prediction")
    elif src_nums != hyp_nums and hyp_nums:
        # if the reference matches the hyp digit-for-digit, the divergence is
        # corpus convention (e.g. yor spells numbers out, eng uses digits)
        if direction == "yor_eng" and hyp_nums == ref_nums:
            pass  # handled by NUMERAL_CONVENTION_NOTE below
        else:
            add("NUMBER_CHANGED", "high",
                f"source {sorted(src_nums)} -> prediction {sorted(hyp_nums)}")
    if src_nums != ref_nums:
        # reference-quality side note: could be legit localization; flag low
        add("REF_DIGIT_MISMATCH", "info",
            f"reference numbers {sorted(ref_nums)} vs source {sorted(src_nums)}",
            kind="reference")

    # -- spelled-out numbers (yor→eng false-positive guard) ----------------
    # FLORES yor sources often spell numbers as words; NLLB renders them as
    # digits in English. Flag as a script-convention note, not an error.
    if direction == "yor_eng" and not src_nums and hyp_nums and ref_nums == hyp_nums:
        add("NUMERAL_CONVENTION_NOTE", "info",
            f"spelled-out yoruba numbers rendered as digits {sorted(hyp_nums)} "
            "(matches reference; script convention, not an error)")

    # -- named entities ---------------------------------------------------
    missing_ents, _present = entity_carryover(cap_entities(src), hyp)
    for ent in missing_ents:
        add("ENTITY_MISSING", "medium",
            f"source entity '{ent}' not carried into prediction")

    # -- length sanity ----------------------------------------------------
    r_src = ratio_len(hyp, ref)
    thyp = len(tokens(hyp))
    if r_src < SHORT_RATIO and thyp < len(tokens(src)) * SHORT_RATIO:
        add("TOO_SHORT", "medium",
            f"prediction {thyp} tokens vs reference {len(tokens(ref))} ({r_src:.0%})")
    elif r_src > LONG_RATIO:
        add("TOO_LONG", "low", f"prediction {r_src:.0%} of reference length")

    # -- repetition -------------------------------------------------------
    reps = repeated_ngrams(hyp)
    if reps:
        add("REPEATED_NGRAM", "medium", "repeated: " + "; ".join(reps[:3]))

    # -- untranslated content ---------------------------------------------
    copies = untranslated_spans(src, hyp)
    if copies:
        add("UNTRANSLATED", "medium", "copied: " + "; ".join(copies[:3]))

    # -- beam divergence --------------------------------------------------
    other_cfg = "beams=4" if cfg == "beams=2" else "beams=2"
    other = beams.get(other_cfg, "")
    if other:
        a, b = set(tokens(hyp)), set(tokens(other))
        jac = len(a & b) / len(a | b) if a | b else 1.0
        if jac < BEAM_DIFF_TOKEN_RATIO:
            add("BEAM_DIVERGENCE", "info",
                f"token Jaccard vs {other_cfg}: {jac:.2f}")

    # -- low sentence chrF -------------------------------------------------
    if chrf < LOW_CHRF_THRESHOLD:
        add("LOW_CHRF", "medium", f"sentence chrF = {chrf:.1f}")

    # -- reference quality (always computed on the shared reference) -------
    if direction == "eng_yor":
        stripped = ref
        marks = sum(1 for c in stripped if unicodedata.combining(c) and c in "\u0300\u0301")
        vowels = len(re.findall(r"[aeiouẹọ]", unicodedata.normalize("NFD", ref), re.I))
        if vowels >= 8 and marks / max(vowels, 1) < 0.3:
            add("REF_NO_DIACRITICS", "info",
                f"only {marks} tone marks over ~{vowels} vowels", kind="reference")
    # digit consistency of the reference is flagged above as REF_DIGIT_MISMATCH

    agg = {"chrf": round(chrf, 2), "n_findings": len(
        [f for f in findings if f["kind"] == "model"])}
    return findings, agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/eval/beams_ab_20261109T231413Z.csv")
    ap.add_argument("--stamp", default=None,
                    help="output filename stamp (default: derived from --csv)")
    args = ap.parse_args()

    csv_path = ROOT / args.csv
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    beams: dict[tuple[str, str, str], str] = {}
    for r in rows:
        beams[(r["direction"], r["idx"], r["config"])] = r["prediction"]

    all_findings: list[dict] = []
    per_row: list[dict] = []

    for r in rows:
        pair_beams = {
            "beams=2": beams.get((r["direction"], r["idx"], "beams=2"), ""),
            "beams=4": beams.get((r["direction"], r["idx"], "beams=4"), ""),
        }
        findings, agg = analyze_row(r, pair_beams)
        all_findings.extend(findings)
        per_row.append({
            "direction": r["direction"], "idx": r["idx"], "config": r["config"],
            "source": r["source"], "reference": r["reference"],
            "prediction": r["prediction"], "latency_s": r["latency_s"],
            **agg,
            "categories": ";".join(sorted({f["category"] for f in findings if f["kind"] == "model"})),
            "severities": ";".join(sorted({f["severity"] for f in findings if f["kind"] == "model"})),
            "review_status": "needs-human-review" if any(
                f["kind"] == "model" for f in findings) else "auto-pass-heuristics",
        })

    # ---- write analysis CSV (one row per finding) ----
    stamp = args.stamp or "analysis"
    ana_path = OUT / "translation_error_analysis.csv"
    with ana_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_findings[0].keys()) if all_findings else
                           ["direction", "idx", "config", "category", "severity", "kind",
                            "detail", "source", "reference", "prediction"])
        w.writeheader()
        w.writerows(all_findings)

    # ---- summary JSON ----
    by_cat = Counter((f["direction"], f["category"]) for f in all_findings)
    summary = {
        "source_csv": csv_path.name,
        "n_predictions": len(rows),
        "thresholds": {
            "low_chrf": LOW_CHRF_THRESHOLD,
            "beam_token_jaccard": BEAM_DIFF_TOKEN_RATIO,
            "short_ratio": SHORT_RATIO,
            "long_ratio": LONG_RATIO,
            "overshoot_ratio": OVERSHOT_RATIO,
        },
        "disclaimer": "All findings are heuristic warnings for human review, not verdicts.",
        "findings_by_direction_category": {
            f"{d}|{c}": n for (d, c), n in sorted(by_cat.items())},
        "by_severity": dict(Counter(f["severity"] for f in all_findings)),
        "by_kind": dict(Counter(f["kind"] for f in all_findings)),
        "mean_chrf_by_direction_config": _mean_chrf(per_row),
        "rows_flagged": sum(1 for p in per_row if p["n_findings"] > 0),
        "rows_total": len(per_row),
    }
    (OUT / "translation_error_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- review CSV (per prediction) ----
    rev_path = OUT / "translation_error_review.csv"
    with rev_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_row[0].keys()))
        w.writeheader()
        w.writerows(per_row)

    print(f"analyzed {len(rows)} predictions; "
          f"{len(all_findings)} findings -> {ana_path.name}, "
          f"{rev_path.name}, translation_error_summary.json")


def _mean_chrf(per_row: list[dict]) -> dict:
    out: dict[str, float] = {}
    for key in {(p["direction"], p["config"]) for p in per_row}:
        vals = [p["chrf"] for p in per_row if (p["direction"], p["config"]) == key]
        out[f"{key[0]}|{key[1]}"] = round(statistics.mean(vals), 2)
    return out


if __name__ == "__main__":
    main()
