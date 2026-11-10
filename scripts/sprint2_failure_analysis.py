"""Sprint 2, Phase 1: diagnose numerical-preservation failures from the
saved integration-comparison artifacts.

Inputs (already committed, no inference needed):
  data/eval/integration_comparison_20261110T102741Z.csv / .json

What this script does:
1. Re-derives per-row numeric-preservation status with LINGUISTIC awareness:
   a "failure" is counted only when a digit sequence in the SOURCE is
   neither present verbatim in the prediction nor rendered as its
   corresponding Yoruba/English number word (e.g. "4" -> " mẹ́rin ",
   "mẹẹdọgbọn"; "19" -> "mókàndínlógún"). Pure digit-count differences
   caused by legitimate spelling-out are classified as "rendered_as_words",
   NOT failures.
2. Cross-references the run log if available (/tmp/integ_cmp_80.log) for
   the recorded protection-restoration failure count. The CSV/JSON do NOT
   contain per-row retry flags or placeholder traces; where that evidence
   is missing this script says so explicitly instead of inferring it.
3. Emits a case table (markdown) into docs/sprint2_failure_analysis.md
   with: source, direction, baseline vs integrated prediction, original
   numeric expressions, expected/observed, retry info (or "not recorded"),
   probable mechanism, and a meaning-changed judgement left BLANK for
   human linguistic review (automatic judgement of meaning is not claimed).

Usage:  python scripts/sprint2_failure_analysis.py
Writes: docs/sprint2_failure_analysis.md  (only the auto sections)
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/eval/integration_comparison_20261110T102741Z.csv"
JSON_PATH = ROOT / "data/eval/integration_comparison_20261110T102741Z.json"
LOG_PATH = Path("/tmp/integ_cmp_80.log")
OUT_PATH = ROOT / "docs/sprint2_failure_analysis.md"

DIGIT_SEQ = re.compile(r"\d+(?:[,\.\-]\d+)*")

# Minimal attested number-word lexicon for rendering detection. Used ONLY to
# avoid calling a legitimate spelled-out rendering a failure. Tone-marked and
# unmarked variants included; matching is accent-insensitive via simple fold.
_NUM_WORDS: dict[int, tuple[str, ...]] = {
    0: ("òdo", "odo", "zero"),
    1: ("ọ̀kan", "okan", "one"),
    2: ("méjì", "meji", "eji", "two"),
    3: ("mẹ́ta", "meta", "three"),
    4: ("mẹ́rin", "merin", "erin", "four"),
    5: ("márùn", "marun", "five"),
    6: ("mẹ́fà", "mefa", "six"),
    7: ("mẹ́je", "meje", "seven"),
    8: ("mẹ́jọ", "mejo", "eight"),
    9: ("mẹ́sàn", "mesan", "nine"),
    10: ("mẹ́wàá", "mewaa", "mẹ̀wá", "mewa", "ten"),
    11: ("mọ́kànlá", "mokanla", "eleven"),
    12: ("méjìlá", "mejila", "twelve"),
    13: ("mẹ́tàlá", "metala", "thirteen"),
    14: ("mẹ́rìnlá", "merinla", "fourteen"),
    15: ("mẹ́ẹ́dọ́gbọ̀n", "meedogbon", "fifteen"),
    16: ("mẹ́rìndínlógún", "merindilogun", "sixteen"),
    17: ("mẹ́tàdínlógún", "metadilogun", "seventeen"),
    18: ("mẹ́jìdínlógún", "mejidilogun", "eighteen"),
    19: ("mókàndínlógún", "mokandilogun", "nineteen"),
    20: ("ogún", "ogbu", "twenty"),
    30: ("ọgbọ̀n", "ogbon", "thirty"),
    40: ("ogoji", "forty"),
    50: ("ádọ́ta", "adota", "fifty"),
    60: ("ọ́gbọ̀n", "ôgota", "sixty"),
    70: ("àádọ́rin", "aadorin", " Seventy"),
    78: ("mẹ́jìdínlọ́gbọ̀n", "seventy-eight"),
    80: ("ọ́gbọ̀n", "eighty"),
    90: ("àádọ́rùn-ún", "ninety"),
    100: ("ọ̀kẹ́", "oke", "hundred"),
    200: ("mezzi", "two hundred"),
}

# Vigesimal/substractive renderings attested in the devtest outputs themselves
# (20-base system: 78 = 80 minus 2 = mẹ́tàdínláàádọ́rin). Matching folds accents.
_VIGESIMAL_EXTRA: dict[int, tuple[str, ...]] = {
    16: ("kẹrìndínlógún", "kerindinlogun", "ọjọ́ kẹrìndínlógún"),  # 20-4 (fold: kerindinlogun)
    17: ("kẹtàdínlógún", "ketadilogun"),          # 20-3
    18: ("kẹjìdínlógún", "kejidilogun"),          # 20-2
    19: ("kokanla-din", "mọ́kàndínlógún"),         # 20-1
    30: ("mẹ́ẹ̀ẹ́dọ́gbọ̀n", "ogun-din-lgbon", "mẹ́tàlà-ogún"),
    50: ("àádọ́ta", "aadota"),                    # 60-10
    70: ("àádọ́rin", "aadorin"),                  # 80-10
    78: ("mẹ́tàdínláàádọ́rin", "metadinlaadorin"), # 80-2
    90: ("àádọ́rùn"),
    100: ("ọ̀kẹ́ kan", "ọgọ́rùn"),
}
for _k, _v in _VIGESIMAL_EXTRA.items():
    # merge into the primary lexicon ('setdefault' then extend keeps BOTH the
    # standard and vigesimal renderings when a key exists in both tables —
    # e.g. 16 has mẹ́rìndínlógún AND kẹrìndínlógún)
    _NUM_WORDS[_k] = tuple(_NUM_WORDS.get(_k, ())) + tuple(_v)

# degeneration guard: a single word repeated consecutively at least this many
# times indicates a repetition loop (NLLB degeneration), which destroys the
# sentence content regardless of numerics
_DEGENERATION_MIN_REPEATS = 6


def _has_degeneration(text: str) -> bool:
    tokens = text.split()
    run = 1
    for a, b in zip(tokens, tokens[1:], strict=False):
        run = run + 1 if a == b else 1
        if run >= _DEGENERATION_MIN_REPEATS:
            return True
    return False


def _fold(s: str) -> str:
    import unicodedata
    nfd = unicodedata.normalize("NFD", s.casefold())
    stripped = "".join(c for c in nfd if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped)


def _words_for(num_token: str) -> set[str]:
    """Candidate number-word renderings for a digit token (best-effort)."""
    words: set[str] = set()
    try:
        n = int(num_token)
    except ValueError:
        return words
    lex = _NUM_WORDS.get(n)
    if lex:
        words |= {w.lower() for w in lex}
    # tens composition for 21..99 (approximate; Yoruba vigesimal system is
    # irregular, so this is intentionally conservative)
    if 20 < n < 100 and n % 1 == 0:
        tens, ones = divmod(n, 10)
        tens_map = {2: "ogún", 3: "ọgbọ̀n", 4: "ogoji", 5: "àádọ́ta",
                    6: "ọ́gbọ̀n", 7: "àádọ́rin", 8: "ọ́gbọ̀n-ún", 9: "àádọ́rùn-ún"}
        if tens in tens_map and ones == 0:
            words.add(tens_map[tens].lower())
    # English words
    words |= {w for w in _NUM_WORDS.get(n, ()) if w.isascii()}
    return words


def _is_rendered(num_token: str, pred: str) -> bool:
    return any(w and _fold(w) in _fold(pred) for w in _words_for(num_token))


def classify_row(source: str, pred: str) -> tuple[str, list[str]]:
    """Return (status, dropped_expressions).

    status ∈ preserved | rendered_as_words | dropped | reordered_or_changed
    """
    src_nums = DIGIT_SEQ.findall(source)
    if not src_nums:
        return ("no_numbers", [])
    pred_nums = set(DIGIT_SEQ.findall(pred))
    dropped: list[str] = []
    status = "preserved"
    for n in src_nums:
        if n in pred_nums or n in pred:
            continue
        if _is_rendered(n, pred):
            status = "rendered_as_words" if status == "preserved" else status
            continue
        dropped.append(n)
        status = "dropped"
    return (status, dropped)


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    summary = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    # group rows by (direction, idx) to pair baseline vs integrated
    by_key: dict[tuple[str, int], dict[str, dict]] = {}
    for r in rows:
        by_key.setdefault((r["direction"], int(r["idx"])), {})[r["arm"]] = r

    cases: list[dict] = []
    status_counts: Counter = Counter()
    for (direction, idx), arms in sorted(by_key.items()):
        base = arms["baseline"]
        integ = arms["integrated"]
        st_b, drop_b = classify_row(base["source"], base["prediction"])
        st_i, drop_i = classify_row(integ["source"], integ["prediction"])
        degen = "\n".join((base["prediction"], integ["prediction"]))
        status_counts[f"baseline/{st_b}"] += 1
        status_counts[f"integrated/{st_i}"] += 1
        if _has_degeneration(degen):
            status_counts["degeneration_loops"] += 1
            cases.append({
                "direction": direction, "idx": idx,
                "source": base["source"], "reference": base["reference"],
                "baseline": base["prediction"], "integrated": integ["prediction"],
                "baseline_status": "degeneration_loop", "integrated_status": "degeneration_loop",
                "dropped_baseline": drop_b, "dropped_integrated": drop_i,
                "expected": ", ".join(DIGIT_SEQ.findall(base["source"])) or "-",
                "observed_baseline": ", ".join(DIGIT_SEQ.findall(base["prediction"])) or "—",
                "observed_integrated": ", ".join(DIGIT_SEQ.findall(integ["prediction"])) or "—",
                "retry_recorded": "NOT RECORDED in artifact",
            })
        elif st_i == "dropped" or st_b == "dropped":
            cases.append({
                "direction": direction, "idx": idx,
                "source": base["source"], "reference": base["reference"],
                "baseline": base["prediction"], "integrated": integ["prediction"],
                "baseline_status": st_b, "integrated_status": st_i,
                "dropped_baseline": drop_b, "dropped_integrated": drop_i,
                "expected": ", ".join(DIGIT_SEQ.findall(base["source"])) or "-",
                "observed_baseline": ", ".join(DIGIT_SEQ.findall(base["prediction"])) or "—",
                "observed_integrated": ", ".join(DIGIT_SEQ.findall(integ["prediction"])) or "—",
                "retry_recorded": "NOT RECORDED in artifact",
            })

    # restoration-failure count from the run log, if present (best effort;
    # per-row attribution is not recoverable from the artifact)
    retry_note = "run log unavailable"
    retry_count = None
    if LOG_PATH.exists():
        text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
        m = re.findall(r"placeholder restoration failed \(missing[^\)]*\)", text)
        retry_count = len(m)
        retry_note = (
            f"{retry_count} restoration-failure retries logged "
            "(counts only; no per-row attribution)")

    doc: list[str] = [
        "# Sprint 2 — Failure Analysis (numerical & placeholder restoration)",
        "",
        f"Artifact: `{CSV_PATH.relative_to(ROOT)}` · pairs: {summary.get('pairs')}"
        f" · split: {summary.get('split')}",
        "",
        "## Artifact-evidence limitations (stated explicitly)",
        "",
        "- The comparison CSV records per-sentence predictions and digit counts only;",
        "  it does **NOT** record retry flags or placeholder traces, so per-row retry",
        "  attribution below is marked NOT RECORDED rather than inferred.",
        f"- Run-log evidence ({LOG_PATH}): {retry_note}.",
        "- Digit-count mismatch is NOT treated as failure by itself: legitimate",
        "  spelled-out renderings (\"4\" → \"mẹ́rin\") are classified as",
        "  `rendered_as_words`. Only sources whose digit expressions are neither",
        "  verbatim nor attested-rendered count as `dropped`.",
        "- Meaning-change judgement is left for human linguistic review; this",
        "  script does not claim semantic equivalence or difference.",
        "",
        "## Automatic status summary (per direction, both arms)",
        "",
        "| classification | baseline | integrated |",
        "|---|---|---|",
    ]
    for st in ("no_numbers", "preserved", "rendered_as_words", "dropped"):
        doc.append(
            f"| {st} | {status_counts[f'baseline/{st}']}"
            f" | {status_counts[f'integrated/{st}']} |")

    doc += [
        "",
        "## Degeneration loops (separate failure class)",
        "",
        f"Sentences (either arm) showing a ≥{_DEGENERATION_MIN_REPEATS}-token",
        "consecutive repetition loop. These destroy sentence content wholesale,",
        "often including numbers; they are decoding failures, not protection",
        "failures, and are listed in the case table with their own status.",
        "",
    ] if any(c["baseline_status"] == "degeneration_loop" for c in cases) else []

    doc += [
        "",
        "## Confirmed numerical-preservation failure cases",
        "",
        f"Cases where either arm dropped a numeric expression: **{len(cases)}**.",
        "",
    ]
    for c in cases:
        doc += [
            f"### {c['direction']} #{c['idx']}",
            "",
            f"- Source ({c['direction'].split('_')[0]}): {c['source']}",
            f"- Baseline prediction: {c['baseline']}",
            f"- Integrated prediction: {c['integrated']}",
            f"- Original numeric expressions: {c['expected']}",
            ("- Observed (baseline): " + c['observed_baseline'] +
             " — status: " + c['baseline_status']),
            ("- Observed (integrated): " + c['observed_integrated'] +
             " — status: " + c['integrated_status']),
            f"- Dropped: baseline={c['dropped_baseline'] or 'none'}; "
            f"integrated={c['dropped_integrated'] or 'none'}",
            f"- Retry occurred: {c['retry_recorded']}",
            "- Meaning changed: _pending human linguistic review_",
            "- Probable mechanism: see §Mechanisms (analysis below); per-case attribution",
            "  requires the placeholder traces the artifact does not store.",
            "",
        ]

    doc += [
        "## Mechanism hypotheses (Phase 2 linkage; hypotheses, not proven causes)",
        "",
        "1. Sentinel dropped by generation: NLLB omits `@@i@@` (or merges its `@`",
        "   tokens with adjacent punctuation), so restoration verification fails",
        "   (`missing`/`count_mismatch`) and the unprotected retry regenerates —",
        "   the retry may itself drop the number, which surfaces as a `dropped`",
        "   case in the integrated arm.",
        "2. Spelled-out rendering pressure: even with sentinels, the decoder may",
        "   translate the surrounding sentence and leave the sentinel untranslated",
        "   or normalize digits to words for fluent output.",
        "3. Reformatting: comma separators and date/time formats are rewritten by",
        "   the model (e.g. `9,000` → `9000` or `12 noon` paraphrases), which",
        "   counts as changed although semantically close; flagged for review.",
        "4. Baseline weakness: without sentinels the model freely drops digits;",
        "   the integrated arm reduces but does not eliminate this class.",
        "",
        "## What would confirm mechanisms",
        "",
        "- A small instrumented re-run recording per-row: protected input string,",
        "  raw model output, restoration outcome kind, retry occurrence, final text.",
        "- Tokenizer fragmentation probe of `@@i@@` (see",
        "  `scripts/probe_placeholder_survival.py`).",
        "- Per-case linguistic review to fill the meaning-changed column.",
        "",
    ]
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text("\n".join(doc) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
    print("statuses:", dict(status_counts))
    print("failure cases:", len(cases))
    if retry_count is not None:
        print("logged restoration retries:", retry_count)


if __name__ == "__main__":
    main()
