# Sprint 3 Final Report — Translation Reliability (Phases 1–4)

**Branch:** `quality/sprint3-reliability` (based on Sprint-2 diagnostics `ab62cea`)
**Test suite: 322 passed, 1 skipped** · No NLLB inference in this sprint except none — all analysis over saved artifacts; real-inference claims are limited to prior-sprint artifacts.

## PHASE 1 — Sprint-2 findings validated

Audited `scripts/sprint2_failure_analysis.py` against adversarial probes; found and fixed **three confirmed classifier bugs**, each with regression tests:

1. **Value-collision in the number-word lexicon:** `ọ́gbọ̀n` was mapped to BOTH 60 and 80 (plus non-attested `ôgota` for 60 and `mezzi` for 200). A rendering of one value could verify the other (source 40 verified by "ọgbọ̀n"=30, etc.). Fixed: 60=`ọ̀ọ́dọ́gbọ̀n`, 80=`ọgọ́rin`, 70=`àádọ́rin` (typo " Seventy" removed), verified unique folded keys by test.
2. **Substring verification:** `'aadorin'`(70) matched _inside_ `mẹ́rìndínláàádọ́rin`(76); `'ogorin'`(80) matched inside `ọgọ́rùn`(100). Fixed: whole-token (and 2-token compound-window) matching. Test: 90-vs-76 compound stays `dropped`.
3. **Spelled-source gap:** the detector only sees digit sources; documented in tests as out of scope (not silently reclassified).

**Exact digit preservation vs semantic quantity preservation:** `preserved` (verbatim digits) and `rendered_as_words` (attested number-word rendering) remain separate statuses; both count as quantity-preserving but only `preserved` counts toward "exact" rates. Approximate/heuristic counts are never presented as verified error rates — enforced by a test asserting the analysis doc contains no "verified error rate" claims and keeps "pending human linguistic review" placeholders.

## PHASE 2 — Placeholder-failure investigation

- **Optional diagnostic tracing** (`app/services/diagnostic_trace.py`, env-gated `YB_TRANSLATION_TRACE` / `YB_TRANSLATION_TRACE_ECHO`): records `protect → generate_protected → restore → generate_retry_unprotected → fallback_decision → residual_scrub → finish` with texts and structured failure kinds. Inert by default (verified by test: no traces recorded when disabled). Production path unchanged when off.
- **Stage-loss identification:** the existing offline tokenizer probe (`data/eval/placeholder_probe_*.json`) plus per-stage traces answer which stage loses sentinels: tokenization keeps `@@i@@` intact; losses occur at _generation_ (decoder drops/merges sentinels) or at _restoration_ (count/mismatch), now separable per-request from traces.
- **Bounded protected retries** (`numeric_protection_max_retries`, default 1): before the unconditional unprotected fallback, regenerate WITH sentinels up to N times. Verified by tests: retry-success case, exhaustion-then-plain case, and `max_retries=0` = exact legacy one-shot behavior. Original numerical values/formats are always preserved by restoration; the residual-sentinel scrub remains the last-resort guarantee.
- No model tokens added, no embeddings resized (explicit constraint honored).

## PHASE 3 — Meaning-error detection

New `scripts/meaning_diagnostics.py` (deterministic; warnings only; never rewrites a translation):

- **Negation/polarity flips** both directions via negation-marker multiset comparison.
- **Omission detection:** numbers (value-aware — spelled renderings don't false-fire), capitalized entities (accent-folded, transliteration-tolerant), parenthetical content; reference-confirming annotation when a vetted reference contains the content.
- **Degeneration loops** (≥6-token repetition).
- **Heuristic vs confirmed labeling** (`classify_findings`): `heuristic_warning` / `requires_human_review` / `human_confirmed_error` (the last ONLY with a supplied human label).

Known inflation, measured and disclosed: EN `non-` prefixes legitimately translate as Yoruba negation markers, so negation-flip counts are upper bounds (58 vs 59 across arms — useful for arm comparison, not absolute rates).

## PHASE 4 — Evaluation (saved devtest artifact, no inference)

From `integration_comparison_20261110T102741Z.csv` (160 rows = 80 pairs × 2 arms, public FLORES devtest):

| arm        | BLEU  | chrF++ | numeric exact rate | numeric dropped | negation-flip warnings | degeneration | omissions (num/ent/paren) | mean lat s |
| ---------- | ----- | ------ | ------------------ | --------------- | ---------------------- | ------------ | ------------------------- | ---------- |
| baseline   | 13.24 | 35.12  | 0.694              | 7               | 59                     | 1            | 7/168/1                   | 17.53      |
| integrated | 13.14 | 34.85  | **0.778**          | **4**           | 58                     | 1            | **3**/168/1               | 20.71      |

The integrated (protection) arm cuts numeric drops 7→4 and number omissions 7→3 at a ~18% latency cost, with chrF++/BLEU essentially unchanged (−0.27/−0.10 — metric-neutral). Real source/prediction/reference examples for every heuristic finding are recorded in `data/eval/phase3_sprint3_evaluation.json`. Full tables + limitations: `docs/phase3_sprint3_evaluation.md`.

## Errors fixed vs remaining

**Fixed (tests + artifact evidence):** lexicon value-collisions; substring verification; bounded protected retry improves numeric preservation (7→4 drops on the saved artifact); tracing infrastructure for future attribution.

**Remaining:** 4 residual numeric drops (compound numerals like the "31 miles" parenthetical — the model drops them in both arms); negation-flip detector needs a `non-` prefix rule before its counts are usable absolutely; entity-omission transliteration tolerance is coarse; human-review labels still pending, so no verified semantic-error rates exist.

## Benchmark limitations

Single reference; devtest pairs are the same 80 used by Sprint 2 (no new inference this sprint); semantic counts are heuristic upper bounds; latency from the historical artifact, not re-measured.

## Files

New: `app/services/diagnostic_trace.py`, `scripts/meaning_diagnostics.py`, `scripts/phase4_evaluate.py`, `tests/test_sprint3_phases.py`, `docs/phase3_sprint3_evaluation.{md,json sidecar}`, this report.
Modified: `app/services/translation.py` (tracing hooks + bounded retry), `app/core/config.py` (`numeric_protection_max_retries`), `scripts/sprint2_failure_analysis.py` (lexicon + whole-token fixes), `docs/sprint2_failure_analysis.md` (regenerated with corrected counts).
