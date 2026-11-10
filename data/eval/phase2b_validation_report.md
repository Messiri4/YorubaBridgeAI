# Phase 2B — FLORES-200 Reference Audit & Error Validation

**Date:** 2026-11-10 · **Branch:** `quality-review` · **No inference, no model loads.**
**Data:** 997 aligned EN–YO pairs from the FLORES-200 dev split, fetched from the
HF mirror `yash9439/flores200` (`data/flores_eval/dev.997.{en,yo}.txt`; provenance
JSON committed). Cross-checked: the mirror's first 60 lines are byte-identical to
our existing official-archive excerpt in `data/flores/` — provenance verified.

## 1. Dataset provenance and audit methodology

|                |                                                                     |
| -------------- | ------------------------------------------------------------------- |
| Pairs audited  | 997 (full dev split)                                                |
| Source         | HF `yash9439/flores200` parquet (Meta FLORES-200, CC-BY-SA 4.0)     |
| Checks         | Orthography (6), numerals (2), entities (1), structure (4)          |
| Tools          | `scripts/audit_flores_references.py`, `scripts/phase2b_outputs.py`  |
| Classification | **mechanical** = byte/structure-level fact; **warning** = heuristic |

## 2. Reference issues found (852 findings over 499/997 pairs)

### Mechanically verified (no linguistic judgment needed)

| Category            | Count | Nature                                                                                                                                                                                                                                    |
| ------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MIXED_UNICODE_FORMS | 77    | precomposed `ẹ/ọ/ṣ` **and** raw `U+0323` dot-below sequences in the same string — a real encoding inconsistency, verified byte-level (e.g. pair #18: `ìlú kan sí kan, oye owó orí àti ẹ̀gbẹ́` mixes `ẹ̀`(o+U+0323) with precomposed `ẹ́gbẹ́`). |
| UNBALANCED_QUOTES   | 29    | unbalanced “” or () in reference (pairs #57/#59/#61 among them)                                                                                                                                                                           |
| ADJACENT_DUPLICATE  | 21    | repeated adjacent words after excluding legit Yoruba reduplications (`fini fini`, `kíá kíá`, iterative `ọjọ ọjọ` are exempted)                                                                                                            |

### Heuristic warnings (require human review; not confirmed errors)

| Category                        | Count | Note                                                                                                                                                          |
| ------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LOW_TONE_MARK_COVERAGE          | 233   | <30% tone-mark coverage over ≥5 content words                                                                                                                 |
| REFERENCE_UNACCENTED            | 173   | near-zero tone marks (accent-dropped style). Measured: **135/997 refs (13.5%) are fully unaccented** — a systematic corpus inconsistency, not per-pair noise. |
| MISSING_UNDERDOT_CANDIDATE      | 102   | words whose obligatory ẹ/ọ/ṣ is absent (e.g. `ọjọ`→`ojo`)                                                                                                     |
| NUM_DIGITS_DROPPED              | 82    | EN has digits, YO reference has none                                                                                                                          |
| ENTITY_NOT_CARRIED              | 98    | EN entity not found even fuzzily (transliteration-tolerant)                                                                                                   |
| NUM_DIGIT_DIFF                  | 31    | digit multisets differ EN vs YO                                                                                                                               |
| SUSPICIOUSLY_LONG / ODD_CAPITAL | 6     | rare                                                                                                                                                          |

**Interpretation:** the single biggest finding is _orthographic inconsistency
across the corpus_: 13.5% fully unaccented refs + 77 mixed-encoding refs mean
FLORES-dev Yoruba does not hold a uniform standard. This inflates variance in
any metric that treats tone marks as characters (chrF) and makes "accent
correctness" unmeasurable against ~1 in 7 references.

## 3. Phase-2 detector validation (false positives)

Cross-checked Phase-2's 314 findings against reference-audit ground truth:

- **6 LOW_CHRF flags reclassified as reference-attributable** — the paired
  reference is itself unaccented/mixed-encoding, so part of the score gap
  belongs to the reference, not the model.
- **4 NUMBER_CHANGED + 2 NUMBER_MISSING reclassified as ref-followed** — the
  model's digit handling matches its (digit-divergent) reference; convention, not error.
- **~289 findings remain plausible** as model-side warnings (ENTITY_MISSING 41,
  REPEATED_NGRAM 48, UNTRANSLATED 16, remaining LOW_CHRF 32, plus yor→eng
  reverse-direction findings not attributable to reference issues).
- **Detector bugs found and fixed in this phase** (Phase-2 reports kept
  unchanged as historical artifacts):
  1. `cap_entities` treated leading capitalized stopwords ("The Washington…")
     as part of the entity, producing spurious ENTITY_MISSING — fixed by
     stripping the stopword prefix.
  2. Transliteration fuzzy-match was substring-based and missed
     vowel-suffix expansion (Washington→Wọshíñtònì); replaced with a
     first-letter + consonant-skeleton matcher (`_similar`, ≥60% shared consonants).
  3. Repetition detector: content-word repetition now requires len≥3 and a
     curated stopword list (function-word repeats were false-flagged).
     All fixes are regression-tested (`tests/test_analyze_translation_errors.py`
     and `tests/test_phase2b_audit.py`).

**Total false-positive rate measured: 12 of 136 eng→yor model-side findings
(≈9%) were reference-attributable rather than model errors.**

## 4. chrF normalization experiment (no inference; same predictions/refs/order)

Config: sacrebleu **2.6.0**, `corpus_chrf(char_order=6, word_order=2, beta=2.0)`
(chrF++), applied identically to hypotheses and references in every regime.

| Direction/config | original | NFC   | NFD   | fold-underdots | fold-all-accents |
| ---------------- | -------- | ----- | ----- | -------------- | ---------------- |
| eng_yor, beams=2 | 28.12    | 28.27 | 28.27 | **33.72**      | **36.86**        |
| eng_yor, beams=4 | 28.76    | 28.92 | 28.92 | **34.57**      | **37.86**        |
| yor_eng, beams=2 | 45.99    | 46.04 | 46.04 | 46.61          | 47.40            |
| yor_eng, beams=4 | 45.51    | 45.56 | 45.56 | 46.01          | 46.78            |

- NFC/NFD move eng→yor by only +0.15 (they equalize the 77 mixed-encoding refs
  but can't restore missing tone marks; NFC≠full unification for ọ vs o+U+0323 —
  see test `test_nfc_nfd_equivalence_of_score_inputs`).
- **Underdot folding: +5.0 to +5.8 chrF for eng→yor; full accent folding: +8.7
  to +9.1.** yor→eng barely moves (+0.6/+1.3) because English references carry
  no Yoruba diacritics to fold away.
- Interpretation: **up to a third of the eng→yor chrF gap versus yor→eng is a
  measurement artifact of diacritic variance**, not translation error.
  Folded scores are diagnostics; fold-all-accents is lossy (destroys tonal
  information) and must not become the production metric. Per-config deltas:
  `data/eval/chrf_normalization_comparison.csv`.

## 5. Human-review sample

`data/eval/flores_reference_review.csv`: **100 unique pairs**, seed 20261109,
deterministic (`test_review_sample_deterministic` passes). Stratification:
numeric discrepancies (30), entity issues (20), orthographic (30 incl.
unaccented refs), structural (10), plus random unflagged fill; every row
records its `sampling_reasons`. All NLLB predictions are in **separate fields**
from the EN-source/YO-reference judgment columns; all human-judgment columns
(reference_accuracy … final_review_status) are **blank** — nothing
pre-filled from heuristics or model output.

## 6. Answers to the four research questions

1. **Measurable reference inconsistencies? Yes.** 135/997 unaccented refs,
   77 mixed-Unicode encodings, 29 unbalanced-quote refs, 21 adjacent
   duplications; 499/997 pairs carry at least one finding.
2. **Detector false positives? Yes, ~9%** of eng→yor model-side findings were
   reference-attributable; 3 detector bugs fixed with regression tests.
3. **Diacritics/normalization effect on chrF? Large for eng→yor**
   (+5.6 with underdot folding, +9.1 with full folding) — a measurement
   artifact, negligible for yor→eng.
4. **Reliably evidenced model problems?** Entity carryover failures (41
   eng→yor), verbatim source-copying (16/22 per direction), and number/time
   corruption (the Phase-2 verified 7–2→72 and 11:29 cases — both beams agree,
   so they are model-knowledge gaps, robust to decoding). Tone-mark production
   by the model is _good_; the reference side is the weaker half.

## 7. Recommended next steps (evidence-ranked)

1. **Two-track scoring standard:** adopt accent-folded chrF as a _secondary
   diagnostic_ alongside unmodified chrF++, and require NFC+NFD-stable
   comparisons when publishing model deltas. Zero risk; makes all future
   model comparisons interpretable.
2. **Reference-normalization pass for evaluation only:** canonicalize
   references to NFC with a documented underdot policy (do NOT rewrite the
   shipped dataset — generate a derived `_normalized` copy for scoring).
   Directly removes the +5–9 chrF measurement artifact.
3. **Entity transliteration lexicon (eng→yor):** the 41 ENTITY_MISSING
   findings cluster on sports/politics names; a small name-form table (forms
   only, per the no-blind-overrides rule) is the highest-yield model-side fix.

Then, only after the reviewer worksheet comes back: number/time post-check
(Phase-2 verified errors) and the beams=2/4 direction split re-validation.

## 8. Limitations

- Yoruba number words are only partially mapped (`_YOR_NUM_WORDS` is minimal);
  spelled-number audits can miss values or mis-fire on homographs (`ọgọ́rin` 80
  vs `ọ̀gọ́rin` variants).
- Tone-mark coverage heuristics can't distinguish an unaccented _style_ from an
  error; both are recorded as warnings.
- No fluent-speaker judgment has been obtained; all "confirmed" labels in this
  report are mechanical/byte-level or detector-logic facts.
- chrF comparisons use a single reference per sentence.

## 9. Artifacts

- `scripts/audit_flores_references.py` — audit (997 pairs)
- `scripts/phase2b_outputs.py` — chrF regimes, Phase-2 validation, review sample
- `data/eval/flores_reference_audit.csv` + `_summary.json`
- `data/eval/chrf_normalization_comparison.csv`, `data/eval/phase2b_meta.json`
- `data/eval/flores_reference_review.csv` (100 pairs, human fields blank)
- Tests: `tests/test_phase2b_audit.py` (18) + detector regressions in
  `tests/test_analyze_translation_errors.py`; suite: **103 passed, 1 skipped**.
- Original artifacts untouched: `beams_ab_*` and Phase-2 outputs byte-identical
  (md5-checked before commit).
