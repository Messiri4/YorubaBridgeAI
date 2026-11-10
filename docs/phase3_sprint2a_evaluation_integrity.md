# Phase 3 — Sprint 2A: Evaluation Integrity & Translation Quality

**Branch:** `quality-review` · **Base:** Sprint 1 `62726e9` · **No NLLB inference, no training.**

## 1. What was inspected

- Evaluation stack: `app/services/scorecard.py`, `scripts/evaluate_quality.py`,
  `scripts/audit_flores_references.py`, `scripts/phase2b_outputs.py`, `configs/evaluation.yaml`.
- Prediction artifacts: `beams_ab_20261109T231413Z.{csv,summary.json}` (all 120 rows intact).
- Datasets: `data/flores_eval/dev.997.{en,yo}.txt` (997 aligned pairs, mirror provenance), devtest untouched.
- Phrase-override mechanism (inspected only — unchanged), scorecard, rubric, all 121 existing tests.

## 2. Why the chrF++ scores differed (fully reproduced)

**The historical numbers were plain chrF; Sprint 1 computed chrF++.** The beam-A/B
experiment used `sacrebleu.corpus_chrf(hyps, refs)` with **default kwargs**
(`char_order=6, word_order=0, beta=2.0` — plain chrF, no word n-grams). Sprint 1
pinned `word_order=2` (true chrF++). Recomputing the identical prediction files
(`sha256 2466d0e9…`) with plain defaults reproduces **all four historical values exactly**:

| arm              | plain chrF (word_order=0) | chrF++ (word_order=2) | BLEU (identical both) |
| ---------------- | ------------------------- | --------------------- | --------------------- |
| eng_yor\|beams=2 | **28.12**                 | 26.07                 | 3.86                  |
| eng_yor\|beams=4 | **28.76**                 | 26.75                 | 4.57                  |
| yor_eng\|beams=2 | **45.99**                 | 43.56                 | 19.21                 |
| yor_eng\|beams=4 | **45.51**                 | 43.01                 | 18.55                 |

BLEU matched exactly in both runs, proving predictions/references/alignment are
unchanged (no data drift). chrF++ scores systematically lower because word
n-gram recall is added with weight β=2 against harsher char-ngram targeting —
expected, not a bug. Resolution: both are valid; **the scorecard standardizes
on chrF++ (word_order=2) and records plain chrF as `chrf_plain` for
continuity with history.**

### Unicode findings (code-point inspection; no dataset blame)

- Corpus census: references and predictions both store underdots predominantly
  as **precomposed U+1EB9/U+1ECD/U+1E63** with combining tone marks
  (U+0300/U+0301) — same encoding family; a wholesale NFC/NFD mismatch claim
  was **wrong**.
- A critical Unicode fact drives the remaining effects: **NFC does not compose
  `o + U+0323` back into ọ** (no canonical composition for dot-below). So
  `NFC(ọ)`-as-o+0323 stays 2 codepoints while the precomposed ọ is 1.
- Controlled single-pair test: identical strings score identically under
  NFC/NFD (no free lunch). Mixed-encoding sentences show **NFD > NFC by ~6.5
  chrF points** — NOT because NFD "fixes" the reference, but because the
  decomposed form creates many shared mark-based character n-grams
  (`letter+0300`, `0301`, `letter+0323`…) that raise _both_ precision and
  recall across **different words**. Measured directly: swapping two vowels
  while keeping marks scores NFD 53.8 vs NFC 48.7 — NFD is **less sensitive to
  vowel identity**, i.e. partially blind to which syllable carries which tone.
- Conclusion: NFD's apparent superiority is a **metric artifact of generic
  combining-mark n-grams**, not evidence that references are defective or that
  NFD is "better". The accent-folded diagnostic (+11.4 over standard) inflates
  for the same reason and remains diagnostic-only. yor→eng shows no regime
  sensitivity (English has no combining marks) — consistent with the mechanism.

## 3. Verified evaluation baseline (Sprint 2A, canonical)

All values from `data/eval/phase3_scorecard_sprint2a.json` (same predictions,
same ordering, pinned sacrebleu 2.6.0, `char_order=6, word_order=2, beta=2.0`;
input sha256 recorded in the file):

| arm              | n   | BLEU  | chrF++ (std) | NFC   | folded (diag) | numeric rate | ent warns | rep warns | mean lat ms |
| ---------------- | --- | ----- | ------------ | ----- | ------------- | ------------ | --------- | --------- | ----------- |
| eng_yor\|beams=2 | 30  | 3.86  | 26.07        | 29.30 | 37.42         | 0.545        | 2         | 0         | 25 666      |
| eng_yor\|beams=4 | 30  | 4.57  | 26.75        | 30.04 | 37.91         | 0.636        | 2         | 0         | 30 453      |
| yor_eng\|beams=2 | 30  | 19.21 | 43.56        | 43.56 | 43.56         | 0.833        | 2         | 0         | 13 480      |
| yor_eng\|beams=4 | 30  | 18.55 | 43.01        | 43.01 | 43.01         | 0.833        | 2         | 0         | 17 490      |

Human-review labels: **unavailable** (pending). Historical scores preserved
untouched in the original artifacts; no score manipulation.

## 4. Detector bugs fixed (each with regression tests)

1. **Repetition FP — legitimate reduplication.** `ọjọ ọjọ` ("day after day")
   was flagged. Fix: tone+underdot-folded whitelist (`REDUPLICATION_WHITELIST`)
   and `_fold_tones` helper. Verified: legit kept, `ìwé ìwé` still flagged.
2. **Repetition FP — non-adjacent content repeats.** The token-pair window
   flagged "the committee … the committee". Fix: pair-window requires full
   adjacency (`tk[i]==tk[i+2] and tk[i+1]==tk[i+3]`); single non-adjacent
   repeats no longer fire. Verified.
3. **Numeric FN — spelled-out numbers.** "11:29" rendered as `aago mọ́kànlá`
   counted as `digits_missing`. Fix: `_NUMBER_WORD_VALUES` + `_pred_has_number_words`
   route word-rendered numerals to a new informational `digits_spelled_out`
   bucket (approximate for Yoruba's vigesimal system — documented limitation).
   Added `mọ́kànlá`(11)/`mẹjìlá`(12)/`kanla` forms; verified end-to-end.
4. Detector behaviors confirmed correct (no change needed): en-dash vs hyphen
   digits compare equal; "50%" vs "50 percent" flags as changed (documented);
   sentence-initial names and verbatim acronyms (WHO) correctly not flagged.

## 5. Prioritized human-review worksheet

`data/eval/flores_priority_review_20.csv` — **20 pairs**, seed `20261110`,
deterministic: 6 meaning-change (verified Phase-2 semantic errors: treasurer→PM,
viable→disadvantaged, 7–2→72, role reversal, 9,000-year drift, actant flip),
4 numeric, 3 entity, 3 omission/low-coverage, 2 orthographic, 2 unflagged
controls. Every row: rank + selection reasons + `pair_idx` tracing to
`dev.997` lines and the A/B arms; the 6 meaning-change rows extend the
worksheet with full dev rows + A/B predictions (marked `extended:meaning-change-priority`).
All judgment fields blank per rubric. Meta: `flores_priority_review_20.meta.json`.

## 6. Test results

Suite: **122 passed, 1 skipped** (3 new Sprint-2A regression tests added;
no failures). Full run: `.venv/bin/python -m pytest`.

## 7. Files

Created: `docs/phase3_sprint2a_evaluation_integrity.md`,
`data/eval/phase3_scorecard_sprint2a.{json,csv,md}`, `data/eval/phase3_manifest_sprint2a.json`,
`data/eval/flores_priority_review_20.csv` (+ `.meta.json`),
`scripts/build_priority_review.py`, `tests/test_phase3_sprint2a.py`.
Modified: `app/services/scorecard.py` (whitelist, `_fold_tones`, `digits_spelled_out`).
Unchanged: all Phase 1/2/2B/Sprint-1 artifacts (md5-verified before commit).

## 8. Recommended next experiment

**Numerical-verification post-check**: verify source numerals/amounts in the
model output (digit or word form) and re-translate/annotate on mismatch —
targets the highest-severity verified errors (7–2→72; 9,000-year lease) with
no model retraining. Validate on dev via this scorecard; hold devtest for final.
