# Phase 3 — Sprint 2C-Q: Numerical Verification Correctness

**Branch:** `quality-review` · **Base:** Sprint 2A `e866936` · **No inference, no downloads.**

## 1. Root cause

The Sprint-2A detector classified any source numeral as "plausibly preserved"
whenever the prediction contained **any** recognized number word:

```python
return bool(folded & set(_NUMBER_WORD_VALUES))   # "eleven" verified 11:29; "twenty" verified 11
```

Failure modes: (a) unrelated number words counted as preservation; (b) partial
matches (11:29 → only "eleven") counted as preservation; (c) the value
dictionary had **conflicting duplicate keys** — `'aadota'` mapped to both 30
and 50 (Python silently kept the last), so tone-folded 30/50 spellings could
verify each other's values. Additionally a **tokenizer bug** was discovered:
Python's `\w` treats a base letter followed by a combining tone mark as a word
boundary, so `àádọ́ta` tokenized as `àádọ` + `ta` and Yoruba number words
silently vanished from lookups.

## 2. Corrected algorithm (`app/services/scorecard.py`)

Per sentence with source digits, `_classify_numeric` returns exactly one of:

| Classification                    | Rule                                                                                                                                                           |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `exact_digits_preserved`          | digit multiset identical (order/format-insensitive; `7-2`≡`7–2`)                                                                                               |
| `verified_number_word_equivalent` | no digits in output, but **every** source numeral group equals a trusted number word's value in the output; source values must themselves be table-supported   |
| `numeric_mismatch_warning`        | digits differ; **or** full-coverage supported words all carry the wrong value (source 11 → "twelve" proves the numeral changed)                                |
| `numeric_review_required`         | output has no digits and equivalence can't be verified: partial coverage, time/date/range/score contexts, compound or unsupported Yoruba values (e.g. 41, 977) |

- **Lookup folding:** `_fold_tones` strips tones+underdots to bare latin; the
  original prediction text is never modified (folding exists only inside the
  detector).
- **Tokenizer fix:** `_strip_combining` removes combining marks _before_
  regex tokenization (`àádọ́ta` now one token); base characters untouched.
- **Dictionary:** duplicate/contradicting keys removed; folded keys now unique
  per value; added attested teen forms `okanla`(11), `ejila`(12).
- **Backward compatibility:** legacy keys kept as aliases —
  `digits_preserved`→exact, `digits_changed`→mismatch, `digits_missing`→
  review_required, `digits_spelled_out`→verified_equivalent (now strictly
  verified, no longer includes unverifiable cases). New `verified_total_rate`
  = (exact + verified)/checked; `consistency_rate` semantics unchanged
  (exact-digit preservation only). **No heuristic count is exposed as a
  verified-error rate** (pinned by test).

## 3. Supported / unsupported forms

**Supported for verification:** units 1–10, teens 11–12, exact tens 20–90,
100, 1000 (Yoruba + English), tone/underdot variants of those.

**Unsupported → `numeric_review_required`:** compound Yoruba numerals
(`ogójì ọ̀kànlá` 41, `aádọ́rùn-ún-dín-ọ̀kànlá` 69 — vigesimal analytic
compounds), dates, clock times, ranges, scores (`7–2`), percentages with unit
mismatch, currencies, and any value >1000 not expressible by a trusted word.

## 4. Before/after warning counts (beam-A/B, saved predictions only)

| arm         | checked | exact | verified-word | mismatch | review-required |
| ----------- | ------- | ----- | ------------- | -------- | --------------- |
| eng_yor\|b2 | 11      | 6     | 1             | 3        | 1               |
| eng_yor\|b4 | 11      | 7     | 1             | 2        | 1               |
| yor_eng\|b2 | 6       | 5     | 0             | 1        | 0               |
| yor_eng\|b4 | 6       | 5     | 0             | 1        | 0               |

Before (Sprint 2A): eng_yor arms reported 2 `digits_spelled_out` each — under
the stricter rule only **1** per eng_yor arm is genuinely verified; the
former "11:29" case is now honestly `numeric_review_required`. `consistency_rate`
is unchanged in all arms (0.5455/0.6364/0.8333/0.8333) since it was already
exact-digit based. **All BLEU/chrF/latency values identical** — no prediction
or metric drift. **These are heuristic-warning counts, not verified
translation-error rates**; verified error rates require the human-review
labels (still unavailable).

## 5. Tests

- New: `tests/test_phase3_sprint2cq.py` — 18 tests covering every brief
  requirement: 11→eleven verified, 11→twelve mismatch, unrelated word not
  verified, 11:29→"eleven" review-required, 7–2→72 mismatch, 9,000→900
  mismatch, 50%→"fifty percent" verified, tone variants resolve, compounds
  review-required, duplicate-key cleanup, empty inputs, multiple/duplicate
  numbers, Unicode equivalence, tokenizer split regression, legacy keys.
- Updated two Sprint-2A tests to the stricter (correct) semantics, with
  comments explaining the change.
- Suite: **153 passed, 1 skipped.**

## 6. Remaining limitations

1. Compound Yoruba numerals are unverified by design — fluent-speaker review
   is the only reliable adjudication (worksheet pending).
2. Unit semantics (%/currency/date-order) are out of scope; digit equality
   can miss a unit change ("50%" → "50 dollars").
3. English number words beyond 12 rely on exact tens; "thirteen"…"nineteen"
   not yet in the table (conservative: review, not verified).
4. Word-equivalence assumes the number word's _value_, not its contextual
   role ("eleven" in a different part of the sentence could coincidentally
   match; multi-number sentences mitigate this).
5. Human-review labels remain unavailable; no verified-error rate exists yet.
