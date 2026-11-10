# Translation Error Analysis Report — beam A/B FLORES dev (30 pairs × 2 directions × 2 beams)

**Source:** `beams_ab_20261109T231413Z.csv` (120 predictions, 60 aligned sentences).
**Method:** deterministic heuristics only (`scripts/analyze_translation_errors.py`);
every finding is a **warning for review**, not a verdict. No model inference, no
downloads. Machine-readable companions:
`translation_error_analysis.csv` (314 findings), `translation_error_summary.json`,
`translation_error_review.csv` (per-prediction review sheet).

## 1. Methodology and limitations

Heuristics (all deterministic, thresholds in `translation_error_summary.json`):
number conservation (digit multisets), named-entity carryover (exact → accent-folded
→ consonant-skeleton transliteration match), length ratios vs reference, repeated
content n-grams, verbatim copied source spans, beam-pair token Jaccard, sentence
chrF < 30, and two _reference-side_ checks (tone marks present, digit fidelity).

Limitations — treat every count as a lower bound or a prompt, never as truth:

- Spelled-out numbers ("ọgọ́rin" = 80) can't be matched to digits without an
  understanding of Yoruba numerals; we handle the frequent cases with a
  numeral-convention guard, but mixed forms may still slip through.
- Entity carryover accepts partial transliterations; genuinely wrong entity
  _spellings_ (Alex Ovechkin → "Overchkin") are flagged as UNTRANSLATED/copied, not judged.
- Semantic errors (negation flips, wrong-but-fluent claims) are invisible to
  character heuristics — those below were found by manual reading, marked as such.
- Sentence chrF against a single noisy reference is a weak signal (verified-error ≠ low-chrf).
- The 30-pair dev sample carries ±1 chrF-scale noise; counts of rare categories
  (≤5) are anecdotal.

## 2. Error counts by category and direction (heuristic findings)

| Category                   | eng→yor (60 preds) | yor→eng (60 preds) | Kind                |
| -------------------------- | ------------------ | ------------------ | ------------------- |
| ENTITY_MISSING             | 41                 | 15                 | model               |
| REPEATED_NGRAM             | 48                 | 18                 | model               |
| LOW_CHRF (<30)             | 38                 | 5                  | model               |
| UNTRANSLATED (copied span) | 16                 | 22                 | model               |
| NUMBER_CHANGED             | 5                  | 10                 | model               |
| NUMBER_MISSING             | 4                  | 0                  | model               |
| TOO_LONG                   | 2                  | 0                  | model               |
| BEAM_DIVERGENCE            | 4                  | 2                  | info                |
| REF_NO_DIACRITICS          | 30 unique refs     | –                  | reference           |
| REF_DIGIT_MISMATCH         | 12                 | 12                 | reference           |
| NUMERAL_CONVENTION_NOTE    | –                  | 8                  | info (not an error) |

104/120 predictions carry at least one model-side warning. Severity: 15 high,
207 medium, 2 low, 90 info.

## 3. Most serious findings (fact-changing; manually confirmed where noted)

**eng→yor**

1. **#3 (both beams) — score corrupted, VERIFIED.** "head to head record … is 7–2"
   → beams=2: `… jẹ̣ 72.` (seventy-two!), beams=4: `jẹ̣ 7 2.` — neither renders the
   7–2 score correctly; the reference uses words (`àmì méje sóódo`). Factual corruption.
2. **#21 — time invented, VERIFIED.** beams=4 inserts `[11:29]` literally and a
   garbled `àádọ́rùn - ún` compound; beams=2 also garbles "11:29" (`aago mọ́kànlá
àádọ́rùn - ún ọdún` — "11 ninety - un year"). Time expression is the model's weakest slot.
3. **#13 — number format drift (both beams).** "9,000-year lease" → beams=2 drops
   "9,000" and writes `ọdún mọ́kàndínlógójì` ("39 years"?!); beams=4 keeps `[9,000]`
   in brackets — different error modes for the same fact.
4. **#25 (both beams) — stat mishandled.** "2 goals and 2 assists" → `góòlù méjì àti
ìsọfúnni/ìrànlọ́wọ́ méjì` but the "5–3" survives only as digits; "2–2 vs 5–3"
   pairing is fragile (flagged NUMBER_CHANGED 2,3,5 → 3,5).
5. **#10 beams=4 — time dropped**, "At 11:20" → `aago mọ́kànlá ààbọ̀` (nonsense "11
   protection"); beams=2 kept "11:20" raw — this is a genuine beams=2 win.

**yor→eng**

6. **#2 (both beams) — office swapped, VERIFIED.** `alákòso ìsúra` (treasurer) →
   "prime minister" in _both_ beams. Factual inversion, unnoticed by chrF (46.9).
7. **#6 (both beams) — causal inversion, VERIFIED.** "when nuclear power becomes
   economically viable" → "if … becomes economically _disadvantaged_"; also reads
   `ìran alágbára ohun ìjagun apanirun` as "powerful generation of nuclear weapons".
   Polarity + entity sense both wrong; beams=4 names "weapons of mass destruction"
   (closer literal) while beams=2 says "nuclear weapons" (closer intent).
8. **#4 (both beams) — actant flip, VERIFIED.** "documents showed fourteen banks
   helped wealthy clients _hide_ money" → "14 banks have helped _investors invest_
   billions … money laundering and mining law". Double distortion; only the digits survive.
9. **#29 — biggest beam divergence (sim 0.16).** beams=2 invents "statement of
   funds"; beams=4's "issued a fine" is closer to `owó ìtanràn` (penalty money) —
   a concrete beams=4 improvement on a legal-procedure sentence.
10. **#22 (both beams) — roles reversed:** "bosses had been reined in _by_
    Provenzano" → "Profenasano led these forces/overthrew these leaders"; names
    also mangled (`Gifani Falkoni`, `Paolo Boselino` vs reference names).

## 4. Beam width: where it changes the result, and which side benefits

26/30 eng→yor and 15/30 yor→eng sentences differ between beams. Consistent pattern:

- **eng→yor, beams=4 better:** #0 "stories" → `storylines` (b4) vs duplicated
  `ìtòlẹ́sẹẹsẹ ìtòlẹ́sẹẹsẹ` (b2) — beams=2 literally repeats "sequel sequel"; #4 b4
  avoids the b2 double `tó wà` glitch; #1 b4's terse `òṣèré ọjọ́ òní` matches the
  reference's framing ("Player of the Day") though it drops the award clause.
- **yor→eng, beams=2 better:** #29 above (b4 hallucination avoided), #10/#18
  numeral handling equal, and corpus scores confirm: chrF 45.99 vs 45.51, BLEU
  19.21 vs 18.55 favor b2; eng→yor favors b4 (28.76/4.57 vs 28.12/3.86).
- **Both-beams-agree failures (systemic, not decoding):** the six verified
  fact-changing errors #2, #3, #4, #6, #13, #25 are near-identical in both arms —
  decoding tweaks cannot fix these; they are model-knowledge gaps.

## 5. Reference-quality findings (kept separate; dataset untouched)

- **6 of 30 eng→yor references have ZERO tone marks** (#4, #5, #9, #12, #16, #22),
  e.g. `Awon iIwe naa fihan ile ifowopamo merinla…` — mixed accent-less Yoruba with
  the typo-like `iIwe`. One more has <10 marks. These depress sentence chrF and
  teach the scorer to reward unaccented output.
- **Reference #3 is itself awkward** (`sàkọọ́lẹ̀ ìforígbárí o ní àmì méje sóódo` —
  head-to-head "record" rendered unusually), which punishes any fluent rendering
  of "7–2"; _both_ beams score LOW_CHRF here partly because of the reference.
- **REF_DIGIT_MISMATCH (24 rows):** references legitimately spell numbers in words
  where the source uses digits ("fourteen banks", "Ní ọjọ́ Kẹrìnlélógún…"); this is
  corpus convention, not error — but it makes digit-recall impossible to score
  mechanically, which is why our number heuristics cross-check the reference too.
- No network/formatting anomalies beyond these; references otherwise align.

**Impact on the evaluation:** with 20% of eng→yor references unaccented and number
style mismatched, FLORES dev eng→yor chrF (~28) understates genuine output quality;
yor→eng references are cleaner (they're native English) — consistent with
yor→eng scoring higher despite equal-or-worse model behavior.

## 6. Recommended improvements (ranked by expected impact × effort ÷ risk)

1. **Number/date/time handling layer (highest impact, low risk).** Wrap the model
   with a deterministic post-check: extract source numerals/times, verify presence
   in output (digits _or_ an accepted Yoruba-numeral wordlist), and re-translate
   or annotate on mismatch. Directly addresses the 6 verified fact-changing
   number/time errors (#3, #13, #21, #25 both beams). Effort: small–medium;
   no model changes.
2. **Normalize evaluation orthography before scoring (high impact on measurement
   fidelity, zero product risk).** Accent-fold _references and hypotheses_ to one
   standard (either compose under-dots or evaluate accent-insensitively) and
   document split conventions. Fixes the 6 unaccented references' scoring penalty
   and the under-dot vs combining-form mismatch noted in the baseline report.
3. **Named-entity transliteration table for eng→yor (medium impact, low risk).**
   The 41 ENTITY_MISSING findings cluster on sports/politics names
   (Ovechkin/Overchkin, Liberal Party, Trinidad & Tobago). A small
   transliteration lexicon injected into prompts/prefix or post-correction beats
   per-sentence luck. Medium effort; must respect the no-blind-overrides rule
   (add only entity _forms_, not sentence translations).
4. **Direction-dependent beam width (small, already measured):** beams=2 for
   yor→eng, beams=4 for eng→yor — pending devtest re-validation.
5. **Fluent-speaker review queue (process fix):** export
   `translation_error_review.csv` filtered to `high` severity + the six verified
   semantic inversions for native-speaker adjudication before any Phase-3 tuning.

## 7. Explicit review-needed list (cannot be settled automatically)

All rows with `needs-human-review` in `translation_error_review.csv` (104), and in
particular the semantic cases that heuristics cannot judge: #2/#6 (yor→eng
polarity and office titles), #3/#21 (time expressions), #22 (roles), and the six
unaccented references.

---

_All artifacts committed under `data/eval/`; the original beam A/B files were not modified._
