# FLORES-200 devtest baseline — raw NLLB-200-distilled-600M (20 pairs/direction)

**Run:** 2026-11-09 UTC · **beams=4, greedy tie-break** · overrides disabled · single model load

## Scores (corpus-level, SacreBLEU)

| Direction        | chrF      | BLEU     | Mean latency    | Wall     |
| ---------------- | --------- | -------- | --------------- | -------- |
| English → Yoruba | **27.44** | **4.55** | 45.1 s/sentence | ~15 min  |
| Yoruba → English | **43.28** | 19.62    | 22.5 s/sentence | ~7.5 min |

Raw CSV: `data/eval/flores_baseline_devtest_20_20261109T201142Z.csv`

## Are the low scores genuine or an artifact? — Both, but dominated by genuine difficulty

**Configurations ruled out as causes:**

- Same 20 aligned pairs both directions (programmatic assertion passed).
- References cross-verified: eng_yor refs == yor_Latn lines; yor_eng refs == eng_Latn lines.
- Diacritics intact in references and predictions (combining marks preserved end-to-end).
- Truncation bug found (`max_length` given _characters_ where tokens expected) — **not a factor here**:
  2000 chars ≫ these ~30-word sentences, so nothing was truncated.
- Metrics: correct corpus-level sacreBLEU chrF (char n-grams) + BLEU (13a tokenizer).

**Genuine difficulty dominates:** FLORES devtest is news/technical text —
long, domain-heavy (medical research, Nobel committee, startups). A 600M
distilled model on such text scores low chrF regardless of implementation,
as established in our earlier FLORES analysis (27.8 there, 76.2 in the
chat domain). BLEU 4.55 additionally suffers from the 20-sentence corpus:
with one reference per sentence and inflected Yoruba morphology, exact
n-gram overlap is nearly impossible; chrF is the meaningful figure.

## Five representative errors per direction

### English → Yoruba

| #   | Type                                                                                | Status       | Evidence (source → hypothesis)                                        |
| --- | ----------------------------------------------------------------------------------- | ------------ | --------------------------------------------------------------------- |
| 1   | Wrong entity: "mice" → `ọ̀dọ́ ọ̀dọ́` (young people/babies)                              | **Verified** | medical article; reversed meaning vs `ẹku` (mice)                     |
| 2   | Register error: "professor" → `akẹ́kọ̀ọ́ alákòókò kíkún` (nonsense compound)           | **Verified** | natural term is `ọ̀jọ̀gbọ́n` (used correctly elsewhere!) — inconsistency |
| 3   | Pronoun error: added "he added" → first-person `Ó fi kún un pé` framing mismatch    | Possible     | needs speaker review of attribution framing                           |
| 4   | "Nova Scotia" transliterated but "Swedish Academy" → `Èdè` ("language") + "Swedish" | **Verified** | committee of Literature mistranslated as language                     |
| 5   | Tone-mark inconsistency on `ọ̀gbógi`/`ògbógi` (expert) between sentences             | Possible     | diacritics present but tonal correctness needs native review          |

### Yoruba → English

| #   | Type                                                                                                                                            | Status       | Evidence                                           |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | -------------------------------------------------- |
| 1   | "mice" → "babies"; reversed disease logic ("had pre-diabetic diabetes")                                                                         | **Verified** | meaning inversion of the mouse-experiment sentence |
| 2   | "medicine" → "military science" (`ìmọ̀ ogun`)                                                                                                    | **Verified** | opposite profession                                |
| 3   | Negation dropped: "we are doing nothing" → "We're going to do something"                                                                        | **Verified** | polarity flip, serious for factual text            |
| 4   | Committee title drift: "permanent secretary of the Nobel Committee for Literature" → "secretary-general of the Swedish Nobel Prize for Fiction" | Possible     | plausible paraphrase, needs review                 |
| 5   | "diabetes cure skepticism" compressed to "doubted the cure" losing the "whether it can be cured" conditional                                    | Possible     | subtle, needs review                               |

**Systemic patterns:** biomedical terms (mice/diabetes) are the weakest
area; titles/professions unstable; negation and pronouns occasionally flip;
diacritics themselves are reliable.

## Latency (Intel Mac, CPU-only)

- eng→yor mean **45.1 s**, max 87.4 s; yor→eng mean **22.5 s**, max 36.2 s.
- Asymmetry expected: Yoruba morphology produces more tokens per sentence.
- Bottleneck = CPU beam search (beams=4 over ~256 new tokens on a 615M model).
- Prior optimizations already landed: beams 8→4 (60% latency cut), beams 4→2 (halves again, tiny quality cost). **Next lever: `num_beams=2` for production WhatsApp traffic** (already the setting for FLORES tuning runs) or dynamic-length `max_new_tokens` proportional to input length (~2× for short chat messages).
- **Not a config problem:** identical parameters gave 3–6 s on chat-length sentences; long formal text is simply more decode steps.

## Reproduce

```bash
PYTHONPATH=$PWD .venv/bin/python scripts/evaluate_flores.py --split devtest --n 20 --beams 4
```

## Recommended next steps

1. Fix the `max_length` character/token confusion in `TranslationService` (currently harmless, structurally wrong).
2. Scale to 200–1012 pairs for stable estimates (20 pairs → ±5 chrF noise).
3. For production latency: switch default to beams=2 + dynamic `max_new_tokens`.
4. Keep devtest **held out**; use dev split for any future tuning experiments.
5. Fluent-speaker review of the 5 "possible" errors above.
