# Translation Quality Evaluation — Baseline Report

**Model:** `facebook/nllb-200-distilled-600M` (local safetensors copy, CPU, beams=4)
**Date:** 2026-11-05 · **Dataset:** `data/eval/en_yor_eval.json` (20 entries)
**Code:** `app/services/evaluation.py` · **Raw outputs:** `data/eval/results.json`

## Headline numbers

| Direction | chrF (primary) | BLEU | Entries | Mean latency |
|---|---|---|---|---|
| English → Yoruba | **37.2** | 29.4 | 11 (9 verified) | ~4.4 s |
| Yoruba → English | **52.9** | 44.2 | 9 (8 verified) | ~2.0 s |

**Why chrF leads:** BLEU's 4-gram precision is meaningless on sentences this
short — a *perfect* single-sentence translation scored BLEU 0.0 in our tests.
chrF (character n-gram F-score) degrades gracefully and reflects Yoruba's
morphology, so chrF is our primary metric; BLEU is reported for convention.

**Yor→Eng outperforming Eng→Yor is expected:** generation into a
low-resource language is harder than understanding it.

## Error taxonomy (recurring patterns found)

1. **Reduplicated/unidiomatic greetings** — "Good morning." → `Àárọ̀ àárọ̀.`
   (natural: `Ẹ kú àárọ̀.`); "Good night." → `Ìròlé dáradára.` ("good evening,
   it's fine" — wrong register). Greetings are highly idiomatic in Yoruba;
   literal translation fails. This is the single biggest quality gap.
2. **Semantic drift on short questions** — "How are you?" → `Báwo ni nǹkan ṣe
   rí fún ẹ?` (over-explicit, ~"how do things appear to you?"); reverse
   direction rendered `Báwo ni?` as "In what way?" — missing the greeting
   sense. `Eèló ni èyí?` ("how much is this?") came back as **"Is this
   nonsense?"** — a serious mistranslation of a common market phrase.
3. **Additive style** — "Thank you very much." → `Mo dúpé lọ́wọ́ yín gan-an`
   (adds emphatic `gan-an` not in source; adds first-person `Mo` where the
   Yoruba register uses bare `O ṣé`).
4. **Tense drift** — `Mo ń lọ` (progressive "I am going") → "I *was* going";
   `Òjò ń rò` → "It *was* raining". Progressive aspect (ń) is read as past.
5. **Diacritics: actually good.** No missing tone marks observed in either
   direction — outputs consistently carry correct tone marks on core
   vocabulary (`ọjà`, `òjò`, `dúpẹ́`, `ṣé`). The earlier suspicion of tone-mark
   loss is not supported by this run.
6. **Provisional entries behaved reasonably** — `conv-003/004` outputs were
   plausible paraphrases; `Ó ti tó.` → "That is enough." is correct.

## What this means for the product

- **Market/shopping phrases (the WhatsApp use case!) are the weakest area**
  (`Eèló ni èyí?` failure is alarming for a commerce assistant). Before
  WhatsApp launch we should either (a) switch to `nllb-200-1.3B`, or (b) add a
  small phrase-base override table for the most common 50–100 exchanges.
- Baseline is recorded: any future change (bigger model, phrase table,
  decoding params) must be re-measured against these numbers.
- Latency on Intel CPU (2–12 s/sentence) is acceptable for asynchronous
  WhatsApp messaging, not for interactive typing.

## Method notes & limitations

- 20 sentences is tiny; scores have wide error bars. chrF differences of ±5
  between runs/configs are not meaningful at this size. Grow the set before
  comparing checkpoints seriously.
- 3 references are marked `verified: false` (provisional); corpus scores
  blend both. Verified-only subset: eng_yor chrF ≈ 35.4 (recompute with
  `run_evaluation` after filtering if needed).
- sacreBLEU's default 13a tokenizer is English-biased; for Yoruba this mostly
  affects BLEU, another reason chrF (character level) is primary here.
- Single reference per entry: MT systems legitimately produce synonyms
  ("Ibo ni" vs "Nibo ni" for "where") that get penalized. Multi-reference
  eval would soften this.
