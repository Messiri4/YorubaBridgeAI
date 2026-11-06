# Translation Quality Evaluation — Baseline Report

**Model:** `facebook/nllb-200-distilled-600M` (local safetensors copy, CPU, beams=4)
**Date:** 2026-11-05 · **Dataset:** `data/eval/en_yor_eval.json` (20 entries)
**Code:** `app/services/evaluation.py` · **Raw outputs:** `data/eval/results.json`

## Headline numbers

| Direction        | chrF (primary) | BLEU | Entries         | Mean latency |
| ---------------- | -------------- | ---- | --------------- | ------------ |
| English → Yoruba | **37.2**       | 29.4 | 11 (9 verified) | ~4.4 s       |
| Yoruba → English | **52.9**       | 44.2 | 9 (8 verified)  | ~2.0 s       |

### After phrase overrides (same eval set, same model)

| Direction        | chrF     | BLEU | Δ chrF |
| ---------------- | -------- | ---- | ------ |
| English → Yoruba | **76.2** | 68.8 | +39.0  |
| Yoruba → English | **90.1** | 86.6 | +37.2  |

The override table (`data/phrases/overrides.json`, 27 curated exact-match
phrases) fixes the greetings/market-phrase failures identified below and
returns them in ~0 ms. **Important caveat:** the eval set overlaps with the
table by design (both were built from the same domains), so these deltas
measure the table's coverage of known weaknesses, not general model quality.
A held-out set of table-absent sentences is needed for honest model numbers.

**Why chrF leads:** BLEU's 4-gram precision is meaningless on sentences this
short — a _perfect_ single-sentence translation scored BLEU 0.0 in our tests.
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
4. **Tense drift** — `Mo ń lọ` (progressive "I am going") → "I _was_ going";
   `Òjò ń rò` → "It _was_ raining". Progressive aspect (ń) is read as past.
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

## FLORES-200 held-out evaluation (the honest test)

**Dataset:** official Meta FLORES-200 (CC-BY-SA 4.0) — 997 dev + 1012 devtest
aligned EN–YO sentences of Wikipedia/news-domain formal text. Prefixes used:
60 dev / 80 devtest lines (`data/flores/`, provenance in its README).
**Deliberately out of the phrase table's chat/commerce domain** — a genuine
leakage-free test of the model itself. Config: beams=2, same 600M model.

### Results

| Split        | Direction | Config                  | chrF     | BLEU | Mean latency |
| ------------ | --------- | ----------------------- | -------- | ---- | ------------ |
| dev (40)     | eng→yor   | baseline (model only)   | **27.8** | 5.3  | 21.6 s       |
| dev (40)     | eng→yor   | full system (overrides) | **27.8** | 5.3  | 25.2 s       |
| devtest (40) | eng→yor   | full system             | **25.1** | 4.7  | 28.3 s       |
| devtest (40) | yor→eng   | full system             | **44.4** | 20.7 | 13.9 s       |

Raw outputs and 8 sample translations per cell:
`data/eval/flores_results.json`, `data/eval/flores_devtest_results.json`.

### Findings

1. **Phrase overrides add ZERO accuracy on held-out, out-of-domain text** —
   baseline and full system scored identically (27.8/27.8) because no FLORES
   sentence matches the 27 curated chat phrases. The honest characterization:
   the table is a _domain patch_ (human-verified correctness + ~0 ms latency
   on WhatsApp-style phrases), NOT a general model improvement. Our earlier
   +39 chrF gain was real but domain-bounded; this run quantifies that
   boundary.
2. **Domain dominates quality:** chrF ≈ 76 on short chat-domain sentences vs
   ≈ 25–28 on formal paragraph text. NLLB-600M is dramatically better at the
   exact use case (WhatsApp-style messages) this project targets.
3. **Direction gap persists:** yor→eng (44.4) ≫ eng→yor (25.1) on formal
   text — generation into a low-resource language remains the hard side.
4. **CPU latency on long formal sentences:** 14–28 s/sentence at beams=2.
   Fine for async messaging; incompatible with interactive use.

### What would actually move FLORES numbers (not done, evidence-based)

- `nllb-200-1.3B`: literature and Meta's own reports show a large chrF jump
  from 600M→1.3B; costs 2× RAM (~5 GB, fits our 16 GB) and ~2× latency.
- Decoding length: many eng→yor outputs truncate mid-sentence
  (`max_new_tokens=256` is short for FLORES' ~25-word sentences); raising it
  would help recall but costs minutes/sentence on this CPU.
- Actual fine-tuning on FLORES dev — explicitly out of scope per project
  mandate (pretrained-only).
