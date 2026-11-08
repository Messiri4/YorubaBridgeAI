# Translation Quality Evaluation — Baseline Report

> **Update 2026-11-06 (`quality-review` branch):** a second investigation
> pass was run on top of this baseline. New findings and measurements are in
> the final section ["2026-11-06 follow-up investigation"](#2026-11-06-follow-up-investigation).
> All baseline numbers below are unchanged and remain the reference point.

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


## 2026-11-06 follow-up investigation

### 1. Root cause of the "Ẹ kú ìrọ̀lẹ́, ọ̀rẹ́ mi." → "Thank you, my friend." bug

**Confirmed cause: the phrase-override lookup, not NLLB itself.**

The override table contains `"ẹ kú ìrọ̀lẹ́." → "Good evening."`, but the
normalizer kept `.` and `,` as part of the key. The input ends with
`..., ọ̀rẹ́ mi.` so normalization produced `ẹ kú ìrọ̀lẹ́, ọ̀rẹ́ mi.` — no
exact key — and the request **fell through to NLLB**, which renders
`ẹ kú ìrọ̀lẹ́` inside a compound sentence as a thanks-idiom ("Thank you,
my friend."). Verified: with a fake model that raises on `generate()`,
the old code called the model for this exact input; `"Ẹ kú ìrọ̀lẹ́."`
(correctly) hit the table.

**Fixes (`quality-review` branch):**

1. `app/services/translation.py` — `PhraseOverrides._normalize` now strips
   `. , ! ? ; :`, collapses whitespace and lowercases (NFC still applied;
   **tone marks are preserved and remain meaningful**). A trailing period no
   longer makes an identical phrase miss the table.
2. `data/phrases/overrides.json` — added explicit greeting+addressee keys:
   `ẹ kú ìrọ̀lẹ́ ọ̀rẹ́ mi → Good evening, my friend.` (plus morning/afternoon
   variants, and the reverse eng→yor entries `good evening, my friend.` /
   `good morning, my friend.`).

Punctuation stripping was deliberately made conservative: a different
addressee (`Ẹ kú àárọ̀, àbúrò mi.`) still misses and goes to the model —
regression-tested in `tests/test_phrase_overrides.py`
(`test_no_false_match_when_recipient_differs`,
`test_diacritics_distinguish_phrases`).

**Verified before/after (real service, quality-review branch):**

| Input (yor→eng) | Before | After |
| --- | --- | --- |
| `Ẹ kú ìrọ̀lẹ́, ọ̀rẹ́ mi.` | Thank you, my friend. (model, ~2 s) | **Good evening, my friend.** (override, <1 ms) |
| `Ẹ kú àárọ̀, ọ̀rẹ́ mi.` | model output (unreliable) | **Good morning, my friend.** (override, <1 ms) |
| `Ẹ kú ìrọ̀lẹ́.` | Good evening. | Good evening. (unchanged) |
| `Ẹ kú àárọ̀, àbúrò mi.` | model (correct behavior) | model (still misses — intended) |

Reverse direction also fixed: `Good evening, my friend.` (eng→yor) used to
come out as `Àrọ̀lé, ọ̀rẹ́ mi.` from the model; it now hits the table and
returns `Ẹ kú ìrọ̀lẹ́, ọ̀rẹ́ mi.` at ~0 ms.

### 2. Error analysis of existing FLORES results (from raw predictions)

Reviewing `data/eval/flores_results.json` / `flores_devtest_results.json`:

- **eng→yor:** diacritics are consistently present (e.g. `àtọ̀gbẹ`,
  `ìṣègùn`) but the sub-dot notation (`ọ̀`, `ẹ̣`, `ṣ̣`) is an under-dot
  encoding that differs from FLORES references' plain combining forms —
  this alone depresses chrF. Real errors: structural rewrites
  (`Ó fi kún un pé:` for "he added"), dropped trailing punctuation on long
  sentences, lexical drift (`ìrànlọ́wọ́` vs ref `asisti` for "assist").
- **yor→eng:** the same sentence renders `ẹ kú ìrọ̀lẹ́` fine in isolation
  but fails in larger contexts; formal-text errors are mostly **semantic**
  (`ẹku` "mice" → "chicks ... baptized"), register drift, and occasional
  negation flips (`a kò ṣe nǹkan` → "I'm going to do something").
- **Truncation:** re-measured — 0/30 dev hypotheses lack terminal
  punctuation at `max_new_tokens=256`, so truncation is not the dominant
  eng→yor error on this subset (contrary to the earlier hypothesis in the
  baseline notes; it may matter on longer devtest paragraphs).

### 3. Decoding-parameter experiment (FLORES **dev** only, devtest untouched)

Config: NLLB-600M safetensors snapshot, CPU (4 torch threads), sequential,
n=30 dev eng→yor sentences, chrF++ = sacrebleu `corpus_chrf(word_order=2)`.

| Config | chrF++ | BLEU | mean latency |
| --- | --- | --- | --- |
| beams=4, max_new=256, length_penalty=1.0 (baseline) | 26.20 | 5.02 | 12.7–15.3 s |
| **beams=4, max_new=400, length_penalty=1.1 (adopted)** | **26.45** | **5.03** | 14.9–15.1 s |
| beams=8, max_new=400, length_penalty=1.1 | 25.87 | 4.35 | 20.6 s |

Only 3/30 outputs changed between baseline and adopted config — a small,
real-but-modest gain, at no truncation cost. beams=8 is worse and slower:
rejected. Adopted as defaults `translation_length_penalty=1.1`
(`app/core/config.py`), documented in `translation.py`.

yor→eng with adopted params on dev (n=30): chrF++ 44.27, BLEU 20.39 —
consistent with the devtest baseline (44.44/20.71); no regression.

Raw hypotheses: `data/eval/flores_dev_improved.json`,
`data/eval/decoding_exp_dev.json`.

### 4. Phrase overrides vs genuine model improvement (the honest split)

- **Override-table gains are domain patches, not model gains.** The
  greeting/addressee fix moves WhatsApp-style inputs from wrong to correct
  instantly, but contributes **zero** on FLORES (no FLORES sentence matches
  a table key). This run's table fix improves *product* quality on the
  target domain; the decoding tweak (+0.25 chrF++ dev) is the only
  *general-model* change, and it is small.
- The 76–90 chrF numbers from the in-house eval set remain dominated by
  table overlap and must not be read as model quality (see baseline
  caveat above — still true).

### 5. Environment / reproducibility notes

- This Intel Mac (macOS x86_64) tops out at torch 2.2.2 — PyTorch stopped
  shipping macOS x86_64 wheels after that. transformers ≥4.57 then refuses
  to `torch.load` the repo's `pytorch_model.bin` (CVE-2025-32434 guard).
  **Workaround (no new downloads):** the HF cache already contained a
  `model.safetensors` snapshot; `.model-local/nllb-600M-fixed/` assembles
  it (symlinks + re-serialized safetensors with tied embeddings) and
  `load_nllb()` now honors a local directory via
  `TRANSLATION_MODEL_ID=.model-local/nllb-600M-fixed`.
  `.model-local/` is git-ignored.
- **Postponed due to resources:** the full 80-line devtest re-evaluation
  (≈30–60 min CPU while other jobs run) and a larger-N dev run. The devtest
  split was deliberately **not** touched; when re-run, use:
  `TRANSLATION_MODEL_ID=.model-local/nllb-600M-fixed python /tmp/deveval.py`
  with `data/flores/devtest.*` and N=80.

### 6. Remaining limitations

1. eng→yor chrF++ ≈ 26 on formal text is a **model-capability ceiling** for
   600M; the documented path to a real jump is `nllb-200-1.3B` (2× RAM,
   fits 16 GB) — not evaluated here to avoid new downloads under resource
   constraints.
2. Under-dot vs combining-diacritic normalization between hypothesis and
   reference likely costs chrF points; a Unicode-compatibility
   normalization step in scoring (not in the product) would make FLORES
   numbers more comparable. Not implemented (scoring-methodology change).
3. The override table stays exact-match; near-miss inputs ("Ẹ kú ìrọ̀lẹ́
   ọ̀rẹ́ mi ọ?" etc.) fall to the model. A fuzzy/segment matcher is future
   work and was deliberately not added blind.
