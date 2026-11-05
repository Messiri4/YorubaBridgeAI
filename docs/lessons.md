# YorubaBridge AI — Implementation Log & Lessons

Running engineering journal: what was built, why, real test results, and
limitations discovered along the way. Newest milestones at the top.

---

## Milestone 1 — Project skeleton (commit 5f5f784)

FastAPI app factory, typed settings (pydantic-settings), health endpoint,
pytest/ruff/mypy tooling, Dockerfile + compose, Makefile.

**Learned:** app-factory pattern; settings validated at startup fail fast;
`@lru_cache` singleton settings with `cache_clear()` for test isolation.

## Milestone 2 — NLLB translation service (commit 02ddd9b)

- `app/services/model_loader.py` — `load_nllb()` + `LoadedModel` bundle +
  `SUPPORTED_LANGUAGES` (`eng`→`eng_Latn`, `yor`→`yor_Latn`, Flores-200 codes).
- `app/services/translation.py` — `TranslationService.translate()`:
  validate → tokenize (source-language prefix) → `generate(forced_bos_token_id)`
  → decode (`skip_special_tokens=True`) → `TranslationResult` with latency.
- `app/models/translation.py` — Pydantic schemas; language regex + non-empty
  text enforced at the API edge, direction check in the service.
- Lifespan-based loading: `import app.main` never touches the model; weights
  load when the server starts.

**Errors hit & fixed:**

- mypy caught `_validate` called but never defined → implemented it.
- transformers 5.x needs torch ≥ 2.6; Intel macOS tops out at torch 2.2.2
  (PyTorch dropped x86_64 macOS wheels) → pinned `transformers<5`, `numpy<2`.
- `create_app()` at import time eagerly loaded the model → moved to lifespan.

**Tests:** 13 mocked unit tests pass in ~3 s, zero network. Smoke tests
(real model) marked and excluded by default.

## Milestone 3 — Redis cache + rate limiting (commit 123096f)

- `app/services/cache.py` — content-addressed keys
  `translate:{src}:{tgt}:{sha256(NFC-normalized text)}`; every Redis call
  wrapped: Redis down → pass-through cache (never a 500).
- `app/core/rate_limit.py` — fixed-window counter (Redis INCR/EXPIRE,
  in-memory fallback), 10 req/60 s default, per-IP via X-Forwarded-For.
- Response gained `"cached": true|false`.

**Learned:** cache-aside pattern; NFC normalization matters because Yoruba
diacritics can be composed or decomposed (ọ vs o + combining dot); graceful
degradation — infrastructure failure should cut performance, not availability.

## Milestone 4 — Real model on Intel Mac (the hard-won lesson)

The checkpoint `facebook/nllb-200-distilled-600M` ships **only**
`pytorch_model.bin` (pickle). Transformers ≥ 4.55 refuses pickle weights on
torch < 2.6 (CVE-2025-32434). Intel Mac caps at torch 2.2.2 → dead end unless
converted. Resolution: one-off script loaded the `.bin` with
`weights_only=True` and re-serialized 512 tensors to `model.safetensors`
(5.61 GB — larger than the `.bin` because safetensors stores tensors
contiguously; pickle deduplicates tied embeddings). Copy lives at
`~/models/nllb-200-distilled-600M-safetensors`, referenced from `.env` via
`TRANSLATION_MODEL_ID` (gitignored — no secret, but machine-specific).

**Verified numbers (Intel CPU):**

- Load from local safetensors: **3.9 s** (memory-mapped)
- Params: **615.1M**, class `M2M100ForConditionalGeneration`, device `cpu`
- First translation 8.6 s; steady-state **1.9–6.8 s/sentence** with beams=4

**How NLLB works (condensed):** encoder–decoder Transformer, one shared
SentencePiece vocab (~256k tokens, byte-fallback so diacritics survive).
Source sequence is _prefixed_ with the source language token; the decoder's
first emitted token is forced to the target language id
(`forced_bos_token_id=tokenizer.convert_tokens_to_ids("yor_Latn")` = 256198).
Generation is autoregressive beam search until `</s>` or `max_new_tokens`.

**Tokenization example (real):**
`"Good morning, how are you?"` →
`[256047, 24718, 64246, 248079, 11657, 2442, 1259, 248130, 2]` =
`['eng_Latn','▁Good','▁morning',',','▁how','▁are','▁you','?','</s>']`.
`▁` marks word starts; combining marks (the dot under ọ) become their own tokens.

## Milestone 5 — Real-model validation before integrations (rule 9)

- `make test-smoke`: **2 passed** (12.4 s) — load + eng→yor + yor→eng round trip.
- 5+5 example battery (all real inference, beams=4):

| English source                     | NLLB Yoruba                      | Yoruba source               | NLLB English                       |
| ---------------------------------- | -------------------------------- | --------------------------- | ---------------------------------- |
| Good morning, my friend.           | Àárọ̀ àárọ̀, ọ̀rẹ́ mi.               | Ẹ kú àárọ̀, ọrẹ́ mi.          | Thank you very much, my friend.    |
| I am learning Yoruba.              | Mo ń kó èdè Yorùbá.              | Mo kọ́ ẹ̀kọ èdè Yorùbá.       | I learned the Yoruba language.     |
| Where is the market?               | Ibo ni ọjà náà wà?               | Nibo ni ọjà wà?             | Where is the market?               |
| The rain fell heavily last night.  | Òjò rò gan-an lálé ọjó kẹ́yìn.    | Òjọ́ ṣubú lágbára àná.       | The sun was setting strongly.      |
| Thank you very much for your help. | Mo dúpé gan-an fún ìrànlọ́wọ́ yín. | E ṣé púpọ̀ fún ìrànlọ́wọ́ yín. | Thank you very much for your help. |

**Quality notes (honest):** round trips preserve meaning; some outputs are
non-idiomatic ("Àárọ̀ àárọ̀" vs natural "Ẹ kú àárọ̀"; "rain" sentence rendered
with shifted tense marks; `ṣubú` misread as "sun setting"). Expected for a
600M low-resource model. Options later: `nllb-200-1.3B` (better, ~2× RAM) or
fine-tuning (out of scope — we use pretrained only).

**Next milestone:** serve the real model through the FastAPI endpoint and
verify end-to-end with curl, including cache and 429 behaviour.
