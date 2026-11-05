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

---

## Milestone 6 — Evaluation baseline (commit 52a0c61)

- `data/eval/en_yor_eval.json`: 20 hand-built entries across greetings,
  conversation, questions, expressions; each flagged `verified: true|false`
  (true = human-confirmed reference, false = provisional).
- `app/services/evaluation.py`: corpus chrF (primary) + BLEU via sacrebleu;
  per-direction scoring; latency captured per entry; results JSON written.
- **Results:** eng→yor chrF **37.2** / BLEU 29.4; yor→eng chrF **52.9** / BLEU 44.2.
- Full error taxonomy in `docs/translation_evaluation.md`. Key finding:
  greetings/market phrases are the weakest (idiomatic gaps); diacritics are
  NOT a problem — tone marks survived every output.

**Learned:** BLEU's 4-gram precision is 0 even for a perfect single-sentence
translation — that failing test taught the metric-selection lesson better than
any lecture. chrF is the honest primary metric at this dataset size.

## Milestone 7 — WhatsApp Cloud API webhook (mocked)

New files:
- `app/services/whatsapp_client.py` — thin Graph API sender
  (`POST /{phone_number_id}/messages`, Bearer token from settings only).
- `app/services/whatsapp_webhook.py` — `verify_signature` (HMAC-SHA256 over
  the RAW request body — re-serializing JSON would break the signature),
  `extract_text_messages` (entries→changes→values→messages, text-only),
  `guess_language` (heuristic: Yoruba markers ṣ/ẹ/ọ/gb/ń/…).
- `app/api/v1/routes_whatsapp.py`:
  - `GET /api/v1/webhook` — Meta verification handshake (echo hub.challenge).
  - `POST /api/v1/webhook` — fail-closed order: no app secret → 503 (can't
    verify anything), bad signature → 403, then translate + reply.
- 8 new tests (signature vectors, payload parsing, handshake, full mocked
  pipeline). Suite: **47 passed**.

**Errors hit & fixed:**
- Tests hit `/webhook` but the router is mounted at `/api/v1/webhook` → 404s.
- Ordering subtlety: with no app secret configured, ANY signature must 503
  (fail closed) — the first version returned 403 for unsigned requests,
  which is wrong because nothing can be verified without the secret.

**Not done (needs Meta developer account, i.e. approval + external signup):**
real app credentials, public webhook URL (ngrok/deployment), live E2E test.

## Milestone 8 — Phrase-override table (quality intervention)

- `data/phrases/overrides.json`: 27 curated exact-match translations for the
  failure domains from the eval taxonomy (greetings, market phrases, politeness).
- `app/services/translation.py`: `PhraseOverrides` class — JSON table keyed by
  NFC-normalized lowercase source phrase, O(1) lookup, direction-specific;
  missing/invalid file degrades to empty table (model-only mode).
- `translate()` consults the table BEFORE the model: exact match returns
  immediately (~0 ms), everything else hits NLLB as before.
- mypy rejected a lambda for normalization → replaced with a typed static
  method (typed codebases want real functions, not lambdas).
- Three existing tests failed *because the table started working*: their
  inputs ("Good morning") now return curated overrides instead of mock model
  output. Fixed by choosing off-table inputs so the model path stays tested.
- Re-measured on the real model: eng→yor chrF **37.2 → 76.2**, yor→eng
  **52.9 → 90.1**. Caveat documented: eval set overlaps the table's domains,
  so deltas measure coverage of known weaknesses, not general model quality.

**Learned:** measure before and after any quality change; be explicit about
what a metric delta does and does not prove. Phrase tables are a legitimate
production MT technique (hybrid RBMT+MT), not cheating — but they only cover
what you anticipated.

## Milestone 9 — Postgres translation history

- `app/services/history.py`: `HistoryStore` — one table
  (`translation_history`: langs, texts, latency, cached, override_hit),
  plain parameterized SQL via psycopg 3, no ORM. Schema bootstraps with
  `CREATE TABLE IF NOT EXISTS`; any failure degrades to a logged warning
  and `record()` returns False — persistence can never break translation.
  `make_history_store()` returns None unless `DATABASE_ENABLED=true`
  (the API runs fine with no database).
- Translate route now records every successful translation, including
  whether it came from the phrase table (`override_hit`).
- `tests/test_history.py`: 5 mocked tests + 1 integration test marked
  `integration` (skipped unless `DATABASE_ENABLED=true`; run via
  `make test-db` after `make db-up`).

**Test-design lessons:**
- Two of my new tests asserted wrong assumptions: commit is called twice
  (schema DDL + insert — by design), and my mock's `side_effect` sequence
  forgot the second DDL call. Fixed the tests, not the code — the code's
  behaviour is correct.
- ruff demanded `contextlib.suppress` over try/except-pass, and an explicit
  `zip(strict=)` — small modernization nits caught automatically.

**Environment limitation (honest):** the live-Postgres integration test
could NOT run — Docker Desktop on this machine is unresponsive (the
`docker` CLI hangs even for `docker ps`; port 5432 stays closed). The
mocked suite fully passes (60 passed). When Docker recovers:
`make db-up && make test-db`.
