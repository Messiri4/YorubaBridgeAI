# YorubaBridge AI

English ↔ Yorùbá text and voice-note translation service, WhatsApp-ready.

## What it does today

- **Neural translation** both directions (NLLB-200-distilled-600M, CPU) with
  a curated phrase-override layer for WhatsApp-style greetings and market
  phrases (human-verified, ~0 ms)
- **Local web UI** at `http://localhost:8000` — type, pick direction, translate
- **REST API**: validation, per-IP rate limiting, optional Redis cache,
  optional Postgres history
- **Voice notes**: upload audio → Whisper STT (auto language detect) → translation
- **WhatsApp webhook** following Meta Cloud API conventions (mock-tested; live
  use needs a Meta developer account)
- **Quality evaluation** on our 20-sentence set and the official FLORES-200
  held-out benchmark

## Measured quality

| Eval set | Direction | chrF | BLEU |
|---|---|---|---|
| Project set (chat domain) | eng→yor | 76.2 (37.2 baseline) | 68.8 |
| Project set (chat domain) | yor→eng | 90.1 (52.9 baseline) | 86.6 |
| FLORES-200 devtest (formal domain) | eng→yor | 25.1 | 4.7 |
| FLORES-200 devtest (formal domain) | yor→eng | 44.4 | 20.7 |

Interpretation: the system is strong at short conversational messages (its
target use case) and weak on formal paragraph text; phrase overrides only
help in-domain (they add nothing on held-out FLORES — documented honestly in
`docs/translation_evaluation.md`).

## Quick start

```bash
make install
cp .env.example .env      # edit TRANSLATION_MODEL_ID if needed
make run                  # then open http://localhost:8000
```

First run downloads the model (see the Intel-Mac note below).

## Commands

| Command        | What it does                          |
|----------------|---------------------------------------|
| `make install` | Create venv + install all deps        |
| `make run`     | API + web UI with autoreload          |
| `make test`    | Fast tests (mocked; real model off)   |
| `make test-smoke` | Real NLLB model tests (~2 min)     |
| `make test-db` | Live-Postgres tests (needs `db-up`)   |
| `make lint`    | ruff + mypy strict                    |
| `make format`  | Auto-format                           |
| `make db-up`   | Start Postgres + Redis containers     |

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web UI |
| `/health` | GET | Liveness |
| `/api/v1/translate` | POST | `{"text","source_language","target_language"}` → `{"translated_text","latency_ms","cached"}` |
| `/api/v1/voice-translate` | POST | multipart `audio` file → transcription + translation (language auto-detected) |
| `/api/v1/webhook` | GET/POST | Meta WhatsApp verification + inbound messages |

## Implementation progress

- [x] Project skeleton, tooling, CI-ready config
- [x] NLLB-200 EN↔YO translation service (mocked + smoke tests)
- [x] Redis cache + rate limiting (graceful degradation)
- [x] Real-model validation on CPU
- [x] Evaluation: project set + FLORES-200 held-out (`docs/translation_evaluation.md`)
- [x] Phrase-override table (in-domain chrF 37.2→76.2 / 52.9→90.1)
- [x] Local web UI (`/`)
- [x] Voice-note translation (faster-whisper + `/api/v1/voice-translate`)
- [x] WhatsApp webhook (mock-tested; live needs Meta credentials)
- [x] Postgres history (live test pending Docker repair)
- [ ] Live WhatsApp E2E (blocked: Meta developer account)
- [ ] Voice output (TTS) for replies

## Note for Intel Mac users

`facebook/nllb-200-distilled-600M` ships only a pickle (`.bin`) checkpoint,
which recent transformers refuses to load with torch < 2.6 (CVE-2025-32434).
Intel Macs are capped at torch 2.2.2, so a one-time conversion to safetensors
is required — see `docs/lessons.md` Milestone 4. Point `TRANSLATION_MODEL_ID`
in `.env` at the converted directory.

## Documentation

- `docs/lessons.md` — engineering journal: every milestone, error and fix
- `docs/translation_evaluation.md` — methodology, results, error taxonomy
- `data/flores/README.md` — FLORES-200 provenance and licensing
