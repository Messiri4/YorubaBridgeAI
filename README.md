# YorubaBridge AI

English ↔ Yoruba text and voice translation service, WhatsApp-ready.

## Stack

Python 3.11 · FastAPI · PyTorch + Hugging Face Transformers · PostgreSQL · Redis · Docker · Meta WhatsApp Cloud API

## Local setup

```bash
make install
cp .env.example .env   # then edit values
make run
```

Open http://localhost:8000/docs for the interactive API docs.

## Commands

| Command        | What it does                      |
| -------------- | --------------------------------- |
| `make install` | Create venv + install all deps    |
| `make run`     | Run the API with autoreload       |
| `make test`    | Run pytest                        |
| `make lint`    | ruff + mypy                       |
| `make format`  | Auto-format with ruff             |
| `make db-up`   | Start Postgres + Redis containers |

## Implementation progress

- [x] Project skeleton, tooling, CI-ready config
- [x] NLLB-200 EN↔YO translation service (mocked + smoke tests)
- [x] Redis cache + rate limiting (graceful degradation)
- [x] Real-model validation on CPU (smoke tests + 10-example battery)
- [x] Engineering journal: [`docs/lessons.md`](docs/lessons.md)
- [ ] End-to-end API verification with the real model
- [ ] WhatsApp Cloud API webhook
- [ ] Postgres persistence (translation history)

## Note for Intel Mac users

`facebook/nllb-200-distilled-600M` ships only a pickle (`.bin`) checkpoint,
which recent transformers refuses to load with torch < 2.6 (CVE-2025-32434).
Intel Macs are capped at torch 2.2.2, so a one-time conversion to safetensors
is required — see `docs/lessons.md` Milestone 4. Point
`TRANSLATION_MODEL_ID` in `.env` at the converted directory.
