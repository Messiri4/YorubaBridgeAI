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
