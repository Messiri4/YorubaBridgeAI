.PHONY: install venv lint format test run db-up db-down

venv:
	python3 -m venv .venv
	.venv/bin/pip install -U pip

install: venv
	.venv/bin/pip install -r requirements-dev.txt

lint:
	.venv/bin/ruff check app tests
	.venv/bin/mypy app

format:
	.venv/bin/ruff format app tests
	.venv/bin/ruff check --fix app tests

test:
	.venv/bin/pytest

# Real-model tests (downloads ~2.4 GB on first run, needs network)
test-smoke:
	.venv/bin/pytest -m smoke

run:
	.venv/bin/uvicorn app.main:app --reload --port 8000

db-up:
	docker compose up -d postgres redis

db-down:
	docker compose down
