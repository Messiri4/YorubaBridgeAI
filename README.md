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

## Local translation review

Run the review CLI from the repository root using the project's Python
environment. It generates an English-to-Yoruba candidate when none is
provided, then back-translates the candidate and checks for possible digit
discrepancies. A digit warning is only a possible mismatch because Yoruba
may spell a number out. The CLI uses the configured local NLLB checkpoint
(`TRANSLATION_MODEL_ID`) and makes no Google Translate or paid API calls.

```powershell
.\.venv\Scripts\python.exe -m app.review_translation "I bought 3 books."
.\.venv\Scripts\python.exe -m app.review_translation "I bought 3 books." --candidate "Mo ra ìwé mẹ́ta."
```

Providing `--candidate` skips forward translation, which is useful when
reviewing a known candidate. Both modes use the local model for Yoruba-to-
English back-translation.

Add `--agent` to have a local Ollama model review the candidate. The model
name can be set with `--ollama-model` or the `OLLAMA_MODEL` environment
variable. Python back-translates the candidate exactly once before asking
Ollama for a single validated JSON decision based on the original English,
back-translation, and digit findings. Both the Ollama model and the
configured NLLB checkpoint must already be available locally.

```powershell
.\.venv\Scripts\python.exe -m app.review_translation "The driver did not deliver 10 boxes." --candidate "Awakọ fi apoti 10 ranṣẹ́." --agent --ollama-model llama3.2
```

The agent reports `needs_review` for suspected errors and only reports
`no_issue_flagged` after back-translation evidence. Failed or invalid agent
actions are reported as `review_incomplete`; numeric-check findings remain
visible regardless of the agent's decision.

`needs_review` and `review_incomplete` results are saved for human review;
unflagged results are saved only with `--audit-no-issue`. The SQLite database
defaults to `.local/translation_reviews.sqlite3` (ignored by Git); use
`--queue-db` or `YORUBA_REVIEW_DB` to choose another path.

```powershell
.\.venv\Scripts\python.exe -m app.review_translation "The driver did not deliver 10 boxes." --candidate "Awakọ fi apoti 10 ranṣẹ́." --agent --audit-no-issue
.\.venv\Scripts\python.exe -m app.review_queue list
.\.venv\Scripts\python.exe -m app.review_queue show 1
.\.venv\Scripts\python.exe -m app.review_queue decide 1 correct --corrected "Awakọ kò fi apoti mẹ́wàá ranṣẹ́." --notes "Restored the negation."
.\.venv\Scripts\python.exe -m app.review_queue summary
```

Reviewer decisions are `approve`, `correct`, or `uncertain`. A correction
requires nonblank Yorùbá text, preserves the original candidate, and cannot
silently replace an existing decision. Summary counts are labeled as results
from the reviewed sample; pending agent-incomplete cases are reported
separately.

### Batch review pilot

`app.review_batch` runs reproducible FLORES `devtest` and blind A/B samples
through the same local back-translation and Gemma decision flow. FLORES mode
audits a locally available subset against the locally available official files,
joins rows by uniquely matching English text to verified IDs, and always
generates fresh NLLB candidates with beams 4, 256 new tokens, length penalty
1.1, phrase overrides disabled, and numeric protection disabled. Blind prompts
contain only the source and candidate text passed through the review flow,
never row IDs, A/B labels, or the private protection key. Results, audit,
summaries, and the SQLite review queue are written to a fresh directory under
ignored `.local`. Dataset defaults are repository-relative under `data/`;
pass the corresponding `--flores-*` or `--blind-csv` arguments when your local
FLORES or blind-packet files are stored elsewhere. The scripts do not depend
on another checkout, a user-home directory, or a session-state path.

```powershell
$env:TRANSLATION_MODEL_ID = 'C:\path\to\existing\nllb-checkpoint'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OLLAMA_MODEL = 'gemma3:4b'
.\.venv\Scripts\python.exe -m app.review_batch --dataset flores --flores-source 'data\flores\devtest.eng_Latn' --flores-reference 'data\flores\devtest.yor_Latn' --flores-official-source 'data\flores\official\eng_Latn.devtest' --flores-official-reference 'data\flores\official\yor_Latn.devtest' --flores-predictions 'data\eval\flores_devtest_dataset\flores_devtest_1012_predictions.csv' --flores-metadata 'data\eval\flores_devtest_dataset\flores_devtest_dataset_metadata.json' --flores-row-ids devtest-0021 devtest-0023 devtest-0064 devtest-0072 devtest-0074 --ollama-model gemma3:4b --audit-no-issue --output-dir '.local\flores-review-pilot'
```

Use a new `--output-dir .local\pilot-name` for every run. The audit JSON
records file hashes, row alignment, normalization outcomes, and any unresolved
rows; unresolved requested rows are excluded and replaced with verified rows
from the same split when available. Specify `--flores-official-source` and
`--flores-official-reference` if the original local files are in a different
location. `--dataset`, `--flores-row-ids`, blind sample options, and
`--ollama-model` are configurable. Add `--blind-answer-key` only when
available: its assignments are read after results are saved and used only to
write aggregate protection-on/off counts, never per-row mapping.

### Phrase-override paired evaluation

`app.phrase_override_eval` runs a separate local paired experiment over exact
override phrases, longer variations, and unrelated controls. Each input is
translated with overrides disabled and enabled using the same NLLB decoding
settings; the service is called directly, bypassing the API translation cache.
The runner records actual override hits, output differences, reference-backed
scores where existing references are available, and separate Gemma review
results. It also writes a randomized human A/B packet and a private condition
key under the selected ignored `.local` output directory. The private key is
not included in reviewer prompts or the public packet. Existing evaluation
references overlap intentionally with some override phrases, so scores on
those rows are not independent evidence of general translation quality.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:TRANSLATION_MODEL_ID = 'C:\path\to\existing\nllb-checkpoint'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OLLAMA_MODEL = 'gemma3:4b'
.\.venv\Scripts\python.exe -m app.phrase_override_eval --output-dir '.local\phrase-override-eval-run' --ollama-model gemma3:4b
```

Choose a fresh `--output-dir` for each run. `--seed` controls the blinded
packet ordering, and `--eval-data` can point to another existing reference
file. METEOR is omitted because no Yorùbá-appropriate scorer/resources are
configured. Saved metadata records input filenames and content hashes rather
than local absolute paths; the private answer key and reviewer-facing notes
remain in the ignored output directory.

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
