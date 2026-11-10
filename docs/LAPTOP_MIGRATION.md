# YorubaBridgeAI — Laptop Migration Guide

Complete procedure to restore this project on a **new laptop** (macOS, Linux, or
Windows) from the GitHub backup at
<https://github.com/Messiri4/YorubaBridgeAI>.

---

## 1. Clone the repository

```bash
# macOS / Linux
git clone https://github.com/Messiri4/YorubaBridgeAI.git
cd YorubaBridgeAI

# Windows (PowerShell)
git clone https://github.com/Messiri4/YorubaBridgeAI.git
cd YorubaBridgeAI
```

If you have SSH keys set up on the new machine you may use
`git@github.com:Messiri4/YorubaBridgeAI.git` instead.

Verify the clone:

```bash
git branch -a          # should list all remote branches
git log --oneline -3   # tip of the checked-out branch
```

## 2. Check out the MENYO research branch

```bash
git checkout research/menyo-domain-adaptation
```

Optionally create a dedicated worktree for it (see §12):

```bash
git worktree add ../YorubaBridgeAI-menyo research/menyo-domain-adaptation
cd ../YorubaBridgeAI-menyo
```

The branch contains `data/menyo20k/raw/` (MENYO-20k TSVs, dev + test splits)
and its `LICENSE` (CC BY-NC 4.0) and `DATA_README.md`.

## 3. Recreate the Python virtual environment

Python **3.11+** is recommended.

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is missing on your branch, the core runtime/eval
dependencies used by the project are:

```bash
pip install fastapi uvicorn torch transformers sentencepiece sacrebleu \
            psycopg[binary] redis python-dotenv pytest
```

(Adjust versions to what your local machine used before migration; the eval
runs were produced with `sacrebleu 2.6.0`.)

## 5. Obtain the NLLB model checkpoint

The project uses **facebook/nllb-200-distilled-600M** from Hugging Face.

- Model: <https://huggingface.co/facebook/nllb-200-distilled-600M>
- License: model weights are released under the CC-BY-NC 4.0 license
  (see the model card for the exact revision you used).
- Revision/commit used previously: the snapshot cached under
  `~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M`.

Download (first run caches automatically):

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
name = "facebook/nllb-200-distilled-600M"
AutoTokenizer.from_pretrained(name)
AutoModelForSeq2SeqLM.from_pretrained(name)
```

The repository intentionally excludes model weights (`models_cache/`,
`*.safetensors`, `*.bin` are git-ignored). A helper script exists on the
research branch to assemble a local snapshot dir from the HF cache:

```bash
# from research/menyo-domain-adaptation
python tools/build_local_snapshot.py   # creates models_cache/nllb-600m/
```

Do **not** commit the resulting `models_cache/` directory.

## 6. Obtain MENYO-20k and FLORES datasets legally

### MENYO-20k — already in the repo

`research/menyo-domain-adaptation` includes the raw TSVs at
`data/menyo20k/raw/` together with its license. If you need a fresh copy:

- Source: MENYO-20k Multitask Domain Adaptation corpus
  (<https://github.com/uds-dmenyo/menyo-20k_MT>).
- License: **CC BY-NC 4.0** — attribution + non-commercial use only.
  Do not use it in a commercial product; keep the LICENSE file alongside it.

### FLORES-200 — download per Meta's terms

- Source: <https://github.com/facebookresearch/flores> (CC-BY-SA 4.0).
- The repo already contains small evaluation slices at `data/flores_eval/`
  with `provenance.json` files recording the upstream source
  (`huggingface:yash9439/flores200`) and license.
- Full dataset:

```bash
python scripts/load_flores.py       # see scripts/ on whatsapp-integration
python scripts/convert_flores_parquet.py
```

Respect both licenses: MENYO is non-commercial, FLORES is share-alike.

## 7. Configure environment variables

Never commit real `.env` files. Copy the template and fill it in:

```bash
cp .env.example .env   # create .env.example on a new branch if absent
```

Minimum keys expected by the app (adjust names to the code you run):

```
DATABASE_URL=postgresql://yoruba:CHANGE_ME@localhost:5432/yorubabridge
REDIS_URL=redis://localhost:6379/0
```

## 8. Start Redis and PostgreSQL

### macOS (Homebrew)

```bash
brew install redis postgresql@16
brew services start redis
brew services start postgresql@16
createdb yorubabridge
```

### Linux (Debian/Ubuntu)

```bash
sudo apt install redis-server postgresql
sudo service redis-server start
sudo service postgresql start
sudo -u postgres createuser --superuser "$USER"
createdb yorubabridge
```

### Windows

Install via the official installers or `winget install Redis.Redis is not
available` — recommended: run both in Docker:

```bash
docker run -d --name yb-redis  -p 6379:6379 redis:7
docker run -d --name yb-pg     -p 5432:5432 \
  -e POSTGRES_PASSWORD=CHANGE_ME -e POSTGRES_DB=yorubabridge postgres:16
```

Never commit the resulting database files; `*.db`, dumps and data dirs stay
out of Git.

## 9. Run FastAPI

```bash
source .venv/bin/activate          # if not already active
uvicorn app.main:app --reload      # adjust the module path to your layout
```

Then open <http://127.0.0.1:8000/docs> for the interactive API docs.

## 10. Run the tests

```bash
pytest -v
```

## 11. Continue the MENYO baseline evaluation

The MENYO baseline evaluation infrastructure lives on
`research/menyo-domain-adaptation`:

```bash
git checkout research/menyo-domain-adaptation   # or the worktree from §2
```

Status at migration time: dataset integration, baseline-eval infra, error
analysis and a feasibility assessment are committed; the next step is running
the baseline decode over `data/menyo20k/raw/test.tsv` with the NLLB-600M
checkpoint from §5 and scoring with the shared eval tooling
(`scripts/evaluate_flores.py`, `scripts/beam_ab_experiment.py` on
`whatsapp-integration`). Follow `docs/evaluation_pipeline.md` for the scoring
schema, and record new manifests/scorecards under `data/eval/`.

## 12. Recreate optional Git worktrees

All branches are on GitHub; recreate whichever worktrees you had:

```bash
git fetch origin
git worktree add ../YorubaBridgeAI-decoding     feature/nllb-optimization
git worktree add ../YorubaBridgeAI-glossary     feature/semantic-glossary
git worktree add ../YorubaBridgeAI-infra        infrastructure-review
git worktree add ../YorubaBridgeAI-integration  quality/sprint2-diagnostics
git worktree add ../YorubaBridgeAI-numeric      feature/numeric-protection
git worktree add ../YorubaBridgeAI-performance  performance/phase3
git worktree add ../YorubaBridgeAI-quality      quality/sprint3-reliability
git worktree add ../YorubaBridgeAI-tonality     feature/yoruba-tonality
git worktree add ../YorubaBridgeAI-website      website-production
git worktree list                                # confirm
```

---

## Post-migration checklist

- [ ] `git worktree list` shows the worktrees you need
- [ ] `pytest -v` passes
- [ ] `.env` recreated locally (never committed)
- [ ] NLLB checkpoint cached / `models_cache/` rebuilt locally
- [ ] Redis + PostgreSQL reachable
- [ ] FLORES/MENYO data present under `data/` with license files intact
