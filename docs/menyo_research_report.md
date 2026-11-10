# MENYO-20k Domain-Adaptation Research Report (Phase 0: dataset + baseline infrastructure)

Branch: `research/menyo-domain-adaptation` · Date: 2026-11-10 · Status: **no fine-tuning performed; no benchmark executed yet (awaiting approval)**.

## 1. What was asked and why

Objective: determine whether domain adaptation with high-quality MENYO-20k
parallel data ([MENYO-20k paper](https://aclanthology.org/2021.mtsummit-research.6/),
[data](https://github.com/uds-lsv/menyo-20k_MT)) improves the existing
NLLB-200-distilled-600M service. Phase 0 = dataset integration + baseline
infrastructure only.

## 2. Repository inspection (Step 1)

- Branch `research/menyo-domain-adaptation` already checked out at commit
  `ab62cea` (sprint2 diagnostics checkpoint); working tree was clean, so no
  uncommitted work was at risk and none was discarded.
- Model loader: `app/services/model_loader.py` (`load_nllb`, local
  safetensors, meta-tensor materialization for tied embeddings). Inference
  interface: `app/services/translation.py::TranslationService.translate`.
- Evaluation: `scripts/evaluate_flores.py`, `scripts/evaluate_quality.py`,
  `app/services/evaluation.py` (`score_direction`: SacreBLEU + chrF via sacrebleu).
- MENYO: **not previously integrated** (grep confirmed). Production paths untouched.

## 3. Dataset integration (Step 2)

- Downloaded **data files only** (plain HTTPS of raw TSVs; no remote code
  executed): `train.tsv`, `dev.tsv`, `test.tsv` + 5 per-domain dev files +
  upstream `LICENSE` and `data/README.md` into `data/menyo20k/raw/`.
  SHA-256 checksums recorded in `data/menyo20k/SHA256SUMS`.
- License verified: **CC BY-NC 4.0** — non-commercial only. Documented in the
  loader docstring, the report, and the feasibility doc.
- Loader: `research/menyo20k/loader.py` (outside production paths):
  NFC normalization (canonical — tone marks ́ ̀ and underdots ẹ ọ ṣ preserved);
  malformed-row rejection; domain provenance from per-domain dev files;
  duplicate + cross-split overlap detection; **test split locked behind
  `allow_test=True`**.
- Measured (this run):
  - train: 10,070 pairs, 0 rejected, 76 duplicate pairs
  - dev: 3,396 pairs, 2 rejected (extra-tab malformed rows), 45 duplicates
  - dev domains matched: news 1076, book 503, tedTalks 430, digital 296, proverbs 228
  - train∩dev exact-pair overlap: **4 pairs** (diagnostic; official splits untouched)

## 4. Baseline evaluation infrastructure (Step 3)

`scripts/evaluate_menyo.py`:

- deterministic stratified sample from the official dev split (round-robin
  across domains; seed = rotation offset; fully reproducible);
- both directions on the **existing** checkpoint via the production
  `TranslationService`, overrides disabled, **caching disabled**;
- metrics: SacreBLEU, chrF++ (char_order=6, word_order=2), numeric
  preservation, placeholder-restoration failures, repeated-output failures,
  mean/median/P95 latency; sentence-level CSV + provenance-complete JSON
  (dataset version+hash, git commit, checkpoint, decoding params, seed,
  sample ids, metric config);
- **cost gate**: prints the runtime estimate and refuses to run without `--yes`.
- Dry-run verified. Estimate at the repo's measured ~2.5 s/sentence CPU:
  **60 pairs × 2 directions ≈ 5 minutes** for the default run — cheap;
  approval still requested per protocol.

## 5. Error analysis (Step 4)

`scripts/analyze_menyo_errors.py`: heuristic flags for negation, numbers,
named entities, low reference overlap (role/order proxy), tense/aspect
markers, omissions, context sensitivity, Yoruba diacritic degradation, and
repetitive generation. Output: review CSV with source, reference, prediction,
domain, flags and **blank** verdict columns (`verdict`, `category_verdict`,
`notes`) for bilingual human review. No human scores are fabricated anywhere.
Tests: `tests/test_menyo_analysis.py`.

## 6. Feasibility assessment (Step 5)

See `docs/menyo_finetuning_feasibility.md`. Summary: full fine-tuning is
impractical on 16 GB Intel macOS; **LoRA (rank 8–16) is feasible**
(~3–5 GB RAM, ~40–90 min/epoch CPU, <1 GB disk); no accelerator available on
this hardware (MPS needs Apple silicon); a cloud GPU would do 3 epochs in
~15–30 min. Both model and dataset are CC-BY-NC 4.0 — any fine-tuned artifact
inherits the non-commercial restriction. Nothing installed or downloaded; no
training started.

## 7. Verification performed

- `pytest tests/test_menyo_loader.py tests/test_menyo_analysis.py` → **18 passed**
- Loader dry-runs on real data (counts above); evaluation script `--dry-run` OK.
- Production suite untouched: no file under `app/` was modified.

## 8. What remains unproven

- All translation quality numbers: the benchmark has not been run yet.
- Whether MENYO fine-tuning helps: fine-tuning not attempted.
- Heuristic error flags: diagnostic only, unvalidated against human judgement
  (review CSV must be filled by bilingual reviewers).
- Cross-domain generalization (e.g. train on news, test on proverbs): future work.
