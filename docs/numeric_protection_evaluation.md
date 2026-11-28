# Reproducing the numeric-protection FLORES comparison

The paired runner compares numeric protection OFF and ON using the same
English sources and Yorùbá references from the 4-beam FLORES-200 dev baseline.
It writes per-row outputs, scores, and machine-path-free metadata under a fresh
Git-ignored `.local` output directory.

The default input file is
`data/eval/beam_eval_dev_beams4_max400_lp11.csv`, a bidirectional baseline
with 30 rows per direction included with this evaluation branch. The runner
filters `direction=eng_yor` and defaults to IDs `dev-0001` through
`dev-0030`. Use `--input-csv` to select another compatible CSV. It must
contain `sentence_id`, `direction`, `source`, `reference`, and `prediction`
columns. Use `--row-ids` to select specific EN-to-YO IDs.

Set `TRANSLATION_MODEL_ID` to an existing local NLLB checkpoint directory.
The runner enables Hugging Face offline mode, sets the decoding parameters to
4 beams, 400 new tokens, length penalty 1.1, and a 2,000-character input
limit, and disables phrase overrides and glossary post-processing. It uses no
remote translation API and does not download weights.

PowerShell example:

```powershell
$env:TRANSLATION_MODEL_ID = 'C:\path\to\existing\nllb-checkpoint'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe scripts\run_numeric_protection_comparison.py --input-csv data\eval\beam_eval_dev_beams4_max400_lp11.csv --output-dir .local\numeric-protection-flores30-run
```

Choose a new `--output-dir` for every run. Model call settings and output
restoration status are recorded in the metadata and per-row files.
