# FLORES-200 Evaluation Pipeline — Reproduction Guide

## Dataset provenance

| Fact           | Value                                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------- |
| Upstream       | Meta AI FLORES-200 ("No Language Left Behind", github.com/facebookresearch/flores)             |
| License        | CC-BY-SA 4.0                                                                                   |
| HF mirror used | `yash9439/flores200` (verified present; single `dev.parquet` / `devtest.parquet` at repo root) |
| Split          | **devtest** — the official evaluation split (dev is reserved for future tuning)                |
| Languages      | `eng_Latn`, `yor_Latn`                                                                         |
| Initial size   | 20 aligned pairs per direction (pipeline smoke validation)                                     |
| Integrity      | Alignment spot-checked manually on loaded pairs (sentence boundaries, diacritics)              |

Because the app pins `numpy<2` (torch 2.2.2 on Intel Mac) while current
pyarrow requires `numpy>=2`, the parquet → text conversion runs in an
**isolated throwaway venv** so the app dependency graph stays untouched.

## Reproduce the dataset

```bash
# one-time converter venv (throwaway)
python3 -m venv /tmp/flores_conv_venv
/tmp/flores_conv_venv/bin/pip install pyarrow
/tmp/flores_conv_venv/bin/python scripts/convert_flores_parquet.py --split devtest --n 20

# scaling up later (e.g. the full 1012-pair devtest):
/tmp/flores_conv_venv/bin/python scripts/convert_flores_parquet.py --split devtest --n 1012
```

Outputs: `data/flores_eval/devtest.<n>.en.txt`, `...yo.txt`, `...provenance.json`.

## Reproduce the evaluation

```bash
cd /Users/user/Desktop/WORK/AI-projects/YorubaBridgeAI
PYTHONPATH=$PWD .venv/bin/python scripts/evaluate_flores.py --split devtest --n 20 --beams 4
```

Key properties (see the script header): single model load reused for both
directions; phrase overrides disabled via an empty table (production
translate() path, raw model measured); deterministic beams=4; per-sentence
latency + totals recorded; **timestamped outputs** so prior results are
never overwritten.

Outputs per run:

- `data/eval/flores_baseline_<split>_<n>_<UTCtimestamp>.csv` — direction, idx,
  source, reference, prediction, latency_s
- `data/eval/flores_baseline_<split>_<n>_<UTCtimestamp>.summary.json` — model,
  split, pairs, beams, overrides flag, timing

## Scores

Filled after the run completes — see `data/eval/flores_baseline_*_summary.json`
and `docs/evaluation_results.md` (generated report with error analysis).
