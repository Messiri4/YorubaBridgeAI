# Local review-agent evaluation artifacts

These are shareable aggregate reports from local-only evaluations. The source
and candidate sentences, reviewer notes, blind assignment keys, SQLite files,
model weights, and per-case Ollama responses are intentionally not included.
Metadata records model families and input filenames, not machine-specific
absolute paths.

## Reports

- [`phrase_overrides_13/`](./phrase_overrides_13/) is the 13-input curated
  phrase-override experiment (five exact phrases, five longer variations, and
  three controls).
- [`phrase_overrides_flores30/`](./phrase_overrides_flores30/) is the paired
  phrase-override comparison on FLORES-200 dev rows `dev-0001` through
  `dev-0030`.
- [`phrase_vs_numeric_flores30/`](./phrase_vs_numeric_flores30/) compares
  phrase overrides and numeric protection on the same 30 FLORES dev rows.

The historical 2-beam and 4-beam FLORES prediction files are
`data/eval/beam_eval_dev_beams2_max400_lp11.csv` and
`data/eval/beam_eval_dev_beams4_max400_lp11.csv` on the
`eval/flores-devtest` branch. The historical benchmark and phrase-override
OFF/ON conditions use different test sets in the 13-input report. The 30-row
reports use matched sources and references; the numeric report's 4-beam,
phrase-OFF, and numeric-OFF outputs were verified identical.

## Reproduction

The 13-input and 30-row phrase experiments use
`python -m app.phrase_override_eval` and `python -m app.review_batch` from the
repository, respectively. Set `TRANSLATION_MODEL_ID` to an existing local
NLLB checkpoint and set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
Review-agent runs additionally require the selected model to be installed in
local Ollama. See the repository README for CLI arguments and local dataset
inputs. Use a fresh `.local` output directory for each run.

Numeric-protection evaluation used the implementation on the
`review/numeric-preservation` branch, with a local NLLB checkpoint and the
verified 4-beam settings recorded in its metadata. Its portable runner is
`scripts/run_numeric_protection_comparison.py` on that branch; see
`docs/numeric_protection_evaluation.md` there for the PowerShell command.
Private run outputs remain in ignored `.local`; only the aggregate scores,
metadata, and chart are published here.

## Interpretation cautions

The five phrase-override outputs exactly match their saved Yorùbá references;
those references may have informed the overrides, so the resulting 100 scores
are in-sample and are not independent quality evidence. FLORES reference
flags are potential review items, not confirmed errors. In the numeric
experiment, the protection matcher also selected five non-digit hyphenated
terms, and eight matched inputs fell back after sentinel restoration failed.
Those behaviors are important context for its lower aggregate scores.
