# Dataset Governance — YorubaBridgeAI evaluation data

## Canonical sources (recorded once, verified by hash)

| File                                                   | Role                                     | Origin                                                                                               | Pairs |
| ------------------------------------------------------ | ---------------------------------------- | ---------------------------------------------------------------------------------------------------- | ----- |
| `data/flores/dev.eng_Latn`, `data/flores/dev.yor_Latn` | original archive excerpt (60 lines each) | official Meta FLORES-200 tarball (CC-BY-SA 4.0)                                                      | 60    |
| `data/flores_eval/dev.997.en.txt`, `dev.997.yo.txt`    | full dev split for experiments           | HF mirror `yash9439/flores200` `dev.parquet`, first 997 rows, extracted via `scripts/load_flores.py` | 997   |
| `data/flores_eval/dev.30.{en,yo}.txt`                  | beam-A/B experiment sample               | seeded (`random.Random(42)`) shuffle of the 997, first 30                                            | 30    |
| `data/flores_eval/devtest.20.*`                        | PAST baseline use only                   | mirror devtest.parquet                                                                               | 20    |

Cross-check performed: mirror `dev.997` lines 1–60 are byte-identical to the
archive excerpt in `data/flores/` — both descend from the official release.
sha256 of every consumed file is recorded per-run by
`scripts/evaluate_quality.py` in the run manifest.

## Split policy

- **dev (997):** all experimentation, decoding sweeps, error analysis, and
  Sprint-level candidate comparisons.
- **devtest:** reserved for the FINAL held-out evaluation of an accepted
  candidate. Never used for tuning, never sampled for review worksheets,
  never loaded in experiment runs. The evaluation CLI's leakage guard
  rejects any prediction whose source sentence is not in the dev files.
- Regression cases (verified model errors from Phase 2/2B) will be stored as
  a _separate_, versioned file `data/eval/regression_cases.json` (pair_idx,
  direction, verified issue, expected property) — never mixed into dev
  training-like material since no training occurs, but kept out of metric
  computation until used as a targeted checklist.

## Review data

- Worksheet: `data/eval/flores_reference_review.csv` (100 pairs, seed
  20261109). Completed reviews will be saved as
  `data/eval/human_reviews_<date>.csv` — the worksheet itself is never
  edited in place.
- Rubric: `docs/human_review_rubric.md`.
- Completed reviews are accepted into the scorecard ONLY via the
  `metrics.human_labels_from` config key, which must point to a completed
  (not worksheet) file.

## Protection rules

1. `scripts/evaluate_quality.py` refuses to overwrite any Phase 1/2/2B
   artifact (PROTECTED set in the module).
2. Original FLORES files are never edited; derived normalized references
   (if later needed) live in a separate `_normalized` namespace.
3. No experiment may add sentences from devtest to any worksheet, manifest,
   or sample file.
