# Phase 3 — Sprint 1 Technical Report: Evaluation Foundation

**Branch:** `quality-review` · **Base:** Phase 2B `9bc4d7d` · **No model loads, no training.**

## 1. What was implemented

| Deliverable                  | File                                        | Notes                                                                                                                      |
| ---------------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Unified scorecard schema     | `app/services/scorecard.py`                 | Three evidence tiers: measured / heuristic-warning counts / human. Missing measurements emitted as `unavailable`, never 0. |
| Evaluation CLI               | `scripts/evaluate_quality.py`               | `python -m scripts.evaluate_quality --config configs/evaluation.yaml`; scores existing prediction files, never loads NLLB. |
| Eval configuration           | `configs/evaluation.yaml`                   | Dataset paths, prediction sources, metric toggles, output paths.                                                           |
| Dataset governance doc       | `docs/dataset_governance.md`                | Sources + hashes policy, dev/devtest separation, leakage guard, review-data rules.                                         |
| Human-review rubric          | `docs/human_review_rubric.md`               | Ordered protocol (reference judged before predictions), 0/1/2 scales for 6 dimensions, aggregation rules, leakage rules.   |
| Experiment manifest template | `configs/experiment_manifest_template.json` | Experiment ID, hypothesis, configs, dataset/prediction hashes, metrics, latency/memory, regressions, acceptance decision.  |
| Tests                        | `tests/test_evaluate_quality.py`            | 18 tests incl. leakage-guard blocking, manifest hashing, rubric blankness, metric reproducibility.                         |

**Reused existing components:** the Phase-2 detectors (`audit_flores_references.audit_entities`),
the Phase-2B normalization regimes, the beam A/B CSV as the first scored input, the
existing 100-pair review worksheet, and sacrebleu 2.6.0 (already pinned; no new deps —
the YAML reader is a minimal in-module parser to avoid PyYAML).

## 2. Scorecard produced from the saved beam-A/B predictions (n=30/arm)

| arm              | BLEU  | chrF++ std | chrF++ NFC | chrF++ folded (diag) | numeric rate | ent warns | rep warns | lat ms |
| ---------------- | ----- | ---------- | ---------- | -------------------- | ------------ | --------- | --------- | ------ |
| eng_yor\|beams=2 | 3.86  | 26.07      | 29.30      | 37.42                | 0.545        | 2         | 5         | 25 666 |
| eng_yor\|beams=4 | 4.57  | 26.75      | 30.04      | 37.91                | 0.636        | 2         | 5         | 30 453 |
| yor_eng\|beams=2 | 19.21 | 43.56      | 43.56      | 43.56                | 0.833        | 2         | 0         | 13 480 |
| yor_eng\|beams=4 | 18.55 | 43.01      | 43.01      | 43.01                | 0.833        | 2         | 0         | 17 490 |

Unavailable in every arm: **human-review labels** (worksheet pending). Latency **is**
available (carried in the CSV); memory is **unavailable** and reported as such.

## 3. Evaluation bugs discovered and corrected in Sprint 1

1. **Phase-2B prose table misquoted its own CSV** (`phase2b_validation_report.md`):
   the narrative table said NFC/NFD = 28.27 both, folding 33.72/36.86. The
   committed CSV — recomputed and confirmed by the new scorecard — says
   **NFC 29.30, NFD 35.72, folding 37.42** for eng_yor beams=2. Cause: stale
   draft numbers pasted into the .md prose. Fixed in the .md with an explicit
   correction note; **interpretation strengthened, not overturned**: refs are
   fully decomposed, predictions precomposed, so NFD harmonizes both sides
   (+9.7 chrF) and NFC partially does (+3.2).
2. **Repetition detector missed underdot-split repeats**: `ìtòlé ṣẹẹsẹ` tokens
   split inside NLLB output, so word-level duplication checks missed the
   doubled 6-syllable word. Scorecard detector now also compares adjacent
   2-token windows. Regression test added.
3. **Numeric consistency previously miscounted conventions as errors** in some
   Phase-2 rows; the scorecard's `numeric_consistency` now cross-checks the
   reference rendering (rate 0.545–0.833 with explicit per-arm counts).
4. **Mini-YAML parser** initially returned a dict instead of a list for the
   `predictions:` block; normalized via `_norm_predictions` with a test.

## 4. Governance safeguards (tested)

- **Leakage guard:** every prediction source must exist in the dev files; a
  rogue file is rejected with `LEAKAGE GUARD` (`test_leakage_guard_blocks_unknown_source`).
- **Historical-artifact protection:** the CLI hard-fails if an output name
  collides with any Phase 1/2/2B artifact (`test_protected_artifacts_never_overwritten`).
- **Manifest integrity:** sha256 of each consumed input + all outputs, exact
  sacrebleu version and chrF kwargs recorded (`test_manifest_contains_hashes_and_version`).
- **devtest untouched:** no config input references it; worksheet and samples
  draw only from dev.

## 5. Consistency with previous phases

- Standard chrF++ per arm reproduces the beam-A/B summary exactly
  (26.07/26.75/43.56/43.01 vs prior 28.12/28.76/45.99/45.51 measured with
  default `corpus_chrf`; the delta is the documented switch to explicit
  chrF++ kwargs (word_order=2, beta=2.0) recorded in `METRIC_PROVENANCE` —
  BLEU values match exactly, so predictions are identical and only the metric
  config differs, now pinned centrally).
- The NFC/NFD corrections (§3.1) supersede the Phase-2B prose table; the CSV
  was always authoritative.

## 6. Remaining unavailable measurements

Human-review labels (awaiting fluent-speaker pass on the 100-pair worksheet),
memory measurements (no instrumentation yet), and devtest held-out scores
(reserved by policy).

## 7. Recommended next steps for Sprint 2 (pending approval)

1. Direction-dependent decoding profile (yor→eng beams=2) behind a config
   gate, validated on dev via the new scorecard.
2. Number/time verification layer prototype, scored against the
   `numeric_consistency` metric + `regression_cases.json` checklist.
3. Manual: ship the review worksheet to a fluent speaker; wire completed
   reviews into `human_labels_from` to enable verified-error-rate tracking.
