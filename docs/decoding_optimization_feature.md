# NLLB decoding optimization (Agent 4 · feature/nllb-optimization)

Status: **module + experiment plan landed; nothing yet validated with real
inference.** All quality statements below cite already-saved artifacts; no new
benchmark numbers are claimed.

## 1. What was inspected (existing evidence)

Generation currently happens in `app/services/translation.py::translate()`:

```python
model.generate(
    **inputs,
    forced_bos_token_id=forced_bos,
    max_new_tokens=settings.translation_max_new_tokens,   # 256
    num_beams=settings.translation_num_beams,             # 4
    length_penalty=settings.translation_length_penalty,   # 1.1
)
```

Truncation handling: tokenizer-side `truncation=True,
max_length=translation_max_input_chars (2000)`; there is **no**
repetition penalty, no `no_repeat_ngram_size`, no `min_new_tokens`, and no
explicit `early_stopping` (transformers default).

Saved results used (not re-measured):

| Source                                                 | Finding                                                                                                                        |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `data/eval/decoding_exp_dev.json` (n=30 dev)           | beams=4/maxnew=256/lp=1.0 → chrF++ 26.20; beams=4/400/1.1 → 26.45 (adopted); beams=8/400/1.1 → 25.87, +60% latency (rejected)  |
| `data/eval/phase3_scorecard_sprint2a.csv`              | beams=4 > beams=2 for eng→yor (26.75 vs 26.07 chrF++) but beams=4 < beams=2 for yor→eng (43.01 vs 43.56), at ~30% more latency |
| `docs/translation_evaluation.md`                       | 0/30 dev hypotheses truncated at max_new_tokens=256; truncation is not the dominant eng→yor error                              |
| `docs/translation_improvement_sprint1.md`, Sprint TI-1 | numeric protection wraps/restores around generate — any decoding change must not break sentinel restoration                    |

## 2. Design review conclusions

- **Beam width**: 4 is the measured sweet spot on dev (n=30); 8 regressed
  quality _and_ speed. The beams=2-vs-4 asymmetry across directions is the
  most interesting unexploited signal → direction-aware beams candidate.
- **Length penalty**: 1.0→1.1 gave a small, consistent eng→yor gain.
  Extending to 1.2 is the natural next step, but n=30 is tiny.
- **Repetition penalty / no-repeat-ngram**: risky for Yoruba — legitimate
  reduplication (`ọjọ ọjọ`, `ìwé ìwé`) is grammatical and the repo's
  repetition detector already carries a reduplication whitelist. Keep
  disabled until a candidate clears that detector.
- **Output-length limits**: flat 256 suffices for FLORES-length inputs, but
  for genuinely long user texts a dynamic budget prevents mid-sentence cuts.
- **Latency**: beam search dominates latency; beams=2 saves ~20–25%
  (scorecard means) — useful as a degraded-mode profile, not a default.

## 3. The module: `app/services/decoding_policy.py`

Pure-logic, dependency-free module (no torch/transformers imports, no
inference). Core pieces:

- `DecodingPolicy` — frozen, validated dataclass with conservative defaults
  (`num_beams=4`, `max_new_tokens=256`, `length_penalty=1.1`; repetition
  controls off).
- `PROFILES` — named presets:
  - `production_default` — bit-equivalent to today's behavior.
  - `conservative_guards` — dynamic input-scaled output budget (≤400).
  - `quality_lp12` — length_penalty=1.2 probe.
  - `fast_latency` — beams=2 fallback profile.
  - `direction_aware_beams` — yor→eng@2, eng→yor@4.
  - `anti_repeat_gentle` — `no_repeat_ngram_size=3` probe.
- `get_policy(settings, profile)` — resolution order: explicit arg >
  `settings.decoding_policy_profile` > env `DECODING_POLICY` >
  `production_default`. Unknown names raise (no silent fallback).
- `build_generate_kwargs(policy, settings, input_tokens, direction)` —
  returns exactly the dict for `model.generate(**kwargs)`.
- `load_policy_file()` — one-off JSON policies for experiments.
- `EXPERIMENT_MATRIX` / `experiment_plan_md()` — defined A/B experiments.

### Backward compatibility

With the default profile and stock settings, `build_generate_kwargs`
returns exactly `{"max_new_tokens": 256, "num_beams": 4, "length_penalty": 1.1}`
— identical to the current hardcoded settings pass-through, including
operator `.env` overrides (the default policy mirrors settings; an
experimental policy is authoritative and ignores ambient settings so runs
can't be contaminated). Optional knobs appear in the kwargs only when
enabled, so `generate` never receives surprise `None`s. Zero behavior
change is enforced by test
`test_kwargs_backward_compatible_with_stock_settings`.

## 4. Integration requirements (for whoever wires it into `translation.py`)

The wiring is deliberately **not** done here (file ownership). Minimal diff
inside `translate()`:

```python
from app.services.decoding_policy import build_generate_kwargs, get_policy

policy = get_policy(self._settings)          # cached per-service
gen_kwargs = build_generate_kwargs(
    policy, self._settings,
    input_tokens=inputs["input_ids"].shape[1],   # enables dynamic budget
    direction=f"{source_language}_{target_language}",
)
generated = model.generate(**inputs, forced_bos_token_id=forced_bos, **gen_kwargs)
```

Notes / caveats:

1. `input_tokens` must be the **token count** of the truncated input
   (`inputs["input_ids"].shape[1]`), not character count.
2. Keep the tokenizer-side `truncation=True/max_length` unchanged —
   `translation_max_input_chars` remains owned by that call.
3. Numeric protection (Sprint TI-1) is orthogonal but every new profile
   must be re-validated with protection on: sentinel restoration can fail
   if a policy induces genuinely different output strings.
4. Tests exist that assert `model.generate.call_args.kwargs` exactly
   (`tests/test_translation_service.py::...num_beams == 4`); they stay
   green because the default profile preserves those values.
5. Optional: add `decoding_policy_profile: str = ""` to `config.py`
   (owned by another agent) so ops selects a profile via `.env`;
   `get_policy` already reads it.

## 5. Controlled A/B experiments (defined, NOT run)

Harness reuse: extend `scripts/beam_ab_experiment.py` (per-cell partials,
resume, same 30 dev pairs, both directions, single model instance,
overrides disabled) so the "arm" becomes a policy profile instead of a
bare beam count. Scoring continues through
`app/services/evaluation.py::score_direction` + `scripts/evaluate_quality.py`
(BLEU, chrF++ standard/NFC/folded, numeric consistency, entity/repetition
warnings, latency).

| id                       | policy                  | question                                                                                              | promotion criteria                                                                                 |
| ------------------------ | ----------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| A0-baseline-fidelity     | `production_default`    | Routing through the module reproduces pre-module predictions byte-identically (gate; any diff = bug). | 0 divergent hypotheses vs `data/eval` beam-A/B predictions.                                        |
| A1-dynamic-budget        | `conservative_guards`   | Input-scaled output budget removes residual truncation on long sentences without quality loss.        | truncation ↓, chrF++ ≥ A0−0.2 both directions, latency ↑ ≤10%.                                     |
| A2-length-penalty        | `quality_lp12`          | Extends measured lp 1.0→1.1 trend one step.                                                           | chrF++ ≥ A0 both directions; no truncation increase.                                               |
| A3-direction-aware-beams | `direction_aware_beams` | Keep yor→eng quality at beams=2 (~25% latency saving) while eng→yor stays at 4.                       | yor→eng chrF++ ≥ A0−0.3 and latency ↓ ≥15%; eng→yor unchanged.                                     |
| A4-anti-repeat           | `anti_repeat_gentle`    | Does `no_repeat_ngram_size=3` cut repetition loops without harming attested reduplication?            | adjacent-repeat warnings ↓; reduplication-whitelist sentences unchanged; chrF++ within −0.2 of A0. |
| A5-fast-fallback         | `fast_latency`          | Degraded-mode fallback profile.                                                                       | documented deltas only; never promoted to default.                                                 |

### How to compare dimensions

- **Translation quality**: corpus `chrF++` (char_order=6, word_order=2,
  beta=2.0 — the reconciled Sprint-2A protocol) plus BLEU, per direction,
  on identical fixed pairs; report both standard and NFC-normalized chrF++
  (underdot policy per `docs/phase2b_validation_report.md`). Use the paired
  design (same pairs, same seed) and report deltas with pair counts; with
  n=30 treat <±0.5 chrF++ as noise until a larger devtest-20/other split
  run is available.
- **Numerical preservation**: `numeric_consistency_rate` and
  `entity_warning_hits` from the existing scorecard; every candidate must
  be ≥ baseline rate, since TI-1 restoration depends on stable outputs.
- **Tonal accuracy**: reuse the accent-folded chrF diagnostic
  (`chrfpp_folded_diagnostic`) as a diacritic-loss indicator, plus manual
  spot-check of underdot-bearing tokens on the priority-review 20 subset;
  do not invent an automated "tone score" without validation.
- **Latency**: total/mean/median wall-clock per pair from the harness
  (already recorded per sentence in the beam-A/B CSVs). Report medians on
  shared, warm CPU; single-run means at 13–30 s/pair are noisy.
- **Repetition/degeneration**: `adjacent_repeat_warnings` +
  `suspiciously_short_warnings` from `evaluate_quality.py`; for A4 also
  verify whitelisted reduplications survive.

## 6. Configurations worth testing with real NLLB later

1. `production_default` — fidelity gate (A0), zero-risk first run.
2. `conservative_guards` — highest practical value: protects long inputs
   from mid-sentence truncation at negligible short-input cost.
3. `direction_aware_beams` — cheapest real latency win backed by the
   existing beams=2-vs-4 direction split.
4. `quality_lp12` — one-step extension of the measured lp trend.
5. `anti_repeat_gentle` — only if repetition warnings recur on larger
   samples; evaluate jointly with the reduplication whitelist.
6. Larger-sample reruns (devtest-20, 50–100 pairs) before promoting any
   candidate: all current evidence is n=30 CPU-only.

Explicitly **not** proposed: beams=8 (already measured worse and slower),
sampling/temperature decoding (breaks the reproducibility assumptions of
the saved-prediction harness), repetition_penalty>1 on Yoruba output
(reduplication risk).
