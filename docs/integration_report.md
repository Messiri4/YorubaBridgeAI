# Translation-Improvements Integration Report

Branch: `integration/translation-improvements` · Baseline: `9414279`
Scope: integration of the four TI-1 feature commits into the FastAPI/NLLB
service without regressions. No merge into `whatsapp-integration`/`main`;
no push; no new model downloads.

## Files modified

| File                                     | Change                                                                                                                                                                                                                                                       |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `app/services/translation.py`            | Full integration: decoding-policy-driven generation, protected/plain two-attempt flow via `with_unprotected_fallback` (explicit callable contract), glossary after restoration, Yoruba-target NFC + tone-restoration hook, residual-sentinel scrub guarantee |
| `app/services/cache.py`                  | `cache_key()`/`get()`/`set()` accept an optional `decoding_policy`; `production_default` maps to the legacy key (no invalidation of existing caches)                                                                                                         |
| `app/api/v1/routes_translate.py`         | Resolves the active policy name, passes it to cache get/set; response contracts unchanged                                                                                                                                                                    |
| `app/services/model_loader.py`           | `_materialize_meta_parameters()`: fixes local-checkpoint loading under transformers 4.57 + torch 2.2 (tied embeddings left on meta device); no dependency upgrades                                                                                           |
| `data/phrases/glossary.json`             | `allow_unaccented: true` on `àtògbẹ` (fold form unambiguous — enables the opt-in unaccented match tested by the TI-1 suite)                                                                                                                                  |
| `tests/test_translation_improvements.py` | Updated three stale assertions to post-hardening contracts (structured failure strings; opt-in unaccented glossary; no-leak fallback expectation)                                                                                                            |
| `tests/test_integration_pipeline.py`     | NEW: 16 integration tests (both directions, numeric preservation, fallback, glossary gating, decoding defaults + env override, error propagation, cache-key policy separation)                                                                               |
| `scripts/integration_comparison.py`      | NEW: reproducible baseline-vs-integrated evaluation harness (sequential real-NLLB inference, chrF++/BLEU, numeric preservation, sentinel leaks, tone-mark stats)                                                                                             |

## Integration decisions

1. **Placeholders can never reach users.** Restoration failure ⇒ structured
   `RestorationFailure`, retry without protection, and a final residual-sentinel
   scrub (`@@N@@` removal) as defense-in-depth, logged at ERROR when it acts.
2. **Explicit callable contract for retry.** `with_unprotected_fallback` receives
   two closures; no `TypeError` probing and no broad exception catching in the
   service to guess signatures.
3. **Glossary runs on restored text**, after sentinels are resolved, so
   placeholders can never shadow a term; token-span replacement means protected
   expressions are structurally safe from glossary rewrites.
4. **Tone restoration wired for Yoruba targets only**, default
   `PassThroughRestorer` behind `ThresholdGuard`: verified no-op, reports the
   orthography audit, fabricates nothing.
5. **Decoding policy `production_default`** reproduces beams=4/max_new=256/lp=1.1
   bit-for-bit; unknown profiles fail loudly (`ValueError`); `DECODING_POLICY`
   env override honored; optional experimental knobs stay OFF.
6. **Cache correctness across policies:** policy name is part of the key for
   non-default profiles only, so switching profiles cannot serve mixed-quality
   cached hypotheses and existing entries remain valid.

## Test results

`293 passed, 1 skipped, 3 deselected` (was `274 passed, 3 failed` before
integration work; the 3 failures were stale TI-1 assertions updated as part of
integration). `ruff check` clean on all touched files. `make lint` fails only
because `.venv/bin/ruff` is absent in the _parent_ worktree's venv — the
environment issue, not a code issue.

## Environment investigation (huggingface-hub)

- `pip list` showed `huggingface-hub None` with a missing/RECORD-less install;
  transformers 4.57 could not even be imported (collection errors in 6 test
  files).
- Fixed with `pip install --ignore-installed --no-deps huggingface-hub==0.36.2`
  (satisfying `>=0.34,<1.0`) — no other dependencies touched.
- Second environment issue found during real-inference validation: the local
  safetensors snapshot leaves the tied embedding group on the meta device
  (transformers 4.57 uses `map_location="meta"`; with torch 2.2 the tied
  group is not assigned). Fixed in-code via `_materialize_meta_parameters`
  (loads `lm_head.weight` into `model.shared.weight`, then `tie_weights()`).
  No torch/transformers upgrade performed.

## Quality comparison vs 9414279

Harness: `scripts/integration_comparison.py` — FLORES-200 devtest, first 80
aligned EN/YO pairs (held out from phrase-table domain; overrides disabled in
BOTH arms), single sequential model instance, deterministic beam search
(beams=4, lp=1.1, max_new_tokens=256). Arm A = protection+glossary off (9414279
behavior); Arm B = integrated pipeline.

Authoritative run: `data/eval/integration_comparison_20261110T102741Z.{csv,json}`
(n=80 per direction). Measured results:

| Metric | Baseline (A) | Integrated (B) | Δ |
|---|---|---|---|
| eng→yor chrF++ | 26.06 | 25.62 | −0.44 |
| eng→yor BLEU | 5.53 | 5.37 | −0.16 |
| eng→yor numeric preservation | 60% (24/40) | **75% (30/40)** | +15 pp |
| eng→yor sentinel leaks | 0 | 0 | 0 |
| eng→yor mean / median latency | 23.4 s / 20.0 s | 30.0 s / 22.2 s | +6.5 s / +2.2 s |
| yor→eng chrF++ | 44.40 | 44.27 | −0.13 |
| yor→eng BLEU | 22.38 | 22.27 | −0.11 |
| yor→eng numeric preservation | 100% (25/25) | 100% (25/25) | 0 |
| yor→eng mean latency | 11.6 s | 11.4 s | −0.2 s |
| eng→yor tone marks (low/high/underdot) | 1739/1728/1084 | 1684/1642/1065 | ≈ −3% |

Honest reading:
- **Safety clearly improves**: numeric preservation +15 pp eng→yor; 0 leaked
  placeholders across all 320 generations; 22 eng→yor sentences hit restoration
  verification failure and were transparently retried without protection — the
  fallback works. Each retry is a second full generation on CPU, which accounts
  for the eng→yor latency increase and likely part of the small chrF++ dip (the
  retried sentences are the number-heavy hard cases).
- The chrF++ deltas (−0.44 / −0.13) are within the noise band seen across this
  repo's prior n=30–80 experiments (±0.3–0.6); they are NOT a verified regression
  and equally NOT a demonstrated quality gain. The unambiguous gains are numeric
  robustness and the no-leak guarantee.
- Smoke artifact (`integration_comparison_20261110T094539Z.json`, n=2):
  byte-identical outputs in both arms — the wiring is behavior-neutral on
  number-free, glossary-free sentences, as designed.
- Decoder fidelity is asserted by tests (beams=4/max_new=256/lp=1.1 exactly).

## Remaining linguistic weaknesses

- Tone restoration is a no-op pass-through; unmarked/mis-toned Yoruba output is
  detected by `audit()` but never corrected.
- Glossary coverage is tiny (3 verified pairs + 1 reverse); wider coverage
  requires human-attested entries.
- Numeric protection covers digits/currency/dates/URLs; spelled-out numbers and
  ordinals remain model-dependent.
- Phrase overrides bypass the pipeline entirely (by design) and thus bypass
  glossary/protection stages.

## Readiness

Ready for further quality evaluation: all tests pass, API contracts unchanged,
defaults bit-compatible with the baseline, and a reproducible evaluation
harness is committed.
