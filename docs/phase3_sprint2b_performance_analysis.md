# Phase 3 Sprint 2B — Inference Performance Analysis

Date: 2026-11-10 · Branch: `performance/phase3` · Model: NLLB-200 distilled 600M
Hardware: Intel MacBook Pro (x86_64), 16 GB RAM, CPU-only inference.
Constraints honored: no new NLLB inference, no downloads, no production changes, no
historical artifact overwrites, no push/merge.

---

## 1. Repository inspection (Task 1)

- Branch: `performance/phase3`; working tree was clean before this sprint's additions.
- Model initialization: `app/services/model_loader.py::load_nllb()` — loads the
  tokenizer then the seq2seq model, `.to("cpu")`, `model.eval()`. Called once per
  process inside the FastAPI lifespan hook (`app/main.py`), **not** at import time.
- Translation service: `app/services/translation.py::TranslationService.translate()` —
  validate → phrase-override lookup (exact match, O(1) dict) → tokenizer call with
  source-lang prefix → `model.generate(...)` with production settings
  (`num_beams=4`, `length_penalty=1.1`, `max_new_tokens=256`) → `batch_decode`.
- Phrase overrides: `PhraseOverrides` (`data/phrases/overrides.json`), NFC-normalized,
  punctuation-stripped exact-match table; override hits bypass the model entirely.
- Redis cache: `TranslationCache` — content-addressed SHA-256 key, read-through in
  `routes_translate.py`; graceful degradation when Redis is unreachable.
- API request flow: rate-limit check → cache GET → override/model → history record →
  cache SET → JSON response. All synchronous on the FastAPI worker thread.
- Profiling tools existing before this sprint: none — only end-to-end latency stats
  embedded in `scripts/beam_ab_experiment.py` (mean/median/max per config).

### Where inference time is spent (architectural reading, not yet instrumented)

The dominant cost must be `model.generate()`: the service is a standard
encoder-decoder autoregressive pipeline where the only super-linear stage is
decoding. Tokenization and detokenization are single-pass Rust-backed
(`NllbTokenizerFast`) operations on ≤2000-char inputs. The profiler built in this
sprint exists precisely to replace this architectural argument with numbers.

---

## 2. Existing latency measurements (Task 2)

Source artifacts (located, not regenerated):

- `data/eval/beams_ab_20261109T231413Z.summary.json` (+ `.csv`, partial JSONL) —
  complete run: 30 dev pairs × 2 directions × beams∈{2,4}.
- `data/eval/beams_ab_partial_20261109T230518Z.jsonl` — earlier interrupted run,
  2 sentences per cell only (kept for provenance; excluded from statistics).
- Quality context: `data/eval/decoding_exp_dev.json`, `beams_ab_*.summary.json` chrF.

Per-sentence statistics computed from `beams_ab_20261109T231413Z.csv` (n=30 each):

| Config  | Direction | Mean s | Median s | p95 s | Min s | Max s |
| ------- | --------- | ------ | -------- | ----- | ----- | ----- |
| beams=2 | eng→yor   | 25.67  | 22.18    | 56.89 | 8.53  | 56.94 |
| beams=2 | yor→eng   | 13.48  | 12.11    | 24.49 | 6.38  | 26.79 |
| beams=4 | eng→yor   | 30.45  | 30.07    | 58.55 | 12.73 | 69.82 |
| beams=4 | yor→eng   | 17.49  | 15.44    | 37.87 | 7.30  | 47.82 |

Verification: the headline numbers in the task brief reproduce exactly
(mean 25.67/13.48/30.45/17.49 s) from the stored per-sentence CSVs.

Derived observations:

- **Beam-search overhead**: beams 2→4 costs +18.6% mean (eng→yor) and +29.8%
  (yor→eng). Quality gain was +0.64 chrF (eng→yor, 28.12→28.76) but **−0.48 chrF**
  (yor→eng, 45.99→45.51): more beams actively hurt the yor→eng direction.
- **Direction asymmetry**: eng→yor is ~1.7–1.9× slower than yor→eng at every
  setting — consistent with longer Yoruba outputs (more generated tokens dominate
  CPU decode time).
- **Length relationship**: Pearson correlation between source character length and
  latency is strong and positive — 0.83 (eng→yor, both beam settings), 0.91/0.81
  (yor→eng, beams=2/4). Output token count, not source length alone, drives cost.
- **Tail risk**: p95 ≈ 2.2× median. Worst-case single requests exceed a minute.

What the stored data cannot answer (instrumentation gaps, see Task 4's profiler):
stage-level split (tokenize vs generate vs decode), cold/warm distinction, peak
memory, and thread-count effects. The original runs had no per-stage timers.

---

## 3. Bottleneck analysis (Task 3)

**Verified** (from artifacts/code, no new inference):

1. Beam-search decoding dominates: latency scales with beams and with output length;
   the decode loop is the only stage whose cost grows super-linearly.
2. Direction asymmetry is real and reproducible (two independent beam settings).
3. Single-request sequential processing: the route handles one text per request and
   awaits nothing — batched requests serialize behind the GIL and the CPU-bound
   generate call, blocking the event loop's thread.
4. Model reuse is already correct (one `LoadedModel` per process; no re-init per
   request). Repeated-initialization risk exists only on reload/workers >1.

**Hypotheses** (plausible, unmeasured — must be profiled before claiming):

- Tokenization/detokenization overhead is small (<5% of request time) because the
  tokenizer is fast Rust code and inputs are short. To be confirmed with the
  profiler's stage split.
- `max_new_tokens=256` allows very long generations for a few pathological inputs
  (p95 tail); capping adaptively could cut tail latency but risks truncation.
- CPU thread configuration (default `torch` thread count on this Intel chip) may be
  suboptimal for small matmul-heavy encoder batches; untested.
- Redis round-trip adds single-digit milliseconds; negligible vs 10–70 s inference.
- FastAPI/Pydantic overhead likewise negligible at these magnitudes.
- Memory pressure on a 16 GB machine (model ≈ 2.4 GB fp32) could cause swap-driven
  latency spikes; needs `peak_rss_mb` evidence from the profiler.

Nothing here claims an optimization works — every hypothesis maps to a benchmark
below.

---

## 4. Reusable profiler (Task 4)

New files:

- `app/services/profiling.py` — library core:
  - `StageTimings` (model_init / tokenize / generate / decode_output / total /
    cold_start / peak_rss_mb), `ProfileReport` with mean/median over **warm runs
    only**, JSON serialization.
  - `TranslationProfiler.profile_run(...)`: takes callables per stage so the caller
    chooses real or mock implementations.
  - `summarize_latencies` (mean/median/p95/min/max), `collect_system_context`
    (platform, torch version + thread count), `_peak_rss_mb` via `resource`.
  - Importing the module performs **no** model work.
- `scripts/profile_inference.py` — CLI. Default invocation is a **dry run**
  (prints context, loads nothing). Explicit flags:
  - `--mock` — synthetic stages (no model), validates instrumentation.
  - `--real-inference` — explicit consent gate required before touching NLLB;
    refuses to run otherwise. Never executed in this sprint.
  - `--runs`, `--samples`, `--limit`, `--direction`, `--output`.

Commands:

```bash
.venv/bin/python scripts/profile_inference.py                  # dry run, no model
.venv/bin/python scripts/profile_inference.py --mock --runs 5  # synthetic pipeline
# Later (explicit, opt-in):
.venv/bin/python scripts/profile_inference.py --real-inference \
    --runs 5 --direction eng_yor \
    --samples data/flores_eval/dev.30.en.txt --limit 3 \
    --output data/eval/profile_sprint2b.json
```

---

## 5. Optimization strategies (Task 5)

Prioritized: low-risk / high-confidence first. Production decoding config is
untouched.

| Strategy                                                                           | Expected benefit                                                             | Quality risk                                                            | Memory             | Complexity                                    | HW fit                           | Benchmark                                                                      |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------ | --------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------ |
| **Direction-specific beam widths** (2 for yor→eng, keep 4 for eng→yor)             | −18–30% yor→eng latency; beams=2 _outscored_ beams=4 on yor→eng chrF (+0.48) | None indicated by data; verify on devtest                               | None               | Very low (one conditional)                    | Full                             | Rerun `beam_ab_experiment.py` split by direction; chrF + latency per direction |
| **Model reuse** (already implemented)                                              | Prevents ~10–30 s cold reloads per request/process restart                   | None                                                                    | None               | Done; verify uvicorn workers=1 or shared load | Full                             | Cold vs warm via profiler's `cold_start` flag                                  |
| **Redis translation cache** (already implemented)                                  | Near-0 ms for repeated texts                                                 | None (deterministic model)                                              | Redis RAM only     | Done                                          | Full                             | Repeat-request microbenchmark via profiler                                     |
| **Reduced generation limits** (e.g. max_new_tokens scaled to input length)         | Cuts p95 tail (56.9 s) without touching typical cases                        | Truncation risk on long sources — needs chrF check                      | Lower              | Low                                           | Full                             | Tail-latency + chrF on the longest dev/devtest pairs                           |
| **CPU threading config** (`torch.set_num_threads`, interop)                        | Possibly 10–40% on Intel MBP; unknown sign                                   | None                                                                    | +RAM per thread    | Very low                                      | Full                             | Sweep threads ∈ {1,2,4,8} with profiler on fixed sentences                     |
| **Efficient batching** (batch concurrent requests)                                 | Higher throughput under load; per-request latency unchanged/worse            | None                                                                    | +activation memory | Medium (async queue)                          | Full                             | Load-test with N parallel requests                                             |
| **Quantization** (int8 dynamic, or PyTorch eager quant)                            | 2–4× speed, ~½ RAM on x86 CPU                                                | Moderate: chrF regression possible, esp. Yoruba's low-resource behavior | ~2.4→~0.7 GB       | Medium                                        | Good (x86 FBGEMM)                | Full devtest chrF + latency before/after                                       |
| **Smaller model** (distilled 130M/600M-en-yo variants, or a dedicated en↔yo model) | 2–5× speed                                                                   | High: chrF drop risk; quality agent's eval gates this                   | Less RAM           | Low to swap                                   | Full                             | Baseline vs candidate chrF on devtest                                          |
| **Alternative runtimes** (ONNX Runtime / CTranslate2 / bettertransformer)          | 2–4× CPU decode speed typical                                                | Conversion + numerics risk                                              | Similar or less    | Medium–High                                   | Good (CTranslate2 strong on CPU) | Port decoder, chrF parity check, then latency A/B                              |

**Top three candidates** (lowest risk first):

1. Direction-specific beam widths — free latency win backed by already-collected
   quality data.
2. Adaptive/capped `max_new_tokens` — targets the p95 tail.
3. CPU thread-count tuning — zero quality risk, measurable via the new profiler.

---

## 6. Controlled benchmark plan (Task 6)

Principles: identical source sentences (`data/flores_eval/dev.30.*.txt`, later the
quality agent's verified devtest sets), same local model weights, same directions,
fixed baseline config (`num_beams=4`, `length_penalty=1.1`, `max_new_tokens=256`),
one warm-up pass discarded per config, ≥5 timed passes, median + p95 + peak RSS per
config, chrF regression gate (no config ships if chrF drops >0.5 vs baseline).

Proposed matrix (each cell = one profiler run):

| ID       | Config                           | Directions       |
| -------- | -------------------------------- | ---------------- |
| BASE     | production (beams=4)             | eng→yor, yor→eng |
| B2       | beams=2 (both directions)        | both             |
| B2-YO    | beams=4 eng→yor, beams=2 yor→eng | both             |
| MX256    | max_new_tokens=256→128 cap       | both             |
| T1/T2/T4 | torch threads 1/2/4              | eng→yor only     |

Status/dependency: this sprint consumes the _existing_ verified artifacts
(`beams_ab_*.json/csv`). The quality agent is producing an upgraded evaluation
framework in a separate worktree; the benchmark runner must switch to that
framework's output format once integrated. That integration is a documented
dependency for the next sprint — the quality worktree was **not** modified here.

Execution stays deferred: no new inference ran in Sprint 2B.

---

## 7. Testing (Task 7)

`tests/test_profiling.py` (all mocked, no model):

- Fake-clock stage timing, cold-start init accounting ✔
- Timer accuracy vs `time.perf_counter` (±30% on a 100 ms busy loop) ✔
- JSON serialization round-trip ✔
- `summarize_latencies` stats + empty-input `ValueError` ✔
- Missing-measurement `KeyError` (no fabricated zeros) ✔
- Warm/cold exclusion in aggregates ✔
- Deterministic reproducibility with a fixed clock ✔
- Stage errors propagate while still recording elapsed time ✔
- Import-must-not-load-model guard (skipped where torch is absent) ✔

Result: `10 passed, 1 skipped` (skip = torch not installed in this venv; the
contract, "import does no model work", holds either way). Ruff clean; CLI dry-run
and `--mock` modes verified by execution.

---

## 8. Files changed

- `app/services/profiling.py` (new)
- `scripts/profile_inference.py` (new)
- `tests/test_profiling.py` (new)
- `configs/benchmark_plan_sprint2b.json` (new)
- `docs/phase3_sprint2b_performance_analysis.md` (this file)

No production code, no historical artifacts, and no other worktrees were touched.
