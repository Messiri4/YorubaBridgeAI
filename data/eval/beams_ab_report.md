# Beam-width A/B experiment — num_beams=2 vs num_beams=4

**Run:** 2026-11-09 UTC · **Split:** FLORES-200 `dev`, 30 aligned EN–YO pairs
**Sampling:** seeded shuffle (`random.Random(42)`) over all 997 dev indices,
first 30 kept — reproducible offline (`data/flores_eval/dev.30.*`, provenance JSON included).
**Model:** `facebook/nllb-200-distilled-600M` (existing local safetensors snapshot,
single model instance reused for all four cells — no second model in memory).
**Identical across arms:** source sentences, references, max_new_tokens=256,
length_penalty=1.1, max input chars, overrides **disabled** (raw model measured),
greedy tie-break defaults. Only `num_beams` differs.

## Results (corpus-level SacreBLEU, both directions)

| Direction | Config      | chrF      | BLEU      | mean lat | median lat | total   |
| --------- | ----------- | --------- | --------- | -------- | ---------- | ------- |
| eng→yor   | beams=2     | 28.12     | 3.86      | 25.67 s  | 22.18 s    | 770.0 s |
| eng→yor   | **beams=4** | **28.76** | **4.57**  | 30.45 s  | 30.07 s    | 913.6 s |
| yor→eng   | **beams=2** | **45.99** | **19.21** | 13.48 s  | 12.11 s    | 404.4 s |
| yor→eng   | beams=4     | 45.51     | 18.55     | 17.49 s  | 15.44 s    | 524.7 s |

Beam width changed the output on **26/30 eng→yor** and **15/30 yor→eng**
sentences, so the arms are meaningfully different decodes, not noise.

## Interpretation

- **eng→yor:** beams=4 wins on both metrics (+0.64 chrF, +0.71 BLEU) at
  ~19% higher latency. Into-Yoruba generation benefits from wider search.
- **yor→eng:** beams=2 wins on both metrics (+0.48 chrF, +0.66 BLEU) at
  ~23% lower latency. Into-English is already well-modeled; extra beams
  drift toward longer, less faithful outputs (the same over-generation the
  baseline report noted with the additive style).
- At n=30, ±0.5–1.0 chrF is within noise (see baseline report caveat);
  treat direction-level deltas as weak signals, but they point the same
  way as latency: beams=2 is strictly better for yor→eng, and slightly
  worse-but-cheaper for eng→yor.

## Recommendation

1. **yor→eng (Yoruba user typing to the bridge): switch default to beams=2** —
   better on both metrics _and_ ~23% faster. This is the safe, immediate change.
2. **eng→yor: keep beams=4.** The quality edge is consistent across both
   metrics; the 19% latency cost is acceptable for async WhatsApp traffic.
   (If latency ever becomes critical, beams=2 costs only ~0.6 chrF.)
3. Direction-dependent beam width requires a one-line conditional in the
   service (`num_beams=2 if target=="eng" else 4`) — deliberately **not**
   applied here to keep production behavior untouched pending owner sign-off;
   the summary JSON contains everything needed to implement it.
4. Re-validate on the held-out devtest split before changing production
   defaults (n=30 dev is a tuning signal, not proof).

## Files

- `data/eval/beams_ab_20261109T231413Z.csv` — 120 rows: config, direction, idx,
  source, reference, prediction, latency_s
- `data/eval/beams_ab_20261109T231413Z.summary.json` — configs, scores, latency stats
- `data/eval/beams_ab_partial_20261109T231413Z.jsonl` — incremental cell saves
  (resume support; each completed (beams, direction) cell is flushed immediately)
- `data/flores_eval/dev.30.{en,yo}.txt`, `dev.30.provenance.json` — sampled pairs

## Reproduce

```bash
cd /Users/user/Desktop/WORK/AI-projects/YorubaBridgeAI-quality
TRANSLATION_MODEL_ID=.model-local/nllb-600M-fixed PYTHONPATH=$PWD \
  .venv/bin/python scripts/beam_ab_experiment.py --split dev --n 30 --beams 2 4
# resume an interrupted run without repeating completed cells:
... scripts/beam_ab_experiment.py --split dev --n 30 --beams 2 4 --resume <stamp>
```

## Procedure notes

- Environment: dedicated worktree venv (`.venv`); the main application venv
  was not modified. sentencepiece had to be installed with `--only-binary`
  (source build fails on this Mac for lack of OpenMP cmake config).
- A 2-pair end-to-end smoke test validated save/resume mechanics first
  (`beams_ab_20261109T230518Z.*`); the resume path was exercised by re-running
  with `--resume`, which replayed 2 cached cells and computed only the new ones.
- Full run took ~34 min CPU, sequential, single model instance (16 GB budget respected).
