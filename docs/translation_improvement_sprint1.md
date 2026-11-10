# Translation Improvement Sprint 1 (TI-1) — Numeric Protection & Glossary

**Branch:** `quality-review` · **Status:** AWAITING APPROVAL (not committed)
**No retraining; uses the existing local safetensors snapshot.**

## 1. Architecture changes

```
translate(text, src, tgt, protect_numerics=None)
  ├─ phrase-override lookup (unchanged, first)
  ├─ NEW: protection.pass-1  (protect_numerics=True only)
  │     emails → URLs → times → ISO dates → thousands/decimals →
  │     scores/ranges → currency → percentages (priority order)
  ├─ NEW: protection.pass-2  (remaining plain numbers; guard against
  │     rewriting inside sentinels)
  ├─ NLLB generate (unchanged decoding: beams=4, lp=1.1)
  ├─ NEW: restoration — verifies every sentinel present exactly once;
  │     on ANY failure returns the RAW model output (safe fallback)
  └─ NEW: glossary.apply (attested eng↔yor terms, context-gated,
        longest-match-first, accent-insensitive)
```

New modules: `app/services/protection.py`, `app/services/glossary.py`.
New data: `data/phrases/glossary.json` (attested mappings only).
Config: `numeric_protection_enabled` (default **True**),
`glossary_enabled` (default **True**), `glossary_path`. Both features are
individually switchable; `protect_numerics=False` restores exact legacy
behavior per call.

**Sentinel choice was empirically validated:** guillemets `«0»` were
normalized away by NLLB's SentencePiece (model emitted bare digits);
`@@0@@` survives generation verbatim in both directions. Sentinel
indexes follow text position, not pattern priority.

## 2. Before/after examples (real NLLB, one model instance, sequential)

| #   | Source                                        | Baseline                                                    | Improved                   | Verdict                            |
| --- | --------------------------------------------- | ----------------------------------------------------------- | -------------------------- | ---------------------------------- |
| 1   | The head-to-head record is 7–2.               | `…jẹ 72.` (score corrupted)                                 | `orí sí orí ni 7–2.`       | **fixed** (verified Phase-2 error) |
| 2   | The show starts at 11:29.                     | `aago mọ́kànlá ó lé mọ́kàndínláàádọ́rùn-ún` (garbled ~"11:39") | `ní 11:29.`                | **fixed**                          |
| 3   | About 9,000 people attended…                  | `ẹgbẹ̀rún mẹ́sàn-án` ("nine hundred/thousand" ambiguous)      | `bí 9,000 àwọn èèyàn`      | **preserved exactly**              |
| 4   | Email support@example.com … refund of $1,000. | email ok, `$1,000` → "ẹgbẹ̀rún kan dọ́là"                     | email + `$1,000` preserved | **improved**                       |
| 5   | Ẹ kú ìrọ̀lẹ́, ọ̀rẹ́ mi.                           | Good evening, my friend.                                    | same                       | **no regression** (override path)  |
| 6   | Òjò ń rò ní ìlú Kano.                         | It is raining in Kano.                                      | same                       | **no regression**                  |
| 7–8 | glossary sentences (àtògbẹ)                   | already correct via model                                   | same                       | **no regression**                  |

Raw data: `data/eval/ti1_improvement_comparison.json`.

## 3. Improvements confirmed

- Score `7–2` no longer corrupts to `72` (the Phase-2 verified error).
- Clock time `11:29` no longer garbles into a wrong spelled compound.
- `9,000`, `$1,000`, emails, URLs round-trip exactly.
- Dropped/duplicated/merged sentinels are detected; the documented safe
  fallback (raw model output) fires instead of silent corruption.
- **Mean latency on the 8-case set: 7 283 ms → 4 311 ms (-41%).** This is
  not a protection effect: baseline beams wander (NLLB spends beams on
  elaborating numerals into long Yoruba compounds), while sentinel text is
  short and committed quickly. Varied per case but uniformly faster here;
  treat as an incidental benefit, not a guarantee.

## 4. Regressions discovered & fixed during the sprint (with tests)

1. Catch-all number regex rewrote digits _inside_ placed sentinels
   (`«1»` → `««1»»`) — fixed with an interior-skip guard; regression test.
2. Sentinel numbering followed pattern priority instead of text position —
   fixed by post-pass renumbering; positional test.
3. Guillemet sentinel destroyed by the tokenizer — replaced with `@@i@@`
   after empirical NLLB probes; diagnostic kept in the module docstring.
4. Glossary: accent-insensitive matching (unaccented `atogbe` now matches
   `àtògbẹ`), longest-match-first, and context gating verified.

## 5. Remaining limitations

- Restoration requires the model to copy sentinels verbatim; if a future
  model reformats them, the safe fallback returns raw output (correctness
  preserved, improvement lost) — logged for monitoring.
- The catch-all still protects lone integers in flowing prose; whether that
  _helps_ fluency long-term needs the FLORES dev A/B (not run here to keep
  inference minimal).
- Glossary contains only 6 attested entries; growth requires verified
  sources (no invented Yoruba, per policy).
- Latency benefit is workload-dependent; re-measure on the full dev split
  before claiming anything beyond these 8 cases.
- Human-review labels remain unavailable; semantic fidelity of outputs 1–4
  was judged by direct numeral comparison (objective), not full fluency.

## 6. Test results

`168 passed, 1 skipped` — including 15 new TI-1 tests (protection
round-trips, corrupted-output fallbacks, glossary direction/context/length
behavior, service-level mock integration). Full real-inference was run once,
sequentially, on the local snapshot; no other agent loaded a model.

## 7. Recommended next changes

1. FLORES-dev A/B of protection on/off (n=997 both arms) using the unified
   scorecard — quantifies the numeric-rate delta corpus-wide.
2. Extend the glossary from the Phase-2 entity findings (sports/politics
   names), keeping the no-invented-Yoruba rule.
3. Wire the completed 20-pair priority review to validate semantic quality
   beyond numeric checks.
