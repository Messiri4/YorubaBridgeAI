# Yoruba Tonality & Orthography Implementation (Agent 3, feature/yoruba-tonality)

Status: **foundation delivered; tone restoration itself is NOT implemented** — see
["What is genuinely implemented"](#what-is-genuinely-implemented) vs
["What remains experimental"](#what-remains-experimental).

Branch: `feature/yoruba-tonality` · Starting commit `9414279`.

---

## 1. Architecture

Two new modules, no changes to files owned by other agents:

```
app/services/yoruba_orthography.py   # deterministic orthography layer (production-safe)
app/services/tone_restoration.py     # interface + safe fallback for future model (experimental)
tests/test_yoruba_tonality_feature.py
docs/yoruba_tonality_implementation.md   (this file)
```

### 1.1 `yoruba_orthography.py` — deterministic, no model

Responsibilities:

| Function                    | Purpose                                                                                                                                                                                                            |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `nfc(text)` / `nfd(text)`   | Unicode NFC/NFD normalization that **preserves** every tone and underdot distinction. NFC composes e.g. `o + U+0323` → `ọ` (U+1ECD) and `o + U+0300` → `ò`; it never deletes marks.                                |
| `split_marks(text)`         | Decompose a word into base letters + ordered combining marks (`U+0300` low, `U+0301` high, `U+0323` underdot).                                                                                                     |
| `count_tone_marks(text)`    | Counts of low / high / underdot marks.                                                                                                                                                                             |
| `has_full_marks(text)`      | Approximate check that every vowel in every word carries a tone mark. **Approximate because mid tone is written as no mark** — a bare word may be legitimately all-mid-tone, not under-marked.                     |
| `audit(text)`               | Returns an `OrthographyReport` of issues **without proposing fixes**: `maybe_missing_tones` (word has vowels, zero low/high marks) and `inconsistent_marking` (same folded word appears both marked and unmarked). |
| `fold_for_comparison(text)` | Diagnostic-only fold (tone + underdot removed). Never user-visible; mirrors the folding pattern already used in `scorecard.py` and `glossary.py`.                                                                  |

Design constraints honored:

- **Unicode normalization is not tone restoration.** `nfc()` makes representations
  consistent; it makes no linguistic choices.
- **No dictionary fabrication.** `audit()` only _flags_; issue details explicitly say the
  form "needs human/model verification".
- **No claim of tonal correctness from string matching.** `has_full_marks` is documented
  as "all vowels marked", not "tones correct".

### 1.2 `tone_restoration.py` — contract + safe fallback

- `ToneRestorer` (typing `Protocol`): a future context-aware diacritization model
  implements `restore(yoruba_text, source_english=None) -> ToneRestorationResult`.
- `ToneRestorationResult` carries `restored_text`, `changed`, `confidence`,
  `per_word_confidence`, `fallback_used`, and a human-readable `reason`.
- `PassThroughRestorer`: the **current default** — NFC-normalizes only, changes no
  linguistic content, reports confidence 0.0 and honest status.
- `ThresholdGuard`: wraps any real model. Below a configurable threshold
  (default `0.85`) the **original input is returned unchanged** and the model's
  suggestion is withheld; per-word confidence below threshold is also blocked.
- `propose_restoration(...)` is the intended single entry point to call **after NLLB
  generation**; swapping in a real model means replacing `DEFAULT_RESTORER` — nothing
  else in the pipeline changes.

### 1.3 Inspection of existing Unicode handling (Task 1)

Prior to this branch, normalization in the codebase was already NFC/NFD-based but
distributed and purpose-specific:

- `app/services/cache.py` — `normalize_for_cache` uses NFC so composed/decomposed
  diacritics share cache keys.
- `app/services/scorecard.py` — `fold_accents` / `_fold_tones` strip marks for
  _diagnostics only_; NFC variants of chrF are reported separately.
- `app/services/glossary.py` — `_strip_marks_lower` for term lookup; token identity is
  folded, output text is never altered.
- `app/services/translation.py` — `PhraseOverrides._normalize` NFC for exact-match keys.
- `app/services/model_loader.py` — notes that decoding is byte-level so diacritics
  survive generation.

None of these normalize _into_ stripped forms in user-visible text — verified by
inspection. The new module centralizes Yoruba-specific mark logic so future agents
reuse `yoruba_orthography` instead of re-implementing folds.

---

## 2. Pretrained Yoruba diacritization: survey (Task 5)

Findings from public research (no models were downloaded or loaded):

- **Orife (Iroro Orife) — soft-attention seq2seq ADA** ("automatic diacritics
  application"): trained on the Yorùbá Bible, Lagos-NWU speech corpus and language
  blogs; poor generalization outside the religious domain
  ([Knowledge 4 All Foundation — Yoruba database project](https://k4all.org/project/database-yoruba/)).
- **Orife et al. — multi-domain ADA**: added conversation, short stories/proverbs,
  books, JW300; evaluated on news (Global Voices) for domain generalization (same source).
- **MENYO-20k** (Adelani et al., MT Summit 2021): multi-domain Yoruba–English parallel
  corpus with curated orthography and standardized splits; their dedicated NMT models
  beat Google (+8.7 BLEU) and M2M (+9.1) into Yoruba, and diacritized training data is
  shown to matter for output understandability
  ([ACL Anthology](https://aclanthology.org/2021.mtsummit-research.6/)).
- **NaijaML** ships a CPU-first Yoruba diacritizer (self-reported ~97.5% on dot-below
  but ~90% on full tonal marks) with small models bundled/offline
  ([github.com/naijaml/naijaml](https://github.com/naijaml/naijaml)). The vendor's own
  honesty about 90% full-tonal accuracy is itself the argument for our
  confidence-threshold fallback: a model that is wrong 1 word in 10 should never be
  allowed to silently rewrite an output sentence.
- **Oyo-T5** (Olawole et al. 2024, YAD dataset): T5 pre-training/fine-tuning with
  Yoruba-specific SentencePiece and heuristic mark-removal augmentation
  ([emergentmind topic page](https://www.emergentmind.com/topics/custom-trained-t5-model)).

**Conclusion:** no verified pretrained Yoruba tone-restoration model is available
locally in this environment, and per project rules nothing was downloaded. The
interface, uncertainty reporting, and fallback are implemented and tested; the actual
model is the next implementation requirement (§5).

---

## 3. Proposed integration after NLLB generation (Task 6)

```
NLLB (translation.py)  →  propose_restoration(pred, source_english)  →  response
                              │
                              ├─ ThresholdGuard
                              │    ├─ confidence ≥ threshold → model's toned text
                              │    └─ below threshold → NLLB output unchanged + reason logged
                              └─ PassThroughRestorer (today: always)
```

Key points:

1. The restorer receives **both** the NLLB Yoruba output and the English source —
   context from the source sentence is exactly what disambiguates tone (e.g. subject
   focus vs. object focus) that word-local models cannot resolve.
2. Restoration is applied **only on English→Yoruba generation output**, never on
   user-supplied Yoruba input (we must not "correct" a native writer).
3. Audit results from `audit()` should be logged with the translation record so human
   reviewers can see _what the system was uncertain about_, not just the final text.

---

## 4. Limitations

- `has_full_marks` and `audit` are **syntactic heuristics**. Mid tone has no written
  mark, so "unmarked vowel" is ambiguous between "all-mid-tone" and "missing marks".
  The code says so in issue details rather than pretending certainty.
- No syllable-structure knowledge (CV vs. V syllables, syllabic nasals like `ń`), so
  mark-count heuristics can over- or under-flag unusual words.
- Elision contractions (apostrophe forms like `'ràn`) are treated word-locally.
- Ambiguous minimal tone pairs (`owa`/`ọ̀wá`/`ọwá`-type distinctions, `kò`, `òṣèṣé`-type
  temporals) are deliberately **not resolved anywhere** in this feature.
- Threshold 0.85 is a starting point, not calibrated against real model confidence
  distributions (no model yet).

---

## 5. Next implementation requirements (for the actual restorer)

1. **Choose a model**: fine-tune a small seq2seq (mT5/T5-small or the Orife-style
   LSTM/GRU ADA architecture) on MENYO-20k + JW300 + Global Voices Yoruba with the
   multi-domain preprocessing from Orife et al.; or evaluate NaijaML's bundled
   diacritizer as a first CPU-viable baseline.
2. **Training data rules**: train only on human-verified diacritized text; hold out a
   domain-matched test set (our FLORES/evaluation splits) — the Orife Bible-model
   failure is the cautionary tale.
3. **Calibration**: measure confidence calibration on the held-out set; set the
   `ThresholdGuard` threshold from the reliability curve, not by intuition.
4. **Per-word confidence**: the interface already accepts `per_word_confidence`; the
   model must output token-level uncertainty (e.g. via per-token entropy) so partial
   fallbacks (restoring only high-confidence words) become possible later.
5. **Evaluation gate**: before enabling in production, show corpus-level chrF gain on
   our evaluation pipeline _and_ zero regressions on fully-marked reference text.
6. **Wiring point**: replace `DEFAULT_RESTORER` in `tone_restoration.py`; no changes to
   `translation.py` are required by design (the integration decision belongs to the
   translation-service owner).
