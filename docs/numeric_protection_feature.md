# Numeric Protection Reliability (feature/numeric-protection)

Owner: Agent 1 · Base commit: `9414279` · Files: `app/services/protection.py`,
`tests/test_numeric_protection_feature.py`, `docs/numeric_protection_feature.md`.

## What the mechanism does

Before NLLB generation, numeric expressions (numbers, scores, dates, times,
currency, percentages, URLs, emails, identifiers) are replaced with opaque
sentinels `@@0@@ … @@N-1@@`, which the model copies through verbatim in both
en↔yor directions. After generation, sentinels are restored to the original
expressions, and every restoration is structurally verified before it is
applied. On any verification failure the **original model output is returned
unchanged** — never a partial or guessed restoration.

## Bugs fixed on this branch

| Bug                                                  | Root cause                                                                                                                                                             | Fix                                                                                                                                                                     |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Nested sentinels (Bug TI-1, all passes)              | Multi-pass design: a later, lower-priority pattern could match digits inside a sentinel emitted by an earlier pass (e.g. `-12.5` → `-«0».5`, `2019-06-04` → `«0»-«1»`) | Single-pass combined alternation (`_MASTER_RE`); `finditer` never rescans a replaced span, so nesting is structurally impossible                                        |
| Wrong expression granularity                         | Same root cause: dates/times split into fragments, currency symbol detached from amount                                                                                | Specificity expressed by alternative order inside one pattern; longest/most-specific wins at every position                                                             |
| Placeholder collision                                | A user's literal `@@3@@` could collide with, or be confused for, a generated sentinel                                                                                  | User sentinel-like tokens are _claimed_: rewritten into fresh sequential sentinels whose stored original is the literal token itself; restoration returns them verbatim |
| Accidental replacement of placeholder-like user text | `restore()` substituted any `@@N@@` in the output                                                                                                                      | Only sentinels actually issued by `protect()` are substituted; unknown indexes are a structured `unknown_sentinel` failure                                              |
| Non-actionable failure reporting                     | `verify_placeholder_structure` returned a bare string                                                                                                                  | `analyze_placeholder_structure` / `restore_outcome` return structured `RestorationFailure` (kind, expected/found, missing, duplicated, unknown lists)                   |
| Sentinel numbering by pattern priority               | Indexes followed pattern order, not reading order                                                                                                                      | Indexes renumbered by left-to-right text position (Bug TI-2 fix preserved)                                                                                              |
| No reusable retry path                               | Restoration failure handling was hard-coded in `translation.py`                                                                                                        | `with_unprotected_fallback()` helper (below)                                                                                                                            |

## Public interface

Unchanged and backwards-compatible:

- `protect(source) -> ProtectionResult` (`.protected_text`, `.placeholders`, `.n`)
- `restore(model_output, protection) -> (text, ok, reason)`
- `verify_placeholder_structure(output, expected_n) -> (ok, reason)`
- `ProtectionResult`

New:

- `RestorationOutcome` — `(text, ok, failure, payload)`; on failure `text` is
  the safe-fallback output.
- `RestorationFailure` — frozen dataclass with `kind` ∈
  `{count_mismatch, duplicated, missing, missing_and_duplicated,
unknown_sentinel}`, plus `expected`, `found`, `missing`, `duplicated`,
  `unknown` tuples.
- `analyze_placeholder_structure(output, expected_map_or_count) -> failure|None`
- `restore_outcome(model_output, protection) -> RestorationOutcome`
- `with_unprotected_fallback(translate_protected, translate_plain, ...)` —
  runs the protected call; if its restoration outcome reports failure, invokes
  `on_failure(failure, raw_text)` (optional), consults `attempt_fallback`
  (optional gate), then calls `translate_plain` (with
  `protect_numerics=False` injected when the callable accepts kwargs).
  Returns `(result, used_fallback, failure)`.

## Failure modes and how they are handled

| Failure mode                                                   | Detection                                                                 | Behavior                                                                                               |
| -------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Model drops a sentinel during generation                       | `missing` (and `missing_and_duplicated` when a dup also occurred)         | Safe fallback: raw model output returned, `ok=False`; integration may retry unprotected via the helper |
| Model duplicates a sentinel                                    | `duplicated`                                                              | Same safe fallback                                                                                     |
| Model alters a sentinel (whitespace tolerated, digits changed) | `missing` for the altered index; tolerated `@@ 3@@` spacing is normalized | Same safe fallback                                                                                     |
| Model invents a sentinel (`@@9@@` never issued)                | `unknown_sentinel`                                                        | Same safe fallback                                                                                     |
| Total count drift with no per-index defect                     | `count_mismatch`                                                          | Same safe fallback                                                                                     |
| User text contains placeholder-like tokens                     | Claimed at protect time                                                   | Restored verbatim; never destroyed or confused                                                         |
| No expressions found (`n == 0`)                                | —                                                                         | Pass-through success                                                                                   |

`translation.py` currently logs the structured reason and returns the raw
model output. To keep unresolved sentinels away from users, integration should
wrap the protected call with `with_unprotected_fallback` (see below).

## Integration requirements (for Agent 2 / translation.py owner)

`translation.py` is NOT edited on this branch. Suggested integration —
one-line change in `translate()`:

```python
from app.services.protection import with_unprotected_fallback, protect

def _translate_once(self, text, src, tgt, protect_numerics):
    ...existing generation code, honoring protect_numerics...

result, used_fallback, failure = with_unprotected_fallback(
    lambda protect_numerics: self._translate_once(text, src, tgt, protect_numerics=True),
    lambda protect_numerics: self._translate_once(text, src, tgt, protect_numerics=False),
)
```

Requirements:

1. `_translate_once(protect_numerics=True)` must return an object whose
   `.payload` is the `RestorationOutcome` from `restore_outcome(...)` (or the
   outcome itself), so the helper can inspect `.ok` / `.failure` / `.text`.
2. Never surface text with `@@N@@` tokens to users: on `used_fallback=True`
   the returned result came from the unprotected retry; otherwise the
   protected result is verified-clean or carries the documented raw fallback.
3. Optionally pass `attempt_fallback` to gate retries (e.g. skip retry when
   the protected translation hit a cache or a phrase override).
4. Keep `numeric_protection_enabled` (config.py) as the global switch; the
   helper does not read settings itself.

## Remaining limitations

- **Decimal comma/point ambiguity:** `9.000,5` (European style) is only
  matched by the dotted-decimal/thousand rules in limited forms; fully
  locale-aware number parsing is out of scope.
- **Overlapping readings:** `2:1` is protected as a score, `11:29` as a time;
  when both readings are plausible the score alternative wins because it is
  ordered first and requires no meridiem — pathological clock times written
  as bare `H:MM` without context could be classified as scores. Both are
  protected (safe), only the stored grouping differs.
- **Ordinals and spelled-out numbers** ("third", "méjì") are not protected —
  the model must still translate them; only digit-based expressions are.
- **Very large numbers of placeholders** (100+ in one sentence) rely on the
  model copying sentinels verbatim; generation-side sentinel loss rates have
  not been measured beyond the Sprint TI-1 diagnostics sample.
- **`@@N@@` in user text** is claimed and restored verbatim, which means the
  model sees one fewer "real" content token than the original text contained;
  acceptable in practice, noted for completeness.
- **Yoruba number words:** digits are passed through unchanged (FLORES/news
  register); no conversion to Yoruba numerals is attempted.

## Tests

`tests/test_numeric_protection_feature.py` — 31 focused unit tests, no model
loaded: repeated numbers, negative numbers, decimals, thousand separators,
percentages (with/without space), times with meridiem, scores with ASCII and
Unicode dashes, colon scores, currency symbols and names, emails, URLs, ISO
and dotted dates, Unicode punctuation neighbors, idempotence, positional
numbering, user-placeholder claiming and collision, structured failure kinds,
the backward-compatible wrappers, and the fallback helper (success, retry,
gate, on_failure hook).

Run: `python -m pytest tests/test_numeric_protection_feature.py -q`
