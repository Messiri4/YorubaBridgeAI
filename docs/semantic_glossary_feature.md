# Semantic Glossary Feature (Sprint TI-1, semantic hardening)

Branch: `feature/semantic-glossary`
Owner: Agent 2 — terminology / semantic correctness
Files: `app/services/glossary.py`, `data/phrases/glossary.json`,
`tests/test_semantic_glossary_feature.py`, this document.

## What the glossary is

A context-aware, bidirectional (English↔Yoruba) verified-term enforcement
layer used before/after MT output. It substitutes only **attested**
translations held in `data/phrases/glossary.json`; it never generates
Yoruba.

## Defects fixed in this sprint

1. **Raw-character offsets corrupted by Unicode normalization.** The old
   implementation matched against NFD-folded text and mapped indices back
   through an incomplete `fold_map` (NFD/NFC length mismatch). Occurrences
   after a decomposed character could be replaced at wrong offsets,
   corrupting neighbouring text. The new implementation tokenizes the raw
   text (whitespace split, never splitting combining marks) and computes
   every replacement span directly on the original string. Composed (ọ)
   and decomposed (o + ◌̣) input produce identical behaviour — see
   `test_composed_and_decomposed_input_agree`.

2. **Unicode-unaware word boundaries.** Regex `\b`-style matching was
   replaced by an explicit token scan over raw text with punctuation-edge
   stripping; token _identity_ is NFC + casefold, so tone marks and
   underdots are compared exactly while surface text is never rewritten.

3. **Context evaluated once for all occurrences.** Previously a context
   satisfied anywhere in the sentence unlocked every occurrence of the
   term. Now each candidate occurrence is checked independently against
   its own ±`CONTEXT_WINDOW` (6) token window
   (`test_same_term_twice_only_contextual_occurrence_fires`).
   The occurrence itself is excluded from satisfying its own context.

4. **Accent-insensitive matching silently merged distinct meanings.**
   Fold-matching is now **opt-in** per entry (`allow_unaccented: true`)
   and automatically **disabled** when two distinct terms in the same
   direction collapse to the same accent-folded form (e.g. ìgbà "time"
   vs igbá "calabash"), with a recorded warning
   (`test_accent_folding_disabled_on_collision`). Context words are
   compared accent-**sensitively**, so tone-distinct contexts do not
   conflate.

5. **Multiple accepted occurrences were collapsed into one giant span.**
   In the old code, when an entry matched twice, both matches fed a single
   `text[start:end]` slice — deleting everything between them. Matches are
   now replaced one-by-one, right-to-left, each with its own span.

6. **Term words satisfying their own context.** A multi-word entry such as
   "Type 1 diabetes" with `context: ["diabetes"]` could never fire, because
   the context word occurred only inside the term. Context words that are
   part of the term are now discounted.

## Entry schema and verification status

```json
{
  "term": "…", "translation": "…",
  "context": ["word", …],          // optional, all-OF, per-occurrence
  "status": "verified",            // "verified" | "assumed"
  "allow_unaccented": false,       // optional, default false
  "comment": "provenance"
}
```

- `verified` — attested (FLORES renderings or established usage) and human-
  reviewed. Applied by `Glossary.apply`.
- `assumed` — plausible but **unreviewed**: loaded and listed by
  `Glossary.unverified()` for human review, **never** substituted
  (`test_assumed_entries_loaded_but_never_applied`).

Current data: all six existing entries are marked `verified` — they are the
attested FLORES-derived mappings already present (diabetes/àrùn àtògbẹ,
Type 1 diabetes, Shark Tank, àtògbẹ, ìtọ ṣúgà, aago ẹnu ọ̀nà). **No new
Yoruba translations were invented in this sprint.**

## Preserving tone marks and underdots

Text outside a replacement is passed through byte-for-byte; a replacement
touches only the exact matched span (`test_tone_marks_preserved_on_
untouched_words`). An entry is applied only when a _verified_ correction
exists; otherwise the accented surface form survives untouched.

## Expanding verified bilingual terminology

1. **Source evidence first.** Only add a mapping backed by (a) FLORES-dev
   / FLORES-test Yoruba renderings, (b) published Yoruba corpora/lexicons
   (e.g. NCRL, Yorùbá Wiktionary entries with citations), or (c) review by
   a fluent Yoruba speaker. Cite the source in `comment`.
2. **Workflow.** Add the candidate with `"status": "assumed"`; run
   `Glossary.unverified()` to get the review queue; a reviewer flips the
   status to `"verified"` only after checking sense, register, and tone
   marks (including underdot letters ẹ/ọ/ṣ).
3. **Ambiguous terms must ship with context.** Any word with more than one
   common sense (bank, light, stroke, Yoruba tone-minimal pairs) needs a
   `context` list naming disambiguating words. Test the entry with both a
   positive and a negative sentence before verification.
4. **Never enable `allow_unaccented` for tone-minimal pairs.** The loader
   refuses it anyway when fold forms collide, but keep the practice.
5. **Add regression tests** in `tests/test_semantic_glossary_feature.py`
   alongside the entry: positive context, negative context, repeated word.

## Test results

`python -m pytest tests/test_semantic_glossary_feature.py -q` → **23 passed**.
Coverage includes: repeated words with divergent decisions per occurrence,
ambiguous terms without context left untouched, NFC/NFD equivalence,
decomposed-offset correctness, tone-mark preservation, accent-fold
collision refusal, longest-match/overlap protection, unverified-entry
blocking, missing/empty glossary fallback.

## Limitations

- Context matching is lexical (bag-of-words within a token window);
  it cannot handle morphological variants ("strokes" ≠ "stroke") or
  word-order-dependent senses.
- No sense inventory: two entries sharing a term differ only by context.
- Tokenization is whitespace + edge-punctuation; intra-word hyphenation
  and clitics are not split.
- `apply` mutates text sequentially and re-tokenizes after each entry;
  O(entries × tokens), fine for current sizes.
- Ignores other agents' files by design: the pipeline must call
  `Glossary.apply(text, direction)` from translation glue code.

## Integration requirements

- Callers construct `Glossary(path=None)` (defaults to
  `data/phrases/glossary.json`; missing file degrades to a no-op) and call
  `apply(text, "eng_yor" | "yor_eng")`.
- Optional feedback channels for UI/logging: `warnings()` (ambiguity
  notices) and `unverified(direction)` (review queue).
- After Agent 1's protection layer runs, glossary application must come
  after protected-span masking so placeholders are never treated as terms.
- Shared per-request instance is safe: `apply` has no mutable class state
  used across calls.
