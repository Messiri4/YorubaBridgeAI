"""Bidirectional verified-term glossary for translation (Sprint TI-1 / semantic).

Design
------
- Terms live in `data/phrases/glossary.json`: verified English↔Yoruba term
  pairs (single words and fixed phrases). Nothing is invented: entries are
  curated by humans (or taken from attested FLORES renderings), never
  machine-generated.
- Application is CONTEXT-SAFE and OCCURRENCE-WISE:
  * every candidate occurrence of a term is evaluated independently — its
    context words must appear within a token window *of that occurrence*;
    other occurrences of the same word elsewhere in the sentence are not
    affected
  * phrase-level entries (multi-word) are preferred over single words and
    never partially clobber each other (overlap rejection)
  * ambiguous words are NOT replaced without their context
- Unicode safety:
  * matching is token-based over the RAW text; character spans always refer
    to the original string, so tone marks and underdots (ọ, ẹ, ṣ, à, ...) are
    preserved byte-for-byte when no verified correction applies
  * token identity keeps diacritics (NFC + casefold): tone-distinct Yoruba
    minimal pairs never merge
  * accent-INSENSITIVE matching is opt-in only: it is refused when the
    folded form is ambiguous (i.e. two distinct entries — or two distinct
    source words with different translations — collapse to the same fold),
    which prevents silently merging distinct meanings such as
    ìgbà ("time") and igbá ("calabash")
- Both directions are supported (eng_yor and yor_eng tables), sharing the
  same context logic.

Schema (data/phrases/glossary.json):
{
  "_meta": {...},
  "eng_yor": [
    {"term": "diabetes", "translation": "àrùn àtògbẹ",
     "context": [], "status": "verified", "comment": "..."},
    ...
  ],
  "yor_eng": [...]
}

Entry fields:
- term / translation : required
- context            : optional list; all-OF semantics, checked per occurrence
- status             : "verified" | "assumed"; only "verified" entries are
                       applied. "assumed" entries are loaded (and surfaced in
                       `unverified()`) for human review but never substituted.
- allow_unaccented   : optional bool; permit accent-folded matching when
                       unambiguous
- comment            : provenance note
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_GLOSSARY_PATH = "data/phrases/glossary.json"

# context window in tokens: term matches are accepted only if every context
# word appears within +/- this many tokens of THAT occurrence.
CONTEXT_WINDOW = 6


@dataclass(frozen=True)
class TermEntry:
    term: str
    translation: str
    context: tuple[str, ...] = ()
    comment: str = ""
    status: str = "verified"          # "verified" | "assumed"
    allow_unaccented: bool = False
    # populated by the loader:
    ambiguous_unaccented: bool = field(default=False)

    @property
    def applicable(self) -> bool:
        return self.status == "verified"


def _normalize_word(word: str) -> str:
    """Identity for matching: NFC + casefold. Tone marks / underdots kept."""
    return unicodedata.normalize("NFC", word).casefold()


def _fold_word(word: str) -> str:
    """Accent-folded identity: tone marks and underdots stripped."""
    nfd = unicodedata.normalize("NFD", word.casefold())
    stripped = "".join(c for c in nfd if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped)


class Glossary:
    """Context-aware bidirectional terminology enforcement."""

    def __init__(self, path: str | None = None) -> None:
        self._entries: dict[str, list[TermEntry]] = {"eng_yor": [], "yor_eng": []}
        self._warnings: list[str] = []
        self._load(path or DEFAULT_GLOSSARY_PATH)

    # ------------------------------------------------------------------ load
    def _load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return  # glossary optional; degrade gracefully
        data = json.loads(p.read_text(encoding="utf-8"))
        for direction, rows in data.items():
            if direction not in self._entries or not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    self._entries[direction].append(TermEntry(
                        term=row["term"], translation=row["translation"],
                        context=tuple(row.get("context", [])),
                        comment=row.get("comment", ""),
                        status=row.get("status", "verified"),
                        allow_unaccented=bool(row.get("allow_unaccented", False)),
                    ))
                except (KeyError, TypeError):
                    continue
        # Ambiguity detection for accent-insensitive matching: if two distinct
        # entries in the same direction share the same accent-folded key, the
        # fold is meaningless (it would merge distinct meanings), so fold
        # matching is disabled for those entries — for BOTH sides of the pair.
        for direction in self._entries:
            by_fold: dict[str, set[str]] = {}
            for e in self._entries[direction]:
                by_fold.setdefault(_fold_word(e.term), set()).add(e.term)
            for e in self._entries[direction]:
                if len(by_fold.get(_fold_word(e.term), ())) > 1:
                    object.__setattr__(e, "ambiguous_unaccented", True)
                    self._warnings.append(
                        f"[{direction}] '{e.term}': accent-folded form is "
                        f"ambiguous; unaccented matching disabled")

    def warnings(self) -> list[str]:
        return list(self._warnings)

    def unverified(self, direction: str | None = None) -> list[TermEntry]:
        """Entries awaiting human verification (loaded but never applied)."""
        dirs = [direction] if direction else list(self._entries)
        return [e for d in dirs for e in self._entries.get(d, [])
                if not e.applicable]

    # ---------------------------------------------------------------- tokens
    @staticmethod
    def _tokens(text: str) -> list[tuple[str, int, int, str]]:
        """Word tokens of the RAW text.

        Returns (core, start, end, raw) tuples. ``core`` is the token with
        edge punctuation stripped; spans always index the ORIGINAL string, so
        precomposed (ọ) and decomposed (o + ◌̣) input yield correct offsets.
        Splitting is on whitespace, not on characters, therefore combining
        marks can never split a token.
        """
        tokens: list[tuple[str, int, int, str]] = []
        for m in re.finditer(r"\S+", text):
            raw = m.group(0)
            core = raw.strip(".,;:!?¡¿\"'“”„«»()[]{}…—–-")
            if not core:
                continue
            tokens.append((core, m.start(0), m.end(0), raw))
        return tokens

    # ----------------------------------------------------------------- apply
    def apply(self, text: str, direction: str) -> tuple[str, list[str]]:
        """Apply glossary substitutions. Returns (new_text, applied_terms).

        Rules:
        - multi-word entries first (longest match), then single words
        - each candidate occurrence is context-checked SEPARATELY
        - overlapping matches are rejected (longest wins)
        - "assumed"-status entries are never applied
        - surface text outside a replacement is untouched (tone marks and
          underdots preserved)
        """
        entries = [e for e in self._entries.get(direction, []) if e.applicable]
        if not entries or not text:
            return text, []

        tokens = self._tokens(text)
        if not tokens:
            return text, []

        entries = sorted(entries, key=lambda e: (-len(e.term.split()), -len(e.term)))
        applied: list[str] = []
        occupied: list[tuple[int, int]] = []  # token-index ranges already claimed

        for entry in entries:
            term_words = [_normalize_word(w) for w in entry.term.split()]
            term_words_fold = [_fold_word(w) for w in entry.term.split()]
            n = len(term_words)

            occurrences: list[tuple[int, int]] = []  # (first_tok_idx, last_tok_idx)
            for i in range(len(tokens) - n + 1):
                span_tokens = tokens[i:i + n]
                ids = [_normalize_word(t[0]) for t in span_tokens]
                folds = [_fold_word(t[0]) for t in span_tokens]
                if ids == term_words:
                    occurrences.append((i, i + n - 1))
                elif (entry.allow_unaccented and not entry.ambiguous_unaccented
                      and folds == term_words_fold):
                    occurrences.append((i, i + n - 1))

            if not occurrences:
                continue

            accepted: list[tuple[int, int]] = []
            for occ in occurrences:
                # overlap rejection: a longer, earlier-sorted entry owns the span
                if any(occ[0] <= b and a <= occ[1] for a, b in occupied):
                    continue
                # context is evaluated FOR THIS OCCURRENCE ONLY
                if entry.context and not self._context_ok_at(
                        tokens, occ, entry.context, tuple(term_words)):
                    continue
                accepted.append(occ)

            if not accepted:
                continue

            for occ in accepted:
                occupied.append(occ)
            # replace each occurrence individually, right-to-left, so text
            # BETWEEN occurrences is preserved
            for occ in sorted(accepted, key=lambda o: -tokens[o[0]][1]):
                start_char = tokens[occ[0]][1]
                end_char = tokens[occ[1]][2]
                text = (text[:start_char] + entry.translation
                        + text[end_char:])
            applied.append(f"{entry.term}→{entry.translation}")

            # re-tokenize: character offsets changed after substitution
            tokens = self._tokens(text)
            if not tokens:
                break
            occupied = []  # positions no longer comparable; fresh scan

        return text, applied

    # --------------------------------------------------------------- context
    @staticmethod
    def _context_ok_at(tokens: list[tuple[str, int, int, str]],
                       occ: tuple[int, int],
                       context: tuple[str, ...],
                       term_words: tuple[str, ...]) -> bool:
        """All context words must appear within CONTEXT_WINDOW tokens of the
        given occurrence (all-OF, accent-sensitive identity via NFC-casefold
        so tone-distinct context words are not conflated).

        Context words that are already part of the term itself (e.g.
        "Type 1 diabetes" with context ["diabetes"]) carry no extra
        disambiguating power and are dropped, otherwise the entry could
        never fire."""
        term_set = {_normalize_word(w) for w in term_words}
        ctx = {_normalize_word(c) for c in context} - term_set
        if not ctx:
            return True
        lo = max(0, occ[0] - CONTEXT_WINDOW)
        hi = min(len(tokens), occ[1] + CONTEXT_WINDOW + 1)
        # the occurrence itself must NOT satisfy its own context
        window = {_normalize_word(t[0]) for i, t in enumerate(tokens[lo:hi], lo)
                  if not (occ[0] <= i <= occ[1])}
        return ctx <= window
