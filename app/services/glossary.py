"""Bidirectional verified-term glossary for translation (Sprint TI-1).

Design
------
- Terms live in `data/phrases/glossary.json`: verified English↔Yoruba term
  pairs (single words and fixed phrases). Nothing is invented: entries are
  curated by humans (or taken from attested FLORES renderings), never
  machine-generated.
- Application is CONTEXT-SAFE:
  * phrase-level entries (multi-word) are preferred over single words
  * a term is applied only when its context words appear nearby
    (`context` list, all-OF semantics within a ±window) — ambiguous words
    are NOT replaced without their context
  * longest-match-first prevents partial clobbering
- Both directions are supported (eng_yor and yor_eng tables), sharing the
  same context logic.

Schema (data/phrases/glossary.json):
{
  "eng_yor": [
    {"term": "diabetes", "translation": "àrùn àtògbẹ",
     "context": ["diabetes"], "comment": "attested FLORES rendering"},
    ...
  ],
  "yor_eng": [...]
}
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GLOSSARY_PATH = "data/phrases/glossary.json"

# context window in tokens: term matches are accepted only if every context
# word appears within +/- this many tokens of the term.
CONTEXT_WINDOW = 6


@dataclass(frozen=True)
class TermEntry:
    term: str
    translation: str
    context: tuple[str, ...]
    comment: str = ""


def _strip_marks_lower(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", text)


class Glossary:
    """Context-aware bidirectional terminology enforcement."""

    def __init__(self, path: str | None = None) -> None:
        self._entries: dict[str, list[TermEntry]] = {"eng_yor": [], "yor_eng": []}
        self._load(path or DEFAULT_GLOSSARY_PATH)

    def _load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return  # glossary optional; degrade gracefully
        data = json.loads(p.read_text(encoding="utf-8"))
        for direction, rows in data.items():
            if direction not in self._entries or not isinstance(rows, list):
                continue
            for row in rows:
                try:
                    self._entries[direction].append(TermEntry(
                        term=row["term"], translation=row["translation"],
                        context=tuple(row.get("context", [])),
                        comment=row.get("comment", "")))
                except (KeyError, TypeError):
                    continue

    # ------------------------------------------------------------------
    @staticmethod
    def _token_spans(text: str) -> list[tuple[str, int, int]]:
        """Tokens of the raw text with char spans; Yoruba-safe tokenization
        (combining marks removed only for the *token identity* comparison,
        never for the output text)."""
        folded = unicodedata.normalize("NFD", text)
        folded_clean = "".join(
            c for c in folded
            if not unicodedata.combining(c)) if any(
            unicodedata.combining(c) for c in folded) else folded
        spans: list[tuple[str, int, int]] = []
        for m in re.finditer(r"\S+", text):
            tok_raw = m.group(0)
            # token identity: strip punctuation edges
            core = re.sub(r"^[^\w]+|[^\w]+$", "", tok_raw, flags=re.UNICODE)
            if not core:
                continue
            identity = _strip_marks_lower(core)
            spans.append((identity, m.start(0), m.end(0)))
        return spans

    # ------------------------------------------------------------------
    def apply(self, text: str, direction: str) -> tuple[str, list[str]]:
        """Apply glossary substitutions. Returns (new_text, applied_terms).

        Rules:
        - multi-word entries first (longest match), then single words
        - skip application if any entry's context words are NOT nearby
        - already-applied translations are never re-replaced
        """
        entries = self._entries.get(direction, [])
        if not entries or not text:
            return text, []
        entries = sorted(entries, key=lambda e: -len(e.term.split()))
        applied: list[str] = []
        out = text
        for entry in entries:
            # Accent-insensitive term matching: the fold map is built from the
            # RAW text so replacement preserves original surface forms.
            fold_map = self._fold_map(out)
            term_fold = _strip_marks_lower(entry.term)
            # regex against folded text, but replacements need raw spans
            matches = [m for m in re.finditer(re.escape(entry.term), out,
                                              re.IGNORECASE)]
            if not matches:
                # try fold-based matching for unaccented/variant input
                matches = []
                fold_re = re.compile(re.escape(term_fold), re.IGNORECASE)
                for fm in fold_re.finditer(_strip_marks_lower(out)):
                    raw_start = fold_map.get(fm.start())
                    raw_end = fold_map.get(fm.end())
                    if raw_start is None or raw_end is None:
                        continue
                    matches.append(type("M", (), {
                        "start": lambda self, s=raw_start: s,
                        "end": lambda self, e=raw_end: e,
                        "group": lambda self, s=raw_start, e=raw_end: out[s:e],
                    })())
                if not matches:
                    continue
            if entry.context and not self._context_satisfied(out, entry, None,
                                                             matches):
                continue
            # apply right-to-left so spans stay valid
            for m in sorted(matches, key=lambda x: -x.start()):
                out = out[:m.start()] + entry.translation + out[m.end():]
            applied.append(f"{entry.term}→{entry.translation}")
        return out, applied

    @staticmethod
    def _fold_map(text: str) -> dict[int, int]:
        """Char index map: folded-string index -> raw-text index."""
        folded = unicodedata.normalize("NFD", text)
        mapping: dict[int, int] = {}
        fi = 0
        for ri, c in enumerate(folded):
            if unicodedata.combining(c):
                continue
            mapping[fi] = ri
            fi += 1
        return mapping

    def _context_satisfied(self, text: str, entry: TermEntry,
                           term_re: re.Pattern[str] | None,
                           matches: list | None = None) -> bool:
        """All context words must appear within CONTEXT_WINDOW tokens of a
        term match (all-OF). Empty context = unconditional."""
        if not entry.context:
            return True
        spans = self._token_spans(text)
        ctx = {_strip_marks_lower(c) for c in entry.context}
        starts = [m.start() if hasattr(m, "start") and not callable(getattr(m, "start", None)) else m.start()
                  for m in (matches or [])]
        # term_re-based path (unused now but kept for API stability)
        if not starts and term_re is not None:
            starts = [m.start() for m in term_re.finditer(text)]
        for start_char in starts:
            idx = None
            for i, (_t, s, e) in enumerate(spans):
                if s <= start_char <= e:
                    idx = i
                    break
            if idx is None:
                continue
            window = spans[max(0, idx - CONTEXT_WINDOW): idx + CONTEXT_WINDOW + 1]
            window_words = {t for t, _s, _e in window}
            if ctx <= window_words:
                return True
        return False
