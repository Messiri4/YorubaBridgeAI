"""Exact-span protection for numbers, identifiers, URLs, and explicit names."""

from __future__ import annotations

import re
import unicodedata

PROTECTED_PATTERNS = (
    re.compile(r"https?://[^\s,;!?]+|www\.[^\s,;!?]+", re.IGNORECASE),
    re.compile(r"\+\d{1,3}(?:[-.\s]\d{2,4}){2,4}\b"),
    re.compile(r"\b(?:\(?\d{2,4}\)?[-.\s]?){2,4}\d{2,4}\b"),
    re.compile(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b"),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:[AaPp]\.?[Mm]\.?|UTC(?:[+-]\d+)?)?\b"),
    re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%\b"),
    re.compile(r"\b\d+(?:\.\d+){1,3}(?:[-+][A-Za-z0-9.-]+)?\b"),
    re.compile(r"\b[A-Z]{1,12}(?:[-_/][A-Z0-9]+)+\b"),
    re.compile(r"\b[A-Za-z0-9]+[-_/][A-Za-z0-9][A-Za-z0-9_-]*\b"),
    re.compile(r"\b\d+(?:[.,]\d+)*\b"),
)

NAME_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z]*(?:[.'’-][A-Z]?[A-Za-z]*)?)"
    r"(?:\s+(?:[A-Z][A-Za-z]*(?:[.'’-][A-Z]?[A-Za-z]*)?)){1,3}\b"
)


class SpanProtector:
    """Temporarily replace exact spans and restore them after translation."""

    def __init__(self) -> None:
        self._spans: dict[str, str] = {}

    def protect(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        matches: list[tuple[int, int, str]] = []
        for match in re.finditer(
            r"\$\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s+(?:million|billion|trillion|thousand))?",
            text,
            re.IGNORECASE,
        ):
            matches.append((match.start(), match.end(), match.group(0)))
        for pattern in PROTECTED_PATTERNS:
            matches.extend((match.start(), match.end(), match.group(0))
                           for match in pattern.finditer(text))

        if not matches:
            for match in NAME_PATTERN.finditer(text):
                matches.append((match.start(), match.end(), match.group(0)))
        else:
            for match in NAME_PATTERN.finditer(text):
                matches.append((match.start(), match.end(), match.group(0)))

        unique: dict[tuple[int, int], str] = {}
        for start, end, span in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
            if any(start < used_end and end > used_start
                   for used_start, used_end in unique):
                continue
            unique[(start, end)] = span

        output = text
        protected = sorted(unique.items(), key=lambda item: item[0][0], reverse=True)
        for (start, end), span in protected:
            placeholder = f"__YBPROTECT_{len(self._spans)}__"
            self._spans[placeholder] = span
            output = output[:start] + placeholder + output[end:]
        return unicodedata.normalize("NFC", output)

    def restore(self, text: str) -> str:
        for placeholder, span in self._spans.items():
            text = text.replace(placeholder, span)
        return unicodedata.normalize("NFC", text)
