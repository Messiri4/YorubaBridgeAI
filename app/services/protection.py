"""Reversible numeric/identifier protection for NLLB translation (Sprint TI-1).

Mechanism
---------
BEFORE generation: numeric expressions, times, dates, URLs, emails, and
identifiers are replaced with opaque sentinel placeholders («0», «1», …)
that NLLB-200 copies through reliably (guillemets are rare in training
data and survive en→yor and yor→eng).

AFTER generation: sentinels are restored to the original expressions.
Every restoration is verified:

  - placeholder count matches exactly, and each placeholder index appears
    exactly once (no missing / duplicated / altered sentinels)
  - the verification failure path is a documented SAFE FALLBACK: the
    ORIGINAL model output is returned unchanged (no silent corruption, no
    partial restoration)

Configuration
-------------
Enabled per-request via `protect_numerics=True` on
`TranslationService.translate`, or globally via settings
`numeric_protection_enabled` (default: enabled). When disabled the
pipeline is a strict pass-through.

Yoruba numerals note: the protector intentionally passes numbers as-is
(digits are valid in modern Yoruba orthography and used in FLORES and
news registers); it does NOT convert digits to Yoruba number words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Sentinel format: @@<i>@@. Measured against the real model (Sprint TI-1
# diagnostics): guillemets «i» are normalized away by NLLB's SentencePiece
# (the model emitted bare digits), while @@i@@ survives generation verbatim
# in both directions. Never use characters the tokenizer merges.
SENTINEL_FMT = "@@{i}@@"
SENTINEL_RE = re.compile(r"@@\s*(\d+)@@")

# Patterns, highest priority first. Each captures complete expressions so
# surrounding punctuation/grammar is untouched.
_PATTERNS: list[re.Pattern[str]] = [
    # emails
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # URLs (http(s), www.)
    re.compile(r"\b(?:https?://|www\.)[^\s]+", re.IGNORECASE),
    # times with colon (24h or am/pm): 11:29, 07:05, 11:29 pm
    re.compile(r"\b\d{1,2}:\d{2}(?:\s?(?:am|pm|AM|PM))?"),
    # dates: ISO 2019-06-04 and dotted 04.06.2019
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}\.\d{2}\.\d{4}\b"),
    # numbers with thousand separators / decimals: 9,000 1,000.50 9.000,5
    re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d+\b"),
    # hyphenated scores/ranges 7–2, 7-2, 2:1 (word-bounded, not a time)
    re.compile(r"(?<![\w:])\b\d{1,4}\s?[–-]\s?\d{1,4}\b(?![\w:])"),
    # currency amounts: $1,000 €200 ₦5,000 £3.50
    re.compile(r"[$€£₦¥]\s?\d[\d.,]*|\b\d[\d.,]*\s?(?:dollars|euros|naira|pounds|yen)\b", re.IGNORECASE),
    # percentages: 50% / 50 %
    re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    # plain integers/decimals (catch-all, last)
    re.compile(r"\b\d+(?:[.,]\d+)?\b"),
]


@dataclass
class ProtectionResult:
    protected_text: str
    placeholders: dict[str, str] = field(default_factory=dict)  # sentinel -> original

    @property
    def n(self) -> int:
        return len(self.placeholders)


def protect(source: str) -> ProtectionResult:
    """Replace supported expressions with sentinels; record originals.

    Two passes:
      1. pattern pass (all non-catch-all patterns, priority order) replaces
         high-specificity expressions (emails, URLs, times, dates,
         separators, scores, currency, percentages).
      2. catch-all pass replaces remaining plain numbers, guarded against
         rewriting text INSIDE existing sentinels (Bug TI-1 fix: the
         catch-all matched digits inside «1», producing nested sentinels).

    Sentinel numbering follows TEXT POSITION (left to right), not pattern
    priority — round-trips read naturally and tests can reason about
    indexes positionally (Bug TI-2 fix).
    """
    placeholders: dict[str, str] = {}

    def _mark(text: str, pattern: re.Pattern[str], is_last_pass: bool) -> str:
        """One replacement pass over `text`; skips sentinel interiors."""
        out_parts: list[str] = []
        last = 0
        for m in pattern.finditer(text):
            if is_last_pass:
                before = text[:m.start()]
                if before.count("@@") % 2 == 1:
                    continue  # inside an unterminated sentinel
                if SENTINEL_RE.search(text[m.start():m.end()]):
                    continue
            out_parts.append(text[last:m.start()])
            # provisional token: \\x00<idx>\\x00 keeps the original ordering
            # info without emitting a final sentinel yet
            idx = provisional_counter[0]
            provisional_counter[0] += 1
            placeholders["@@" + str(idx) + "@@"] = m.group(0)
            out_parts.append("@@" + str(idx) + "@@")
            last = m.end()
        out_parts.append(text[last:])
        return "".join(out_parts)

    provisional_counter = [0]
    text = source
    for pat in _PATTERNS[:-1]:
        text = _mark(text, pat, False)
    text = _mark(text, _PATTERNS[-1], True)

    # Renumber by text position: find sentinel occurrences left-to-right.
    order = [m.group(0) for m in SENTINEL_RE.finditer(text)]
    remap = {old: new_idx for new_idx, old in enumerate(order)}
    final_placeholders = {SENTINEL_FMT.format(i=remap[k]): v
                          for k, v in placeholders.items()}

    def _renumber(m: re.Match[str]) -> str:
        return SENTINEL_FMT.format(i=remap[m.group(0)])

    text = SENTINEL_RE.sub(_renumber, text)
    return ProtectionResult(protected_text=text, placeholders=final_placeholders)


def verify_placeholder_structure(output: str, expected_n: int) -> tuple[bool, str]:
    """Structural verification of the model output's sentinel usage."""
    found = SENTINEL_RE.findall(output)
    if len(found) != expected_n:
        return False, f"expected {expected_n} sentinels, found {len(found)}"
    counts: dict[str, int] = {}
    for i in found:
        counts[i] = counts.get(i, 0) + 1
    dups = [k for k, v in counts.items() if v > 1]
    if dups:
        return False, f"duplicated sentinels: {dups}"
    missing = [str(i) for i in range(expected_n) if str(i) not in counts]
    if missing:
        return False, f"missing sentinels: {missing}"
    return True, "ok"


def restore(model_output: str, protection: ProtectionResult) -> tuple[str, bool, str]:
    """Restore sentinels to originals.

    Returns (text, ok, reason). On ANY structural failure the ORIGINAL model
    output is returned unchanged (safe fallback: never a partial or guessed
    restoration). """
    if protection.n == 0:
        return model_output, True, "nothing to restore"
    ok, reason = verify_placeholder_structure(model_output, protection.n)
    if not ok:
        return model_output, False, reason
    restored = SENTINEL_RE.sub(
        lambda m: protection.placeholders[m.group(0)], model_output)
    return restored, True, "ok"
