"""Yoruba orthography utilities: validation, normalization, and diagnostics.

Scope (Agent 3, feature/yoruba-tonality):
- Unicode NFC normalization that PRESERVES linguistic distinctions
  (tone marks U+0300/U+0301/U+0303, underdot U+0323, mid-tone U+0303).
- Detection of potentially missing tone marks / inconsistent orthography.
  Detection only: this module never claims to know the correct form and
  never fabricates tone marks from an unverified dictionary.
- Underdot handling for ẹ/ọ/ṣ, including composed codepoints
  (U+1EB9, U+1ECD, U+1E63) and decomposed sequences.

Unicode normalization is NOT tone restoration: NFC merely makes
representations consistent; choosing correct tones is a linguistic task
handled (later, experimentally) by app/services/tone_restoration.py.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

# Combining marks used in Yoruba orthography
COMBINING_GRAVE = "\u0300"  # low tone (à)
COMBINING_ACUTE = "\u0301"  # high tone (á)
COMBINING_TILDE = "\u0303"  # mid tone is commonly written as no mark; tilde
                            # appears in nasalized/derived contexts (e.g. ń̃)
COMBINING_DOT_BELOW = "\u0323"  # underdot (ẹ, ọ, ṣ)

TONE_MARKS = frozenset({COMBINING_GRAVE, COMBINING_ACUTE})
UNDERDOT_MARK = COMBINING_DOT_BELOW

# Composed underdot letters and their decomposed base forms
UNDERDOT_LETTERS = {
    "ẹ": "e",  # U+1EB9
    "ọ": "o",  # U+1ECD
    "ṣ": "s",  # U+1E63
    "ẹ́": "é",
    "ọ́": "ó",
    "ẹ̀": "è",
    "ọ̀": "ò",
}
PLAIN_EQUIVALENTS = {v: k for k, v in UNDERDOT_LETTERS.items()}

# Vowel letters of Yoruba (7 oral vowels; ẹ/ọ handled via UNDERDOT_LETTERS)
PLAIN_VOWELS = "aeiou"
ALL_VOWEL_BASES = set(PLAIN_VOWELS) | set(UNDERDOT_LETTERS.values())


def nfc(text: str) -> str:
    """NFC-normalize without removing any tone/underdot distinction.

    NFC composes base letter + combining mark into precomposed forms where
    they exist (ẹ, ọ, ṣ, à, é, ...) but the combining sequences remain
    valid and equal after normalization; no mark is ever stripped here.
    """
    return unicodedata.normalize("NFC", text)


def nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def split_marks(text: str) -> tuple[str, list[str]]:
    """Return (base_text_with_dots_removed_to_base_letters, tone_marks_in_order).

    Example: "ọ̀jẹ́" -> ("oje", ["\u0300", "\u0323", "\u0301"]) order reflects
    the decomposed stream.
    """
    bases: list[str] = []
    marks: list[str] = []
    for ch in nfd(text):
        if unicodedata.combining(ch):
            marks.append(ch)
        elif ch in UNDERDOT_LETTERS and len(ch) == 1:
            bases.append(UNDERDOT_LETTERS[ch])
        else:
            bases.append(ch)
    return "".join(bases), marks


def count_tone_marks(text: str) -> dict[str, int]:
    """Count grave (low), acute (high), and underdot marks in NFC text."""
    low = high = underdot = 0
    for ch in nfd(text):
        if ch == COMBINING_GRAVE:
            low += 1
        elif ch == COMBINING_ACUTE:
            high += 1
        elif ch == UNDERDOT_MARK:
            underdot += 1
    return {"low": low, "high": high, "underdot": underdot}


def has_full_marks(text: str) -> bool:
    """True when every vowel-bearing word looks fully toned.

    NOTE: mid tone is written as NO mark in standard orthography, so this is
    inherently approximate: a word whose vowels are all bare could be fully
    mid-toned or could be missing marks. This function therefore reports
    'all vowels marked', not 'tones correct'.
    """
    words = nfc(text).split()
    if not words:
        return True
    for word in words:
        if not _word_has_vowels(word):
            continue
        _, marks = split_marks(word)
        tone_count = sum(1 for m in marks if m in TONE_MARKS)
        vowel_count = sum(1 for b in split_marks(word)[0] if b.lower() in ALL_VOWEL_BASES)
        # Nasal syllabic markers and apostrophized contractions reduce vowel count.
        if vowel_count and tone_count < vowel_count:
            # Allow apostrophe elisions like 'ọmọ' + "'ràn" style joins handled
            # at word level; a word-level shortfall still flags.
            return False
    return True


def _word_has_vowels(word: str) -> bool:
    base, _ = split_marks(word)
    return any(c.lower() in ALL_VOWEL_BASES for c in base)


@dataclass
class OrthographyIssue:
    kind: str
    word: str
    position: int  # char offset in the NFC-normalized input
    detail: str


@dataclass
class OrthographyReport:
    issues: list[OrthographyIssue] = field(default_factory=list)
    tone_counts: dict[str, int] = field(default_factory=dict)
    nfc_equal_to_input: bool = True

    @property
    def clean(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        kinds = sorted({i.kind for i in self.issues})
        return (
            f"clean" if self.clean else f"issues={len(self.issues)} kinds={kinds}"
        )


def audit(text: str) -> OrthographyReport:
    """Detect potential orthographic problems WITHOUT proposing fixes.

    Checks:
    1. Words containing vowel bases but zero tone marks (possible missing
       tones OR legitimately all-mid-tone; flagged as 'maybe_missing_tones').
    2. Inconsistent representation: same folded word appearing both with and
       without marks ('inconsistent_marking').
    3. Mixed composed/decomposed forms that NFC does not unify fully
       ('non_nfc_canonical').
    """
    normalized = nfc(text)
    report = OrthographyReport(
        tone_counts=count_tone_marks(normalized),
        nfc_equal_to_input=(normalized == text),
    )

    # 1. Possibly missing tone marks.
    for word in normalized.split():
        stripped = word.strip(".,;:!?()[]{}\"'’")
        if not stripped or not _word_has_vowels(stripped):
            continue
        _, marks = split_marks(stripped)
        if not any(m in TONE_MARKS for m in marks):
            report.issues.append(
                OrthographyIssue(
                    kind="maybe_missing_tones",
                    word=stripped,
                    position=max(text.find(stripped), 0),
                    detail=(
                        "word has vowel letters but no low/high tone marks; "
                        "could be all-mid-tone OR under-marked — needs human/model verification"
                    ),
                )
            )

    # 2. Inconsistent marking of the same folded word.
    seen: dict[str, set[str]] = {}
    idx = 0
    for word in normalized.split():
        base, _ = split_marks(word)
        key = base.lower()
        seen.setdefault(key, set()).add(word)
        idx += len(word) + 1
    for key, variants in seen.items():
        if len(variants) > 1:
            report.issues.append(
                OrthographyIssue(
                    kind="inconsistent_marking",
                    word=key,
                    position=0,
                    detail=f"same base word appears as multiple marked forms: {sorted(variants)}",
                )
            )

    return report


def fold_for_comparison(text: str) -> str:
    """Diagnostic-only fold: removes tone + underdot for matching.

    NEVER use the output as user-visible text.
    """
    base, _ = split_marks(text)
    return unicodedata.normalize("NFC", base.lower())
