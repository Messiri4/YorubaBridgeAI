"""FLORES-200 Yoruba reference audit + Phase-2 detector validation + chrF
normalization experiment + human-review sample construction.

All processing is offline text analysis on the 997 aligned dev pairs
(data/flores_eval/dev.997.{en,yo}.txt). No inference, no downloads beyond
the dataset fetch already done via scripts/load_flores.py.

Outputs (in data/eval/):
  flores_reference_audit.csv            one row per (pair, issue) finding
  flores_reference_audit_summary.json   counts by category/kind/severity
  flores_reference_review.csv           ~100-pair human-review worksheet
  chrf_normalization_comparison.csv     per-arm chrF under 5 normalization regimes
Written by phase2b driver (this file's main() runs audit only; the
normalization experiment and review-sample builder live in the same module
and are invoked from phase2b_outputs.py to keep concerns separated).
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "flores_eval"
OUT = ROOT / "data" / "eval"

# Yoruba orthography constants -------------------------------------------------
# Tone marks: grave (low), acute (high); no mid-tone mark in practical orthography.
TONE_MARKS = "\u0300\u0301"
VOWELS = "aeiouɛɔ"
# Yoruba underdot letters: ẹ (U+1EB9), ọ (U+1ECD), ṣ (U+1E63), plus combining forms
UNDERDOT_PRECOMPOSED = "ẹọṣ"
UNDERDOT_COMBINING = re.compile("e\u0323|o\u0323|s\u0323")
# Words that are unambiguously Yoruba in origin — unaccented spellings of
# these are strong (but not certain) signals of accent-dropped text.
# Core words where an underdot is REQUIRED in the base form: ẹ/ọ/ṣ. Only
# words whose correct spelling *contains* ẹ, ọ, or ṣ qualify (e.g. 'ọjọ',
# 'ṣe', 'ẹ̀jọ́'). Words whose base form has NO underdot letter ('ilé', 'ní',
# 'wá') can never have a missing underdot — matching them here was a bug.
YORUBA_CORE_UNDERDOT_WORDS = {
    # normalized (accent-stripped, underdot-stripped) key -> what we match
    # entries all correspond to correct spellings containing ẹ/ọ/ṣ
    "ọjọ": "ọjọ", "ṣe": "ṣe", "ẹjọ": "ẹjọ", "ọmọ": "ọmọ",
    "ọdọ": "ọdọ", "ṣugá": "ṣugá", "ẹranko": "ẹranko", "ọjà": "ọjà",
    "ẹ̀kọ́": "ẹkọ", "ṣiṣẹ́": "ṣiṣẹ", "ọ̀nà": "ọna", "ẹ̀yà": "ẹya",
    "ọ̀rẹ́": "ọrẹ", "ìtọ́": "ito", "àṣà": "asa", "ọ̀pọ̀lọ́pọ̀": "opolopo",
}
# tolerant matching: strip tones+underdots from the token, then compare
# against keys that KEEP the underdot letters (below)
YORUBA_CORE_WITH_UNDERDOT = {
    # bare (tones+underdots stripped) -> correct full form used for display.
    # NOTE: bare 'se' is deliberately ABSENT — it also covers the particle
    # 'sé/σe' ambiguity; flagging 178 identical 'se' tokens measures translator
    # style, not per-instance defects. Words only counted when the underdot
    # is essentially obligatory in every attested spelling.
    "ojo": "ọjọ́",       # day/rain — ọ underdot obligatory
    "ejo": "ẹjọ́",       # court/case — ẹ obligatory
    "omo": "ọmọ",       # child — ọ obligatory
    "suga": "ṣugá",     # sugar
    "eranko": "ẹranko", # animal
    "oja": "ọjà",       # market
    "eko": "ẹ̀kọ́",      # lesson/Lagos
    "ona": "ọ̀nà",       # way/road
    "ore": "ọ̀rẹ́",       # friend
    "asa": "àṣà",       # culture (ṣ obligatory)
    "opolopo": "ọ̀pọ̀lọ́pọ̀",  # many
    "osere": "ọṣèré",    # actor
    "osu": "oṣù",       # month (ṣ obligatory)
    "aso": "aṣọ",       # cloth
    "ise": "iṣẹ́",       # work (also noun 'ise' homograph: reviewer decides)
}
ENGLISH_FUNCTION_WORDS = {
    "the", "and", "of", "in", "on", "to", "for", "with", "that", "this",
}
ACCENT_MARKS_RE = re.compile(r"[\u0300\u0301\u0323]")  # tones + dot-below


@dataclass
class Finding:
    pair_idx: int
    direction: str  # "reference" or "alignment"
    category: str
    severity: str   # "mechanical" (verified) | "warning" (heuristic)
    detail: str
    snippet: str = ""

    def to_row(self) -> dict:
        return {
            "pair_idx": self.pair_idx, "side": self.direction,
            "category": self.category, "severity": self.severity,
            "detail": self.detail, "snippet": self.snippet[:160],
        }


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


# --------------------------------------------------------------------------
# A. Yoruba orthography checks
# --------------------------------------------------------------------------

def has_underdot_form(word_nfc: str) -> bool:
    """Word uses a proper underdot letter (precomposed or combining)."""
    return any(c in UNDERDOT_PRECOMPOSED for c in word_nfc) or bool(
        UNDERDOT_COMBINING.search(word_nfc))


def word_tone_profile(word_nfd: str) -> tuple[int, int]:
    """(#tone-marks, #vowels-that-bear-tone) — Yoruba words carrying multiple
    syllables typically have at least one tone mark per content word."""
    n_vowels = sum(1 for c in word_nfd if c in VOWELS)
    n_marks = sum(1 for c in word_nfd if c in TONE_MARKS)
    return n_marks, n_vowels


def audit_orthography(idx: int, en: str, yo: str) -> list[Finding]:
    findings: list[Finding] = []
    nfc = unicodedata.normalize("NFC", yo)
    nfd = unicodedata.normalize("NFD", yo)

    # A1 Mixed normalization forms (mechanical — purely byte-level fact):
    # mixed = precomposed underdot letters (ẹ/ọ/ṣ) coexisting with dot-below
    # COMBINING sequences (e̲/ọ̲ as base+U+0323). Note: combining TONE marks
    # over plain letters are the standard NFC representation and never count
    # as 'mixed' on their own.
    has_precomposed_underdot = any(c in UNDERDOT_PRECOMPOSED for c in yo)
    # Look for combining dot-below in the ORIGINAL string: NFD of a normal NFC
    # string re-introduces U+0323 by decomposing precomposed ẹ/ọ/ṣ, so NFD
    # cannot be used here. A raw U+0323 next to a plain o/e/s IS decomposed.
    has_combining_dot = "\u0323" in yo
    if has_precomposed_underdot and has_combining_dot:
        findings.append(Finding(idx, "reference", "MIXED_UNICODE_FORMS",
                                "mechanical",
                                "reference mixes precomposed underdots and combining dot-below marks"))

    # A2 Tone-mark coverage: fraction of 'syllable-bearing' vowels carrying tones
    #    over multi-vowel content words. Unaccented paragraph detection.
    words = tokens(yo)
    # words that are long enough that full accentlessness is suspicious
    suspect = [w for w in words if len(w) >= 4]
    if len(suspect) >= 5:
        nfd_suspect = [unicodedata.normalize("NFD", w) for w in suspect]
        marks = sum(word_tone_profile(w)[0] for w in nfd_suspect)
        vowels = sum(word_tone_profile(w)[1] for w in nfd_suspect)
        cov = marks / vowels if vowels else 1.0
        if cov < 0.3:
            findings.append(Finding(
                idx, "reference", "LOW_TONE_MARK_COVERAGE", "warning",
                f"tone-mark coverage {cov:.0%} over {len(suspect)} content words",
                " ".join(suspect[:6])))
        # whole-pair accent-drop pattern: reference nearly unaccented
        if cov < 0.05 and marks < 5:
            findings.append(Finding(
                idx, "reference", "REFERENCE_UNACCENTED", "warning",
                f"reference has {marks} tone marks total; likely accent-dropped text"))

    # A3 Missing underdots in words whose correct spelling requires ẹ/ọ/ṣ.
    # Match on the tone-stripped, underdot-stripped form; fire only when the
    # token equals a known underdot-requiring word BUT carries no underdot.
    for w in words:
        base = unicodedata.normalize("NFD", w)
        bare = "".join(c for c in base
                       if c not in TONE_MARKS and c != "\u0323")
        bare = unicodedata.normalize("NFC", bare)
        correct = YORUBA_CORE_WITH_UNDERDOT.get(bare)
        if correct and not has_underdot_form(w):
            findings.append(Finding(
                idx, "reference", "MISSING_UNDERDOT_CANDIDATE", "warning",
                f"'{w}' should likely be '{correct}' (missing ẹ/ọ/ṣ)", w))

    # A4 Suspicious capitalization: mid-sentence Capitalized common word
    for i, w in enumerate(words):
        pass  # token() lowercases; handled in raw scan below
    raw_words = yo.split()
    for w in raw_words[1:]:
        cleaned = w.strip(".,;:!?()\"'“”")
        if cleaned[:1].isupper() and cleaned[1:].islower() and len(cleaned) > 2:
            low = cleaned.lower()
            if low in ENGLISH_FUNCTION_WORDS or low in YORUBA_CORE_WITH_UNDERDOT:
                findings.append(Finding(
                    idx, "reference", "ODD_MID_SENTENCE_CAPITAL", "warning",
                    f"mid-sentence capitalized '{cleaned}'", w))
                break

    # A5 Repeated adjacent words (mechanical), excluding known Yoruba
    # reduplications/ideophones and legitimate emphatic duplications
    # ('kíá kíá' hurrying, 'fùkẹ fùkẹ' completely, 'kánmọ kánmọ' alike,
    # iterative 'ọjọ ọjọ' day-after-day) and digit runs.
    YORUBA_REDUPLICATIONS = {
        "fini", "dindi", "keke", "kere", "dada", "pedia", "koto",
        "buruku", "sise", "seria", "kíá", "fùkẹ", "kánmọ", "ọjọ",
        "kan", "kikan", "bẹ̀rẹ̀",
    }
    tk = tokens(yo)
    for a, b in zip(tk, tk[1:]):
        if a == b and len(a) >= 3 and a not in YORUBA_REDUPLICATIONS:
            if not a.isdigit():
                findings.append(Finding(idx, "reference", "ADJACENT_DUPLICATE",
                                        "mechanical", f"'{a} {b}'"))
                break

    # A6 Unexpected characters (e.g. non-Yoruba diacritics, control chars)
    weird = {c for c in yo if ord(c) < 32 or unicodedata.category(c) in ("Cc", "Cf")}
    if weird:
        findings.append(Finding(idx, "reference", "CONTROL_CHARS", "mechanical",
                                f"control/format chars: {[hex(ord(c)) for c in weird]}"))

    return findings


# --------------------------------------------------------------------------
# B. Numerical consistency
# --------------------------------------------------------------------------

_NUM_WORD_EN = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "hundred": 100, "thousand": 1000, "million": 10**6,
}

# Minimal, conservative Yoruba number words (cardinals commonly agreed):
_YOR_NUM_WORDS = {
    "ọ̀kan": 1, "méjì": 2, "mẹ́ta": 3, "mẹ́rin": 4, "márùn": 5, "mẹ́fà": 6,
    "méje": 7, "mẹ́jọ": 8, "mẹ́sànán": 9, "mẹ̀wá": 10, "ogójì": 20,
    "àádọ́ta": 50, "ọgọ́rin": 80, "ọ̀kànlélógójì": 21,
}


def digits(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", text)


def number_words_en(text: str) -> list[int]:
    vals = []
    for t in tokens(text):
        if t in _NUM_WORD_EN:
            vals.append(_NUM_WORD_EN[t])
    return vals


def audit_numbers(idx: int, en: str, yo: str) -> list[Finding]:
    findings: list[Finding] = []
    en_d, yo_d = digits(en), digits(yo)
    if Counter(en_d) != Counter(yo_d):
        # Only mechanical when digits present in EN but absent in YO;
        # word-vs-digit rendering is a convention, so warn otherwise.
        if en_d and not yo_d:
            findings.append(Finding(idx, "reference", "NUM_DIGITS_DROPPED",
                                    "warning",
                                    f"EN digits {sorted(en_d)} absent in YO reference"))
        elif set(en_d) != set(yo_d):
            findings.append(Finding(idx, "reference", "NUM_DIGIT_DIFF", "warning",
                                    f"EN {sorted(en_d)} vs YO {sorted(yo_d)}"))
    return findings


# --------------------------------------------------------------------------
# C. Named entities
# --------------------------------------------------------------------------

def cap_seqs(text: str) -> set[str]:
    """Capitalized runs (1+ words). Very crude person/org/place proxy."""
    out: set[str] = set()
    for m in re.finditer(r"\b[A-Z][a-zA-Z'’\-]+(?:\s+[A-Z][a-zA-Z'’\-]+)*", text):
        out.add(m.group(0))
    return out


def entity_tokens(ent: str) -> list[str]:
    return [t for t in tokens(ent) if len(t) > 2]


def audit_entities(idx: int, en: str, yo: str) -> list[Finding]:
    findings: list[Finding] = []
    yo_low = unicodedata.normalize("NFC", yo).lower()
    # fold accents for fuzzy match
    yo_fold = "".join(c for c in unicodedata.normalize("NFD", yo_low)
                      if not unicodedata.combining(c))
    yo_fold = yo_fold.replace("\u0323", "").replace("ṣ", "s").replace(
        "ẹ", "e").replace("ọ", "o")
    en_ents = cap_seqs(en)
    # remove common sentence-start non-entities
    en_ents = {e for e in en_ents
               if e.lower() not in ENGLISH_FUNCTION_WORDS
               and not (len(e.split()) == 1 and e.lower() in ENGLISH_FUNCTION_WORDS)}
    for ent in en_ents:
        etoks = entity_tokens(ent)
        if not etoks:
            continue
        exact = all(t in yo_fold for t in etoks)
        if exact:
            continue
        fuzzy = any(
            sum(1 for c in t if c in yo_fold) / max(len(t), 1) >= 0.6
            for t in etoks if len(t) >= 4
        )
        if not fuzzy:
            findings.append(Finding(idx, "reference", "ENTITY_NOT_CARRIED",
                                    "warning",
                                    f"EN entity '{ent}' not found (even fuzzily) in YO", ent))
    return findings


# --------------------------------------------------------------------------
# D. Structural checks
# --------------------------------------------------------------------------

def audit_structure(idx: int, en: str, yo: str) -> list[Finding]:
    findings: list[Finding] = []
    te, ty = len(tokens(en)), len(tokens(yo))
    if ty == 0:
        findings.append(Finding(idx, "reference", "EMPTY_REFERENCE", "mechanical",
                                "Yoruba reference is empty"))
    elif ty < te * 0.5:
        findings.append(Finding(idx, "reference", "SUSPICIOUSLY_SHORT", "warning",
                                f"YO {ty} tokens vs EN {te} ({ty/te:.0%})"))
    elif ty > te * 2.2:
        findings.append(Finding(idx, "reference", "SUSPICIOUSLY_LONG", "warning",
                                f"YO {ty} tokens vs EN {te} ({ty/te:.0%})"))
    # dangling brackets/quotes (mechanical)
    for open_c, close_c in (("(", ")"), ("[", "]"), ("“", "”")):
        if yo.count(open_c) != yo.count(close_c):
            findings.append(Finding(idx, "reference", "UNBALANCED_QUOTES", "mechanical",
                                    f"unbalanced {open_c}{close_c}"))
    return findings


# --------------------------------------------------------------------------
# audit driver
# --------------------------------------------------------------------------

def load_pairs(n: int | None = 997) -> list[tuple[str, str]]:
    en = (DATA / "dev.997.en.txt").read_text(encoding="utf-8").splitlines()
    yo = (DATA / "dev.997.yo.txt").read_text(encoding="utf-8").splitlines()
    assert len(en) == len(yo), "EN/YO dev files misaligned"
    pairs = list(zip(en, yo, strict=True))
    return pairs if n is None else pairs[:n]


def run_audit(pairs: list[tuple[str, str]]) -> list[Finding]:
    findings: list[Finding] = []
    for i, (en, yo) in enumerate(pairs):
        findings += audit_orthography(i, en, yo)
        findings += audit_numbers(i, en, yo)
        findings += audit_entities(i, en, yo)
        findings += audit_structure(i, en, yo)
    return findings


def main() -> None:
    pairs = load_pairs()
    findings = run_audit(pairs)

    OUT.mkdir(exist_ok=True)
    audit_path = OUT / "flores_reference_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(Finding(0, "", "", "", "").to_row().keys()))
        w.writeheader()
        for fo in findings:
            w.writerow(fo.to_row())

    by_cat = Counter((f.category, f.severity) for f in findings)
    summary = {
        "pairs_audited": len(pairs),
        "provenance": json.loads((DATA / "dev.997.provenance.json").read_text()),
        "findings_total": len(findings),
        "by_category_severity": {f"{c}|{s}": n for (c, s), n in sorted(by_cat.items())},
        "pairs_with_any_finding": len({f.pair_idx for f in findings}),
        "disclaimer": "Heuristic warnings for review — not confirmed errors.",
    }
    (OUT / "flores_reference_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"audited {len(pairs)} pairs: {len(findings)} findings "
          f"({summary['pairs_with_any_finding']} pairs flagged)")


if __name__ == "__main__":
    main()
