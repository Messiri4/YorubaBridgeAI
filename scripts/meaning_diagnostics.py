"""Sprint 3 PHASE 3: meaning-error detection heuristics.

Deterministic, documented checks over (source, prediction[, reference]):

  detect_negation_flip   polarity reversal between source and prediction,
                         both directions (yor↔eng), via negation-marker
                         multisets.
  detect_omissions       numbers, capitalized entities, and parenthesized
                         content present in the source but absent from the
                         prediction (reference-verified when available).
  detect_degeneration    repeated-token output loops (decoder degeneration).
  classify_findings      separates heuristic warnings from reference-
                         verifiable content checks; nothing is auto-rewritten
                         and nothing is called "confirmed" without a human.

All outputs are WARNINGS for review. The module never modifies a translation.
"""
from __future__ import annotations

import re
import unicodedata

# Negation markers ----------------------------------------------------------------
def _neg_fold(m: str) -> str:
    nfd = unicodedata.normalize("NFD", m.lower())
    return "".join(c for c in nfd if not unicodedata.combining(c))


_YOR_NEG_RAW = ("kò", "kìí", "má", "máà", "ìkọ́", "kíì", "kì", "èṣe")
# folded at import: matching happens in folded space so tone variants unify
_YOR_NEG = {_neg_fold(m) for m in _YOR_NEG_RAW}
_ENG_NEG = {"not", "no", "never", "n't", "neither", "nor", "without"}

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    # fold tones/underdots for matching only (identity comparison)
    folded = _neg_fold(text)
    return _TOKEN.findall(folded)


def _negation_count(text: str, is_yoruba: bool) -> int:
    toks = _tokens(text)
    markers = _YOR_NEG if is_yoruba else _ENG_NEG
    return sum(1 for t in toks if t in markers)


def detect_negation_flip(source: str, prediction: str) -> str:
    """'negation_flip_suspected' when the negation-marker count differs
    between source and prediction (either direction); 'ok' otherwise.

    Deliberately blunt: double negation, rhetorical negation and marker
    mistranslations can fool it — which is why the output is a warning, and
    why 'ok' does not certify polarity correctness."""
    src_yoruba = any(ord(c) > 127 for c in source)
    tgt_yoruba = not src_yoruba
    ns = _negation_count(source, src_yoruba)
    np = _negation_count(prediction, tgt_yoruba)
    if ns != np:
        return "negation_flip_suspected"
    return "ok"


# Omissions ------------------------------------------------------------------------
_NUM = re.compile(r"\d+(?:[,\.\-]\d+)*")
_CAP = re.compile(r"\b[A-Z][a-zA-Z'’-]{2,}\b")
_PAREN = re.compile(r"\([^()]{3,}\)")


def detect_omissions(source: str, prediction: str,
                     reference: str | None = None) -> list[str]:
    """Content present in the source but absent from the prediction.

    Numbers: compared by value — a prediction may legitimately spell numbers
    out as words (4 → mẹ́rin), so a number is 'omitted' only when neither the
    digits nor a known number-word rendering appears. Entities are compared
    accent-folded (transliteration tolerant via prefix match). Parenthetical
    content is checked as a whole unit.
    """
    findings: list[str] = []
    pred_fold = _fold_text(prediction)

    for num in _NUM.findall(source):
        if num in prediction:
            continue
        if _rendered_as_words(num, pred_fold):
            continue
        findings.append(f"omitted_number:{num}")

    for ent in _CAP.findall(source):
        if ent.lower() in ("the", "a", "an"):
            continue
        fold_ent = _fold_text(ent)
        if fold_ent in pred_fold:
            continue
        # transliteration tolerance: shared 4-char prefix
        if any(tok.startswith(fold_ent[:4]) for tok in pred_fold.split()
               if len(tok) >= 4):
            continue
        findings.append(f"omitted_entity:{ent}")

    for paren in _PAREN.findall(source):
        core = paren.strip("()")
        # digits-only parens like (31 miles) are covered by number/entity checks
        if any(ch.isdigit() for ch in core) or _fold_text(core) in pred_fold:
            continue
        findings.append(f"omitted_parenthetical:{paren}")

    # Reference-verified omissions: if a reference DOES contain the item, the
    # omission is objectively present in a human-vetted rendering (upgrades
    # confidence but is still not a full semantic verdict).
    if reference:
        for f in list(findings):
            item = f.split(":", 1)[1]
            if item in reference or _fold_text(item) in _fold_text(reference):
                findings[f.index(f)] = f + "|reference_confirms_content"
    return findings


def _fold_text(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s.lower())
    return unicodedata.normalize("NFC", "".join(
        c for c in nfd if not unicodedata.combining(c)))


# reuse the vetted whole-token lexicon from the sprint-2 analysis
def _rendered_as_words(num: str, pred_fold: str) -> bool:
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from scripts.sprint2_failure_analysis import _is_rendered
    return _is_rendered(num, pred_fold)


# Degeneration ----------------------------------------------------------------------

def detect_degeneration(text: str, min_repeats: int = 6) -> str:
    toks = text.split()
    run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        if run >= min_repeats:
            return "repetition_loop"
    return "ok"


# Classification ---------------------------------------------------------------------

def classify_findings(findings: list[dict]) -> list[dict]:
    """Attach a status to each finding:
      heuristic_warning        detector-only suspicion (negation flips)
      requires_human_review    content check, needs a fluent speaker
      human_confirmed_error    ONLY when a human label is supplied
    Heuristic counts are never reported as verified error rates."""
    out = []
    for f in findings:
        g = dict(f)
        if g.get("human_label") == "confirmed_error":
            g["status"] = "human_confirmed_error"
        elif g["kind"] in ("negation_flip", "repetition_loop"):
            g["status"] = "heuristic_warning"
        else:
            g["status"] = "requires_human_review"
        out.append(g)
    return out
