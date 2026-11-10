"""Unified translation-quality scorecard (Phase 3 Sprint 1).

One schema for all evaluation runs. Metrics are grouped by evidence tier:

  - measured:   computed from the given predictions + references (BLEU,
                chrF++ variants, latency stats) — always present when the
                inputs exist.
  - heuristic:  warning COUNTS from deterministic detectors (numeric,
                entity, repetition/omission). These are *counts of
                suspicions*, never error rates.
  - human:      verified labels from the review worksheet. Blank/unset
                fields are reported as None ("unavailable"), never 0.
"""
from __future__ import annotations

import statistics
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import sacrebleu

SACREBLEU_VERSION = sacrebleu.__version__
CHRF_KWARGS = dict(char_order=6, word_order=2, beta=2.0)  # chrF++ canonical


# --------------------------------------------------------------------------
# normalization regimes (shared with phase2b_outputs)
# --------------------------------------------------------------------------

def fold_accents(text: str) -> str:
    """Diagnostic lossy fold: strip tones + dot-below, map precomposed
    underdot letters to plain latin. NOT a production metric."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", text)


def fold_underdots_only(text: str) -> str:
    out = []
    for c in unicodedata.normalize("NFD", text):
        if c == "\u0323":
            continue
        out.append({"ẹ": "e", "ọ": "o", "ṣ": "s"}.get(c, c))
    return unicodedata.normalize("NFC", "".join(out))


REGIMES: dict[str, Any] = {
    "standard": lambda t: t,
    "nfc": lambda t: unicodedata.normalize("NFC", t),
    "accent_folded_diagnostic": fold_accents,
}


# --------------------------------------------------------------------------
# heuristic detectors (thin adapters over the phase-2 logic; deterministic)
# --------------------------------------------------------------------------

_DIGITS = __import__("re").compile(r"\d+(?:[.,]\d+)?")

# Number words with RELIABLY determined values. Keys are tone- and underdot-
# FOLDED (see _fold_tones); the table is looked up via the same folding, so
# Yoruba tone/underdot variants resolve to a single key without altering the
# original text anywhere. Entries are restricted to forms whose value is
# unambiguous: units 1-10, teens 11-12 (the only ones attested consistently in
# FLORES text), exact tens, 100 and 1000. COMPOUND Yoruba numerals (e.g.
# 'ogójì-ọ̀kànlá' 41, 'aádọ́rùn-ún-dín-ọ̀kànlá' 69) are deliberately EXCLUDED —
# the vigesimal system produces long analytic compounds we cannot verify
# without risking wrong equivalences; those become review_required.
#
# Sprint 2C-Q (duplicate/contradiction cleanup): the Sprint-2A dict contained
# conflicting duplicate keys (Python silently kept the last: 'aadota' mapped
# to both 30 and 50; 'àádọ́ta' likewise) and several tone-folded variants that
# collided (e.g. 'aadota' can fold from both 30 and 50 spellings). Folded keys
# are now UNIQUE per value; contradictions removed.
_NUMBER_WORD_VALUES = {
    # Yoruba units (folded keys)
    "okan": 1, "meji": 2, "meta": 3, "merin": 4, "marun": 5,
    "mefa": 6, "meje": 7, "mejo": 8, "mesan": 9, "mewa": 10,
    "mokanla": 11, "mejila": 12, "okanla": 11, "ejila": 12,
    # exact tens (folded): ogójì(40)/àádọ́ta(50)/ọgọ́rin(80) are distinct keys
    "ogun": 20, "ogoji": 40, "aadota": 50, "aadorin": 70, "ogorin": 80,
    "aadorun": 90, "ogorun": 100,
    # English
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}


def _strip_combining(text: str) -> str:
    """Remove combining marks (tones + dot-below) WITHOUT altering any base
    character. Needed because Python's \w treats a base letter followed by a
    combining tone mark as a word boundary (e.g. 'àádọ́ta' tokenizes as
    'àádọ'+'ta'), which silently splits Yoruba words. Unicode normalization
    of the ORIGINAL text is never changed — this only affects tokenization
    inside detectors."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _pred_number_word_values(pred: str) -> set[int]:
    """Set of numeric values the prediction's number words denote EXACTLY.

    Looks up each folded token in _NUMBER_WORD_VALUES; multi-token exact
    matches like 'fifty percent' are handled by the caller for unit checks.
    Only unambiguous single-word values are returned; compound or partially
    matching expressions are NOT included (they become review_required)."""
    import re
    words = re.findall(r"[^\W_]+", _strip_combining(pred.lower()), re.UNICODE)
    values: set[int] = set()
    for w in words:
        v = _NUMBER_WORD_VALUES.get(_fold_tones(w))
        if v is not None:
            values.add(v)
    return values


def _classify_numeric(s_digits: list[str], p_digits: list[str],
                      pred: str, src: str) -> str:
    """Sprint 2C-Q classification for one sentence with source digits.

    Returns one of:
      exact_digits_preserved        digit multiset identical
      verified_number_word_equivalent  no digits in pred, but every source
                                    numeral group has a word counterpart of
                                    EQUAL VALUE (single-token trusted forms)
      numeric_mismatch_warning      digits differ (incl. value-changing errors
                                    like 7–2→72 or 9,000→900)
      numeric_review_required       no digits in pred and equivalence can't be
                                    verified (compound Yoruba numerals, partial
                                    matches like 11:29→'eleven', times/ranges)
    """
    if sorted(s_digits) == sorted(p_digits):
        return "exact_digits_preserved"
    if not p_digits:
        # Split source numeral groups into comparable value tokens.
        # Source values from digits: e.g. ['11', '29'] for 11:29;
        # ['9,000'] -> 9000; ['7','2'] -> 7,2.
        src_vals: list[int] = []
        ambiguous = False
        for g in s_digits:
            clean = g.replace(",", "").replace(".", "")
            if ":" in src[max(0, src.find(g) - 1): src.find(g) + len(g) + 1]:
                ambiguous = True  # time-like context (e.g. 11:29)
            try:
                src_vals.append(int(clean))
            except ValueError:
                ambiguous = True
        pred_vals = _pred_number_word_values(pred)
        if not pred_vals:
            return "numeric_review_required"  # no verifiable word either
        if ambiguous:
            return "numeric_review_required"  # times/ranges need human review
        # Compound/unsupported source values (e.g. 41, 977 — anything outside
        # the trusted single-word table) can't be word-verified: a Yoruba
        # compound rendering would produce multiple partial tokens we cannot
        # responsibly equate. Review rather than verify.
        if any(v not in set(_NUMBER_WORD_VALUES.values()) for v in src_vals):
            return "numeric_review_required"
        if all(v in pred_vals for v in src_vals):
            return "verified_number_word_equivalent"
        # A supported word with the WRONG value proves the numeral changed
        # (source 11 -> 'twelve'): verified mismatch, not mere suspicion.
        # Only when pred values cover ALL source values would it be equivalent;
        # wrong-value coverage of every group is a definite error.
        if len(src_vals) == len(pred_vals) and len(src_vals) > 0:
            return "numeric_mismatch_warning"
        # Partial coverage (e.g. source 11 & 29, pred only 'eleven') -> review
        return "numeric_review_required"
    return "numeric_mismatch_warning"


def numeric_consistency(preds: list[str], srcs: list[str]) -> dict:
    """Per-sentence numeric classification (Sprint 2C-Q).

    Classifications per sentence with source digits:
      exact_digits_preserved / verified_number_word_equivalent /
      numeric_mismatch_warning / numeric_review_required

    Schema change vs Sprint 2A (documented, intentional):
      - 'digits_spelled_out' renamed conceptually to
        verified_number_word_equivalent + numeric_review_required split:
        legacy keys 'digits_preserved'/'digits_missing'/'digits_changed' are
        kept for backward compatibility, and legacy 'digits_spelled_out' now
        counts ONLY verified_number_word_equivalent (no longer includes
        unverifiable cases). New key 'review_required' added.
      - 'consistency_rate' remains exact-digit preservation rate (unchanged
        semantics); a new 'verified_total_rate' adds word-verified cases.
    """
    counts = {
        "exact_digits_preserved": 0,
        "verified_number_word_equivalent": 0,
        "numeric_mismatch_warning": 0,
        "numeric_review_required": 0,
    }
    checked = 0
    for s, p in zip(srcs, preds):
        s_d, p_d = _DIGITS.findall(s), _DIGITS.findall(p)
        if not s_d:
            continue
        checked += 1
        cls = _classify_numeric(s_d, p_d, p, s)
        counts[cls] += 1
    return {
        "sentences_with_source_digits": checked,
        # explicit classification counts (Sprint 2C-Q)
        **counts,
        # backward-compatible aliases (documented)
        "digits_preserved": counts["exact_digits_preserved"],
        "digits_missing": counts["numeric_review_required"],
        "digits_changed": counts["numeric_mismatch_warning"],
        "digits_spelled_out": counts["verified_number_word_equivalent"],
        "consistency_rate": (round(counts["exact_digits_preserved"] / checked, 4)
                             if checked else None),
        "verified_total_rate": (round(
            (counts["exact_digits_preserved"]
             + counts["verified_number_word_equivalent"]) / checked, 4)
            if checked else None),
    }


def named_entity_consistency(preds: list[str], srcs: list[str]) -> dict:
    """Capitalized-run carryover (transliteration-tolerant, deterministic).
    Heuristic only: the count is a *warning*, not an error rate."""
    sys_path_helper = __import__("pathlib").Path(__file__).resolve().parents[1]
    import sys
    if str(sys_path_helper) not in sys.path:
        sys.path.insert(0, str(sys_path_helper))
    from scripts.audit_flores_references import audit_entities  # reuse logic

    carried = not_carried = total = 0
    for s, p in zip(srcs, preds):
        # audit_entities is written for EN->YO but is symmetric enough for a
        # carryover count: swap roles when the prediction is the English side.
        if _looks_like_english(p) and not _looks_like_english(s):
            findings = audit_entities(0, p, s)
        else:
            findings = audit_entities(0, s, p)
        for f in findings:
            total += 1
            if f.category == "ENTITY_NOT_CARRIED":
                not_carried += 1
    carried = total - not_carried
    return {
        "entity_checks": total,
        "entity_warning_hits": not_carried,
        "carryover_rate": round(carried / total, 4) if total else None,
    }


def _looks_like_english(text: str) -> bool:
    """Crude script check: mostly basic-latin letters and common English words."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_share = sum(1 for c in letters if ord(c) < 128) / len(letters)
    return ascii_share > 0.9


# Legitimate Yoruba reduplications / ideophones (emphasis or iteration), and
# common emphatic English doubling — repeated occurrences of these are normal,
# not defects. Tone-stripped keys so accented variants match.
REDUPLICATION_WHITELIST = {
    "fini", "dindi", "keke", "kere", "dada", "pedia", "koto", "buruku",
    "sise", "seria", "kia", "fuke", "kanmo", "ojo", "kan", "kikan",
    "bere", "najia", "yoyo", "soso", "ga", "kekerere",
}


def _fold_tones(tok: str) -> str:
    """Fold tones AND underdots to bare latin — matches REDUPLICATION_WHITELIST
    keys, which are stored tone- AND underdot-stripped."""
    text = "".join(c for c in unicodedata.normalize("NFD", tok)
                   if c not in "\u0300\u0301" and c != "\u0323")
    text = text.replace("ẹ", "e").replace("ọ", "o").replace("ṣ", "s")
    return text


def repetition_omission_warnings(preds: list[str], refs: list[str]) -> dict:
    """Repetition warnings (token or token-pair repeated adjacently — token
    pairs catch underdot-split words like 'ìtòlé ṣẹẹsẹ' repeating) plus
    suspiciously short outputs vs reference.

    Legitimate reduplications (REDUPLICATION_WHITELIST, tone-stripped) and
    repeated FUNCTION words are never flagged (Bug 2A-1 fix: 'ọjọ ọjọ'
    'day after day' is good Yoruba; also non-adjacent repeats of content
    words were false-flagged by an earlier draft — only ADJACENT repeats
    count).
    """
    import re
    repeated = short = 0
    for h, r in zip(preds, refs):
        tk = [t for t in re.findall(r"[^\W_]+", h.lower(), re.UNICODE) if len(t) >= 3]
        # tone-fold tokens for whitelist comparison
        tfold = [_fold_tones(t) for t in tk]
        dup = False
        for i in range(len(tk) - 1):
            if tk[i] == tk[i+1] and tfold[i] not in REDUPLICATION_WHITELIST:
                dup = True
                break
        if not dup and len(tk) >= 4:
            for i in range(len(tk) - 3):
                if tk[i] == tk[i+2] and tk[i+1] == tk[i+3] \
                        and tfold[i] not in REDUPLICATION_WHITELIST:
                    dup = True
                    break
        if dup:
            repeated += 1
        tr = len(re.findall(r"[^\W_]+", r, re.UNICODE)) or 1
        if len(tk) < tr * 0.5:
            short += 1
    return {
        "adjacent_repeat_warnings": repeated,
        "suspiciously_short_warnings": short,
    }


# --------------------------------------------------------------------------
# scorecard assembly
# --------------------------------------------------------------------------

@dataclass
class ScorecardRow:
    arm: str                     # e.g. "eng_yor|beams=2"
    n_sentences: int
    bleu: float | None
    chrf_standard: float | None
    chrf_nfc: float | None
    chrf_folded_diagnostic: float | None
    numeric: dict
    entities: dict
    repetition: dict
    latency: dict
    human_labels: dict
    unavailable: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "arm": self.arm, "n_sentences": self.n_sentences,
            "bleu": self.bleu,
            "chrfpp_standard": self.chrf_standard,
            "chrfpp_nfc": self.chrf_nfc,
            "chrfpp_folded_diagnostic": self.chrf_folded_diagnostic,
            "numeric_consistency": self.numeric,
            "named_entity_consistency": self.entities,
            "repetition_omission_warnings": self.repetition,
            "latency": self.latency,
            "human_labels": self.human_labels,
        }
        d["unavailable"] = self.unavailable
        return d


def build_scorecard_row(arm: str, preds: list[str], refs: list[str],
                        srcs: list[str], latencies: list[float] | None = None,
                        human_labels: dict | None = None) -> ScorecardRow:
    unavailable: list[str] = []
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    std = sacrebleu.corpus_chrf(preds, [refs], **CHRF_KWARGS).score
    nfc = sacrebleu.corpus_chrf([unicodedata.normalize("NFC", p) for p in preds],
                                [[unicodedata.normalize("NFC", r) for r in refs]],
                                **CHRF_KWARGS).score
    folded = sacrebleu.corpus_chrf([fold_accents(p) for p in preds],
                                   [[fold_accents(r) for r in refs]],
                                   **CHRF_KWARGS).score
    if latencies:
        lat = {
            "mean_ms": round(statistics.mean(latencies), 1),
            "median_ms": round(statistics.median(latencies), 1),
            "total_s": round(sum(latencies), 1),
        }
    else:
        lat = {}
        unavailable.append("latency")
    if not human_labels:
        human_labels = {}
        unavailable.append("human_review_labels")
    return ScorecardRow(
        arm=arm, n_sentences=len(preds), bleu=round(bleu, 2),
        chrf_standard=round(std, 2), chrf_nfc=round(nfc, 2),
        chrf_folded_diagnostic=round(folded, 2),
        numeric=numeric_consistency(preds, srcs),
        entities=named_entity_consistency(preds, srcs),
        repetition=repetition_omission_warnings(preds, refs),
        latency=lat, human_labels=human_labels, unavailable=unavailable,
    )


METRIC_PROVENANCE = {
    "sacrebleu": SACREBLEU_VERSION,
    "chrf_kwargs": CHRF_KWARGS,
    "bleu_tokenizer": "13a",
    "note": "heuristic detector counts are warnings, not error rates; "
            "missing measurements are reported as unavailable, never zero.",
}
