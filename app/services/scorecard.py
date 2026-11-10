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


# Spelled-out number words whose VALUES count as digit equivalents when a
# prediction renders a numeral in words (Bug 2A-2 fix: '11:29' rendered as
# 'aago mọ́kànlá lékirin-dín-logójì' is semantically faithful). Values are
# approximate for Yoruba's vigesimal system; only confident core terms.
_NUMBER_WORD_VALUES = {
    "ọ̀kan": 1, "meji": 2, "méjì": 2, "meta": 3, "mẹ́ta": 3, "merin": 4,
    "mẹ́rin": 4, "marun": 5, "márùn-ún": 5, "mẹ́fà": 6, "meje": 7, "méje": 7,
    "mejo": 8, "mẹ́jọ": 8, "mesan": 9, "mẹ́sànán": 9, "mẹ̀wá": 10,
    "mokanla": 11, "mọ́kànlá": 11, "kanla": 11, "mejila": 12, "méjìlá": 12,
    "aadota": 30, "àádọ́ta": 30, "ogoji": 40, "ogójì": 40, "aadota": 50,
    "àádọ́ta": 50, "ogorin": 80, "ọgọ́rin": 80, "ogorun": 100, "ọ̀ọ́dúnrún": 100,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}


def _pred_has_number_words(pred: str, needed: list[str]) -> bool:
    """True if the prediction contains any known number word (tone/underdot-
    folded). Conservative by design: we accept word-rendering as *plausibly*
    faithful and route it to the informational 'digits_spelled_out' bucket
    rather than counting it as an error. Value-exact verification of Yoruba's
    vigesimal numerals is deliberately out of scope (documented limitation)."""
    import re
    words = {t for t in re.findall(r"[^\W_]+",
             unicodedata.normalize("NFC", pred.lower()), re.UNICODE)}
    folded = {_fold_tones(w) for w in words}
    return bool(folded & set(_NUMBER_WORD_VALUES))


def numeric_consistency(preds: list[str], srcs: list[str]) -> dict:
    """Per-sentence digit multiset agreement. Report counts + rate; the rate
    is over sentences where the source contains digits at all.

    Bug 2A-2 fix: a prediction that renders source numerals as number WORDS
    (e.g. 11:29 -> 'aago mọ́kànlá') is no longer counted as digits_missing;
    it moves to a separate 'digits_spelled_out' bucket (informational, since
    word-value verification is approximate for Yoruba's vigesimal numerals).
    """
    both = changed = missing = spelled = checked = 0
    for s, p in zip(srcs, preds):
        s_d, p_d = _DIGITS.findall(s), _DIGITS.findall(p)
        if not s_d:
            continue
        checked += 1
        if sorted(s_d) == sorted(p_d):
            both += 1
        elif not p_d:
            if _pred_has_number_words(p, s_d):
                spelled += 1
            else:
                missing += 1
        else:
            changed += 1
    denom = checked
    return {
        "sentences_with_source_digits": checked,
        "digits_preserved": both,
        "digits_missing": missing,
        "digits_changed": changed,
        "digits_spelled_out": spelled,
        "consistency_rate": round(both / denom, 4) if denom else None,
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
