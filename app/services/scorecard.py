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


def numeric_consistency(preds: list[str], srcs: list[str]) -> dict:
    """Per-sentence digit multiset agreement. Report counts + rate; the rate
    is over sentences where the source contains digits at all."""
    both = changed = missing = checked = 0
    for s, p in zip(srcs, preds):
        s_d, p_d = _DIGITS.findall(s), _DIGITS.findall(p)
        if not s_d:
            continue
        checked += 1
        if sorted(s_d) == sorted(p_d):
            both += 1
        elif not p_d:
            missing += 1
        else:
            changed += 1
    return {
        "sentences_with_source_digits": checked,
        "digits_preserved": both,
        "digits_missing": missing,
        "digits_changed": changed,
        "consistency_rate": round(both / checked, 4) if checked else None,
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


def repetition_omission_warnings(preds: list[str], refs: list[str]) -> dict:
    """Adjacent repetition (a token or token-pair followed by itself — token
    pairs catch underdot-split words like 'ìtòlé ṣẹẹsẹ' repeating) and
    suspiciously short outputs vs reference."""
    import re
    repeated = short = 0
    for h, r in zip(preds, refs):
        tk = [t for t in re.findall(r"[^\W_]+", h.lower(), re.UNICODE) if len(t) >= 3]
        dup = any(a == b for a, b in zip(tk, tk[1:]))
        if not dup and len(tk) >= 4:
            dup = any(tk[i:i+2] == tk[i+2:i+4] for i in range(len(tk) - 3))
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
