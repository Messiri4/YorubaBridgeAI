"""Experimental interface for context-aware Yoruba tone restoration.

STATUS: EXPERIMENTAL — no pretrained diacritization model is loaded or
downloaded here. This module defines ONLY the contract (Protocol), the
result types, and a deterministic safe fallback that passes text through
unchanged. See docs/yoruba_tonality_implementation.md for integration plans
after NLLB generation.

Design principles:
- Unicode normalization is not tone restoration.
- No fabrication of tone marks from an unverified dictionary.
- Uncertainty is first-class: a restorer must report confidence; below
  threshold we fall back to the input unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.services.yoruba_orthography import audit, nfc

CONFIDENCE_THRESHOLD_DEFAULT = 0.85


@dataclass
class ToneRestorationResult:
    """Outcome of a (potential) tone-restoration attempt."""

    restored_text: str          # text actually usable (may equal input)
    changed: bool               # did we alter the text at all?
    confidence: float           # 0.0-1.0 from the model, if any
    per_word_confidence: dict[str, float] = field(default_factory=dict)
    fallback_used: bool = False  # True when we returned the input unchanged
    reason: str = ""             # why fallback occurred / status note


class ToneRestorer(Protocol):
    """Contract for a future context-aware tone-restoration component.

    A real implementation (e.g. a fine-tuned seq2seq/mT5-style Yoruba
    diacritizer) receives the NLLB OUTPUT plus the source ENGLISH sentence
    as context and returns its best fully-toned Yoruba hypothesis with
    calibrated confidence per word.
    """

    def restore(
        self,
        yoruba_text: str,
        source_english: str | None = None,
    ) -> ToneRestorationResult: ...


class PassThroughRestorer:
    """Safe fallback restorer: never changes text, reports honestly.

    Used when no verified pretrained model is available (current state),
    or as the runtime guard under any real model below threshold.
    """

    def restore(
        self,
        yoruba_text: str,
        source_english: str | None = None,
    ) -> ToneRestorationResult:
        normalized = nfc(yoruba_text)
        report = audit(normalized)
        return ToneRestorationResult(
            restored_text=normalized,
            changed=(normalized != yoruba_text),
            confidence=0.0,
            fallback_used=True,
            reason=(
                "no verified tone-restoration model configured; "
                f"orthography audit: {report.summary()}"
            ),
        )


class ThresholdGuard:
    """Wraps any ToneRestorer and enforces safe fallback behavior.

    If the underlying restorer's confidence is below ``threshold`` the
    original input is returned unchanged and the model's suggestion is
    discarded (reported only in ``reason`` for audit purposes).
    """

    def __init__(
        self,
        restorer: ToneRestorer | None = None,
        threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
    ) -> None:
        # None => PassThroughRestorer (current production reality).
        self.restorer = restorer if restorer is not None else PassThroughRestorer()
        self.threshold = threshold

    def restore(
        self,
        yoruba_text: str,
        source_english: str | None = None,
    ) -> ToneRestorationResult:
        result = self.restorer.restore(yoruba_text, source_english)
        if result.fallback_used or result.confidence < self.threshold:
            if result.restored_text != yoruba_text:
                return ToneRestorationResult(
                    restored_text=yoruba_text,
                    changed=False,
                    confidence=result.confidence,
                    per_word_confidence=result.per_word_confidence,
                    fallback_used=True,
                    reason=(
                        f"confidence {result.confidence:.2f} below threshold "
                        f"{self.threshold:.2f}; original text kept; "
                        f"model suggestion withheld"
                    ),
                )
            result.reason = result.reason or "below-threshold or passthrough; text unchanged"
            result.fallback_used = True
            return result
        return result


DEFAULT_RESTORER: ToneRestorer = ThresholdGuard()


def propose_restoration(
    yoruba_text: str,
    source_english: str | None = None,
) -> ToneRestorationResult:
    """Entry point intended to be called after NLLB generation.

    Today this is a documented no-op with diagnostics. Wiring a real model
    means supplying a ThresholdGuard(CustomModelRestorer(), threshold=...) —
    nothing else in the pipeline should change.
    """
    return DEFAULT_RESTORER.restore(yoruba_text, source_english)
