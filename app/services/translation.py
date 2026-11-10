"""Bidirectional English <-> Yoruba translation service.

Business logic lives here; model construction lives in `model_loader.py`.
The service depends on the `LoadedModel` bundle, so unit tests can inject
mocks and run without downloading anything.

Integration (branch integration/translation-improvements):
- reversible numeric/identifier protection (app/services/protection.py)
  with structured failure reporting and an explicit unprotected-retry
  contract (`with_unprotected_fallback`); unresolved placeholders are
  NEVER returned to API users — on restoration failure the plain model
  output is regenerated without protection, and as a last resort every
  residual sentinel is stripped before the response leaves the service.
- context-aware verified-term glossary (app/services/glossary.py),
  applied AFTER restoration so placeholders cannot shadow terms, and
  token-span based so it can never corrupt protected expressions.
- Yoruba NFC normalization + orthography auditing and the tone-restoration
  interface (PassThroughRestorer default; no diacritics are fabricated),
  wired only for Yoruba TARGET output.
- decoding-policy selection (app/services/decoding_policy.py) with the
  production_default profile reproducing the legacy generation parameters
  bit-for-bit.
"""

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.config import get_settings
from app.services.model_loader import SUPPORTED_LANGUAGES, LoadedModel

logger = logging.getLogger(__name__)

# Public codes clients may send in API requests.
LANGUAGE_CODES = frozenset(SUPPORTED_LANGUAGES)


from typing import TYPE_CHECKING  # noqa: E402  (baseline import order kept)

if TYPE_CHECKING:
    from app.services.glossary import Glossary


class PhraseOverrides:
    """Exact-match translation table loaded from a JSON file.

    Keys are NFC-normalized, punctuation-stripped, lowercased source
    phrases; lookup is O(1). Final punctuation and case are ignored so
    that e.g. "Ẹ kú ìrọ̀lẹ́, ọ̀rẹ́ mi." hits the same key as a plain
    greeting form where appropriate. A missing/invalid file degrades to
    an empty table (model-only).
    """

    # Characters ignored when building/looking up table keys. NLLB's
    # SentencePiece treats '.' and ',' as separate tokens, so a trailing
    # period must NOT make an otherwise-identical phrase miss the table.
    _STRIPPED_CHARS = str.maketrans("", "", ".,!?;:")

    def __init__(self, path: str | None = None) -> None:
        import json

        self._table: dict[str, dict[str, str]] = {"eng_yor": {}, "yor_eng": {}}
        path = path or get_settings().phrase_overrides_path
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            for direction in ("eng_yor", "yor_eng"):
                self._table[direction] = {
                    PhraseOverrides._normalize(k): v for k, v in raw.get(direction, {}).items()
                }
            logger.info(
                "Phrase overrides loaded: %d eng_yor, %d yor_eng",
                len(self._table["eng_yor"]),
                len(self._table["yor_eng"]),
            )
        except FileNotFoundError:
            logger.warning("Phrase overrides file not found: %s (model-only mode)", path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Phrase overrides unusable (%s); model-only mode", exc)

    @staticmethod
    def _normalize(text: str) -> str:
        """NFC-normalize, strip punctuation, collapse and lowercase a phrase.

        Root cause of the "Ẹ kú ìrọ̀lẹ́, ọ̀rẹ́ mi." → "Thank you, my friend."
        bug: the old normalizer kept '.' and ',', so a comma-separated
        greeting missed its exact-match key and fell through to NLLB, which
        renders "ẹ kú ìrọ̀lẹ́" as "thank you" in compound sentences.
        """
        import unicodedata

        text = unicodedata.normalize("NFC", text).translate(PhraseOverrides._STRIPPED_CHARS)
        return " ".join(text.split()).strip().lower()

    def lookup(self, text: str, source_language: str, target_language: str) -> str | None:
        """Return the override translation, or None when no exact match."""
        key = f"{source_language}_{target_language}"
        return self._table.get(key, {}).get(self._normalize(text))


class TokenizerProtocol(Protocol):
    """Minimal typing surface of the tokenizer used by the service."""

    src_lang: str
    tgt_lang: str

    def __call__(
        self, text: str, return_tensors: str, truncation: bool = ..., max_length: int = ...
    ) -> object: ...


class TranslationError(Exception):
    """Raised when translation cannot be performed for a known reason."""


@dataclass(frozen=True)
class TranslationResult:
    """Outcome of one translation call."""

    translated_text: str
    source_language: str
    target_language: str
    latency_ms: float


# Residual-sentinel scrubber: last-resort guarantee that no sentinel issued
# by protection.protect() can reach an API consumer, even if every upstream
# verification and fallback path somehow failed.
_RESIDUAL_SENTINEL_RE = re.compile(r"@@\s*\d+@@")


class TranslationService:
    """Translates text between English (eng) and Yoruba (yor) using NLLB.

    A phrase-override table is consulted first for exact matches (curated
    fixes for known model weaknesses); everything else goes to the model.

    Attributes:
        loaded: LoadedModel bundle (model, tokenizer, device).

    Example:
        >>> service = TranslationService(loaded)
        >>> service.translate("Hello", source_language="eng", target_language="yor")
        TranslationResult(...)
    """

    def __init__(self, loaded: LoadedModel) -> None:
        self._loaded = loaded
        self._settings = get_settings()
        self._overrides: PhraseOverrides | None = None
        self._glossary: Glossary | None = None  # lazy; see translate()
        self._restorer = None  # lazy tone restorer (interface contract)

    @property
    def overrides(self) -> PhraseOverrides:
        """Lazy-loaded phrase table (so construction stays cheap for tests)."""
        if self._overrides is None:
            self._overrides = PhraseOverrides()
        return self._overrides

    def _validate(self, text: str, source_language: str, target_language: str) -> None:
        """Validate inputs; raise TranslationError on any problem.

        Checks: non-empty text after stripping, both languages supported,
        and source != target. Character limits are enforced upstream by
        the Pydantic schema; here we only guard the service-level invariants.
        """
        if not text or not text.strip():
            raise TranslationError("text must be a non-empty string")
        if source_language not in LANGUAGE_CODES:
            raise TranslationError(
                f"unsupported source_language '{source_language}'. "
                f"Supported: {sorted(LANGUAGE_CODES)}"
            )
        if target_language not in LANGUAGE_CODES:
            raise TranslationError(
                f"unsupported target_language '{target_language}'. "
                f"Supported: {sorted(LANGUAGE_CODES)}"
            )
        if source_language == target_language:
            raise TranslationError("source_language and target_language must differ")

    def _diagnostic_trace(self):
        """Sprint 3 PHASE 2: optional per-stage tracing for placeholder
        diagnostics. Delegates to app.services.diagnostic_trace, which is
        inert unless YB_TRANSLATION_TRACE is enabled, so the production path
        never pays for diagnostics. The caller sets ``trace.source_text``
        right after obtaining it."""
        try:
            from app.services.diagnostic_trace import diagnostic_trace
            return diagnostic_trace()
        except Exception:  # noqa: BLE001 - diagnostics must never break serving
            logger.debug("diagnostic trace unavailable", exc_info=True)
            return None

    # ------------------------------------------------------------------ core

    def _generate(self, text: str, src: str, tgt: str) -> str:
        """Run one NLLB generation pass with the resolved decoding policy."""
        from app.services.decoding_policy import build_generate_kwargs, get_policy

        tokenizer = self._loaded.tokenizer
        model = self._loaded.model

        # 1. Tokenize with the source language prefix (NLLB convention).
        tokenizer.src_lang = src
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._settings.translation_max_input_chars,
        )

        # 2. Force the decoder to start with the target language token.
        forced_bos = tokenizer.convert_tokens_to_ids(tgt)

        # 3. Autoregressive beam-search generation, parameterized by the
        #    decoding policy. The default ("production_default") mirrors the
        #    settings-driven behavior exactly (beams=4, max_new=256, lp=1.1),
        #    so switching here is output-neutral unless a profile is chosen.
        policy = get_policy(settings=self._settings)
        input_tokens = None
        input_ids = getattr(inputs, "input_ids", None)
        shape = getattr(input_ids, "shape", None)
        if shape is not None and len(shape) >= 2:
            input_tokens = int(shape[1])
        generate_kwargs = build_generate_kwargs(
            policy, self._settings, input_tokens=input_tokens,
            direction=f"{self._current_source}_{self._current_target}",
        )
        generated = model.generate(**inputs, forced_bos_token_id=forced_bos, **generate_kwargs)

        # 4. Detokenize. skip_special_tokens drops </s> and language tags;
        #    Yoruba diacritics are preserved because decoding is byte-level.
        return tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

    def _translate_once(self, text: str, source_language: str, target_language: str,
                        protect_numerics: bool) -> str:
        """One translation pass WITHOUT restoration; returns raw model text.

        Called through `with_unprotected_fallback`, which owns the
        protected/unprotected retry decision (explicit callable contract,
        no signature guessing).
        """
        src = SUPPORTED_LANGUAGES[source_language]
        tgt = SUPPORTED_LANGUAGES[target_language]
        return self._generate(text, src, tgt)

    def _finish(self, text: str, source_language: str, target_language: str) -> str:
        """Post-generation enrichment shared by both protected/plain paths.

        Order matters: glossary runs on fully restored text (placeholders
        cannot shadow terms), then NFC normalization, then the tone-
        restoration interface — Yoruba target only, and the default
        restorer is a no-op pass-through that never fabricates diacritics.
        """
        from app.services.yoruba_orthography import nfc as yoruba_nfc

        # Verified-term glossary (post-generation; context-aware, Unicode
        # safe, occurrence-wise). Applied on restored text.
        if self._settings.glossary_enabled:
            from app.services.glossary import Glossary
            if self._glossary is None:
                self._glossary = Glossary(self._settings.glossary_path)
            text, applied = self._glossary.apply(
                text, f"{source_language}_{target_language}")
            if applied:
                logger.info("glossary applied: %s", applied)

        is_yoruba_target = target_language == "yor"

        # NFC normalization for Yoruba output (preserves every tone/underdot
        # distinction; makes composed/decomposed representations consistent).
        if is_yoruba_target:
            text = yoruba_nfc(text)

            # Tone-restoration interface (experiment). Default is the
            # PassThroughRestorer behind a ThresholdGuard: it never changes
            # text and never fabricates diacritics.
            try:
                from app.services.tone_restoration import propose_restoration
                if self._restorer is None:
                    self._restorer = propose_restoration
                outcome = self._restorer(text, None)
                text = outcome.restored_text
                if outcome.fallback_used:
                    logger.debug("tone restoration: %s", outcome.reason)
            except Exception:  # noqa: BLE001 - tonality must never break serving
                logger.warning("tone-restoration stage failed; text unchanged", exc_info=True)

        return text

    # ------------------------------------------------------------------- API

    def translate(
        self, text: str, source_language: str, target_language: str,
        protect_numerics: bool | None = None,
    ) -> "TranslationResult":
        """Translate `text` from source_language to target_language.

        Args:
            text: Input text (non-empty, within max_input_chars).
            source_language: "eng" or "yor".
            target_language: "eng" or "yor" (must differ from source).
            protect_numerics: reversible sentinel protection for numbers,
                times, dates, currency, URLs, emails and identifiers
                (Sprint TI-1). None = use the global setting. On
                restoration failure the request is retried WITHOUT
                protection (safe fallback), and a final residual-sentinel
                scrub guarantees no placeholder ever reaches the caller.

        Returns:
            TranslationResult with translated text and latency.

        Raises:
            TranslationError: on invalid input or unsupported languages.
        """
        from app.services.protection import (
            protect as prot_wrap,
        )
        from app.services.protection import (
            restore_outcome as prot_restore_outcome,
        )
        from app.services.protection import (
            with_unprotected_fallback,
        )

        start = time.perf_counter()
        self._validate(text, source_language, target_language)

        # Keep direction for the (optional) direction-aware decoding policy.
        self._current_source = source_language
        self._current_target = target_language

        protection_enabled = (self._settings.numeric_protection_enabled
                              if protect_numerics is None else protect_numerics)
        protection = prot_wrap(text) if protection_enabled else None
        protected_text = protection.protected_text if protection is not None else text
        trace = self._diagnostic_trace()  # no-op unless YB_TRANSLATION_TRACE
        trace.source_text = text  # bound by the tracer (no-op otherwise)
        if protection is not None:
            trace.stage("protect", source=text, protected=protected_text,
                        n_placeholders=protection.n,
                        placeholders=dict(protection.placeholders))

        # Phrase-override table: exact matches bypass the model entirely.
        # This fixes known model weaknesses (greeting idioms, market
        # phrases) at zero latency cost.
        override = self.overrides.lookup(text, source_language, target_language)
        if override is not None:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info("Phrase override hit for %s->%s", source_language, target_language)
            return TranslationResult(
                translated_text=override,
                source_language=source_language,
                target_language=target_language,
                latency_ms=round(latency_ms, 1),
            )

        if protection is not None:
            # Protected path with a structured, explicit retry contract:
            # the first callable translates with sentinels and returns the
            # restoration outcome; the second (plain) callable is invoked
            # by the helper ONLY when verification fails. No signature
            # guessing, no broad TypeError catching in this module.
            def _protected_attempt() -> "object":

                raw = self._translate_once(
                    protected_text, source_language, target_language,
                    protect_numerics=True)
                if trace is not None:
                    trace.stage("generate_protected", raw_model_output=raw)
                outcome = prot_restore_outcome(raw, protection)
                if trace is not None:
                    trace.stage("restore", ok=outcome.ok,
                                failure=str(outcome.failure) if outcome.failure else None,
                                restored_text=getattr(outcome, "text", raw))
                return outcome

            def _plain_attempt() -> str:
                plain = self._translate_once(
                    text, source_language, target_language, protect_numerics=False)
                if trace is not None:
                    trace.stage("generate_retry_unprotected", raw_model_output=plain)
                return plain

            # Sprint 3 PHASE 2: bounded protected retries BEFORE the
            # unconditional unprotected fallback. Each retry regenerates WITH
            # sentinels. numeric_protection_max_retries=0 restores the legacy
            # one-shot contract exactly (protected once, then plain retry).
            max_protected_retries = max(
                0, int(self._settings.numeric_protection_max_retries))
            outcome = None
            failure = None
            for attempt in range(1 + max_protected_retries):
                if attempt > 0 and trace is not None:
                    trace.stage("protected_retry", attempt=attempt)
                outcome = _protected_attempt()
                if outcome.ok:
                    failure = None
                    break
                failure = outcome.failure
            if outcome is not None and outcome.ok:
                used_fallback = False
                translated = outcome.text
            else:
                # Protected attempts exhausted: documented safe fallback is
                # one plain (unprotected) regeneration, then the residual
                # sentinel scrub below as defense-in-depth.
                translated = _plain_attempt()
                used_fallback = True
            if trace is not None:
                trace.stage("fallback_decision", used_fallback=used_fallback,
                            failure=str(failure) if failure else None)
            if failure is not None:
                logger.warning(
                    "placeholder restoration failed (%s)%s",
                    failure, "; retried without protection" if used_fallback else "")
        else:
            # Plain path (protection off, or nothing to protect).
            translated = self._translate_once(
                text, source_language, target_language, protect_numerics=False)

        # Hard guarantee: unresolved placeholders can never reach users.
        # After the safe-fallback retry there should be none; this scrub is
        # defense-in-depth, logged loudly when it has to act.
        if "@@" in translated:
            if trace is not None:
                trace.stage("residual_scrub", before=translated)
            scrubbed = _RESIDUAL_SENTINEL_RE.sub("", translated)
            cleaned = re.sub(r"\s{2,}", " ", scrubbed).strip(" ,;")
            if cleaned != translated:
                logger.error("residual placeholder scrubbed from output")
                translated = cleaned

        translated = self._finish(translated, source_language, target_language)
        if trace is not None:
            trace.stage("finish", final_text=translated)
            trace.close()

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Translated %s->%s in %.1f ms", source_language, target_language, latency_ms)
        return TranslationResult(
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            latency_ms=round(latency_ms, 1),
        )
