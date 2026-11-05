"""Bidirectional English <-> Yoruba translation service.

Business logic lives here; model construction lives in `model_loader.py`.
The service depends on the `LoadedModel` bundle, so unit tests can inject
mocks and run without downloading anything.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.config import get_settings
from app.services.model_loader import SUPPORTED_LANGUAGES, LoadedModel

logger = logging.getLogger(__name__)

# Public codes clients may send in API requests.
LANGUAGE_CODES = frozenset(SUPPORTED_LANGUAGES)


class PhraseOverrides:
    """Exact-match translation table loaded from a JSON file.

    Keys are NFC-normalized lowercase source phrases; lookup is O(1).
    A missing/invalid file degrades to an empty table (model-only).
    """

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
        """NFC-normalize, strip and lowercase a phrase for table lookup."""
        import unicodedata

        return unicodedata.normalize("NFC", text).strip().lower()

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

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> "TranslationResult":
        """Translate `text` from source_language to target_language.

        Args:
            text: Input text (non-empty, within max_input_chars).
            source_language: "eng" or "yor".
            target_language: "eng" or "yor" (must differ from source).

        Returns:
            TranslationResult with translated text and latency.

        Raises:
            TranslationError: on invalid input or unsupported languages.
        """
        start = time.perf_counter()
        self._validate(text, source_language, target_language)

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

        src = SUPPORTED_LANGUAGES[source_language]
        tgt = SUPPORTED_LANGUAGES[target_language]

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

        # 3. Autoregressive beam-search generation.
        generated = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_new_tokens=self._settings.translation_max_new_tokens,
            num_beams=self._settings.translation_num_beams,
        )

        # 4. Detokenize. skip_special_tokens drops </s> and language tags;
        #    Yoruba diacritics are preserved because decoding is byte-level.
        translated = tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Translated %s->%s in %.1f ms", source_language, target_language, latency_ms)
        return TranslationResult(
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            latency_ms=round(latency_ms, 1),
        )
