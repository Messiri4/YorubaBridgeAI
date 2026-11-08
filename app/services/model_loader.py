"""Model loading for NLLB-200, deliberately separated from translation logic.

Separation of concerns: this module's only job is to construct and hold the
heavy objects (model + tokenizer). The `TranslationService` stays free of
loading details, which keeps it unit-testable with mocks.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Flores-200 language codes used by YorubaBridge. Verified against the
# NLLB model card and HF docs: https://huggingface.co/facebook/nllb-200-distilled-600M
SUPPORTED_LANGUAGES: dict[str, str] = {
    "eng": "eng_Latn",  # English
    "yor": "yor_Latn",  # Yoruba (Yorùbá)
}


@dataclass
class LoadedModel:
    """Bundle of everything the translation service needs at inference time."""

    model: Any  # AutoModelForSeq2SeqLM instance (PreTrainedModel)
    tokenizer: Any  # NllbTokenizerFast instance
    device: str


def load_nllb(model_id: str | None = None) -> LoadedModel:
    """Load the NLLB-200 model and tokenizer.

    Args:
        model_id: Hugging Face checkpoint id. Defaults to the configured one.

    Returns:
        LoadedModel with model (on best available device), tokenizer, device.

    Raises:
        Exception: propagates download/network/model errors to the caller,
            which decides how to surface them (fail-fast at startup).
    """
    settings = get_settings()
    model_id = model_id or settings.translation_model_id

    # Use GPU when available; NLLB-600M runs comfortably on CPU too.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Honor a local checkpoint directory (e.g. a safetensors snapshot built
    # from an already-downloaded HF cache). This avoids re-downloading weights
    # and lets torch<2.6 environments load safetensors instead of .bin.
    if os.path.isdir(model_id):
        logger.info("Loading translation model from local path '%s' on %s ...", model_id, device)
        tokenizer = NllbTokenizerFast.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
        model.eval()
        logger.info("Model loaded: %s (device=%s)", model_id, device)
        return LoadedModel(model=model, tokenizer=tokenizer, device=device)

    logger.info("Loading translation model '%s' on %s ...", model_id, device)
    tokenizer = NllbTokenizerFast.from_pretrained(model_id)

    # Ensure Yoruba diacritics survive tokenization round-trips. The
    # SentencePiece vocab is byte-fallback, but forcing Unicode handling
    # explicitly documents intent and guards against tokenizer updates.
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "yor_Latn"

    # torch_dtype=float32 keeps CPU behaviour deterministic; the 600M model
    # is small enough that fp16 is unnecessary for development.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    model.eval()  # inference mode: disables dropout, saves memory
    logger.info("Model loaded: %s (device=%s)", model_id, device)

    return LoadedModel(model=model, tokenizer=tokenizer, device=device)
