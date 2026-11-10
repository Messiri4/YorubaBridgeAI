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


def _materialize_meta_parameters(model: Any, model_dir: str) -> Any:
    """Materialize any parameters left on the meta device after loading.

    transformers 4.57 loads safetensors shards with map_location="meta"
    and then assigns tensors one-by-one via ``load_state_dict(assign=True)``;
    with tied embeddings (NLLB shares lm_head/embed_tokens) the tied group
    can remain on the meta device when torch < 2.6 is installed. Those
    "Cannot copy out of meta tensor" parameters make ``model.to(device)``
    fail. Here the missing tensors are loaded directly from the local
    checkpoint (safetensors first, torch.load fallback with the
    transformers safety gate for .bin) and assigned into the tied slots.
    No downloads happen: only an already-present local directory is read.
    """
    meta_keys = [k for k, v in model.state_dict().items() if v.device.type == "meta"]
    if not meta_keys:
        return model

    weights_path = os.path.join(model_dir, "model.safetensors")
    state: dict[str, Any] = {}
    if os.path.exists(weights_path):
        from safetensors import safe_open

        with safe_open(weights_path, framework="pt", device="cpu") as f:
            available = set(f.keys())
            # Tied-group aliases: the checkpoint stores ONE tensor
            # ('lm_head.weight'); model.shared.weight / embed_tokens alias it.
            tied_aliases = [k for k in available if k.endswith("lm_head.weight")]
            for key in meta_keys:
                if key in available:
                    state[key] = f.get_tensor(key)
                elif key == "model.shared.weight" and tied_aliases:
                    state[key] = f.get_tensor(tied_aliases[0])
                elif key.endswith(".weight") and key.rsplit(".", 1)[0] + ".weight" in available:
                    state[key] = f.get_tensor(key)
    else:
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(bin_path):
            from transformers.utils import check_torch_load_is_safe

            check_torch_load_is_safe()  # raises on torch<2.6 (CVE-2025-32434)
            full = torch.load(bin_path, map_location="cpu", weights_only=True)
            for key in meta_keys:
                if key in full:
                    state[key] = full[key]

    if not state:
        logger.warning(
            "meta-device parameters present (%s) but no loadable checkpoint "
            "found in %s; letting .to() fail loudly", meta_keys[:3], model_dir
        )
        return model

    from transformers.modeling_utils import _load_parameter_into_model

    # IMPORTANT: materialize the canonical tied root first. For M2M100/NLLB
    # that is ``model.shared.weight``: ``tie_weights()`` re-points every
    # other tied slot (embed_tokens, lm_head) AT the shared parameter, so
    # assigning a real tensor to shared and then re-tying restores the whole
    # group. Assigning to lm_head first and re-tying would re-link shared to
    # a still-meta tensor and wipe the fix (verified empirically).
    ordered_keys = sorted(state,
                          key=lambda k: (0 if k.endswith("shared.weight") else 1, k))
    for key in ordered_keys:
        _load_parameter_into_model(model, key, state[key])
        logger.info("materialized meta parameter '%s' from local checkpoint", key)

    # Re-establish the tying after assignment (assign=True replaces the
    # parameter object, which can break weight sharing for tied groups).
    try:
        model.tie_weights()
    except Exception:  # noqa: BLE001 - tying is best-effort here
        logger.warning("tie_weights() after meta materialization failed", exc_info=True)
    return model


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
        logger.info(
            "Loading translation model from local path '%s' on %s ...", model_id, device
        )
        tokenizer = NllbTokenizerFast.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model = _materialize_meta_parameters(model, model_id)
        model = model.to(device)
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
