"""POST /api/v1/translate — English <-> Yoruba translation endpoint."""

import logging

from fastapi import APIRouter, HTTPException, Request

from app.core.rate_limit import RateLimiter, client_identifier
from app.models.translation import TranslationRequest, TranslationResponse
from app.services.cache import TranslationCache
from app.services.translation import TranslationError, TranslationService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["translation"])


def get_service(request: Request) -> TranslationService:
    """Fetch the TranslationService created at startup (app.state)."""
    service: TranslationService = request.app.state.translation_service
    return service


@router.post("/translate", response_model=TranslationResponse)
def translate(payload: TranslationRequest, request: Request) -> TranslationResponse:
    """Translate text between English and Yoruba.

    Pipeline: rate-limit check -> cache lookup -> model inference ->
    cache store. Validation (language codes, empty text, direction) is
    done by the Pydantic schema before this handler runs.
    """
    # 1. Rate limit (raises 429 when exceeded).
    limiter: RateLimiter = request.app.state.rate_limiter
    limiter.check(client_identifier(request))

    cache: TranslationCache = request.app.state.translation_cache

    # 2. Cache hit -> skip the model entirely.
    hit = cache.get(payload.text, payload.source_language, payload.target_language)
    if hit is not None:
        return TranslationResponse(**hit, cached=True)

    # 3. Cache miss -> real inference.
    try:
        result = get_service(request).translate(
            payload.text,
            source_language=payload.source_language,
            target_language=payload.target_language,
        )
    except TranslationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = TranslationResponse(**result.__dict__, cached=False)

    # 4. Store for next time; the original (non-cached) latency is stored
    #    so repeat consumers see the computation cost, not ~0.0.
    cache.set(
        payload.text,
        payload.source_language,
        payload.target_language,
        result.__dict__,
    )
    return response
