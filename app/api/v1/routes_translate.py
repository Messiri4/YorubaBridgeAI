"""POST /api/v1/translate — English <-> Yoruba translation endpoint."""

import logging

from fastapi import APIRouter, Request

from app.models.translation import TranslationRequest, TranslationResponse
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

    Validation (language codes, empty text, direction) happens in the
    Pydantic schema before this handler runs; model inference errors are
    surfaced as HTTP 500 by FastAPI's default handler.
    """
    try:
        result = get_service(request).translate(
            payload.text,
            source_language=payload.source_language,
            target_language=payload.target_language,
        )
    except TranslationError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TranslationResponse(**result.__dict__)
