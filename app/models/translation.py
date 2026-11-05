"""Pydantic request/response schemas for the translation API."""

from pydantic import BaseModel, Field, model_validator

SUPPORTED_LANGUAGE_CODES = frozenset({"eng", "yor"})


class TranslationRequest(BaseModel):
    """POST /api/v1/translate request body.

    Attributes:
        text: source text, 1..2000 chars (non-empty enforced).
        source_language: "eng" or "yor".
        target_language: "eng" or "yor", must differ from source.
    """

    text: str = Field(min_length=1, max_length=2000)
    source_language: str = Field(pattern="^(eng|yor)$")
    target_language: str = Field(pattern="^(eng|yor)$")

    @model_validator(mode="after")
    def languages_must_differ(self) -> "TranslationRequest":
        if self.source_language == self.target_language:
            raise ValueError("source_language and target_language must differ")
        return self


class TranslationResponse(BaseModel):
    """POST /api/v1/translate response body."""

    translated_text: str
    source_language: str
    target_language: str
    latency_ms: float
    cached: bool = False
