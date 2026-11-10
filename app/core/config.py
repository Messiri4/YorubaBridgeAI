"""Typed application settings loaded from environment / .env.

Why pydantic-settings: it validates types at startup, so a misconfigured
service fails fast with a clear error instead of at request time.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for the service.

    Attributes are populated from environment variables (case-insensitive)
    and the local `.env` file. Secrets are never stored in this file.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_name: str = "YorubaBridgeAI"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    database_url: str = "postgresql+psycopg://localhost/yorubabridge"
    redis_url: str = "redis://localhost:6379/0"

    # Translation model settings (NLLB-200 distilled 600M)
    # NOTE: facebook/nllb-* checkpoints are CC-BY-NC 4.0 (non-commercial).
    # Override via .env to swap checkpoints (e.g. an open-licensed one) later.
    translation_model_id: str = "facebook/nllb-200-distilled-600M"
    translation_max_new_tokens: int = 256
    translation_num_beams: int = 4
    translation_length_penalty: float = 1.1
    translation_max_input_chars: int = 2000

    # Caching & rate limiting (Redis-backed; degrades gracefully without it)
    cache_ttl_seconds: int = 3600
    rate_limit_max_requests: int = 10
    rate_limit_window_seconds: int = 60

    # WhatsApp Cloud API credentials (used in a later milestone)
    whatsapp_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_verify_token: str = ""
    whatsapp_app_secret: str = ""
    whatsapp_api_version: str = "v20.0"

    # Voice notes (STT). 'tiny' fits CPU/RAM budget; override in .env if you
    # upgrade hardware.
    speech_model_size: str = "tiny"

    # Phrase-base overrides (exact-match table applied before the model)
    phrase_overrides_path: str = "data/phrases/overrides.json"

    # Sprint TI-1: reversible numeric/identifier protection around NLLB
    # generation (times, dates, scores, currency, URLs, emails). Safe
    # fallback on restoration failure; disable to restore exact prior behavior.
    numeric_protection_enabled: bool = True

    # Sprint 3 PHASE 2: bounded protected retry — before the unconditional
    # unprotected fallback, re-attempt generation WITH sentinels up to N
    # times. The retry costs one extra generation pass per attempt but keeps
    # numeric values exact far more often than the plain retry; 0 disables
    # (legacy one-shot behavior).
    numeric_protection_max_retries: int = 1

    # Verified-term glossary (data/phrases/glossary.json). Applied after
    # generation; context-aware; contains only attested mappings.
    glossary_enabled: bool = True
    glossary_path: str = "data/phrases/glossary.json"

    # Persistence (Postgres). Disabled by default: the API runs fine
    # without a database until you set DATABASE_ENABLED=true.
    database_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings instance.

    `lru_cache` makes this a singleton per process so the .env is parsed once.
    Tests can override it via `get_settings.cache_clear()`.
    """
    return Settings()
