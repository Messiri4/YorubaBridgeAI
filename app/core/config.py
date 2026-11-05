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

    # WhatsApp Cloud API credentials (used in a later milestone)
    whatsapp_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_verify_token: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings instance.

    `lru_cache` makes this a singleton per process so the .env is parsed once.
    Tests can override it via `get_settings.cache_clear()`.
    """
    return Settings()
