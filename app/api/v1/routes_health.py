"""Health check endpoints for liveness/readiness probes."""

from typing import Any

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe.

    Returns basic app identity only; never leaks configuration values
    such as tokens or connection strings.
    """
    settings = get_settings()
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}
