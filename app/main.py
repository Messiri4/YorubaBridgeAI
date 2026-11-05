"""FastAPI application factory and ASGI entrypoint."""
from fastapi import FastAPI

from app.api.v1.routes_health import router as health_router
from app.api.v1.routes_translate import router as translate_router
from app.core.config import get_settings
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    """Build the FastAPI application.

    The factory pattern keeps app construction explicit and lets tests
    create isolated app instances.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.include_router(health_router)
    app.include_router(translate_router, prefix="/api/v1")
    return app


app = create_app()
