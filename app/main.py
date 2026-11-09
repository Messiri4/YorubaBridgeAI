"""FastAPI application factory and ASGI entrypoint."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.v1.routes_health import router as health_router
from app.api.v1.routes_translate import router as translate_router
from app.api.v1.routes_voice import router as voice_router
from app.api.v1.routes_whatsapp import router as whatsapp_router
from app.core.config import get_settings
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


def create_app(load_model: bool = True) -> FastAPI:
    """Build the FastAPI application.

    Args:
        load_model: load the NLLB model during startup (lifespan). Tests
            pass False so importing/creating an app stays cheap.

    The factory pattern keeps app construction explicit and lets tests
    create isolated app instances. Model loading happens in the lifespan
    hook, NOT at import time, so `import app.main` never downloads weights.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        import redis as redis_lib

        from app.core.rate_limit import RateLimiter
        from app.services.cache import TranslationCache

        # Shared Redis connection pool (None-safe: both components degrade
        # gracefully when Redis is unreachable).
        redis_client = None
        try:
            redis_client = redis_lib.Redis.from_url(
                settings.redis_url, decode_responses=True, socket_connect_timeout=2
            )
            redis_client.ping()
            logger.info("Redis connected for caching and rate limiting")
        except Exception:  # noqa: BLE001
            redis_client = None
            logger.warning("Redis unavailable: cache off, in-memory rate limiting")

        application.state.translation_cache = TranslationCache(redis_client)
        application.state.rate_limiter = RateLimiter(lambda: redis_client)

        from app.services.whatsapp_dedup import MessageDeduplicator

        application.state.wa_dedup = MessageDeduplicator(lambda: redis_client)

        from app.services.history import make_history_store

        history_store = make_history_store()
        application.state.history_store = history_store

        # Automatic retention: if history is enabled, delete rows older than
        # RETENTION_DAYS once at startup and then daily. Runs in the
        # event loop; failures are logged, never raised (history.py).
        if history_store is not None:
            import asyncio

            from app.services.history import RetentionCleanup

            cleanup = RetentionCleanup(
                connection_factory=history_store._connection_factory,  # noqa: SLF001
                retention_days=settings.retention_days,
            )

            async def _retention_loop() -> None:
                # to_thread keeps the blocking psycopg I/O OFF the event
                # loop so the API stays responsive during cleanup.
                while True:
                    await asyncio.to_thread(cleanup.cleanup)
                    await asyncio.sleep(24 * 3600)

            application.state.retention_task = asyncio.create_task(_retention_loop())

        if load_model:
            from app.services.model_loader import load_nllb
            from app.services.translation import TranslationService

            loaded = load_nllb()
            application.state.translation_service = TranslationService(loaded)
        yield
        # Teardown: cancel the retention loop if one was scheduled.
        task = getattr(application.state, "retention_task", None)
        if task is not None:
            task.cancel()

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(translate_router, prefix="/api/v1")
    app.include_router(whatsapp_router, prefix="/api/v1")
    app.include_router(voice_router, prefix="/api/v1")

    # Speech service is lazy-loading; construction is cheap.
    from app.api.v1.routes_voice import purge_stale_audio
    from app.services.speech import SpeechService

    purge_stale_audio()  # remove leftovers from a previous abnormal exit
    app.state.speech_service = SpeechService()

    # Local demo UI (single static page; no template engine needed).
    from fastapi.responses import FileResponse

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    return app


# Module-level instance used by `uvicorn app.main:app`. Import is cheap;
# heavy model loading happens only when the server actually starts.
app = create_app()
