import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api.container import build_container
from app.api.legacy import auth_legacy, chatbot_legacy, measurements_legacy
from app.api.v1 import assistant, auth, health, measurements, users
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.logging import RequestIdMiddleware, configure_logging
from app.domain.ports import Clock
from app.infrastructure.mongo.client import Database, create_client
from app.infrastructure.mongo.indexes import ensure_indexes

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    database: Database | None = None,
    clock: Clock | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Application factory.

    Nothing heavy happens at import time: the Mongo client and every collaborator are created in the lifespan, and the
    optional legacy OCR model is loaded lazily on its first request. ``database``, ``clock`` and ``http_client``
    are injection points for tests.
    """
    settings = settings or get_settings()
    configure_logging(json_output=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = None if database is not None else create_client(settings)
        db = database if database is not None else client[settings.db_name]  # type: ignore[index]
        async with http_client or httpx.AsyncClient() as http:
            if database is None:
                await ensure_indexes(db)
            app.state.container = build_container(settings, db, http, clock)
            logger.info("TensoScan API started (environment=%s)", settings.environment)
            yield
        if client is not None:
            await client.close()

    app = FastAPI(
        title="TensoScan API",
        description="Authentication, measurements and assistant API of the TensoScan app",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    register_error_handlers(app)

    for router in (health.router, auth.router, users.router, measurements.router, assistant.router):
        app.include_router(router)
    for legacy_router in (auth_legacy.router, chatbot_legacy.router, measurements_legacy.router):
        app.include_router(legacy_router)
    if settings.enable_legacy_ocr:
        from app.api.legacy import ocr_legacy

        app.include_router(ocr_legacy.router)

    return app
