import logging
from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.deps import ContainerDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["operations"])


def _version() -> str:
    try:
        return version("tensoscan-api")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree without installation
        return "0.0.0"


@router.get("/health")
async def health(container: ContainerDep) -> JSONResponse:
    try:
        await container.db.command("ping")
    except Exception:
        logger.exception("Health check: database unreachable")
        return JSONResponse(status_code=503, content={"status": "unavailable", "version": _version()})
    return JSONResponse(content={"status": "ok", "version": _version()})
