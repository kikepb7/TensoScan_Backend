"""On-server digit recognition of the previous version. Disabled by default (``TENSOSCAN_ENABLE_LEGACY_OCR``).

The model is loaded lazily on the first request, so a missing model file only affects these routes (503) and never
stops the API from starting. Images are processed in memory and never written to disk. Spec B-002 removes this module.
"""

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.deps import ContainerDep, CurrentUser
from app.api.legacy.headers import deprecated
from app.domain.errors import FieldIssue, UpstreamUnavailable, ValidationFailed

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
router = APIRouter(prefix="/ocr", tags=["legacy"], dependencies=[Depends(deprecated("/v1/measurements"))])


class LegacyOcr:
    def __init__(self, model_path: Path | None) -> None:
        self._model_path = model_path
        self._lock = threading.Lock()
        self._recognizer: Any = None
        self._display: Any = None

    def _load(self) -> None:
        with self._lock:
            if self._display is not None:
                return
            try:
                if self._model_path is None:
                    raise FileNotFoundError("TENSOSCAN_LEGACY_OCR_MODEL_PATH is not configured")
                from app.infrastructure.legacy_ocr.display_recognizer import DisplayRecognizerService
                from app.infrastructure.legacy_ocr.image_processor import ImageProcessorService
                from app.infrastructure.legacy_ocr.keras_recognizer import KerasNumberRecognizer

                self._recognizer = KerasNumberRecognizer(str(self._model_path))
                self._display = DisplayRecognizerService(self._recognizer, ImageProcessorService())
            except Exception as error:
                logger.exception("Legacy OCR is unavailable")
                raise UpstreamUnavailable("The recognition service is not available") from error

    def recognize_digit(self, image_bytes: bytes) -> dict[str, Any]:
        import io

        from PIL import Image

        self._load()
        return self._recognizer.predict(Image.open(io.BytesIO(image_bytes))).model_dump()

    def recognize_display(self, image_bytes: bytes) -> dict[str, Any]:
        self._load()
        return self._display.recognize_display(image_bytes).model_dump()


async def _read_image(image: UploadFile) -> bytes:
    if not (image.content_type or "").startswith("image/"):
        raise ValidationFailed([FieldIssue("image", "NOT_AN_IMAGE")])
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValidationFailed([FieldIssue("image", "TOO_LARGE")])
    return content


def _ocr(container: ContainerDep) -> LegacyOcr:
    state = container.legacy_ocr
    if state is None:
        raise UpstreamUnavailable("The recognition service is disabled")
    return state


@router.post("/recognize")
async def recognize_numbers(
    user: CurrentUser, container: ContainerDep, image: UploadFile = File(...)
) -> dict[str, Any]:
    content = await _read_image(image)
    return await run_in_threadpool(_ocr(container).recognize_digit, content)


@router.post("/display-recognize")
async def display_recognize(
    user: CurrentUser, container: ContainerDep, image: UploadFile = File(...)
) -> dict[str, Any]:
    content = await _read_image(image)
    result = await run_in_threadpool(_ocr(container).recognize_display, content)
    await container.db["recognition_results"].insert_one(
        {
            "user_id": ObjectId(user.id),
            "filename": image.filename or "",
            "result": result,
            "timestamp": datetime.now(UTC),
        }
    )
    return result
