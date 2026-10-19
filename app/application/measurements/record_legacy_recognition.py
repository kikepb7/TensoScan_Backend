import logging
from uuid import uuid4

from app.domain.measurements.legacy import LEGACY_MODEL_VERSION, legacy_confidence, parse_legacy_integer
from app.domain.measurements.model import Measurement, Reading, RecognitionInfo
from app.domain.ports import Clock, MeasurementRepository

logger = logging.getLogger(__name__)


class RecordLegacyRecognition:
    """Stores what the deprecated on-server recognition (``POST /ocr/display-recognize``) read from a photo.

    DOCUMENTED EXCEPTION (design §5): unlike ``CreateMeasurement`` this does NOT apply ``ReadingRules``. The route
    has always stored whatever the model produced and the route is deprecated; it goes away with the legacy OCR
    (REQ-B002-13). A result that is not made of digits cannot be a ``Reading`` and is not stored.
    """

    def __init__(self, measurements: MeasurementRepository, clock: Clock) -> None:
        self._measurements = measurements
        self._clock = clock

    async def execute(
        self, *, user_id: str, high_pressure: object, low_pressure: object, pulse: object, confidence: object
    ) -> Measurement | None:
        values = [parse_legacy_integer(value) for value in (high_pressure, low_pressure, pulse)]
        if values[0] is None or values[1] is None or values[2] is None:
            logger.warning("Legacy recognition produced a non-numeric result; it was not stored (user %s)", user_id)
            return None
        now = self._clock.now().replace(microsecond=0)
        draft = Measurement(
            id=None,
            user_id=user_id,
            client_id=uuid4(),
            reading=Reading(values[0], values[1], values[2]),
            measured_at=now,
            created_at=None,
            source="camera",
            recognition=RecognitionInfo(legacy_confidence(confidence), LEGACY_MODEL_VERSION, frozenset()),
        )
        stored, _ = await self._measurements.add_idempotent(draft, created_at=now)
        return stored
