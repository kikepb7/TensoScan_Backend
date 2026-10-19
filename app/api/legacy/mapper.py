from datetime import UTC, datetime

from pydantic import BaseModel

from app.domain.measurements.model import Measurement


class LegacyResult(BaseModel):
    high_pressure: str
    low_pressure: str
    pulse: str
    confidence: float


class LegacyMeasurement(BaseModel):
    measurement_id: str
    user_id: str
    filename: str
    result: LegacyResult
    timestamp: datetime  # naive on purpose: the previous version never sent a zone


class LegacyMeasurementMapper:
    """Adapts a ``Measurement`` to the shape the previous app version expects (design §5)."""

    @staticmethod
    def to_legacy(measurement: Measurement) -> LegacyMeasurement:
        recognition = measurement.recognition
        return LegacyMeasurement(
            measurement_id=measurement.id or "",
            user_id=measurement.user_id,
            filename="",  # v1 never stores photos (C-SEC-05)
            result=LegacyResult(
                high_pressure=str(measurement.reading.systolic),
                low_pressure=str(measurement.reading.diastolic),
                pulse=str(measurement.reading.pulse),
                confidence=recognition.confidence if recognition else 1.0,
            ),
            timestamp=measurement.measured_at.astimezone(UTC).replace(tzinfo=None),
        )
