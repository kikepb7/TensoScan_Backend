from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from app.domain.errors import MeasurementInvalid
from app.domain.measurements.model import Measurement, MeasurementSource, RecognitionInfo, as_utc
from app.domain.measurements.rules import ReadingRules
from app.domain.ports import Clock, MeasurementRepository


@dataclass(frozen=True)
class NewMeasurement:
    """What the client sent. Values are optional here so that a missing one is reported as ``REQUIRED`` by the rules."""

    user_id: str
    client_id: UUID
    systolic: int | None
    diastolic: int | None
    pulse: int | None
    measured_at: datetime
    source: MeasurementSource
    recognition: RecognitionInfo | None


@dataclass(frozen=True)
class CreatedMeasurement:
    measurement: Measurement
    created: bool  # False: an identical request had already been stored (idempotent replay)


class CreateMeasurement:
    """REQ-B002-01..05: validates with the shared reading rules and stores the measurement idempotently."""

    def __init__(self, measurements: MeasurementRepository, clock: Clock) -> None:
        self._measurements = measurements
        self._clock = clock

    async def execute(self, command: NewMeasurement) -> CreatedMeasurement:
        now = self._clock.now()
        evaluation = ReadingRules.evaluate(command.systolic, command.diastolic, command.pulse)
        issues = [
            *evaluation.errors,
            *ReadingRules.validate_measured_at(command.measured_at, now),
            *ReadingRules.validate_recognition(command.source, command.recognition),
        ]
        if issues or evaluation.reading is None:
            raise MeasurementInvalid(issues, "The measurement is not valid")
        # Warnings (reading-rules §3.2) never block: the user already confirmed them in the app.
        draft = Measurement(
            id=None,
            user_id=command.user_id,
            client_id=command.client_id,
            reading=evaluation.reading,
            # Second precision, like the canonical payload, so that a replay returns exactly what the first call did.
            measured_at=as_utc(command.measured_at).replace(microsecond=0),
            created_at=None,
            source=command.source,
            recognition=command.recognition,
        )
        stored, created = await self._measurements.add_idempotent(draft, created_at=now.replace(microsecond=0))
        return CreatedMeasurement(stored, created)
