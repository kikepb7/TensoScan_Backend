from datetime import datetime
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, Field, StrictInt
from pydantic.types import UUID4

from app.application.measurements.create_measurement import NewMeasurement
from app.application.measurements.list_measurements import MeasurementsListing
from app.domain.measurements.model import Measurement, MeasurementSource, ReadingField, RecognitionInfo


class RecognitionSchema(BaseModel):
    """Range and length limits are checked by ``ReadingRules`` so that they answer ``MEASUREMENT_INVALID``."""

    confidence: float
    model_version: str
    edited_fields: list[ReadingField] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, recognition: RecognitionInfo) -> "RecognitionSchema":
        return cls(
            confidence=recognition.confidence,
            model_version=recognition.model_version,
            edited_fields=sorted(recognition.edited_fields),
        )

    def to_domain(self) -> RecognitionInfo:
        return RecognitionInfo(self.confidence, self.model_version, frozenset(self.edited_fields))


class CreateMeasurementRequest(BaseModel):
    client_id: UUID4
    # Optional on purpose: a missing value is reported as MEASUREMENT_INVALID / REQUIRED, as in reading-rules.md.
    systolic: StrictInt | None = None
    diastolic: StrictInt | None = None
    pulse: StrictInt | None = None
    measured_at: AwareDatetime
    source: MeasurementSource
    recognition: RecognitionSchema | None = None

    def to_command(self, user_id: str) -> NewMeasurement:
        return NewMeasurement(
            user_id=user_id,
            client_id=self.client_id,
            systolic=self.systolic,
            diastolic=self.diastolic,
            pulse=self.pulse,
            measured_at=self.measured_at,
            source=self.source,
            recognition=self.recognition.to_domain() if self.recognition else None,
        )


class MeasurementResponse(BaseModel):
    id: str
    client_id: UUID
    systolic: int
    diastolic: int
    pulse: int
    measured_at: datetime
    created_at: datetime
    source: MeasurementSource
    recognition: RecognitionSchema | None

    @classmethod
    def from_domain(cls, measurement: Measurement) -> "MeasurementResponse":
        # Stored measurements always carry an id and a creation time.
        return cls(
            id=measurement.id or "",
            client_id=measurement.client_id,
            systolic=measurement.reading.systolic,
            diastolic=measurement.reading.diastolic,
            pulse=measurement.reading.pulse,
            measured_at=measurement.measured_at,
            created_at=measurement.created_at or measurement.measured_at,
            source=measurement.source,
            recognition=RecognitionSchema.from_domain(measurement.recognition) if measurement.recognition else None,
        )


class MeasurementPageResponse(BaseModel):
    items: list[MeasurementResponse]
    next_cursor: str | None

    @classmethod
    def from_domain(cls, listing: MeasurementsListing) -> "MeasurementPageResponse":
        return cls(
            items=[MeasurementResponse.from_domain(item) for item in listing.items],
            next_cursor=listing.next_cursor,
        )
