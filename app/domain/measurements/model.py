from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

MeasurementSource = Literal["camera", "manual"]
ReadingField = Literal["systolic", "diastolic", "pulse"]

SOURCES: tuple[MeasurementSource, ...] = ("camera", "manual")
READING_FIELDS: tuple[ReadingField, ...] = ("systolic", "diastolic", "pulse")


def as_utc(value: datetime) -> datetime:
    """Aware datetimes are converted to UTC; naive ones are assumed to already be UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True)
class Reading:
    """The three values of one measurement. Range rules live in ``ReadingRules``, not in the constructor, so that the
    documented legacy exception (display-recognize skips validation) can still build one."""

    systolic: int
    diastolic: int
    pulse: int


@dataclass(frozen=True)
class RecognitionInfo:
    confidence: float
    model_version: str
    edited_fields: frozenset[ReadingField]


@dataclass(frozen=True)
class Measurement:
    id: str | None
    user_id: str
    client_id: UUID
    reading: Reading
    measured_at: datetime
    created_at: datetime | None
    source: MeasurementSource
    recognition: RecognitionInfo | None


@dataclass(frozen=True)
class MeasurementPage:
    """One page of a keyset-paginated listing; ``has_more`` says whether another page follows."""

    items: list[Measurement]
    has_more: bool
