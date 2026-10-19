"""Data handed to a ``ReportRenderer``: everything already computed, so renderers only draw."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, tzinfo

from app.domain.measurements.model import Measurement


@dataclass(frozen=True)
class FieldStats:
    mean: float
    minimum: int
    maximum: int


@dataclass(frozen=True)
class ReportSummary:
    count: int
    systolic: FieldStats
    diastolic: FieldStats
    pulse: FieldStats

    @classmethod
    def of(cls, measurements: Sequence[Measurement]) -> "ReportSummary":
        """Raises ``ValueError`` for an empty sequence: there is nothing to summarise."""
        if not measurements:
            raise ValueError("A summary needs at least one measurement")

        def stats(values: list[int]) -> FieldStats:
            return FieldStats(sum(values) / len(values), min(values), max(values))

        return cls(
            count=len(measurements),
            systolic=stats([m.reading.systolic for m in measurements]),
            diastolic=stats([m.reading.diastolic for m in measurements]),
            pulse=stats([m.reading.pulse for m in measurements]),
        )


@dataclass(frozen=True)
class MeasurementsReport:
    user_name: str
    generated_at: datetime
    timezone: tzinfo
    range_from: datetime | None
    range_to: datetime | None
    measurements: Sequence[Measurement]  # oldest first
    summary: ReportSummary
