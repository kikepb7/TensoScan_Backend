"""Validation of a reading: exactly ``specs/contracts/reading-rules.md`` §2-3 (checked by the golden cases)."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import ClassVar

from app.domain.errors import FieldIssue
from app.domain.measurements.model import MeasurementSource, RecognitionInfo

REQUIRED = "REQUIRED"
OUT_OF_RANGE = "OUT_OF_RANGE"
SYSTOLIC_NOT_GREATER_THAN_DIASTOLIC = "SYSTOLIC_NOT_GREATER_THAN_DIASTOLIC"
IN_FUTURE = "IN_FUTURE"
TOO_OLD = "TOO_OLD"
NOT_ALLOWED = "NOT_ALLOWED"
TOO_LONG = "TOO_LONG"

MAX_MODEL_VERSION_LENGTH = 64


@dataclass(frozen=True)
class ReadingEvaluation:
    errors: list[FieldIssue]
    warnings: list[str]


class ReadingRules:
    RANGES: ClassVar[dict[str, tuple[int, int]]] = {
        "systolic": (60, 260),
        "diastolic": (30, 160),
        "pulse": (30, 220),
    }
    FUTURE_TOLERANCE: ClassVar[timedelta] = timedelta(minutes=5)
    EARLIEST_MEASURED_AT: ClassVar[datetime] = datetime(2000, 1, 1, tzinfo=UTC)

    @classmethod
    def evaluate(cls, systolic: int | None, diastolic: int | None, pulse: int | None) -> ReadingEvaluation:
        """Errors (§3.1, block saving) and, only when there are none, warnings (§3.2, never block)."""
        values = {"systolic": systolic, "diastolic": diastolic, "pulse": pulse}
        errors: list[FieldIssue] = []
        for name, (low, high) in cls.RANGES.items():
            value = values[name]
            if value is None:
                errors.append(FieldIssue(name, REQUIRED))
            elif not low <= value <= high:
                errors.append(FieldIssue(name, OUT_OF_RANGE))
        blood_pressure_ok = not any(issue.field in ("systolic", "diastolic") for issue in errors)
        if blood_pressure_ok and systolic is not None and diastolic is not None and systolic <= diastolic:
            errors.append(FieldIssue("systolic", SYSTOLIC_NOT_GREATER_THAN_DIASTOLIC))
        if errors or systolic is None or diastolic is None or pulse is None:
            return ReadingEvaluation(errors, [])
        return ReadingEvaluation(errors, cls._warnings(systolic, diastolic, pulse))

    @staticmethod
    def _warnings(systolic: int, diastolic: int, pulse: int) -> list[str]:
        warnings: list[str] = []
        if systolic >= 180 or diastolic >= 120:
            warnings.append("VERY_HIGH_PRESSURE")
        if systolic < 90 or diastolic < 60:
            warnings.append("LOW_PRESSURE")
        if pulse < 40 or pulse > 150:
            warnings.append("UNUSUAL_PULSE")
        pulse_pressure = systolic - diastolic
        if pulse_pressure < 20 or pulse_pressure > 100:
            warnings.append("UNUSUAL_PULSE_PRESSURE")
        return warnings

    @classmethod
    def validate_measured_at(cls, measured_at: datetime, now: datetime) -> list[FieldIssue]:
        if measured_at > now + cls.FUTURE_TOLERANCE:
            return [FieldIssue("measured_at", IN_FUTURE)]
        if measured_at < cls.EARLIEST_MEASURED_AT:
            return [FieldIssue("measured_at", TOO_OLD)]
        return []

    @staticmethod
    def validate_recognition(source: MeasurementSource, recognition: RecognitionInfo | None) -> list[FieldIssue]:
        """``recognition`` is mandatory for ``camera`` and forbidden for ``manual`` (api-v1.md §3.1)."""
        if recognition is None:
            return [FieldIssue("recognition", REQUIRED)] if source == "camera" else []
        if source == "manual":
            return [FieldIssue("recognition", NOT_ALLOWED)]
        issues: list[FieldIssue] = []
        if not 0.0 <= recognition.confidence <= 1.0:  # also false for NaN
            issues.append(FieldIssue("recognition.confidence", OUT_OF_RANGE))
        if not recognition.model_version:
            issues.append(FieldIssue("recognition.model_version", REQUIRED))
        elif len(recognition.model_version) > MAX_MODEL_VERSION_LENGTH:
            issues.append(FieldIssue("recognition.model_version", TOO_LONG))
        return issues
