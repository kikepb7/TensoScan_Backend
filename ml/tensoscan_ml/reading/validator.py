"""Reading validation (`reading-rules.md` sections 2 and 3).

Errors block saving; warnings allow it after an explicit confirmation and are only evaluated when there are no errors.
`POSSIBLE_DUPLICATE` belongs to the app (it needs the stored history) and is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Final

SYSTOLIC_RANGE: Final = (60, 260)
DIASTOLIC_RANGE: Final = (30, 160)
PULSE_RANGE: Final = (30, 220)

FUTURE_TOLERANCE: Final = timedelta(minutes=5)
OLDEST_MEASUREMENT: Final = datetime(2000, 1, 1, tzinfo=UTC)

# Warning thresholds (section 3.2).
VERY_HIGH_SYSTOLIC: Final = 180
VERY_HIGH_DIASTOLIC: Final = 120
LOW_SYSTOLIC: Final = 90
LOW_DIASTOLIC: Final = 60
UNUSUAL_PULSE_LOW: Final = 40
UNUSUAL_PULSE_HIGH: Final = 150
PULSE_PRESSURE_LOW: Final = 20
PULSE_PRESSURE_HIGH: Final = 100

_RANGES: Final = {"systolic": SYSTOLIC_RANGE, "diastolic": DIASTOLIC_RANGE, "pulse": PULSE_RANGE}


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    code: str


@dataclass(frozen=True)
class ValidationResult:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_reading(
    systolic: int | None,
    diastolic: int | None,
    pulse: int | None,
    *,
    measured_at: datetime | None = None,
    now: datetime | None = None,
) -> ValidationResult:
    """Validates a reading. `measured_at` is checked only when given (that is, when saving)."""
    values = {"systolic": systolic, "diastolic": diastolic, "pulse": pulse}
    errors: list[ValidationIssue] = []
    failed: set[str] = set()
    for name, value in values.items():
        if value is None:
            errors.append(ValidationIssue(name, "REQUIRED"))
            failed.add(name)
            continue
        low, high = _RANGES[name]
        if not low <= value <= high:
            errors.append(ValidationIssue(name, "OUT_OF_RANGE"))
            failed.add(name)

    pair_checked = not ({"systolic", "diastolic"} & failed)
    if pair_checked and systolic is not None and diastolic is not None and systolic <= diastolic:
        errors.append(ValidationIssue("systolic", "SYSTOLIC_NOT_GREATER_THAN_DIASTOLIC"))

    if measured_at is not None:
        errors.extend(_date_errors(measured_at, now or datetime.now(UTC)))

    if errors or systolic is None or diastolic is None or pulse is None:
        return ValidationResult(errors=errors)
    return ValidationResult(warnings=_warnings(systolic, diastolic, pulse))


def _date_errors(measured_at: datetime, now: datetime) -> list[ValidationIssue]:
    if measured_at.tzinfo is None:
        raise ValueError("measured_at must be timezone-aware (C-API-03)")
    issues: list[ValidationIssue] = []
    if measured_at > now + FUTURE_TOLERANCE:
        issues.append(ValidationIssue("measured_at", "IN_FUTURE"))
    if measured_at < OLDEST_MEASUREMENT:
        issues.append(ValidationIssue("measured_at", "TOO_OLD"))
    return issues


def _warnings(systolic: int, diastolic: int, pulse: int) -> list[str]:
    warnings: list[str] = []
    if systolic >= VERY_HIGH_SYSTOLIC or diastolic >= VERY_HIGH_DIASTOLIC:
        warnings.append("VERY_HIGH_PRESSURE")
    if systolic < LOW_SYSTOLIC or diastolic < LOW_DIASTOLIC:
        warnings.append("LOW_PRESSURE")
    if pulse < UNUSUAL_PULSE_LOW or pulse > UNUSUAL_PULSE_HIGH:
        warnings.append("UNUSUAL_PULSE")
    pulse_pressure = systolic - diastolic
    if pulse_pressure < PULSE_PRESSURE_LOW or pulse_pressure > PULSE_PRESSURE_HIGH:
        warnings.append("UNUSUAL_PULSE_PRESSURE")
    return warnings
