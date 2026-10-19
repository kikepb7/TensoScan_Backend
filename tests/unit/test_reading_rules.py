"""``ReadingRules`` against the shared golden cases (``specs/contracts/reading-golden-cases.json``, never copied)."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app.domain.errors import FieldIssue
from app.domain.measurements.model import RecognitionInfo
from app.domain.measurements.rules import ReadingRules

GOLDEN = Path(__file__).resolve().parents[2] / "specs" / "contracts" / "reading-golden-cases.json"
VALIDATION_CASES: list[dict[str, Any]] = json.loads(GOLDEN.read_text())["validation_cases"]
NOW = datetime(2026, 10, 19, 8, 0, tzinfo=UTC)


def test_the_golden_file_provides_validation_cases() -> None:
    assert len(VALIDATION_CASES) >= 14


@pytest.mark.parametrize("case", VALIDATION_CASES, ids=[case["name"] for case in VALIDATION_CASES])
def test_golden_validation_cases_match_errors_and_warnings(case: dict[str, Any]) -> None:
    values = case["input"]
    expected_errors = [FieldIssue(error["field"], error["code"]) for error in case["errors"]]

    actual = ReadingRules.evaluate(values["systolic"], values["diastolic"], values["pulse"])

    assert actual.errors == expected_errors
    assert actual.warnings == case["warnings"]


def test_every_missing_value_is_reported_in_field_order() -> None:
    expected = [
        FieldIssue("systolic", "REQUIRED"),
        FieldIssue("diastolic", "REQUIRED"),
        FieldIssue("pulse", "REQUIRED"),
    ]

    actual = ReadingRules.evaluate(None, None, None).errors

    assert actual == expected


def test_all_errors_are_returned_not_only_the_first() -> None:
    expected = [FieldIssue("systolic", "OUT_OF_RANGE"), FieldIssue("pulse", "OUT_OF_RANGE")]

    actual = ReadingRules.evaluate(300, 80, 10).errors

    assert actual == expected


def test_the_systolic_diastolic_check_is_skipped_when_either_has_its_own_error() -> None:
    expected = [FieldIssue("diastolic", "OUT_OF_RANGE")]

    actual = ReadingRules.evaluate(100, 200, 70).errors

    assert actual == expected


def test_the_systolic_diastolic_check_still_applies_when_only_the_pulse_is_wrong() -> None:
    expected = [FieldIssue("pulse", "REQUIRED"), FieldIssue("systolic", "SYSTOLIC_NOT_GREATER_THAN_DIASTOLIC")]

    actual = ReadingRules.evaluate(80, 80, None).errors

    assert actual == expected


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (timedelta(0), []),
        (timedelta(minutes=5), []),
        (timedelta(minutes=5, seconds=1), [FieldIssue("measured_at", "IN_FUTURE")]),
        (timedelta(days=1), [FieldIssue("measured_at", "IN_FUTURE")]),
        (-timedelta(days=30), []),
    ],
)
def test_measured_at_may_be_at_most_five_minutes_ahead(delta: timedelta, expected: list[FieldIssue]) -> None:
    actual = ReadingRules.validate_measured_at(NOW + delta, NOW)

    assert actual == expected


@pytest.mark.parametrize(
    ("measured_at", "expected"),
    [
        (datetime(2000, 1, 1, tzinfo=UTC), []),
        (datetime(1999, 12, 31, 23, 59, 59, tzinfo=UTC), [FieldIssue("measured_at", "TOO_OLD")]),
        # 2000-01-01T00:30+02:00 is still 1999-12-31 in UTC.
        (datetime(2000, 1, 1, 0, 30, tzinfo=UTC) - timedelta(hours=2), [FieldIssue("measured_at", "TOO_OLD")]),
    ],
)
def test_measured_at_may_not_precede_the_year_2000_in_utc(measured_at: datetime, expected: list[FieldIssue]) -> None:
    actual = ReadingRules.validate_measured_at(measured_at, NOW)

    assert actual == expected


def test_a_camera_measurement_requires_recognition() -> None:
    expected = [FieldIssue("recognition", "REQUIRED")]

    actual = ReadingRules.validate_recognition("camera", None)

    assert actual == expected


def test_a_manual_measurement_forbids_recognition() -> None:
    expected = [FieldIssue("recognition", "NOT_ALLOWED")]

    actual = ReadingRules.validate_recognition("manual", RecognitionInfo(0.9, "digits-1.0.0", frozenset()))

    assert actual == expected


def test_a_manual_measurement_without_recognition_is_coherent() -> None:
    assert ReadingRules.validate_recognition("manual", None) == []


def test_a_coherent_camera_measurement_is_accepted() -> None:
    assert (
        ReadingRules.validate_recognition("camera", RecognitionInfo(0.0, "mlkit-text-v2", frozenset({"pulse"}))) == []
    )


@pytest.mark.parametrize(
    ("recognition", "expected"),
    [
        (RecognitionInfo(1.01, "m", frozenset()), [FieldIssue("recognition.confidence", "OUT_OF_RANGE")]),
        (RecognitionInfo(-0.01, "m", frozenset()), [FieldIssue("recognition.confidence", "OUT_OF_RANGE")]),
        (RecognitionInfo(float("nan"), "m", frozenset()), [FieldIssue("recognition.confidence", "OUT_OF_RANGE")]),
        (RecognitionInfo(0.5, "", frozenset()), [FieldIssue("recognition.model_version", "REQUIRED")]),
        (RecognitionInfo(0.5, "v" * 65, frozenset()), [FieldIssue("recognition.model_version", "TOO_LONG")]),
        (RecognitionInfo(0.5, "v" * 64, frozenset()), []),
    ],
)
def test_recognition_limits_follow_the_contract(recognition: RecognitionInfo, expected: list[FieldIssue]) -> None:
    actual = ReadingRules.validate_recognition("camera", recognition)

    assert actual == expected
