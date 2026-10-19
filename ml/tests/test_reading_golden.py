"""CA-B003-03: the official parser and validator pass every golden case of `specs/contracts/`."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tensoscan_ml.config import contracts_dir
from tensoscan_ml.reading import FrameStatus, SymbolDetection, parse_detections, validate_reading
from tensoscan_ml.reading.models import Box


def _load() -> dict[str, Any]:
    with (contracts_dir() / "reading-golden-cases.json").open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


_GOLDEN = _load()
_PARSER_CASES = [case for case in _GOLDEN["parser_cases"] if not case.get("deprecated")]
_VALIDATION_CASES = [case for case in _GOLDEN["validation_cases"] if not case.get("deprecated")]


def test_golden_file_is_the_expected_contract_version_and_is_not_empty() -> None:
    assert _GOLDEN["contract_version"].startswith("1.")
    assert len(_PARSER_CASES) >= 14
    assert len(_VALIDATION_CASES) >= 14


@pytest.mark.parametrize("case", _PARSER_CASES, ids=[case["name"] for case in _PARSER_CASES])
def test_parser_reproduces_the_golden_case(case: dict[str, Any]) -> None:
    detections = [SymbolDetection.from_dict(raw) for raw in case["detections"]]
    expected = case["expected"]

    actual = parse_detections(detections)

    assert actual.status.value == expected["status"]
    assert actual.values == (expected["systolic"], expected["diastolic"], expected["pulse"])


@pytest.mark.parametrize("case", _VALIDATION_CASES, ids=[case["name"] for case in _VALIDATION_CASES])
def test_validator_reproduces_the_golden_case(case: dict[str, Any]) -> None:
    given = case["input"]

    actual = validate_reading(given["systolic"], given["diastolic"], given["pulse"])

    assert [{"field": issue.field, "code": issue.code} for issue in actual.errors] == case["errors"]
    assert actual.warnings == case["warnings"]


def _detection(symbol: str, left: float, top: float, right: float, bottom: float, confidence: float = 0.9) -> Any:
    return SymbolDetection(symbol, Box(left, top, right, bottom), confidence)


def test_parser_confidence_of_a_field_is_the_mean_of_its_glyphs() -> None:
    detections = [
        _detection("1", 0.10, 0.1, 0.18, 0.4, 0.6),
        _detection("2", 0.20, 0.1, 0.30, 0.4, 0.8),
        _detection("0", 0.10, 0.5, 0.20, 0.8, 0.9),
        _detection("0", 0.22, 0.5, 0.32, 0.8, 0.9),
        _detection("7", 0.10, 0.82, 0.16, 0.97, 0.9),
        _detection("0", 0.18, 0.82, 0.24, 0.97, 0.9),
    ]

    actual = parse_detections(detections)

    assert actual.status is FrameStatus.COMPLETE
    assert actual.systolic is not None
    assert actual.systolic.confidence == pytest.approx(0.7)


def test_parser_never_swaps_systolic_and_diastolic() -> None:
    detections = [
        _detection("6", 0.1, 0.1, 0.2, 0.4),
        _detection("0", 0.22, 0.1, 0.32, 0.4),
        _detection("1", 0.1, 0.5, 0.15, 0.8),
        _detection("2", 0.17, 0.5, 0.27, 0.8),
        _detection("0", 0.29, 0.5, 0.39, 0.8),
    ]

    actual = parse_detections(detections)

    assert actual.systolic is not None and actual.diastolic is not None
    assert (actual.systolic.value, actual.diastolic.value) == (60, 120)


def test_validator_flags_every_blocking_error_in_the_documented_order() -> None:
    actual = validate_reading(None, 500, 10)

    assert [(i.field, i.code) for i in actual.errors] == [
        ("systolic", "REQUIRED"),
        ("diastolic", "OUT_OF_RANGE"),
        ("pulse", "OUT_OF_RANGE"),
    ]
    assert actual.warnings == []
    assert not actual.is_valid


def test_validator_dates_are_checked_only_when_saving() -> None:
    now = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)

    future = validate_reading(128, 82, 67, measured_at=now + timedelta(minutes=6), now=now)
    tolerated = validate_reading(128, 82, 67, measured_at=now + timedelta(minutes=5), now=now)
    old = validate_reading(128, 82, 67, measured_at=datetime(1999, 12, 31, 23, 59, tzinfo=UTC), now=now)

    assert [(i.field, i.code) for i in future.errors] == [("measured_at", "IN_FUTURE")]
    assert tolerated.errors == []
    assert [(i.field, i.code) for i in old.errors] == [("measured_at", "TOO_OLD")]


def test_validator_rejects_naive_datetimes() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_reading(128, 82, 67, measured_at=datetime(2026, 1, 1))  # noqa: DTZ001
