import base64
import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.domain.errors import FieldIssue, ValidationFailed
from app.domain.measurements.cursor import PageCursor, is_valid_measurement_id

OBJECT_ID = "6713a1b2c3d4e5f600112233"


def test_a_cursor_survives_an_encode_decode_round_trip() -> None:
    expected = PageCursor(datetime(2026, 10, 19, 6, 15, 30, 123000, tzinfo=UTC), OBJECT_ID)

    actual = PageCursor.decode(expected.encode())

    assert actual == expected


def test_the_cursor_is_url_safe_and_unpadded() -> None:
    token = PageCursor(datetime(2026, 10, 19, tzinfo=UTC), OBJECT_ID).encode()

    assert "=" not in token and "+" not in token and "/" not in token


def test_a_cursor_with_an_offset_is_normalised_to_utc() -> None:
    madrid = datetime(2026, 10, 19, 8, 15, tzinfo=timezone(timedelta(hours=2)))

    actual = PageCursor.decode(PageCursor(madrid, OBJECT_ID).encode())

    assert actual.measured_at == datetime(2026, 10, 19, 6, 15, tzinfo=UTC)
    assert actual.measured_at.utcoffset() == timedelta(0)


def _token(payload: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")


@pytest.mark.parametrize(
    "token",
    [
        "",
        "not base64 at all!",
        "%%%%",
        base64.urlsafe_b64encode(b"not json").decode(),
        _token([1, 2]),
        _token("text"),
        _token({"m": "2026-10-19T06:15:00+00:00"}),
        _token({"i": OBJECT_ID}),
        _token({"m": "yesterday", "i": OBJECT_ID}),
        _token({"m": 12345, "i": OBJECT_ID}),
        _token({"m": "2026-10-19T06:15:00", "i": OBJECT_ID}),  # naive
        _token({"m": "2026-10-19T06:15:00+00:00", "i": "short"}),
        _token({"m": "2026-10-19T06:15:00+00:00", "i": 7}),
    ],
)
def test_garbage_is_rejected_as_a_validation_error_on_the_cursor_field(token: str) -> None:
    with pytest.raises(ValidationFailed) as raised:
        PageCursor.decode(token)

    assert raised.value.details == [FieldIssue("cursor", "INVALID")]


def test_measurement_ids_are_object_ids() -> None:
    assert is_valid_measurement_id(OBJECT_ID)
    assert not is_valid_measurement_id("not-an-id")
    assert not is_valid_measurement_id(OBJECT_ID + "0")
