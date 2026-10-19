from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

from app.domain.measurements.canonical import canonical_payload, hash_payload, payload_hash
from app.domain.measurements.model import Measurement, Reading, RecognitionInfo

CLIENT_ID = UUID("5b0b6f0e-6a1f-4d8e-9d0a-2f1c3b7e9a10")


def _measurement(**overrides: object) -> Measurement:
    base = Measurement(
        id=None,
        user_id="u1",
        client_id=CLIENT_ID,
        reading=Reading(128, 82, 67),
        measured_at=datetime(2026, 10, 19, 6, 15, tzinfo=UTC),
        created_at=None,
        source="camera",
        recognition=RecognitionInfo(0.93, "digits-1.0.0", frozenset({"pulse", "systolic"})),
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_the_payload_has_the_documented_shape() -> None:
    expected = {
        "client_id": "5b0b6f0e-6a1f-4d8e-9d0a-2f1c3b7e9a10",
        "systolic": 128,
        "diastolic": 82,
        "pulse": 67,
        "measured_at": "2026-10-19T06:15:00Z",
        "source": "camera",
        "recognition": {"confidence": 0.93, "model_version": "digits-1.0.0", "edited_fields": ["pulse", "systolic"]},
    }

    actual = canonical_payload(_measurement())

    assert actual == expected


def test_the_hash_does_not_depend_on_the_order_of_the_keys() -> None:
    payload = canonical_payload(_measurement())
    shuffled = dict(reversed(list(payload.items())))
    shuffled["recognition"] = dict(reversed(list(payload["recognition"].items())))

    assert list(shuffled) != list(payload)
    assert hash_payload(shuffled) == hash_payload(payload)


def test_the_hash_does_not_depend_on_the_order_of_edited_fields() -> None:
    first = _measurement(recognition=RecognitionInfo(0.93, "m", frozenset(["pulse", "diastolic", "systolic"])))
    second = _measurement(recognition=RecognitionInfo(0.93, "m", frozenset(["systolic", "pulse", "diastolic"])))

    assert payload_hash(first) == payload_hash(second)
    assert canonical_payload(first)["recognition"]["edited_fields"] == ["diastolic", "pulse", "systolic"]


def test_the_same_instant_in_another_offset_has_the_same_hash() -> None:
    madrid = timezone(timedelta(hours=2))
    local = _measurement(measured_at=datetime(2026, 10, 19, 8, 15, tzinfo=madrid))

    assert payload_hash(local) == payload_hash(_measurement())


def test_sub_second_precision_and_confidence_noise_are_ignored() -> None:
    noisy = _measurement(
        measured_at=datetime(2026, 10, 19, 6, 15, 0, 987654, tzinfo=UTC),
        recognition=RecognitionInfo(0.930001, "digits-1.0.0", frozenset({"pulse", "systolic"})),
    )

    assert payload_hash(noisy) == payload_hash(_measurement())


def test_any_meaningful_change_alters_the_hash() -> None:
    reference = payload_hash(_measurement())
    variants = [
        _measurement(reading=Reading(128, 82, 68)),
        _measurement(measured_at=datetime(2026, 10, 19, 6, 16, tzinfo=UTC)),
        _measurement(source="manual", recognition=None),
        _measurement(recognition=RecognitionInfo(0.93, "digits-1.0.1", frozenset({"pulse", "systolic"}))),
        _measurement(recognition=RecognitionInfo(0.93, "digits-1.0.0", frozenset({"pulse"}))),
    ]

    assert all(payload_hash(variant) != reference for variant in variants)


def test_server_side_fields_are_not_part_of_the_payload() -> None:
    stored = _measurement(id="abc", user_id="other", created_at=datetime(2026, 1, 1, tzinfo=UTC))

    assert payload_hash(stored) == payload_hash(_measurement())
