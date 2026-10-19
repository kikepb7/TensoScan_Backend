"""Migration m001 (``recognition_results`` -> ``measurements``) on the ``db`` fixture, DST cases and a second run."""

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid5

import httpx
import pytest
from bson import ObjectId

from app.domain.measurements.model import Measurement, Reading, RecognitionInfo
from app.infrastructure.mongo.measurement_repository import MongoMeasurementRepository
from app.infrastructure.mongo.migrations import m001_measurements as m001
from app.infrastructure.mongo.migrations.m001_measurements import (
    NAMESPACE_TENSOSCAN,
    localize_legacy_time,
    run_migration,
)
from tests.conftest import bearer, register_and_login

RUN_AT = datetime(2026, 10, 19, 9, 0, tzinfo=UTC)
USER = ObjectId()


def legacy(
    high: Any,
    low: Any,
    pulse: Any,
    timestamp: Any = datetime(2026, 1, 1, 12),
    *,
    user: Any = None,
    confidence: Any = 0.91,
) -> dict[str, Any]:
    result = {"high_pressure": high, "low_pressure": low, "pulse": pulse, "confidence": confidence}
    return {
        "user_id": USER if user is None else user,
        "filename": "photo.jpg",
        "result": result,
        "timestamp": timestamp,
    }


async def seed(db: Any, *documents: dict[str, Any]) -> list[ObjectId]:
    ids = []
    for document in documents:
        ids.append((await db["recognition_results"].insert_one(document)).inserted_id)
    return ids


async def test_valid_documents_are_copied_and_invalid_ones_are_reported_not_migrated(db: Any) -> None:
    good, bad_letter, empty, too_high, missing_value = await seed(
        db,
        legacy("128", "82", "67", datetime(2026, 1, 15, 12, 0)),
        legacy("12O", "82", "67", datetime(2026, 1, 15, 12, 0)),  # letter O instead of zero
        legacy("", "", "", datetime(2026, 1, 15, 12, 0)),
        legacy("300", "82", "67", datetime(2026, 1, 15, 12, 0)),
        {"user_id": USER, "result": {"high_pressure": "120"}, "timestamp": datetime(2026, 1, 15)},
    )

    report = await run_migration(db, now=RUN_AT)

    assert (report.scanned, report.migrated, report.already_migrated) == (5, 1, 0)
    assert report.skipped == [
        {"legacy_id": str(bad_letter), "reason": "NOT_NUMERIC:high_pressure"},
        {"legacy_id": str(empty), "reason": "NOT_NUMERIC:high_pressure"},
        {"legacy_id": str(too_high), "reason": "OUT_OF_RANGE:systolic"},
        {"legacy_id": str(missing_value), "reason": "NOT_NUMERIC:low_pressure"},
    ]
    stored = await db["measurements"].find({}).to_list(length=None)
    assert len(stored) == 1
    assert stored[0]["legacy_id"] == good


async def test_a_migrated_document_has_the_documented_shape(db: Any) -> None:
    (legacy_id,) = await seed(db, legacy("128", "82", "67", datetime(2026, 1, 15, 12, 0), confidence=0.87))

    await run_migration(db, now=RUN_AT)

    stored = await db["measurements"].find_one({})
    assert stored is not None
    assert stored["user_id"] == USER
    assert stored["client_id"] == str(uuid5(NAMESPACE_TENSOSCAN, str(legacy_id)))
    assert (stored["systolic"], stored["diastolic"], stored["pulse"]) == (128, 82, 67)  # integers, not strings
    assert stored["source"] == "camera"
    assert stored["recognition"] == {"confidence": 0.87, "model_version": "legacy-keras", "edited_fields": []}
    # 12:00 in Madrid in January (CET, UTC+1) is 11:00 UTC, for both instants.
    expected = datetime(2026, 1, 15, 11, 0, tzinfo=UTC)
    assert stored["measured_at"].astimezone(UTC) == expected
    assert stored["created_at"].astimezone(UTC) == expected
    assert stored["legacy_id"] == legacy_id
    assert len(stored["payload_hash"]) == 64


async def test_running_the_migration_twice_creates_no_duplicates(db: Any) -> None:
    await seed(
        db,
        legacy("128", "82", "67", datetime(2026, 1, 15, 12, 0)),
        legacy("135", "85", "70", datetime(2026, 2, 1, 8, 30)),
        legacy("12O", "82", "67", datetime(2026, 2, 2, 8, 30)),
    )

    first = await run_migration(db, now=RUN_AT)
    second = await run_migration(db, now=RUN_AT)

    assert (first.migrated, first.already_migrated) == (2, 0)
    assert (second.migrated, second.already_migrated) == (0, 2)
    assert [item["legacy_id"] for item in second.skipped] == [item["legacy_id"] for item in first.skipped]
    assert await db["measurements"].count_documents({}) == 2


async def test_a_new_legacy_document_is_picked_up_by_a_later_run(db: Any) -> None:
    await seed(db, legacy("128", "82", "67", datetime(2026, 1, 15, 12, 0)))
    await run_migration(db, now=RUN_AT)
    await seed(db, legacy("130", "80", "65", datetime(2026, 1, 16, 12, 0)))

    report = await run_migration(db, now=RUN_AT)

    assert (report.migrated, report.already_migrated) == (1, 1)
    assert await db["measurements"].count_documents({}) == 2


async def test_the_legacy_collection_is_left_untouched(db: Any) -> None:
    await seed(
        db,
        legacy("128", "82", "67", datetime(2026, 1, 15, 12, 0)),
        legacy("12O", "82", "67", datetime(2026, 1, 15, 12, 0)),
    )
    before = await db["recognition_results"].find({}).sort("_id", 1).to_list(length=None)

    await run_migration(db, now=RUN_AT)
    await run_migration(db, now=RUN_AT)

    assert await db["recognition_results"].find({}).sort("_id", 1).to_list(length=None) == before


async def test_a_dry_run_reports_but_writes_nothing(db: Any) -> None:
    await seed(
        db,
        legacy("128", "82", "67", datetime(2026, 1, 15, 12, 0)),
        legacy("300", "82", "67", datetime(2026, 1, 15, 12, 0)),
    )

    report = await run_migration(db, dry_run=True, now=RUN_AT)

    assert (report.dry_run, report.migrated, len(report.skipped)) == (True, 1, 1)
    assert await db["measurements"].count_documents({}) == 0
    assert await db["migration_reports"].count_documents({}) == 0


async def test_a_dry_run_after_the_real_run_sees_everything_as_already_migrated(db: Any) -> None:
    await seed(db, legacy("128", "82", "67", datetime(2026, 1, 15, 12, 0)))
    await run_migration(db, now=RUN_AT)

    report = await run_migration(db, dry_run=True, now=RUN_AT)

    assert (report.migrated, report.already_migrated) == (0, 1)


async def test_the_report_is_stored_in_migration_reports(db: Any) -> None:
    (bad,) = await seed(db, legacy("12O", "82", "67", datetime(2026, 1, 15, 12, 0)))
    await seed(db, legacy("128", "82", "67", datetime(2026, 1, 15, 12, 0)))

    await run_migration(db, now=RUN_AT)

    stored = await db["migration_reports"].find_one({})
    assert stored is not None
    assert stored["migration"] == "m001"
    assert stored["ran_at"].astimezone(UTC) == RUN_AT
    assert (stored["scanned"], stored["migrated"], stored["legacy_tz"]) == (2, 1, "Europe/Madrid")
    assert stored["skipped"] == [{"legacy_id": str(bad), "reason": "NOT_NUMERIC:high_pressure"}]
    assert {"ambiguous_local_times", "nonexistent_local_times"} <= stored.keys()


async def test_the_report_is_logged_without_any_reading_value(db: Any, caplog: pytest.LogCaptureFixture) -> None:
    await seed(
        db, legacy("128", "82", "67", datetime(2026, 1, 15, 12, 0)), legacy("12O", "82", "67", datetime(2026, 1, 15))
    )
    caplog.set_level("INFO", logger=m001.logger.name)

    await run_migration(db, now=RUN_AT)

    line = next(record.getMessage() for record in caplog.records if "finished" in record.getMessage())
    logged = json.loads(line.split(": ", 1)[1])
    assert logged["migrated"] == 1 and logged["skipped"][0]["reason"] == "NOT_NUMERIC:high_pressure"
    assert "128" not in line and "12O" not in line


# --- daylight saving time (Europe/Madrid, 2026: spring forward on 29 March, fall back on 25 October) -----------------


async def test_the_ambiguous_autumn_hour_takes_the_first_occurrence_and_is_counted(db: Any) -> None:
    await seed(db, legacy("128", "82", "67", datetime(2026, 10, 25, 2, 30)))

    report = await run_migration(db, now=RUN_AT)

    stored = await db["measurements"].find_one({})
    assert stored is not None
    assert stored["measured_at"].astimezone(UTC) == datetime(2026, 10, 25, 0, 30, tzinfo=UTC)  # 02:30 CEST (UTC+2)
    assert (report.ambiguous_local_times, report.nonexistent_local_times) == (1, 0)


async def test_the_nonexistent_spring_hour_moves_one_hour_forward_and_is_counted(db: Any) -> None:
    await seed(db, legacy("128", "82", "67", datetime(2026, 3, 29, 2, 30)))

    report = await run_migration(db, now=RUN_AT)

    stored = await db["measurements"].find_one({})
    assert stored is not None
    assert stored["measured_at"].astimezone(UTC) == datetime(2026, 3, 29, 1, 30, tzinfo=UTC)  # 03:30 CEST (UTC+2)
    assert (report.ambiguous_local_times, report.nonexistent_local_times) == (0, 1)


async def test_ordinary_times_around_the_changes_are_not_flagged(db: Any) -> None:
    await seed(
        db,
        legacy("120", "80", "60", datetime(2026, 3, 29, 1, 59)),
        legacy("121", "80", "60", datetime(2026, 3, 29, 3, 0)),
        legacy("122", "80", "60", datetime(2026, 10, 25, 1, 59)),
        legacy("123", "80", "60", datetime(2026, 10, 25, 3, 0)),
    )

    report = await run_migration(db, now=RUN_AT)

    assert (report.migrated, report.ambiguous_local_times, report.nonexistent_local_times) == (4, 0, 0)


@pytest.mark.parametrize(
    ("wall_clock", "expected_utc", "expected_issue"),
    [
        (datetime(2026, 7, 1, 12, 0), datetime(2026, 7, 1, 10, 0, tzinfo=UTC), None),
        (datetime(2026, 1, 1, 12, 0), datetime(2026, 1, 1, 11, 0, tzinfo=UTC), None),
        (datetime(2026, 3, 29, 1, 59, 59), datetime(2026, 3, 29, 0, 59, 59, tzinfo=UTC), None),
        (datetime(2026, 3, 29, 2, 0), datetime(2026, 3, 29, 1, 0, tzinfo=UTC), "nonexistent"),
        (datetime(2026, 3, 29, 2, 59, 59), datetime(2026, 3, 29, 1, 59, 59, tzinfo=UTC), "nonexistent"),
        (datetime(2026, 3, 29, 3, 0), datetime(2026, 3, 29, 1, 0, tzinfo=UTC), None),
        (datetime(2026, 10, 25, 1, 59, 59), datetime(2026, 10, 24, 23, 59, 59, tzinfo=UTC), None),
        (datetime(2026, 10, 25, 2, 0), datetime(2026, 10, 25, 0, 0, tzinfo=UTC), "ambiguous"),
        (datetime(2026, 10, 25, 2, 59, 59), datetime(2026, 10, 25, 0, 59, 59, tzinfo=UTC), "ambiguous"),
        (datetime(2026, 10, 25, 3, 0), datetime(2026, 10, 25, 2, 0, tzinfo=UTC), None),
    ],
)
def test_localizing_around_the_madrid_changes(
    wall_clock: datetime, expected_utc: datetime, expected_issue: str | None
) -> None:
    from zoneinfo import ZoneInfo

    actual = localize_legacy_time(wall_clock, ZoneInfo("Europe/Madrid"))

    assert (actual.utc, actual.issue) == (expected_utc, expected_issue)


async def test_the_legacy_zone_can_be_chosen(db: Any) -> None:
    await seed(db, legacy("128", "82", "67", datetime(2026, 1, 15, 12, 0)))

    await run_migration(db, legacy_tz="America/New_York", now=RUN_AT)

    stored = await db["measurements"].find_one({})
    assert stored is not None
    assert stored["measured_at"].astimezone(UTC) == datetime(2026, 1, 15, 17, 0, tzinfo=UTC)  # EST: UTC-5


# --- other legacy shapes -------------------------------------------------------------------------------------------


async def test_odd_but_recoverable_legacy_documents_are_migrated(db: Any) -> None:
    owner_text = str(ObjectId())
    await seed(
        db,
        legacy(" 128 ", 82, "067", "2026-01-15T12:00:00"),  # padded, integer and zero-led values; ISO string time
        legacy("120", "80", "60", datetime(2026, 1, 16, 12), confidence=7),  # confidence out of range is clamped
        legacy("121", "80", "60", datetime(2026, 1, 17, 12), user=owner_text),
        legacy("122", "80", "60", "2026-01-18T12:00:00+00:00"),  # string that already carries its offset
    )

    report = await run_migration(db, now=RUN_AT)

    assert (report.migrated, report.skipped) == (4, [])
    by_systolic = {d["systolic"]: d async for d in db["measurements"].find({})}
    assert by_systolic[128]["pulse"] == 67 and by_systolic[128]["diastolic"] == 82
    assert by_systolic[128]["measured_at"].astimezone(UTC) == datetime(2026, 1, 15, 11, 0, tzinfo=UTC)
    assert by_systolic[120]["recognition"]["confidence"] == 1.0
    assert by_systolic[121]["user_id"] == ObjectId(owner_text)
    assert by_systolic[122]["measured_at"].astimezone(UTC) == datetime(2026, 1, 18, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        (
            {"result": {"high_pressure": "1", "low_pressure": "1", "pulse": "1"}, "timestamp": datetime(2026, 1, 1)},
            "MISSING_USER",
        ),
        ({"user_id": USER, "timestamp": datetime(2026, 1, 1)}, "MISSING_RESULT"),
        (legacy("120", "80", "60", None), "INVALID_TIMESTAMP"),
        (legacy("120", "80", "60", "yesterday"), "INVALID_TIMESTAMP"),
        (legacy("120", "80", True), "NOT_NUMERIC:pulse"),
        (legacy("70", "80", "60"), "SYSTOLIC_NOT_GREATER_THAN_DIASTOLIC:systolic"),
        (
            legacy("20", "20", "600", datetime(2026, 1, 1)),
            "OUT_OF_RANGE:systolic,OUT_OF_RANGE:diastolic,OUT_OF_RANGE:pulse",
        ),
    ],
)
async def test_documents_that_cannot_be_migrated_are_skipped_with_their_reason(
    db: Any, document: dict[str, Any], reason: str
) -> None:
    document.setdefault("timestamp", datetime(2026, 1, 1))
    (legacy_id,) = await seed(db, document)

    report = await run_migration(db, now=RUN_AT)

    assert report.skipped == [{"legacy_id": str(legacy_id), "reason": reason}]
    assert await db["measurements"].count_documents({}) == 0


# --- the migrated data in the API ----------------------------------------------------------------------------------


async def test_migrated_measurements_show_up_in_v1_and_a_reconciling_insert_is_a_replay(
    client: httpx.AsyncClient, db: Any
) -> None:
    tokens = await register_and_login(client)
    owner = ObjectId(tokens["user"]["id"])
    (legacy_id,) = await seed(db, legacy("128", "82", "67", datetime(2026, 1, 15, 12, 0), user=owner, confidence=0.9))
    await run_migration(db, now=RUN_AT)
    client_id = uuid5(NAMESPACE_TENSOSCAN, str(legacy_id))

    listing = await client.get("/v1/measurements", headers=bearer(tokens))
    # The same measurement expressed as the API would receive it: the stored payload hash must match (no 409).
    same = Measurement(
        id=None,
        user_id=str(owner),
        client_id=client_id,
        reading=Reading(128, 82, 67),
        measured_at=datetime(2026, 1, 15, 11, 0, tzinfo=UTC),
        created_at=None,
        source="camera",
        recognition=RecognitionInfo(0.9, "legacy-keras", frozenset()),
    )
    replayed, created = await MongoMeasurementRepository(db).add_idempotent(same, created_at=RUN_AT)

    [item] = listing.json()["items"]
    assert (item["client_id"], item["systolic"], item["source"]) == (str(client_id), 128, "camera")
    assert item["measured_at"] == item["created_at"] == "2026-01-15T11:00:00Z"
    assert created is False
    assert replayed.id == item["id"]


def test_the_namespace_is_pinned_because_changing_it_would_duplicate_every_migrated_measurement() -> None:
    assert UUID("62300507-fe11-4834-a79f-d5f2ab79dde9") == NAMESPACE_TENSOSCAN
    assert uuid5(NAMESPACE_TENSOSCAN, "6713a1b2c3d4e5f600112233") == UUID("16e1ade0-7e2d-501f-9965-9a3282801f2e")


# --- command line --------------------------------------------------------------------------------------------------


def test_the_command_line_has_dry_run_and_legacy_tz_with_madrid_as_default() -> None:
    parser = m001.build_parser()

    defaults = parser.parse_args([])
    explicit = parser.parse_args(["--dry-run", "--legacy-tz", "UTC"])

    assert (defaults.dry_run, defaults.legacy_tz) == (False, "Europe/Madrid")
    assert (explicit.dry_run, explicit.legacy_tz) == (True, "UTC")


def test_an_unknown_legacy_zone_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        m001.main(["--legacy-tz", "Mars/Olympus"])

    assert stop.value.code == 2
    assert "not an IANA time zone" in capsys.readouterr().err


async def test_the_command_runs_against_the_configured_database(
    db: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from contextlib import asynccontextmanager

    await seed(db, legacy("128", "82", "67", datetime(2026, 1, 15, 12, 0)))

    @asynccontextmanager
    async def fake_open(_: Any) -> Any:
        yield db

    monkeypatch.setattr(m001, "open_database", fake_open)
    monkeypatch.setenv("TENSOSCAN_MONGODB_URL", "mongodb://unused")
    monkeypatch.setenv("TENSOSCAN_SECRET_KEY", "s" * 40)

    report = await m001._execute(m001.Settings(), legacy_tz="Europe/Madrid", dry_run=False)

    assert report.migrated == 1
    assert await db["measurements"].count_documents({}) == 1
