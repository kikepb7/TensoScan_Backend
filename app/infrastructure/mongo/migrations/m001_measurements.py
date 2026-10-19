"""m001: copies the legacy ``recognition_results`` into ``measurements`` (spec B-002, REQ-B002-11, design §7).

    uv run python -m app.infrastructure.mongo.migrations.m001_measurements [--dry-run] [--legacy-tz Europe/Madrid]

* Idempotent: every legacy document gets ``client_id = uuid5(NAMESPACE_TENSOSCAN, str(legacy _id))`` and is written with
  ``update_one(..., {"$setOnInsert": ...}, upsert=True)``, so a second run creates nothing.
* The legacy collection is never modified.
* Legacy timestamps were written by ``datetime.now()`` in the server's local time (naive) and PyMongo stored them as if
  they were UTC. The wall-clock fields of the stored value are therefore reinterpreted in ``--legacy-tz`` (decision of
  the owner: ``Europe/Madrid``) and converted to UTC. At a DST change, an ambiguous autumn hour resolves to its first
  occurrence (``fold=0``) and a non-existent spring hour moves one hour forward; both are counted in the report.
* Invalid documents (non-numeric or out-of-range values, no owner, no timestamp) are skipped and listed in the report.
  The report is logged (ids and reasons only, never readings, C-SEC-03) and, in a real run, stored in
  ``migration_reports``.
"""

import argparse
import asyncio
import json
import logging
import math
import re
import sys
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from bson import ObjectId

from app.core.config import Settings
from app.domain.measurements.model import Measurement, RecognitionInfo
from app.domain.measurements.rules import ReadingRules
from app.infrastructure.mongo.client import Database, create_client
from app.infrastructure.mongo.indexes import create_indexes
from app.infrastructure.mongo.measurement_repository import to_document

logger = logging.getLogger(__name__)

# Fixed forever: the app reconciles measurements by ``client_id``, so changing it would duplicate every migrated one.
NAMESPACE_TENSOSCAN = uuid.UUID("62300507-fe11-4834-a79f-d5f2ab79dde9")
MIGRATION_ID = "m001"
LEGACY_COLLECTION = "recognition_results"
REPORTS_COLLECTION = "migration_reports"
LEGACY_MODEL_VERSION = "legacy-keras"
DEFAULT_LEGACY_TZ = "Europe/Madrid"

_DIGITS = re.compile(r"\d{1,4}")
_LEGACY_FIELDS = (("high_pressure", "systolic"), ("low_pressure", "diastolic"), ("pulse", "pulse"))

LocalTimeIssue = Literal["ambiguous", "nonexistent"]


@dataclass(frozen=True)
class LocalizedTime:
    utc: datetime
    issue: LocalTimeIssue | None


def localize_legacy_time(wall_clock: datetime, zone: ZoneInfo) -> LocalizedTime:
    """Interprets a naive wall-clock time in ``zone`` and returns it in UTC (see the module docstring for DST)."""
    first = wall_clock.replace(tzinfo=zone, fold=0)
    second = wall_clock.replace(tzinfo=zone, fold=1)
    if first.utcoffset() == second.utcoffset():
        return LocalizedTime(first.astimezone(UTC), None)
    if first.astimezone(UTC).astimezone(zone).replace(tzinfo=None) == wall_clock:
        return LocalizedTime(first.astimezone(UTC), "ambiguous")  # the hour repeated in autumn: first occurrence
    shifted = (wall_clock + timedelta(hours=1)).replace(tzinfo=zone)  # the hour skipped in spring
    return LocalizedTime(shifted.astimezone(UTC), "nonexistent")


@dataclass
class MigrationReport:
    ran_at: datetime
    dry_run: bool
    legacy_tz: str
    scanned: int = 0
    migrated: int = 0  # created by this run (or, in a dry run, that it would create)
    already_migrated: int = 0
    skipped: list[dict[str, str]] = field(default_factory=list)
    ambiguous_local_times: int = 0
    nonexistent_local_times: int = 0

    def to_document(self) -> dict[str, Any]:
        return {
            "migration": MIGRATION_ID,
            "ran_at": self.ran_at,
            "dry_run": self.dry_run,
            "legacy_tz": self.legacy_tz,
            "scanned": self.scanned,
            "migrated": self.migrated,
            "already_migrated": self.already_migrated,
            "skipped": self.skipped,
            "ambiguous_local_times": self.ambiguous_local_times,
            "nonexistent_local_times": self.nonexistent_local_times,
        }


def _parse_digits(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and _DIGITS.fullmatch(value.strip()):
        return int(value.strip())
    return None


def _wall_clock(value: object) -> datetime | LocalizedTime | None:
    """The naive local time of a legacy ``timestamp``; a string carrying its own offset is already an instant."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)  # tz_aware clients label the stored wall clock as UTC: drop the label
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return LocalizedTime(parsed.astimezone(UTC), None) if parsed.tzinfo else parsed
    return None


def _confidence(result: dict[str, Any]) -> float:
    value = result.get("confidence")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(value):
        return min(1.0, max(0.0, float(value)))
    return 1.0


@dataclass(frozen=True)
class _Converted:
    measurement: Measurement
    issue: LocalTimeIssue | None


def convert_legacy_document(document: dict[str, Any], zone: ZoneInfo) -> _Converted | str:
    """Returns the measurement to write, or the reason why the document is not migrated."""
    owner = document.get("user_id")
    if isinstance(owner, str) and ObjectId.is_valid(owner):
        owner = ObjectId(owner)
    if not isinstance(owner, ObjectId):
        return "MISSING_USER"
    result = document.get("result")
    if not isinstance(result, dict):
        return "MISSING_RESULT"
    values: dict[str, int] = {}
    for legacy_name, name in _LEGACY_FIELDS:
        parsed = _parse_digits(result.get(legacy_name))
        if parsed is None:
            return f"NOT_NUMERIC:{legacy_name}"
        values[name] = parsed
    evaluation = ReadingRules.evaluate(values["systolic"], values["diastolic"], values["pulse"])
    if evaluation.reading is None:
        return ",".join(f"{issue.code}:{issue.field}" for issue in evaluation.errors)
    wall_clock = _wall_clock(document.get("timestamp"))
    if wall_clock is None:
        return "INVALID_TIMESTAMP"
    localized = wall_clock if isinstance(wall_clock, LocalizedTime) else localize_legacy_time(wall_clock, zone)
    measured_at = localized.utc.replace(microsecond=0)
    measurement = Measurement(
        id=None,
        user_id=str(owner),
        client_id=uuid.uuid5(NAMESPACE_TENSOSCAN, str(document["_id"])),
        reading=evaluation.reading,
        measured_at=measured_at,
        created_at=measured_at,
        source="camera",
        recognition=RecognitionInfo(_confidence(result), LEGACY_MODEL_VERSION, frozenset()),
    )
    return _Converted(measurement, localized.issue)


async def run_migration(
    db: Database, *, legacy_tz: str = DEFAULT_LEGACY_TZ, dry_run: bool = False, now: datetime | None = None
) -> MigrationReport:
    zone = ZoneInfo(legacy_tz)
    report = MigrationReport(ran_at=(now or datetime.now(UTC)), dry_run=dry_run, legacy_tz=legacy_tz)
    target = db["measurements"]
    async for document in db[LEGACY_COLLECTION].find({}).sort("_id", 1):
        report.scanned += 1
        converted = convert_legacy_document(document, zone)
        if isinstance(converted, str):
            report.skipped.append({"legacy_id": str(document["_id"]), "reason": converted})
            continue
        if converted.issue == "ambiguous":
            report.ambiguous_local_times += 1
        elif converted.issue == "nonexistent":
            report.nonexistent_local_times += 1
        measurement = converted.measurement
        key = {"user_id": ObjectId(measurement.user_id), "client_id": str(measurement.client_id)}
        if dry_run:
            exists = await target.find_one(key, projection={"_id": 1}) is not None
            report.already_migrated += exists
            report.migrated += not exists
            continue
        content = {k: v for k, v in to_document(measurement, converted.measurement.measured_at).items() if k not in key}
        content["legacy_id"] = document["_id"]
        outcome = await target.update_one(key, {"$setOnInsert": content}, upsert=True)
        if outcome.upserted_id is not None:
            report.migrated += 1
        else:
            report.already_migrated += 1
    logger.info("Migration %s finished: %s", MIGRATION_ID, json.dumps(_loggable(report)))
    if not dry_run:
        await db[REPORTS_COLLECTION].insert_one(report.to_document())
    return report


def _loggable(report: MigrationReport) -> dict[str, Any]:
    return {**report.to_document(), "ran_at": report.ran_at.isoformat()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.infrastructure.mongo.migrations.m001_measurements",
        description="Copy the legacy recognition_results into measurements (idempotent).",
    )
    parser.add_argument("--dry-run", action="store_true", help="report what would happen without writing anything")
    parser.add_argument(
        "--legacy-tz",
        default=DEFAULT_LEGACY_TZ,
        help="IANA zone of the naive legacy timestamps (default: %(default)s)",
    )
    return parser


@asynccontextmanager
async def open_database(settings: Settings) -> AsyncIterator[Database]:
    client = create_client(settings)
    try:
        yield client[settings.db_name]
    finally:
        await client.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        ZoneInfo(args.legacy_tz)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        parser.error(f"--legacy-tz {args.legacy_tz!r} is not an IANA time zone")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()  # required values come from the environment
    report = asyncio.run(_execute(settings, legacy_tz=args.legacy_tz, dry_run=args.dry_run))
    sys.stdout.write(json.dumps(_loggable(report), indent=2) + "\n")
    return 0


async def _execute(settings: Settings, *, legacy_tz: str, dry_run: bool) -> MigrationReport:
    async with open_database(settings) as db:
        if not dry_run:
            await create_indexes(db)  # idempotency relies on the unique (user_id, client_id) index
        return await run_migration(db, legacy_tz=legacy_tz, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
