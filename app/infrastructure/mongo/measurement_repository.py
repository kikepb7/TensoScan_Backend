from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.domain.errors import ClientIdConflict
from app.domain.measurements.canonical import payload_hash
from app.domain.measurements.cursor import PageCursor
from app.domain.measurements.model import Measurement, MeasurementPage, Reading, RecognitionInfo
from app.infrastructure.mongo.client import Database

COLLECTION = "measurements"
_NEWEST_FIRST = [("measured_at", -1), ("_id", -1)]
_INSERT_ATTEMPTS = 3


class MongoMeasurementRepository:
    """Collection ``measurements``. Every query carries ``user_id``, so one user can never reach another's data."""

    def __init__(self, db: Database) -> None:
        self._items = db[COLLECTION]

    async def add_idempotent(self, measurement: Measurement, *, created_at: datetime) -> tuple[Measurement, bool]:
        """ADR-B002-01: the unique ``(user_id, client_id)`` index decides who wins a race; the loser compares hashes."""
        user_id = ObjectId(measurement.user_id)
        for _ in range(_INSERT_ATTEMPTS):
            document = to_document(measurement, created_at)
            try:
                result = await self._items.insert_one(document)
            except DuplicateKeyError:
                existing = await self._items.find_one({"user_id": user_id, "client_id": str(measurement.client_id)})
                if existing is None:
                    continue  # it was deleted between the two calls: simply try to insert again
                if existing.get("payload_hash") == document["payload_hash"]:
                    return _to_measurement(existing), False
                raise ClientIdConflict("A different measurement already uses this client_id") from None
            return _to_measurement({**document, "_id": result.inserted_id}), True
        raise RuntimeError("Could not store the measurement")  # pragma: no cover - needs delete/insert ping-pong

    async def list_page(
        self,
        user_id: str,
        *,
        since: datetime | None,
        until: datetime | None,
        limit: int,
        after: PageCursor | None,
    ) -> MeasurementPage:
        if not ObjectId.is_valid(user_id):
            return MeasurementPage([], False)
        query = _query(ObjectId(user_id), since, until, after)
        # One extra row tells whether another page follows without a second count query.
        documents = await self._items.find(query).sort(_NEWEST_FIRST).limit(limit + 1).to_list(length=limit + 1)
        return MeasurementPage([_to_measurement(d) for d in documents[:limit]], has_more=len(documents) > limit)

    async def list_all(self, user_id: str, *, since: datetime | None, until: datetime | None) -> list[Measurement]:
        if not ObjectId.is_valid(user_id):
            return []
        documents = (
            await self._items.find(_query(ObjectId(user_id), since, until, None))
            .sort(_NEWEST_FIRST)
            .to_list(length=None)
        )
        return [_to_measurement(d) for d in documents]

    async def delete(self, user_id: str, measurement_id: str) -> bool:
        if not (ObjectId.is_valid(user_id) and ObjectId.is_valid(measurement_id)):
            return False
        result = await self._items.delete_one({"_id": ObjectId(measurement_id), "user_id": ObjectId(user_id)})
        return result.deleted_count > 0

    async def delete_all_for_user(self, user_id: str) -> int:
        if not ObjectId.is_valid(user_id):
            return 0
        result = await self._items.delete_many({"user_id": ObjectId(user_id)})
        return result.deleted_count


def _query(
    user_id: ObjectId, since: datetime | None, until: datetime | None, after: PageCursor | None
) -> dict[str, Any]:
    query: dict[str, Any] = {"user_id": user_id}
    window: dict[str, datetime] = {}
    if since is not None:
        window["$gte"] = since
    if until is not None:
        window["$lte"] = until
    if window:
        query["measured_at"] = window
    if after is None:
        return query
    keyset = {
        "$or": [
            {"measured_at": {"$lt": after.measured_at}},
            {"measured_at": after.measured_at, "_id": {"$lt": ObjectId(after.id)}},
        ]
    }
    return {"$and": [query, keyset]}


def to_document(measurement: Measurement, created_at: datetime) -> dict[str, Any]:
    recognition = measurement.recognition
    return {
        "user_id": ObjectId(measurement.user_id),
        "client_id": str(measurement.client_id),
        "systolic": measurement.reading.systolic,
        "diastolic": measurement.reading.diastolic,
        "pulse": measurement.reading.pulse,
        "measured_at": measurement.measured_at,
        "created_at": created_at,
        "source": measurement.source,
        "recognition": None
        if recognition is None
        else {
            "confidence": recognition.confidence,
            "model_version": recognition.model_version,
            "edited_fields": sorted(recognition.edited_fields),
        },
        "payload_hash": payload_hash(measurement),
        "legacy_id": None,
    }


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _to_measurement(document: dict[str, Any]) -> Measurement:
    stored = document.get("recognition")
    recognition = (
        None
        if stored is None
        else RecognitionInfo(
            confidence=float(stored["confidence"]),
            model_version=stored["model_version"],
            edited_fields=frozenset(stored.get("edited_fields", [])),
        )
    )
    return Measurement(
        id=str(document["_id"]),
        user_id=str(document["user_id"]),
        client_id=UUID(document["client_id"]),
        reading=Reading(document["systolic"], document["diastolic"], document["pulse"]),
        measured_at=_aware(document["measured_at"]),
        created_at=_aware(document["created_at"]),
        source=document["source"],
        recognition=recognition,
    )
