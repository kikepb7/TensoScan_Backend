"""``MongoMeasurementRepository`` against the ``db`` fixture (real MongoDB in CI, in-memory double locally)."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.domain.errors import ClientIdConflict
from app.domain.measurements.cursor import PageCursor
from app.domain.measurements.model import Measurement, Reading, RecognitionInfo
from app.infrastructure.mongo.measurement_repository import MongoMeasurementRepository

BASE = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
CREATED = datetime(2026, 10, 19, 8, 0, tzinfo=UTC)


def _user() -> str:
    return str(ObjectId())


def _measurement(user_id: str, *, at: datetime = BASE, client_id: UUID | None = None, **changes: Any) -> Measurement:
    base = Measurement(
        id=None,
        user_id=user_id,
        client_id=client_id or uuid4(),
        reading=Reading(128, 82, 67),
        measured_at=at,
        created_at=None,
        source="camera",
        recognition=RecognitionInfo(0.93, "digits-1.0.0", frozenset({"pulse"})),
    )
    return replace(base, **changes)


@pytest.fixture
def repository(db: Any) -> MongoMeasurementRepository:
    return MongoMeasurementRepository(db)


async def test_a_new_measurement_is_stored_with_id_and_server_timestamp(
    repository: MongoMeasurementRepository,
) -> None:
    user_id = _user()
    draft = _measurement(user_id)

    stored, created = await repository.add_idempotent(draft, created_at=CREATED)

    assert created is True
    assert stored.id is not None and ObjectId.is_valid(stored.id)
    assert replace(stored, id=None, created_at=None) == draft
    assert stored.created_at == CREATED


async def test_the_same_payload_again_returns_the_existing_document_without_a_duplicate(
    repository: MongoMeasurementRepository, db: Any
) -> None:
    draft = _measurement(_user())
    first, _ = await repository.add_idempotent(draft, created_at=CREATED)

    second, created = await repository.add_idempotent(draft, created_at=CREATED + timedelta(hours=1))

    assert created is False
    assert second == first
    assert second.created_at == CREATED
    assert await db["measurements"].count_documents({}) == 1


async def test_the_same_client_id_with_a_different_payload_is_a_conflict(
    repository: MongoMeasurementRepository,
) -> None:
    draft = _measurement(_user())
    await repository.add_idempotent(draft, created_at=CREATED)

    with pytest.raises(ClientIdConflict):
        await repository.add_idempotent(replace(draft, reading=Reading(128, 82, 68)), created_at=CREATED)


async def test_the_same_client_id_may_be_used_by_two_users(repository: MongoMeasurementRepository) -> None:
    client_id = uuid4()

    _, first = await repository.add_idempotent(_measurement(_user(), client_id=client_id), created_at=CREATED)
    _, second = await repository.add_idempotent(_measurement(_user(), client_id=client_id), created_at=CREATED)

    assert (first, second) == (True, True)


async def test_two_simultaneous_identical_inserts_yield_one_creation_and_one_replay(
    repository: MongoMeasurementRepository, db: Any
) -> None:
    draft = _measurement(_user())

    results = await asyncio.gather(
        repository.add_idempotent(draft, created_at=CREATED), repository.add_idempotent(draft, created_at=CREATED)
    )

    assert sorted(created for _, created in results) == [False, True]
    assert results[0][0].id == results[1][0].id
    assert await db["measurements"].count_documents({}) == 1


async def test_the_unique_index_on_user_and_client_id_is_enforced_by_the_database(db: Any) -> None:
    user_id, client_id = ObjectId(), str(uuid4())
    await db["measurements"].insert_one({"user_id": user_id, "client_id": client_id})

    with pytest.raises(DuplicateKeyError):
        await db["measurements"].insert_one({"user_id": user_id, "client_id": client_id})


async def test_the_listing_is_newest_first_with_ties_broken_by_descending_id(
    repository: MongoMeasurementRepository,
) -> None:
    user_id = _user()
    old, _ = await repository.add_idempotent(_measurement(user_id, at=BASE), created_at=CREATED)
    tie_a, _ = await repository.add_idempotent(_measurement(user_id, at=BASE + timedelta(hours=1)), created_at=CREATED)
    tie_b, _ = await repository.add_idempotent(_measurement(user_id, at=BASE + timedelta(hours=1)), created_at=CREATED)
    newest, _ = await repository.add_idempotent(_measurement(user_id, at=BASE + timedelta(days=1)), created_at=CREATED)

    page = await repository.list_page(user_id, since=None, until=None, limit=50, after=None)

    assert [item.id for item in page.items] == [newest.id, tie_b.id, tie_a.id, old.id]
    assert page.has_more is False


async def test_120_measurements_are_paged_50_50_20_without_repeats(repository: MongoMeasurementRepository) -> None:
    user_id = _user()
    # Every third measurement shares its instant with the previous one, so the tie-break is exercised too.
    for index in range(120):
        at = BASE + timedelta(minutes=index - index % 3)
        await repository.add_idempotent(_measurement(user_id, at=at), created_at=CREATED)

    pages, cursor = [], None
    while True:
        page = await repository.list_page(user_id, since=None, until=None, limit=50, after=cursor)
        pages.append(page)
        if not page.has_more:
            break
        last = page.items[-1]
        cursor = PageCursor(last.measured_at, last.id or "")

    items = [item for page in pages for item in page.items]
    assert [len(page.items) for page in pages] == [50, 50, 20]
    assert [page.has_more for page in pages] == [True, True, False]
    assert len({item.id for item in items}) == 120
    assert [(i.measured_at, i.id) for i in items] == sorted(((i.measured_at, i.id) for i in items), reverse=True)


async def test_a_page_that_exactly_fills_the_limit_reports_no_more_pages(
    repository: MongoMeasurementRepository,
) -> None:
    user_id = _user()
    for index in range(3):
        await repository.add_idempotent(_measurement(user_id, at=BASE + timedelta(minutes=index)), created_at=CREATED)

    page = await repository.list_page(user_id, since=None, until=None, limit=3, after=None)

    assert (len(page.items), page.has_more) == (3, False)


async def test_since_and_until_filter_by_measured_at_inclusively(repository: MongoMeasurementRepository) -> None:
    user_id = _user()
    for day in range(5):
        await repository.add_idempotent(_measurement(user_id, at=BASE + timedelta(days=day)), created_at=CREATED)

    page = await repository.list_page(
        user_id, since=BASE + timedelta(days=1), until=BASE + timedelta(days=3), limit=50, after=None
    )
    everything = await repository.list_all(user_id, since=BASE + timedelta(days=1), until=None)

    assert [item.measured_at for item in page.items] == [BASE + timedelta(days=d) for d in (3, 2, 1)]
    assert len(everything) == 4


async def test_filters_and_cursor_work_together(repository: MongoMeasurementRepository) -> None:
    user_id = _user()
    for day in range(6):
        await repository.add_idempotent(_measurement(user_id, at=BASE + timedelta(days=day)), created_at=CREATED)
    first = await repository.list_page(user_id, since=BASE, until=BASE + timedelta(days=4), limit=2, after=None)
    last = first.items[-1]

    second = await repository.list_page(
        user_id,
        since=BASE,
        until=BASE + timedelta(days=4),
        limit=2,
        after=PageCursor(last.measured_at, last.id or ""),
    )

    assert [item.measured_at for item in first.items] == [BASE + timedelta(days=d) for d in (4, 3)]
    assert [item.measured_at for item in second.items] == [BASE + timedelta(days=d) for d in (2, 1)]
    assert second.has_more is True


async def test_a_user_only_sees_and_deletes_their_own_measurements(repository: MongoMeasurementRepository) -> None:
    owner, other = _user(), _user()
    mine, _ = await repository.add_idempotent(_measurement(owner), created_at=CREATED)
    await repository.add_idempotent(_measurement(other), created_at=CREATED)

    seen = await repository.list_all(other, since=None, until=None)
    stolen = await repository.delete(other, mine.id or "")
    still_there = await repository.list_all(owner, since=None, until=None)
    removed = await repository.delete(owner, mine.id or "")
    again = await repository.delete(owner, mine.id or "")

    assert [item.user_id for item in seen] == [other]
    assert stolen is False
    assert [item.id for item in still_there] == [mine.id]
    assert (removed, again) == (True, False)


@pytest.mark.parametrize("bad_id", ["", "not-an-id", "123"])
async def test_deleting_a_malformed_id_is_just_not_found(repository: MongoMeasurementRepository, bad_id: str) -> None:
    assert await repository.delete(_user(), bad_id) is False


async def test_a_malformed_user_id_matches_nothing(repository: MongoMeasurementRepository) -> None:
    page = await repository.list_page("nope", since=None, until=None, limit=10, after=None)

    assert (page.items, page.has_more) == ([], False)
    assert await repository.list_all("nope", since=None, until=None) == []
    assert await repository.delete_all_for_user("nope") == 0


async def test_delete_all_for_user_only_removes_that_users_measurements(
    repository: MongoMeasurementRepository,
) -> None:
    leaving, staying = _user(), _user()
    for _ in range(3):
        await repository.add_idempotent(_measurement(leaving), created_at=CREATED)
    await repository.add_idempotent(_measurement(staying), created_at=CREATED)

    removed = await repository.delete_all_for_user(leaving)

    assert removed == 3
    assert await repository.list_all(leaving, since=None, until=None) == []
    assert len(await repository.list_all(staying, since=None, until=None)) == 1


async def test_a_manual_measurement_round_trips_without_recognition(repository: MongoMeasurementRepository) -> None:
    draft = _measurement(_user(), source="manual", recognition=None)

    stored, _ = await repository.add_idempotent(draft, created_at=CREATED)
    listed = (await repository.list_all(stored.user_id, since=None, until=None))[0]

    assert listed == stored
    assert listed.recognition is None
