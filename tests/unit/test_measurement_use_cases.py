from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from app.application.measurements.create_measurement import CreateMeasurement, NewMeasurement
from app.application.measurements.delete_measurement import DeleteMeasurement
from app.application.measurements.erase_measurements import EraseAllMeasurements
from app.application.measurements.list_measurements import ListAllMeasurements, ListMeasurements
from app.domain.errors import ClientIdConflict, FieldIssue, MeasurementInvalid, NotFound, ValidationFailed
from app.domain.measurements.model import RecognitionInfo
from tests.support.fakes import FakeClock, InMemoryMeasurementRepository, RecordingMeasurementEraser

USER = "a" * 24
OTHER_USER = "b" * 24
CLIENT_ID = UUID("5b0b6f0e-6a1f-4d8e-9d0a-2f1c3b7e9a10")


def _command(**changes: object) -> NewMeasurement:
    base = NewMeasurement(
        user_id=USER,
        client_id=CLIENT_ID,
        systolic=128,
        diastolic=82,
        pulse=67,
        measured_at=datetime(2026, 10, 19, 6, 15, tzinfo=UTC),
        source="camera",
        recognition=RecognitionInfo(0.93, "digits-1.0.0", frozenset({"pulse"})),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


@pytest.fixture
def repository() -> InMemoryMeasurementRepository:
    return InMemoryMeasurementRepository()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sut(repository: InMemoryMeasurementRepository, clock: FakeClock) -> CreateMeasurement:
    return CreateMeasurement(repository, clock)


async def test_a_valid_measurement_is_created_with_the_server_timestamp(
    sut: CreateMeasurement, clock: FakeClock
) -> None:
    result = await sut.execute(_command())

    assert result.created is True
    assert result.measurement.id is not None
    assert result.measurement.created_at == clock.now()
    assert (result.measurement.reading.systolic, result.measurement.reading.pulse) == (128, 67)


async def test_the_same_command_again_is_a_replay_of_the_stored_measurement(
    sut: CreateMeasurement, repository: InMemoryMeasurementRepository, clock: FakeClock
) -> None:
    first = await sut.execute(_command())
    clock.advance(minutes=10)

    second = await sut.execute(_command())

    assert second.created is False
    assert second.measurement == first.measurement
    assert len(repository.items) == 1


async def test_the_same_client_id_with_other_values_is_a_conflict(sut: CreateMeasurement) -> None:
    await sut.execute(_command())

    with pytest.raises(ClientIdConflict):
        await sut.execute(_command(pulse=68))


async def test_the_same_client_id_for_another_user_is_a_new_measurement(sut: CreateMeasurement) -> None:
    await sut.execute(_command())

    result = await sut.execute(_command(user_id=OTHER_USER))

    assert result.created is True


async def test_an_invalid_measurement_lists_every_problem(sut: CreateMeasurement, repository) -> None:  # type: ignore[no-untyped-def]
    command = _command(
        systolic=300,
        pulse=None,
        measured_at=datetime(2026, 10, 19, 9, 0, tzinfo=UTC),
        source="manual",
    )
    expected = [
        FieldIssue("systolic", "OUT_OF_RANGE"),
        FieldIssue("pulse", "REQUIRED"),
        FieldIssue("measured_at", "IN_FUTURE"),
        FieldIssue("recognition", "NOT_ALLOWED"),
    ]

    with pytest.raises(MeasurementInvalid) as raised:
        await sut.execute(command)

    assert raised.value.details == expected
    assert repository.items == []


async def test_measured_at_at_the_five_minute_tolerance_is_accepted(sut: CreateMeasurement, clock: FakeClock) -> None:
    result = await sut.execute(_command(measured_at=clock.now() + timedelta(minutes=5)))

    assert result.created is True


async def test_measured_at_before_2000_is_rejected(sut: CreateMeasurement) -> None:
    with pytest.raises(MeasurementInvalid) as raised:
        await sut.execute(_command(measured_at=datetime(1999, 12, 31, tzinfo=UTC)))

    assert raised.value.details == [FieldIssue("measured_at", "TOO_OLD")]


@pytest.mark.parametrize(
    ("systolic", "diastolic", "pulse"), [(185, 100, 80), (88, 55, 70), (120, 80, 38), (100, 90, 70)]
)
async def test_warning_values_never_block_the_creation(
    sut: CreateMeasurement, systolic: int, diastolic: int, pulse: int
) -> None:
    result = await sut.execute(_command(systolic=systolic, diastolic=diastolic, pulse=pulse))

    assert result.created is True


async def test_an_offset_and_sub_second_time_is_stored_as_utc_seconds(sut: CreateMeasurement) -> None:
    from datetime import timezone

    local = datetime(2026, 10, 19, 8, 15, 0, 999999, tzinfo=timezone(timedelta(hours=2)))

    result = await sut.execute(_command(measured_at=local))

    assert result.measurement.measured_at == datetime(2026, 10, 19, 6, 15, tzinfo=UTC)
    assert result.measurement.measured_at.microsecond == 0


async def test_a_replay_with_a_different_sub_second_part_is_still_the_same_payload(sut: CreateMeasurement) -> None:
    await sut.execute(_command())

    replay = await sut.execute(_command(measured_at=datetime(2026, 10, 19, 6, 15, 0, 500, tzinfo=UTC)))

    assert replay.created is False


async def _fill(repository: InMemoryMeasurementRepository, clock: FakeClock, how_many: int) -> None:
    sut = CreateMeasurement(repository, clock)
    for index in range(how_many):
        await sut.execute(
            _command(client_id=uuid4(), measured_at=datetime(2026, 10, 1, tzinfo=UTC) + timedelta(hours=index))
        )


async def test_listing_returns_a_next_cursor_that_leads_to_the_rest(
    repository: InMemoryMeasurementRepository, clock: FakeClock
) -> None:
    await _fill(repository, clock, 5)
    sut = ListMeasurements(repository)

    first = await sut.execute(user_id=USER, since=None, until=None, limit=2, cursor=None)
    second = await sut.execute(user_id=USER, since=None, until=None, limit=2, cursor=first.next_cursor)
    third = await sut.execute(user_id=USER, since=None, until=None, limit=2, cursor=second.next_cursor)

    assert [len(page.items) for page in (first, second, third)] == [2, 2, 1]
    assert first.next_cursor is not None and second.next_cursor is not None and third.next_cursor is None
    seen = [m.id for page in (first, second, third) for m in page.items]
    assert len(set(seen)) == 5


async def test_listing_with_an_invalid_cursor_is_a_validation_error(repository: InMemoryMeasurementRepository) -> None:
    sut = ListMeasurements(repository)

    with pytest.raises(ValidationFailed) as raised:
        await sut.execute(user_id=USER, since=None, until=None, limit=10, cursor="garbage")

    assert raised.value.details == [FieldIssue("cursor", "INVALID")]


async def test_naive_filter_dates_are_read_as_utc(repository: InMemoryMeasurementRepository, clock: FakeClock) -> None:
    await _fill(repository, clock, 3)
    sut = ListMeasurements(repository)

    page = await sut.execute(user_id=USER, since=datetime(2026, 10, 1, 1), until=None, limit=10, cursor=None)

    assert len(page.items) == 2


async def test_list_all_returns_everything_of_the_user(
    repository: InMemoryMeasurementRepository, clock: FakeClock
) -> None:
    await _fill(repository, clock, 3)

    actual = await ListAllMeasurements(repository).execute(user_id=USER)

    assert len(actual) == 3
    assert await ListAllMeasurements(repository).execute(user_id=OTHER_USER) == []


async def test_deleting_an_own_measurement_removes_it(
    repository: InMemoryMeasurementRepository, sut: CreateMeasurement
) -> None:
    created = await sut.execute(_command())

    await DeleteMeasurement(repository).execute(user_id=USER, measurement_id=created.measurement.id or "")

    assert repository.items == []


async def test_deleting_someone_elses_or_an_unknown_measurement_is_not_found(
    repository: InMemoryMeasurementRepository, sut: CreateMeasurement
) -> None:
    created = await sut.execute(_command())
    delete = DeleteMeasurement(repository)

    with pytest.raises(NotFound):
        await delete.execute(user_id=OTHER_USER, measurement_id=created.measurement.id or "")
    with pytest.raises(NotFound):
        await delete.execute(user_id=USER, measurement_id="0" * 24)

    assert len(repository.items) == 1


async def test_erasing_everything_adds_up_what_every_store_removed() -> None:
    first, second = RecordingMeasurementEraser(), RecordingMeasurementEraser()

    removed = await EraseAllMeasurements([first, second]).delete_all_for_user(USER)

    assert removed == 0
    assert first.erased == second.erased == [USER]
