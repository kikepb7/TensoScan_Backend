from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.application.measurements.generate_report import GenerateMeasurementsReport
from app.domain.errors import FieldIssue, NotFound, ValidationFailed
from app.domain.measurements.model import Measurement, Reading
from app.domain.measurements.report import MeasurementsReport, ReportSummary
from app.domain.user import User
from tests.support.fakes import FakeClock, InMemoryMeasurementRepository

USER = User(id="a" * 24, email="ana@example.com", name="Ana", last_name="García", password_hash="x")


class CapturingRenderer:
    def __init__(self) -> None:
        self.reports: list[MeasurementsReport] = []

    def render(self, report: MeasurementsReport) -> bytes:
        self.reports.append(report)
        return b"%PDF-fake"


def _measurement(user_id: str, at: datetime, systolic: int = 120) -> Measurement:
    return Measurement(
        id=None,
        user_id=user_id,
        client_id=uuid4(),
        reading=Reading(systolic, 80, 60),
        measured_at=at,
        created_at=None,
        source="manual",
        recognition=None,
    )


@pytest.fixture
def repository() -> InMemoryMeasurementRepository:
    return InMemoryMeasurementRepository()


@pytest.fixture
def renderer() -> CapturingRenderer:
    return CapturingRenderer()


@pytest.fixture
def sut(repository: InMemoryMeasurementRepository, renderer: CapturingRenderer) -> GenerateMeasurementsReport:
    return GenerateMeasurementsReport(repository, renderer, FakeClock(), "Europe/Madrid")


async def _store(repository: InMemoryMeasurementRepository, *measurements: Measurement) -> None:
    for measurement in measurements:
        await repository.add_idempotent(measurement, created_at=datetime(2026, 10, 19, tzinfo=UTC))


async def test_the_report_gets_the_measurements_oldest_first_with_their_summary(
    sut: GenerateMeasurementsReport, repository: InMemoryMeasurementRepository, renderer: CapturingRenderer
) -> None:
    base = datetime(2026, 10, 10, 8, tzinfo=UTC)
    await _store(
        repository,
        _measurement(USER.id, base + timedelta(days=2), 140),
        _measurement(USER.id, base, 100),
        _measurement("b" * 24, base, 250),
    )

    result = await sut.execute(user=USER, since=None, until=None, timezone=None)

    report = renderer.reports[0]
    assert result.content == b"%PDF-fake"
    assert [m.reading.systolic for m in report.measurements] == [100, 140]
    assert report.summary == ReportSummary.of(list(report.measurements))
    assert report.summary.systolic.mean == 120
    assert report.user_name == "Ana García"


async def test_without_tz_the_configured_default_zone_is_used(
    sut: GenerateMeasurementsReport, repository: InMemoryMeasurementRepository, renderer: CapturingRenderer
) -> None:
    await _store(repository, _measurement(USER.id, datetime(2026, 10, 10, tzinfo=UTC)))

    await sut.execute(user=USER, since=None, until=None, timezone=None)
    await sut.execute(user=USER, since=None, until=None, timezone="Asia/Tokyo")

    assert [str(report.timezone) for report in renderer.reports] == ["Europe/Madrid", "Asia/Tokyo"]


async def test_the_filename_carries_the_local_generation_date(
    repository: InMemoryMeasurementRepository, renderer: CapturingRenderer
) -> None:
    await _store(repository, _measurement(USER.id, datetime(2026, 10, 10, tzinfo=UTC)))
    late_evening_utc = FakeClock(datetime(2026, 10, 19, 23, 30, tzinfo=UTC))  # already the 20th in Madrid
    sut = GenerateMeasurementsReport(repository, renderer, late_evening_utc, "Europe/Madrid")

    madrid = await sut.execute(user=USER, since=None, until=None, timezone=None)
    utc = await sut.execute(user=USER, since=None, until=None, timezone="UTC")

    assert (madrid.filename, utc.filename) == ("tensoscan-report-20261020.pdf", "tensoscan-report-20261019.pdf")


async def test_no_measurements_is_not_found(sut: GenerateMeasurementsReport) -> None:
    with pytest.raises(NotFound):
        await sut.execute(user=USER, since=None, until=None, timezone=None)


async def test_an_invalid_zone_is_reported_before_looking_for_data(sut: GenerateMeasurementsReport) -> None:
    with pytest.raises(ValidationFailed) as raised:
        await sut.execute(user=USER, since=None, until=None, timezone="Mars/Olympus")

    assert raised.value.details == [FieldIssue("tz", "INVALID_TIMEZONE")]


async def test_the_range_limits_the_measurements(
    sut: GenerateMeasurementsReport, repository: InMemoryMeasurementRepository, renderer: CapturingRenderer
) -> None:
    base = datetime(2026, 10, 10, tzinfo=UTC)
    await _store(repository, *[_measurement(USER.id, base + timedelta(days=d), 100 + d) for d in range(5)])

    await sut.execute(user=USER, since=base + timedelta(days=1), until=base + timedelta(days=2), timezone=None)

    report = renderer.reports[0]
    assert [m.reading.systolic for m in report.measurements] == [101, 102]
    assert (report.range_from, report.range_to) == (base + timedelta(days=1), base + timedelta(days=2))


def test_a_summary_of_nothing_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ReportSummary.of([])


def test_summary_statistics() -> None:
    day = datetime(2026, 1, 1, tzinfo=UTC)
    readings = [Reading(100, 60, 50), Reading(140, 90, 70)]
    items = [replace(_measurement("u", day), reading=reading) for reading in readings]

    summary = ReportSummary.of(items)

    assert summary.count == 2
    assert (summary.systolic.mean, summary.systolic.minimum, summary.systolic.maximum) == (120, 100, 140)
    assert (summary.diastolic.mean, summary.pulse.maximum) == (75, 70)
