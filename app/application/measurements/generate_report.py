import asyncio
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.domain.errors import FieldIssue, NotFound, ValidationFailed
from app.domain.measurements.model import as_utc
from app.domain.measurements.report import MeasurementsReport, ReportSummary
from app.domain.ports import Clock, MeasurementRepository, ReportRenderer
from app.domain.user import User


@dataclass(frozen=True)
class GeneratedReport:
    content: bytes
    filename: str


def parse_timezone(name: str) -> ZoneInfo:
    """Raises ``ValidationFailed`` (field ``tz``) when ``name`` is not an IANA zone."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError) as error:
        raise ValidationFailed([FieldIssue("tz", "INVALID_TIMEZONE")], "tz is not a valid IANA time zone") from error


class GenerateMeasurementsReport:
    """REQ-B002-08/09: the PDF report of the user's measurements, with local times in the requested zone."""

    def __init__(
        self,
        measurements: MeasurementRepository,
        renderer: ReportRenderer,
        clock: Clock,
        default_timezone: str,
    ) -> None:
        self._measurements = measurements
        self._renderer = renderer
        self._clock = clock
        self._default_timezone = default_timezone

    async def execute(
        self, *, user: User, since: datetime | None, until: datetime | None, timezone: str | None
    ) -> GeneratedReport:
        zone = parse_timezone(self._default_timezone if timezone is None else timezone)
        since_utc = as_utc(since) if since else None
        until_utc = as_utc(until) if until else None
        found = await self._measurements.list_all(user.id, since=since_utc, until=until_utc)
        if not found:
            raise NotFound("No measurements found")
        now = self._clock.now()
        report = MeasurementsReport(
            user_name=f"{user.name} {user.last_name}".strip(),
            generated_at=now,
            timezone=zone,
            range_from=since_utc,
            range_to=until_utc,
            measurements=sorted(found, key=lambda m: (m.measured_at, m.id or "")),
            summary=ReportSummary.of(found),
        )
        content = await asyncio.to_thread(self._renderer.render, report)
        return GeneratedReport(content, f"tensoscan-report-{now.astimezone(zone):%Y%m%d}.pdf")
