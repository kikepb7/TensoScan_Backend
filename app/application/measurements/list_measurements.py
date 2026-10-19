from dataclasses import dataclass
from datetime import datetime

from app.domain.measurements.cursor import PageCursor
from app.domain.measurements.model import Measurement, as_utc
from app.domain.ports import MeasurementRepository


@dataclass(frozen=True)
class MeasurementsListing:
    items: list[Measurement]
    next_cursor: str | None


class ListMeasurements:
    """REQ-B002-06: the user's measurements, newest first, one cursor-paginated page at a time."""

    def __init__(self, measurements: MeasurementRepository) -> None:
        self._measurements = measurements

    async def execute(
        self,
        *,
        user_id: str,
        since: datetime | None,
        until: datetime | None,
        limit: int,
        cursor: str | None,
    ) -> MeasurementsListing:
        """Raises ``ValidationFailed`` (field ``cursor``) when the cursor is not one this API issued."""
        after = PageCursor.decode(cursor) if cursor else None
        page = await self._measurements.list_page(
            user_id,
            since=as_utc(since) if since else None,
            until=as_utc(until) if until else None,
            limit=limit,
            after=after,
        )
        last = page.items[-1] if page.items else None
        next_cursor = PageCursor(last.measured_at, last.id).encode() if page.has_more and last and last.id else None
        return MeasurementsListing(page.items, next_cursor)


class ListAllMeasurements:
    """Unpaginated listing, for the legacy routes that never had pagination."""

    def __init__(self, measurements: MeasurementRepository) -> None:
        self._measurements = measurements

    async def execute(self, *, user_id: str) -> list[Measurement]:
        return await self._measurements.list_all(user_id, since=None, until=None)
