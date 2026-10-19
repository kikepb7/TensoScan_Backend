from collections.abc import Sequence

from app.domain.ports import MeasurementEraser


class EraseAllMeasurements:
    """Account deletion must erase every store that ever held measurements: the current collection and, while it
    exists, the legacy one. Implements the ``MeasurementEraser`` port by delegating to several of them."""

    def __init__(self, erasers: Sequence[MeasurementEraser]) -> None:
        self._erasers = erasers

    async def delete_all_for_user(self, user_id: str) -> int:
        removed = 0
        for eraser in self._erasers:
            removed += await eraser.delete_all_for_user(user_id)
        return removed
