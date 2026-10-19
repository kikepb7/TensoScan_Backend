from app.domain.errors import NotFound
from app.domain.ports import MeasurementRepository


class DeleteMeasurement:
    """REQ-B002-07: deletes one of the user's measurements; anything else is 404, without revealing it exists."""

    def __init__(self, measurements: MeasurementRepository) -> None:
        self._measurements = measurements

    async def execute(self, *, user_id: str, measurement_id: str) -> None:
        if not await self._measurements.delete(user_id, measurement_id):
            raise NotFound("Measurement not found")
