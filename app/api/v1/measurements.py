from datetime import datetime
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Path, Query, Response, status

from app.api.deps import ContainerDep, CurrentUser
from app.api.schemas.errors import ErrorResponse
from app.api.schemas.measurements import CreateMeasurementRequest, MeasurementPageResponse, MeasurementResponse

router = APIRouter(prefix="/v1/measurements", tags=["measurements"])

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200


@router.post(
    "",
    response_model=MeasurementResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_200_OK: {"model": MeasurementResponse, "description": "Idempotent replay of a stored measurement"},
        status.HTTP_409_CONFLICT: {"model": ErrorResponse, "description": "CLIENT_ID_CONFLICT"},
        HTTPStatus.UNPROCESSABLE_ENTITY: {"model": ErrorResponse, "description": "MEASUREMENT_INVALID"},
    },
)
async def create_measurement(
    body: CreateMeasurementRequest, response: Response, user: CurrentUser, container: ContainerDep
) -> MeasurementResponse:
    result = await container.create_measurement.execute(body.to_command(user.id))
    if not result.created:
        response.status_code = status.HTTP_200_OK
    return MeasurementResponse.from_domain(result.measurement)


@router.get(
    "",
    response_model=MeasurementPageResponse,
    responses={HTTPStatus.UNPROCESSABLE_ENTITY: {"model": ErrorResponse, "description": "VALIDATION_ERROR"}},
)
async def list_measurements(
    user: CurrentUser,
    container: ContainerDep,
    since: Annotated[datetime | None, Query(alias="from")] = None,
    until: Annotated[datetime | None, Query(alias="to")] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> MeasurementPageResponse:
    listing = await container.list_measurements.execute(
        user_id=user.id, since=since, until=until, limit=limit, cursor=cursor
    )
    return MeasurementPageResponse.from_domain(listing)


@router.delete(
    "/{id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse, "description": "NOT_FOUND"}},
)
async def delete_measurement(
    id: Annotated[str, Path()],  # noqa: A002 - the contract names the path parameter {id}
    user: CurrentUser,
    container: ContainerDep,
) -> Response:
    await container.delete_measurement.execute(user_id=user.id, measurement_id=id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
