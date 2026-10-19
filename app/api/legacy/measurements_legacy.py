"""Measurement routes of the previous version, now reading and writing the ``measurements`` collection.

They answer in the old shape through ``LegacyMeasurementMapper`` (design §5) so that an old app and a new one see the
same data. Spec B-002 (REQ-B002-13) removes them once the new app is adopted.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Response
from fastapi.responses import HTMLResponse

from app.api.deps import ContainerDep, CurrentUser
from app.api.legacy.headers import deprecated, deprecation_headers
from app.api.legacy.mapper import LegacyMeasurement, LegacyMeasurementMapper
from app.api.legacy.measurements_html import render_measurements_html
from app.api.legacy.usage import record_legacy_usage
from app.application.measurements.generate_report import parse_timezone
from app.domain.errors import FieldIssue, NotFound, ValidationFailed
from app.domain.measurements.cursor import is_valid_measurement_id

router = APIRouter(
    prefix="/ocr",
    tags=["legacy"],
    dependencies=[Depends(deprecated("/v1/measurements")), Depends(record_legacy_usage)],
)


@router.get("/measurements", response_model=list[LegacyMeasurement])
async def list_measurements(user: CurrentUser, container: ContainerDep) -> list[LegacyMeasurement]:
    measurements = await container.list_all_measurements.execute(user_id=user.id)
    # The previous version listed oldest first (insertion order); v1 lists newest first.
    return [LegacyMeasurementMapper.to_legacy(m) for m in reversed(measurements)]


@router.get("/user/measurements/html", response_class=HTMLResponse)
async def measurements_html(user: CurrentUser, container: ContainerDep) -> HTMLResponse:
    measurements = await container.list_all_measurements.execute(user_id=user.id)
    if not measurements:
        raise NotFound("No measurements found")
    zone = parse_timezone(container.settings.default_report_tz)
    return HTMLResponse(
        content=render_measurements_html(measurements, zone), headers=deprecation_headers("/v1/measurements")
    )


@router.get("/user/measurements/pdf")
async def measurements_pdf(user: CurrentUser, container: ContainerDep) -> Response:
    report = await container.generate_report.execute(user=user, since=None, until=None, timezone=None)
    return Response(
        content=report.content,
        media_type="application/pdf",
        headers={
            **deprecation_headers("/v1/measurements/report?format=pdf"),
            "Content-Disposition": "attachment; filename=mediciones.pdf",
        },
    )


@router.delete("/remove/measurements/{measurement_id}", status_code=204)
async def remove_measurement(
    user: CurrentUser,
    container: ContainerDep,
    measurement_id: Annotated[str, Path()],
) -> None:
    if not is_valid_measurement_id(measurement_id):
        raise ValidationFailed([FieldIssue("measurement_id", "INVALID")])
    await container.delete_measurement.execute(user_id=user.id, measurement_id=measurement_id)
