"""Measurement routes of the previous version, reading the ``recognition_results`` collection.

Spec B-002 (T-B002-07) re-points them to the new ``measurements`` collection and finally removes them.
"""

from typing import Any

from bson import ObjectId
from fastapi import APIRouter, Depends, Path, Response
from fastapi.responses import HTMLResponse

from app.api.deps import ContainerDep, CurrentUser
from app.api.legacy.headers import deprecated, deprecation_headers
from app.api.legacy.measurements_html import render_measurements_html
from app.application.measurements.generate_report import parse_timezone
from app.domain.errors import FieldIssue, NotFound, ValidationFailed
from app.infrastructure.legacy_ocr.models import Measurement

router = APIRouter(prefix="/ocr", tags=["legacy"], dependencies=[Depends(deprecated("/v1/measurements"))])


def _user_object_id(user_id: str) -> ObjectId:
    return ObjectId(user_id)


async def _results_of(container: ContainerDep, user_id: str) -> list[dict[str, Any]]:
    cursor = container.db["recognition_results"].find({"user_id": _user_object_id(user_id)})
    return await cursor.to_list(length=None)


@router.get("/measurements", response_model=list[Measurement])
async def list_measurements(user: CurrentUser, container: ContainerDep) -> list[dict[str, Any]]:
    return [
        {
            "measurement_id": str(item["_id"]),
            "user_id": str(item["user_id"]),
            "filename": item["filename"],
            "result": {
                "high_pressure": item["result"]["high_pressure"],
                "low_pressure": item["result"]["low_pressure"],
                "pulse": item["result"]["pulse"],
                "confidence": item["result"]["confidence"],
            },
            "timestamp": item["timestamp"],
        }
        for item in await _results_of(container, user.id)
    ]


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
    measurement_id: str = Path(...),
) -> None:
    if not ObjectId.is_valid(measurement_id):
        raise ValidationFailed([FieldIssue("measurement_id", "INVALID")])
    result = await container.db["recognition_results"].delete_one(
        {"_id": ObjectId(measurement_id), "user_id": _user_object_id(user.id)}
    )
    if result.deleted_count == 0:
        raise NotFound("Measurement not found")
