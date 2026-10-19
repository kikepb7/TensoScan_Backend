"""Measurement routes of the previous version, reading the ``recognition_results`` collection.

Spec B-002 (T-B002-07) re-points them to the new ``measurements`` collection and finally removes them.
"""

from typing import Any

from bson import ObjectId
from fastapi import APIRouter, Depends, Path
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse

from app.api.deps import ContainerDep, CurrentUser
from app.api.legacy.headers import deprecated
from app.domain.errors import FieldIssue, NotFound, ValidationFailed
from app.infrastructure.legacy_ocr.models import Measurement
from app.infrastructure.legacy_reports.html_render import render_measurements_html
from app.infrastructure.legacy_reports.pdf_generator import generate_pdf_from_html

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
    measurements = await _results_of(container, user.id)
    if not measurements:
        raise NotFound("No measurements found")
    return HTMLResponse(content=render_measurements_html(measurements))


@router.get("/user/measurements/pdf")
async def measurements_pdf(user: CurrentUser, container: ContainerDep) -> StreamingResponse:
    measurements = await _results_of(container, user.id)
    if not measurements:
        raise NotFound("No measurements found")
    pdf_file = await run_in_threadpool(generate_pdf_from_html, render_measurements_html(measurements))
    return StreamingResponse(
        pdf_file,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=mediciones.pdf"},
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
