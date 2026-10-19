"""Request helpers shared by the measurement tests."""

from typing import Any
from uuid import uuid4

import httpx

from tests.conftest import bearer

# The fixed test clock reads 2026-10-19T08:00:00Z.
MEASURED_AT = "2026-10-19T06:15:00Z"


def body(**changes: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "client_id": str(uuid4()),
        "systolic": 128,
        "diastolic": 82,
        "pulse": 67,
        "measured_at": MEASURED_AT,
        "source": "camera",
        "recognition": {"confidence": 0.93, "model_version": "digits-1.0.0", "edited_fields": ["pulse"]},
    }
    payload.update(changes)
    return {key: value for key, value in payload.items() if value is not ...}


async def post(client: httpx.AsyncClient, tokens: dict[str, Any], payload: dict[str, Any]) -> httpx.Response:
    return await client.post("/v1/measurements", headers=bearer(tokens), json=payload)
