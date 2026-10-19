"""Request/response examples of ``specs/contracts/examples/`` are recorded from the real API and kept in sync.

The files are what the app team copies into their own tests, so they must never drift from the implementation:

* by default this test replays the scenarios and fails if a committed example differs from what the API returns;
* ``TENSOSCAN_UPDATE_EXAMPLES=1 uv run pytest tests/contract/test_api_examples.py`` rewrites them.

Only two things are normalised, because they differ on every run: the Mongo ``id`` of each measurement (replaced by a
stable ``6713a1b2c3d4e5f6000000NN`` in order of appearance, cursors are re-encoded accordingly) and the bearer token
(never recorded).
"""

import json
import os
from pathlib import Path
from typing import Any

import httpx

from app.domain.measurements.cursor import PageCursor
from tests.conftest import bearer, register_and_login

EXAMPLES = Path(__file__).resolve().parents[2] / "specs" / "contracts" / "examples"
UPDATE = os.getenv("TENSOSCAN_UPDATE_EXAMPLES") == "1"
CLIENT_ID = "5b0b6f0e-6a1f-4d8e-9d0a-2f1c3b7e9a10"
UNKNOWN_ID = "6713a1b2c3d4e5f600000099"
AUTHORIZATION = "Bearer <access_token>"


class Recorder:
    def __init__(self, client: httpx.AsyncClient, tokens: dict[str, Any]) -> None:
        self._client = client
        self._tokens = tokens
        self._ids: dict[str, str] = {}
        self.examples: dict[str, dict[str, Any]] = {}

    def _stable(self, real_id: str) -> str:
        return self._ids.setdefault(real_id, f"6713a1b2c3d4e5f6{len(self._ids) + 1:08x}")

    def _normalise(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._normalise_field(key, item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._normalise(item) for item in value]
        return value

    def _normalise_field(self, key: str, value: Any) -> Any:
        if key == "id" and isinstance(value, str):
            return self._stable(value)
        if key == "next_cursor" and isinstance(value, str):
            cursor = PageCursor.decode(value)
            return PageCursor(cursor.measured_at, self._stable(cursor.id)).encode()
        return self._normalise(value)

    async def call(
        self, name: str, description: str, method: str, path: str, body: dict[str, Any] | None = None
    ) -> httpx.Response:
        response = await self._client.request(method, path, headers=bearer(self._tokens), json=body)
        request: dict[str, Any] = {"method": method, "path": path, "headers": {"Authorization": AUTHORIZATION}}
        if body is not None:
            request["headers"]["Content-Type"] = "application/json"
            request["body"] = body
        answer: dict[str, Any] = {"status": response.status_code}
        if response.content:
            answer["body"] = self._normalise(response.json())
        self.examples[name] = {"description": description, "request": request, "response": answer}
        return response


def _payload(**changes: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "client_id": CLIENT_ID,
        "systolic": 128,
        "diastolic": 82,
        "pulse": 67,
        "measured_at": "2026-10-19T06:15:00Z",
        "source": "camera",
        "recognition": {"confidence": 0.93, "model_version": "digits-1.0.0", "edited_fields": ["pulse"]},
    }
    payload.update(changes)
    return payload


async def _record_all(client: httpx.AsyncClient) -> dict[str, dict[str, Any]]:
    recorder = Recorder(client, await register_and_login(client))
    created = await recorder.call(
        "measurements-create.201",
        "A new measurement: 201 Created with the stored resource (id and created_at are set by the server).",
        "POST",
        "/v1/measurements",
        _payload(),
    )
    await recorder.call(
        "measurements-create-replay.200",
        "The same request again (idempotent retry): 200 OK with the same resource and id; nothing is duplicated.",
        "POST",
        "/v1/measurements",
        _payload(),
    )
    await recorder.call(
        "measurements-create-conflict.409",
        "Same client_id but a different pulse: 409 CLIENT_ID_CONFLICT.",
        "POST",
        "/v1/measurements",
        _payload(pulse=68),
    )
    await recorder.call(
        "measurements-create-invalid.422",
        "A reading that breaks reading-rules.md: 422 MEASUREMENT_INVALID with one detail per failure.",
        "POST",
        "/v1/measurements",
        _payload(client_id="0b9a4f5e-3c1d-4b7a-8e2f-6d5c4b3a2f10", systolic=80, pulse=None),
    )
    for minutes, systolic in ((10, 120), (20, 131)):
        await client.post(
            "/v1/measurements",
            headers=bearer(recorder._tokens),
            json=_payload(
                client_id=f"7c1d2e3f-4a5b-4c6d-8e7f-00000000000{minutes // 10}",
                systolic=systolic,
                measured_at=f"2026-10-19T06:{minutes}:00Z",
            ),
        )
    assert created.status_code == 201
    await recorder.call(
        "measurements-list-page.200",
        "First page of the listing (limit=2, newest first); next_cursor leads to the following page.",
        "GET",
        "/v1/measurements?limit=2",
    )
    await recorder.call(
        "error-envelope.404",
        "Every 4xx/5xx has the same envelope {error: {code, message, details}}. Here: a measurement that does not "
        "exist (or belongs to someone else).",
        "DELETE",
        f"/v1/measurements/{UNKNOWN_ID}",
    )
    return recorder.examples


def _render(example: dict[str, Any]) -> str:
    return json.dumps(example, indent=2, ensure_ascii=False) + "\n"


async def test_the_committed_examples_are_what_the_api_really_answers(client: httpx.AsyncClient) -> None:
    recorded = await _record_all(client)

    if UPDATE:
        EXAMPLES.mkdir(parents=True, exist_ok=True)
        for name, example in recorded.items():
            (EXAMPLES / f"{name}.json").write_text(_render(example))
    on_disk = {path.stem: path.read_text() for path in sorted(EXAMPLES.glob("*.json"))}

    assert on_disk == {name: _render(example) for name, example in recorded.items()}, (
        "specs/contracts/examples is out of date: run `TENSOSCAN_UPDATE_EXAMPLES=1 uv run pytest "
        "tests/contract/test_api_examples.py` and commit the result (and sync it to the app repository)"
    )


async def test_the_examples_cover_the_documented_outcomes(client: httpx.AsyncClient) -> None:
    recorded = await _record_all(client)

    statuses = {name: example["response"]["status"] for name, example in recorded.items()}

    assert statuses == {
        "measurements-create.201": 201,
        "measurements-create-replay.200": 200,
        "measurements-create-conflict.409": 409,
        "measurements-create-invalid.422": 422,
        "measurements-list-page.200": 200,
        "error-envelope.404": 404,
    }
    replay = recorded["measurements-create-replay.200"]["response"]["body"]
    assert replay == recorded["measurements-create.201"]["response"]["body"]
    page = recorded["measurements-list-page.200"]["response"]["body"]
    assert len(page["items"]) == 2 and page["next_cursor"]
