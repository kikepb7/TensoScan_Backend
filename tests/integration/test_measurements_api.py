"""``/v1/measurements`` over HTTP: creation, idempotency, validation, listing, pagination, isolation and deletion."""

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest

from tests.conftest import NEW_USER, bearer, register_and_login

GOLDEN = Path(__file__).resolve().parents[2] / "specs" / "contracts" / "reading-golden-cases.json"
VALIDATION_CASES: list[dict[str, Any]] = json.loads(GOLDEN.read_text())["validation_cases"]
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


@pytest.fixture
async def tokens(client: httpx.AsyncClient) -> dict[str, Any]:
    return await register_and_login(client)


async def test_a_valid_measurement_is_created_with_201_and_the_documented_shape(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    payload = body()

    response = await post(client, tokens, payload)

    assert response.status_code == 201
    created = response.json()
    assert created == {
        "id": created["id"],
        "client_id": payload["client_id"],
        "systolic": 128,
        "diastolic": 82,
        "pulse": 67,
        "measured_at": "2026-10-19T06:15:00Z",
        "created_at": "2026-10-19T08:00:00Z",
        "source": "camera",
        "recognition": {"confidence": 0.93, "model_version": "digits-1.0.0", "edited_fields": ["pulse"]},
    }
    assert len(created["id"]) == 24


async def test_a_manual_measurement_has_no_recognition(client: httpx.AsyncClient, tokens: dict[str, Any]) -> None:
    response = await post(client, tokens, body(source="manual", recognition=None))

    assert response.status_code == 201
    assert response.json()["recognition"] is None


async def test_repeating_the_same_request_answers_200_with_the_same_resource_and_stores_one_document(
    client: httpx.AsyncClient, tokens: dict[str, Any], db: Any, clock: Any
) -> None:
    payload = body()
    first = await post(client, tokens, payload)
    clock.advance(minutes=10)

    second = await post(client, tokens, payload)

    assert (first.status_code, second.status_code) == (201, 200)
    assert second.json() == first.json()
    assert await db["measurements"].count_documents({}) == 1


async def test_the_replay_ignores_the_order_of_edited_fields_and_the_offset_notation(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    recognition = {"confidence": 0.93, "model_version": "m", "edited_fields": ["pulse", "systolic"]}
    original = body(recognition=recognition)
    await post(client, tokens, original)
    again = {
        **original,
        "measured_at": "2026-10-19T08:15:00+02:00",
        "recognition": {**recognition, "edited_fields": ["systolic", "pulse"]},
    }

    response = await post(client, tokens, again)

    assert response.status_code == 200


async def test_the_same_client_id_with_a_different_pulse_is_a_409_conflict(
    client: httpx.AsyncClient, tokens: dict[str, Any], db: Any
) -> None:
    payload = body()
    await post(client, tokens, payload)

    response = await post(client, tokens, {**payload, "pulse": 68})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CLIENT_ID_CONFLICT"
    assert await db["measurements"].count_documents({}) == 1


async def test_two_simultaneous_identical_posts_answer_201_and_200(
    client: httpx.AsyncClient, tokens: dict[str, Any], db: Any
) -> None:
    payload = body()

    responses = await asyncio.gather(post(client, tokens, payload), post(client, tokens, payload))

    assert sorted(response.status_code for response in responses) == [200, 201]
    assert responses[0].json() == responses[1].json()
    assert await db["measurements"].count_documents({}) == 1


@pytest.mark.parametrize("case", VALIDATION_CASES, ids=[case["name"] for case in VALIDATION_CASES])
async def test_golden_validation_cases_over_http(
    client: httpx.AsyncClient, tokens: dict[str, Any], case: dict[str, Any]
) -> None:
    payload = body(**case["input"])

    response = await post(client, tokens, payload)

    if case["errors"]:
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "MEASUREMENT_INVALID"
        assert response.json()["error"]["details"] == case["errors"]
    else:  # warnings never block (REQ-B002-05)
        assert response.status_code == 201


async def test_a_missing_value_is_reported_as_required_not_as_a_schema_error(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    response = await post(client, tokens, body(pulse=...))

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "MEASUREMENT_INVALID",
        "message": "The measurement is not valid",
        "details": [{"field": "pulse", "code": "REQUIRED"}],
    }


@pytest.mark.parametrize(
    ("changes", "details"),
    [
        ({"measured_at": "2026-10-19T08:06:00Z"}, [{"field": "measured_at", "code": "IN_FUTURE"}]),
        ({"measured_at": "1999-12-31T23:59:59Z"}, [{"field": "measured_at", "code": "TOO_OLD"}]),
        ({"recognition": None}, [{"field": "recognition", "code": "REQUIRED"}]),
        ({"source": "manual"}, [{"field": "recognition", "code": "NOT_ALLOWED"}]),
        (
            {"recognition": {"confidence": 1.5, "model_version": "", "edited_fields": []}},
            [
                {"field": "recognition.confidence", "code": "OUT_OF_RANGE"},
                {"field": "recognition.model_version", "code": "REQUIRED"},
            ],
        ),
        (
            {"recognition": {"confidence": 0.5, "model_version": "v" * 65, "edited_fields": []}},
            [{"field": "recognition.model_version", "code": "TOO_LONG"}],
        ),
        (
            {"systolic": 300, "pulse": 10, "measured_at": "2026-10-20T00:00:00Z"},
            [
                {"field": "systolic", "code": "OUT_OF_RANGE"},
                {"field": "pulse", "code": "OUT_OF_RANGE"},
                {"field": "measured_at", "code": "IN_FUTURE"},
            ],
        ),
    ],
)
async def test_invalid_dates_and_source_recognition_combinations_answer_measurement_invalid(
    client: httpx.AsyncClient, tokens: dict[str, Any], changes: dict[str, Any], details: list[dict[str, str]]
) -> None:
    response = await post(client, tokens, body(**changes))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "MEASUREMENT_INVALID"
    assert response.json()["error"]["details"] == details


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"client_id": "not-a-uuid"}, "client_id"),
        ({"client_id": "5b0b6f0e-6a1f-1d8e-9d0a-2f1c3b7e9a10"}, "client_id"),  # not version 4
        ({"measured_at": "2026-10-19T06:15:00"}, "measured_at"),  # no time zone
        ({"measured_at": ...}, "measured_at"),
        ({"source": "scanner"}, "source"),
        ({"systolic": "128"}, "systolic"),
        ({"systolic": 128.5}, "systolic"),
        ({"systolic": True}, "systolic"),
        ({"recognition": {"confidence": 0.9, "model_version": "m", "edited_fields": ["height"]}}, "recognition"),
    ],
)
async def test_structurally_wrong_bodies_answer_a_generic_validation_error(
    client: httpx.AsyncClient, tokens: dict[str, Any], changes: dict[str, Any], field: str
) -> None:
    response = await post(client, tokens, body(**changes))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert response.json()["error"]["details"][0]["field"].startswith(field)


async def test_measurements_require_authentication(client: httpx.AsyncClient) -> None:
    responses = [
        await client.post("/v1/measurements", json=body()),
        await client.get("/v1/measurements"),
        await client.delete("/v1/measurements/" + "a" * 24),
    ]

    assert [response.status_code for response in responses] == [401, 401, 401]


async def _seed(client: httpx.AsyncClient, tokens: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """``count`` measurements; every third shares its instant with the previous one (exercises the tie-break)."""
    created = []
    for index in range(count):
        minute = index - index % 3
        at = f"2026-10-{1 + minute // 1440:02d}T{minute % 1440 // 60:02d}:{minute % 60:02d}:00Z"
        response = await post(client, tokens, body(measured_at=at))
        assert response.status_code == 201, response.text
        created.append(response.json())
    return created


async def test_120_measurements_are_paged_50_50_20_in_order_without_repeats(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    await _seed(client, tokens, 120)

    pages, cursor = [], None
    while True:
        params: dict[str, Any] = {"limit": 50, **({"cursor": cursor} if cursor else {})}
        response = await client.get("/v1/measurements", headers=bearer(tokens), params=params)
        assert response.status_code == 200
        pages.append(response.json())
        cursor = response.json()["next_cursor"]
        if cursor is None:
            break

    items = [item for page in pages for item in page["items"]]
    assert [len(page["items"]) for page in pages] == [50, 50, 20]
    assert len({item["id"] for item in items}) == 120
    keys = [(item["measured_at"], item["id"]) for item in items]
    assert keys == sorted(keys, reverse=True)


async def test_the_default_limit_is_50_and_the_listing_is_empty_for_a_new_user(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    empty = await client.get("/v1/measurements", headers=bearer(tokens))
    await _seed(client, tokens, 51)

    first = await client.get("/v1/measurements", headers=bearer(tokens))

    assert empty.json() == {"items": [], "next_cursor": None}
    assert len(first.json()["items"]) == 50
    assert first.json()["next_cursor"] is not None


async def test_from_and_to_filter_by_measured_at(client: httpx.AsyncClient, tokens: dict[str, Any]) -> None:
    for day in (10, 11, 12, 13):
        await post(client, tokens, body(measured_at=f"2026-10-{day}T10:00:00Z"))

    response = await client.get(
        "/v1/measurements",
        headers=bearer(tokens),
        params={"from": "2026-10-11T10:00:00Z", "to": "2026-10-12T12:00:00+02:00"},
    )

    assert [item["measured_at"] for item in response.json()["items"]] == [
        "2026-10-12T10:00:00Z",
        "2026-10-11T10:00:00Z",
    ]


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 201}, {"limit": "many"}, {"from": "yesterday"}])
async def test_out_of_range_query_parameters_are_validation_errors(
    client: httpx.AsyncClient, tokens: dict[str, Any], params: dict[str, Any]
) -> None:
    response = await client.get("/v1/measurements", headers=bearer(tokens), params=params)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


async def test_the_limit_bounds_are_accepted(client: httpx.AsyncClient, tokens: dict[str, Any]) -> None:
    for limit in (1, 200):
        response = await client.get("/v1/measurements", headers=bearer(tokens), params={"limit": limit})
        assert response.status_code == 200


async def test_an_invalid_cursor_is_a_validation_error_on_the_cursor_field(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    response = await client.get("/v1/measurements", headers=bearer(tokens), params={"cursor": "garbage"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert response.json()["error"]["details"] == [{"field": "cursor", "code": "INVALID"}]


async def test_users_never_see_each_others_measurements(client: httpx.AsyncClient, tokens: dict[str, Any]) -> None:
    other = await register_and_login(client, {**NEW_USER, "email": "other@example.com"})
    mine = (await _seed(client, tokens, 2))[0]

    theirs = await client.get("/v1/measurements", headers=bearer(other))
    stolen = await client.delete(f"/v1/measurements/{mine['id']}", headers=bearer(other))
    still_mine = await client.get("/v1/measurements", headers=bearer(tokens))

    assert theirs.json() == {"items": [], "next_cursor": None}
    assert stolen.status_code == 404
    assert stolen.json()["error"]["code"] == "NOT_FOUND"
    assert len(still_mine.json()["items"]) == 2


async def test_two_users_may_use_the_same_client_id(client: httpx.AsyncClient, tokens: dict[str, Any]) -> None:
    other = await register_and_login(client, {**NEW_USER, "email": "other@example.com"})
    payload = body()

    first = await post(client, tokens, payload)
    second = await post(client, other, payload)

    assert (first.status_code, second.status_code) == (201, 201)
    assert first.json()["id"] != second.json()["id"]


async def test_deleting_an_own_measurement_answers_204_and_then_404(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    created = (await post(client, tokens, body())).json()

    deleted = await client.delete(f"/v1/measurements/{created['id']}", headers=bearer(tokens))
    again = await client.delete(f"/v1/measurements/{created['id']}", headers=bearer(tokens))
    listing = await client.get("/v1/measurements", headers=bearer(tokens))

    assert (deleted.status_code, deleted.content) == (204, b"")
    assert again.status_code == 404
    assert listing.json()["items"] == []


@pytest.mark.parametrize("bad_id", ["not-an-id", "123", "0" * 24])
async def test_deleting_a_malformed_or_unknown_id_is_404(
    client: httpx.AsyncClient, tokens: dict[str, Any], bad_id: str
) -> None:
    response = await client.delete(f"/v1/measurements/{bad_id}", headers=bearer(tokens))

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


async def test_a_deleted_measurement_can_be_created_again_with_its_client_id(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    payload = body()
    created = (await post(client, tokens, payload)).json()
    await client.delete(f"/v1/measurements/{created['id']}", headers=bearer(tokens))

    again = await post(client, tokens, payload)

    assert again.status_code == 201
    assert again.json()["id"] != created["id"]
