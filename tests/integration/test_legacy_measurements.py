"""Legacy measurement routes on the new ``measurements`` collection (REQ-B002-12, 14; CA-B002-10)."""

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from bson import ObjectId

from app.api.legacy.ocr_legacy import LegacyOcr
from app.core.config import Settings
from app.main import create_app
from tests.conftest import NEW_USER, bearer, register_and_login
from tests.support.fakes import FakeClock
from tests.support.measurements import body, post


@pytest.fixture
async def tokens(client: httpx.AsyncClient) -> dict[str, Any]:
    return await register_and_login(client)


async def test_measurements_created_by_v1_are_listed_by_the_legacy_route_in_the_old_shape(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    created = (await post(client, tokens, body(systolic=128, diastolic=82, pulse=67))).json()

    response = await client.get("/ocr/measurements", headers=bearer(tokens))

    assert response.status_code == 200
    assert response.json() == [
        {
            "measurement_id": created["id"],
            "user_id": tokens["user"]["id"],
            "filename": "",
            "result": {"high_pressure": "128", "low_pressure": "82", "pulse": "67", "confidence": 0.93},
            "timestamp": "2026-10-19T06:15:00",
        }
    ]
    assert response.headers["Deprecation"] == "true"


async def test_a_manual_measurement_has_confidence_one_in_the_legacy_shape(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    await post(client, tokens, body(source="manual", recognition=None))

    [item] = (await client.get("/ocr/measurements", headers=bearer(tokens))).json()

    assert item["result"]["confidence"] == 1.0


async def test_the_legacy_timestamp_is_utc_without_a_zone(client: httpx.AsyncClient, tokens: dict[str, Any]) -> None:
    await post(client, tokens, body(measured_at="2026-10-19T08:15:00+02:00"))

    [item] = (await client.get("/ocr/measurements", headers=bearer(tokens))).json()

    assert item["timestamp"] == "2026-10-19T06:15:00"


async def test_the_legacy_listing_is_oldest_first_like_before(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    for day in (12, 10, 11):
        await post(client, tokens, body(measured_at=f"2026-10-{day}T10:00:00Z"))

    items = (await client.get("/ocr/measurements", headers=bearer(tokens))).json()

    assert [item["timestamp"] for item in items] == [f"2026-10-{day}T10:00:00" for day in (10, 11, 12)]


async def test_deleting_through_the_legacy_route_removes_the_v1_measurement(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    created = (await post(client, tokens, body())).json()

    deleted = await client.delete(f"/ocr/remove/measurements/{created['id']}", headers=bearer(tokens))
    again = await client.delete(f"/ocr/remove/measurements/{created['id']}", headers=bearer(tokens))
    listing = await client.get("/v1/measurements", headers=bearer(tokens))

    assert (deleted.status_code, again.status_code) == (204, 404)
    assert listing.json()["items"] == []


async def test_a_user_only_sees_their_own_measurements_through_the_legacy_route(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    other = await register_and_login(client, {**NEW_USER, "email": "other@example.com"})
    await post(client, other, body())

    response = await client.get("/ocr/measurements", headers=bearer(tokens))

    assert response.json() == []


async def test_legacy_routes_require_authentication(client: httpx.AsyncClient) -> None:
    responses = [
        await client.get("/ocr/measurements"),
        await client.get("/ocr/user/measurements/pdf"),
        await client.get("/ocr/user/measurements/html"),
        await client.delete("/ocr/remove/measurements/" + "a" * 24),
    ]

    assert [response.status_code for response in responses] == [401] * 4


# --- usage log (REQ-B002-14) -----------------------------------------------------------------------------------


async def test_every_legacy_request_is_logged_with_route_and_user_id(
    client: httpx.AsyncClient, tokens: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    created = (await post(client, tokens, body())).json()
    caplog.set_level(logging.INFO, logger="app.api.legacy.usage")
    headers = bearer(tokens)

    await client.get("/ocr/measurements", headers=headers)
    await client.get("/ocr/user/measurements/pdf", headers=headers)
    await client.get("/ocr/user/measurements/html", headers=headers)
    await client.delete(f"/ocr/remove/measurements/{created['id']}", headers=headers)
    await client.get("/me", headers=headers)
    await client.post("/ai/chatbot", headers=headers, json={"prompt": "hi"})

    records = [r for r in caplog.records if r.name == "app.api.legacy.usage"]
    assert [r.legacy_route for r in records] == [  # type: ignore[attr-defined]
        "GET /ocr/measurements",
        "GET /ocr/user/measurements/pdf",
        "GET /ocr/user/measurements/html",
        "DELETE /ocr/remove/measurements/{measurement_id}",
        "GET /me",
        "POST /ai/chatbot",
    ]
    assert {r.user_id for r in records} == {tokens["user"]["id"]}  # type: ignore[attr-defined]
    assert all(tokens["user"]["id"] in r.getMessage() for r in records)


async def test_the_unauthenticated_legacy_routes_are_logged_with_the_route_only(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="app.api.legacy.usage")

    await client.post("/register", json=NEW_USER)
    await client.post("/login", data={"username": NEW_USER["email"], "password": NEW_USER["password"]})

    records = [r for r in caplog.records if r.name == "app.api.legacy.usage"]
    assert [r.legacy_route for r in records] == ["POST /register", "POST /login"]  # type: ignore[attr-defined]
    assert all(r.user_id is None for r in records)  # type: ignore[attr-defined]


async def test_v1_requests_are_not_counted_as_legacy_usage(
    client: httpx.AsyncClient, tokens: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="app.api.legacy.usage")

    await post(client, tokens, body())
    await client.get("/v1/measurements", headers=bearer(tokens))

    assert [r for r in caplog.records if r.name == "app.api.legacy.usage"] == []


# --- POST /ocr/display-recognize (needs ENABLE_LEGACY_OCR) ----------------------------------------------------------


@pytest.fixture
def recognized() -> dict[str, Any]:
    """What the (replaced) recognizer answers; a test edits it to simulate other readings."""
    return {"high_pressure": "128", "low_pressure": "82", "pulse": "67", "confidence": 0.8}


@pytest.fixture
async def legacy_ocr_client(
    settings: Settings, db: Any, clock: FakeClock, monkeypatch: pytest.MonkeyPatch, recognized: dict[str, Any]
) -> AsyncIterator[httpx.AsyncClient]:
    """An app with the deprecated OCR enabled and the (TensorFlow) recognizer replaced by a canned answer."""
    monkeypatch.setattr(LegacyOcr, "recognize_display", lambda _self, _image: dict(recognized))
    app = create_app(settings.model_copy(update={"enable_legacy_ocr": True}), database=db, clock=clock)
    async with LifespanManager(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            yield http


IMAGE = {"image": ("display.jpg", b"not-really-a-jpeg", "image/jpeg")}


async def test_display_recognize_stores_into_the_new_collection_and_v1_sees_it(
    legacy_ocr_client: httpx.AsyncClient, db: Any
) -> None:
    tokens = await register_and_login(legacy_ocr_client)

    response = await legacy_ocr_client.post("/ocr/display-recognize", headers=bearer(tokens), files=IMAGE)
    listing = await legacy_ocr_client.get("/v1/measurements", headers=bearer(tokens))

    assert response.status_code == 200
    assert response.json() == {"high_pressure": "128", "low_pressure": "82", "pulse": "67", "confidence": 0.8}
    [item] = listing.json()["items"]
    assert (item["systolic"], item["diastolic"], item["pulse"], item["source"]) == (128, 82, 67, "camera")
    assert item["recognition"] == {"confidence": 0.8, "model_version": "legacy-keras", "edited_fields": []}
    assert item["measured_at"] == item["created_at"] == "2026-10-19T08:00:00Z"
    assert await db["recognition_results"].count_documents({}) == 0


async def test_each_display_recognize_call_is_its_own_measurement_with_a_random_client_id(
    legacy_ocr_client: httpx.AsyncClient,
) -> None:
    tokens = await register_and_login(legacy_ocr_client)

    for _ in range(2):
        await legacy_ocr_client.post("/ocr/display-recognize", headers=bearer(tokens), files=IMAGE)
    items = (await legacy_ocr_client.get("/v1/measurements", headers=bearer(tokens))).json()["items"]

    assert len(items) == 2
    assert items[0]["client_id"] != items[1]["client_id"]


async def test_display_recognize_skips_range_validation_as_documented(
    legacy_ocr_client: httpx.AsyncClient, recognized: dict[str, Any]
) -> None:
    """DOCUMENTED EXCEPTION (design §5): the deprecated route stores what the model read, even out of range."""
    tokens = await register_and_login(legacy_ocr_client)
    recognized.update(high_pressure="300", low_pressure="20", pulse="5")
    rejected_by_v1 = await post(legacy_ocr_client, tokens, body(systolic=300, diastolic=20, pulse=5))

    response = await legacy_ocr_client.post("/ocr/display-recognize", headers=bearer(tokens), files=IMAGE)
    listing = await legacy_ocr_client.get("/ocr/measurements", headers=bearer(tokens))

    assert rejected_by_v1.status_code == 422  # the same values through v1 are invalid...
    assert response.status_code == 200
    [item] = listing.json()
    assert item["result"] == {"high_pressure": "300", "low_pressure": "20", "pulse": "5", "confidence": 0.8}


async def test_display_recognize_does_not_store_a_result_that_is_not_numeric(
    legacy_ocr_client: httpx.AsyncClient, db: Any, recognized: dict[str, Any]
) -> None:
    tokens = await register_and_login(legacy_ocr_client)
    recognized.update(high_pressure="", pulse="1x")

    response = await legacy_ocr_client.post("/ocr/display-recognize", headers=bearer(tokens), files=IMAGE)

    assert response.status_code == 200  # the caller still gets what the model produced
    assert await db["measurements"].count_documents({}) == 0


async def test_display_recognize_is_logged_as_legacy_usage(
    legacy_ocr_client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    tokens = await register_and_login(legacy_ocr_client)
    caplog.set_level(logging.INFO, logger="app.api.legacy.usage")

    await legacy_ocr_client.post("/ocr/display-recognize", headers=bearer(tokens), files=IMAGE)

    [record] = [r for r in caplog.records if r.name == "app.api.legacy.usage"]
    assert record.legacy_route == "POST /ocr/display-recognize"  # type: ignore[attr-defined]
    assert record.user_id == tokens["user"]["id"]  # type: ignore[attr-defined]


async def test_account_deletion_also_erases_measurements_stored_in_the_legacy_collection(
    client: httpx.AsyncClient, db: Any, tokens: dict[str, Any]
) -> None:
    await post(client, tokens, body())
    await db["recognition_results"].insert_one({"user_id": ObjectId(tokens["user"]["id"]), "result": {}})

    response = await client.request(
        "DELETE", "/v1/users/me", headers=bearer(tokens), json={"password": NEW_USER["password"]}
    )

    assert response.status_code == 204
    assert await db["measurements"].count_documents({}) == 0
    assert await db["recognition_results"].count_documents({}) == 0
