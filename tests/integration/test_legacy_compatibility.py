"""The previous app version keeps working against the same routes while the new one is rolled out."""

from datetime import UTC, datetime
from typing import Any

import bcrypt
import httpx
from bson import ObjectId

from tests.conftest import NEW_USER, bearer, register_and_login


async def test_legacy_register_login_and_me_keep_their_shapes(client: httpx.AsyncClient) -> None:
    registered = await client.post("/register", json=NEW_USER)
    login = await client.post("/login", data={"username": NEW_USER["email"], "password": NEW_USER["password"]})
    me = await client.get("/me", headers={"Authorization": f"Bearer {login.json()['access_token']}"})

    assert registered.status_code == 200
    assert registered.json() == {"msg:": "User registered successfully"}
    assert {"access_token", "token_type"} <= login.json().keys()
    assert me.json()["email"] == NEW_USER["email"]
    assert registered.headers["Deprecation"] == "true"
    assert "/v1/auth/register" in registered.headers["Link"]


async def test_a_user_created_by_the_previous_version_can_still_log_in_and_is_upgraded_to_argon2(
    client: httpx.AsyncClient, db: Any
) -> None:
    legacy_hash = bcrypt.hashpw(b"old-password", bcrypt.gensalt(rounds=4)).decode()
    await db["users"].insert_one(
        {"name": "Luis", "last_name": "Gil", "email": "luis@example.com", "hashed_password": legacy_hash}
    )

    login = await client.post("/login", data={"username": "luis@example.com", "password": "old-password"})

    assert login.status_code == 200
    stored = await db["users"].find_one({"email": "luis@example.com"})
    assert stored["password_hash"].startswith("$argon2id$")
    assert "hashed_password" not in stored
    again = await client.post("/v1/auth/login", data={"username": "luis@example.com", "password": "old-password"})
    assert again.status_code == 200


async def test_legacy_chatbot_route_answers_with_the_old_shape(client: httpx.AsyncClient) -> None:
    tokens = await register_and_login(client)

    response = await client.post(
        "/ai/chatbot", headers=bearer(tokens), json={"prompt": "hi", "history": [{"role": "user", "content": "x"}]}
    )

    assert response.status_code == 200
    assert response.json() == {"response": "Hello from the model"}


async def test_legacy_measurements_are_listed_and_deleted_with_the_old_shape(
    client: httpx.AsyncClient, db: Any
) -> None:
    tokens = await register_and_login(client)
    user_id = ObjectId(tokens["user"]["id"])
    result = {"high_pressure": "128", "low_pressure": "82", "pulse": "67", "confidence": 0.9}
    inserted = await db["recognition_results"].insert_one(
        {"user_id": user_id, "filename": "a.jpg", "result": result, "timestamp": datetime(2026, 1, 2, tzinfo=UTC)}
    )

    listing = await client.get("/ocr/measurements", headers=bearer(tokens))
    deleted = await client.delete(f"/ocr/remove/measurements/{inserted.inserted_id}", headers=bearer(tokens))
    missing = await client.delete(f"/ocr/remove/measurements/{inserted.inserted_id}", headers=bearer(tokens))

    assert listing.status_code == 200
    assert listing.json()[0]["measurement_id"] == str(inserted.inserted_id)
    assert listing.json()[0]["result"] == result
    assert deleted.status_code == 204
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "NOT_FOUND"


async def test_a_user_cannot_delete_someone_elses_legacy_measurement(client: httpx.AsyncClient, db: Any) -> None:
    owner = await register_and_login(client)
    intruder = await register_and_login(client, {**NEW_USER, "email": "intruder@example.com"})
    inserted = await db["recognition_results"].insert_one(
        {
            "user_id": ObjectId(owner["user"]["id"]),
            "filename": "a.jpg",
            "result": {"high_pressure": "1", "low_pressure": "1", "pulse": "1", "confidence": 1.0},
            "timestamp": datetime(2026, 1, 2, tzinfo=UTC),
        }
    )

    response = await client.delete(f"/ocr/remove/measurements/{inserted.inserted_id}", headers=bearer(intruder))

    assert response.status_code == 404
    assert await db["recognition_results"].count_documents({}) == 1


async def test_an_invalid_measurement_id_is_a_validation_error_not_a_server_error(client: httpx.AsyncClient) -> None:
    tokens = await register_and_login(client)

    response = await client.delete("/ocr/remove/measurements/not-an-id", headers=bearer(tokens))

    assert response.status_code == 422
    assert response.json()["error"]["details"] == [{"field": "measurement_id", "code": "INVALID"}]


async def test_the_legacy_ocr_routes_are_not_even_registered_when_the_feature_is_off(
    client: httpx.AsyncClient,
) -> None:
    tokens = await register_and_login(client)

    response = await client.post(
        "/ocr/display-recognize", headers=bearer(tokens), files={"image": ("a.jpg", b"x", "image/jpeg")}
    )

    assert response.status_code == 404
