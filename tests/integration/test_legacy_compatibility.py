"""The previous app version keeps working against the same routes while the new one is rolled out."""

from typing import Any

import bcrypt
import httpx

from tests.conftest import NEW_USER, bearer, register_and_login
from tests.support.measurements import body, post


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


async def test_a_user_cannot_delete_someone_elses_legacy_measurement(client: httpx.AsyncClient, db: Any) -> None:
    owner = await register_and_login(client)
    intruder = await register_and_login(client, {**NEW_USER, "email": "intruder@example.com"})
    created = (await post(client, owner, body())).json()

    response = await client.delete(f"/ocr/remove/measurements/{created['id']}", headers=bearer(intruder))

    assert response.status_code == 404
    assert await db["measurements"].count_documents({}) == 1


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


async def test_the_legacy_html_report_is_a_plain_escaped_table_without_scripts_or_external_resources(
    client: httpx.AsyncClient,
) -> None:
    tokens = await register_and_login(client)
    await post(client, tokens, body(systolic=131, diastolic=84, pulse=66))

    response = await client.get("/ocr/user/measurements/html", headers=bearer(tokens))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<td>131</td><td>84</td><td>66</td><td>19-10-2026 08:15</td>" in response.text
    lowered = response.text.lower()
    assert "<script" not in lowered and "http://" not in lowered and "https://" not in lowered
    assert response.headers["Deprecation"] == "true"


async def test_the_legacy_pdf_report_reuses_the_new_renderer(client: httpx.AsyncClient) -> None:
    tokens = await register_and_login(client)
    await post(client, tokens, body())

    response = await client.get("/ocr/user/measurements/pdf", headers=bearer(tokens))

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


async def test_the_legacy_reports_are_404_when_there_is_nothing_to_report(client: httpx.AsyncClient) -> None:
    tokens = await register_and_login(client)

    html = await client.get("/ocr/user/measurements/html", headers=bearer(tokens))
    pdf = await client.get("/ocr/user/measurements/pdf", headers=bearer(tokens))

    assert (html.status_code, pdf.status_code) == (404, 404)
