import asyncio
from datetime import timedelta
from typing import Any

import httpx
import jwt

from tests.conftest import NEW_USER, bearer, register_and_login
from tests.support.fakes import FakeClock


async def test_register_returns_the_new_user_without_secrets(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/auth/register", json={**NEW_USER, "email": " Ana@Example.com "})

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "ana@example.com"
    assert set(body) == {"id", "email", "name", "last_name"}


async def test_registering_twice_answers_conflict_with_the_error_envelope(client: httpx.AsyncClient) -> None:
    await client.post("/v1/auth/register", json=NEW_USER)

    response = await client.post("/v1/auth/register", json={**NEW_USER, "email": "ANA@example.com"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "EMAIL_ALREADY_REGISTERED"


async def test_simultaneous_registrations_with_the_same_email_create_one_account(client: httpx.AsyncClient) -> None:
    results = await asyncio.gather(*(client.post("/v1/auth/register", json=NEW_USER) for _ in range(5)))

    assert sorted(response.status_code for response in results) == [201, 409, 409, 409, 409]


async def test_invalid_registration_data_lists_the_wrong_fields(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/auth/register", json={"name": "", "email": "nope", "password": "123"})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert {detail["field"] for detail in error["details"]} == {"name", "last_name", "email", "password"}


async def test_login_returns_an_access_and_a_refresh_token(client: httpx.AsyncClient) -> None:
    tokens = await register_and_login(client)

    assert tokens["token_type"] == "bearer"
    assert tokens["expires_in"] == 1800
    assert tokens["access_token"] and tokens["refresh_token"]


async def test_login_with_a_wrong_password_and_with_an_unknown_email_answer_the_same(client: httpx.AsyncClient) -> None:
    await client.post("/v1/auth/register", json=NEW_USER)

    wrong = await client.post("/v1/auth/login", data={"username": NEW_USER["email"], "password": "wrong-password"})
    unknown = await client.post("/v1/auth/login", data={"username": "nobody@example.com", "password": "whatever-pass"})

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()
    assert wrong.json()["error"]["code"] == "INVALID_CREDENTIALS"


async def test_me_returns_the_profile_of_the_token_owner(client: httpx.AsyncClient) -> None:
    tokens = await register_and_login(client)

    response = await client.get("/v1/users/me", headers=bearer(tokens))

    assert response.status_code == 200
    assert response.json() == tokens["user"]


async def test_me_without_a_token_answers_unauthorized_with_the_error_envelope(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/users/me")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


async def test_an_expired_access_token_is_reported_as_expired(client: httpx.AsyncClient, clock: FakeClock) -> None:
    tokens = await register_and_login(client)
    clock.advance(minutes=31)

    response = await client.get("/v1/users/me", headers=bearer(tokens))

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "TOKEN_EXPIRED"


async def test_a_token_with_a_malformed_subject_is_unauthorized_not_a_server_error(
    client: httpx.AsyncClient, clock: FakeClock, settings: Any
) -> None:
    now = clock.now()
    forged = jwt.encode(
        {"sub": "not-an-object-id", "iat": now, "exp": now + timedelta(minutes=5), "type": "access"},
        settings.secret_key.get_secret_value(),
        algorithm="HS256",
    )

    response = await client.get("/v1/users/me", headers={"Authorization": f"Bearer {forged}"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


async def test_refresh_returns_a_new_pair_and_invalidates_the_old_refresh_token(client: httpx.AsyncClient) -> None:
    tokens = await register_and_login(client)

    renewed = await client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    reused = await client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})

    assert renewed.status_code == 200
    assert renewed.json()["refresh_token"] != tokens["refresh_token"]
    assert reused.status_code == 401


async def test_reusing_a_rotated_refresh_token_also_kills_the_new_one(client: httpx.AsyncClient) -> None:
    tokens = await register_and_login(client)
    renewed = (await client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})).json()

    await client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})  # stolen token replayed
    legitimate = await client.post("/v1/auth/refresh", json={"refresh_token": renewed["refresh_token"]})

    assert legitimate.status_code == 401


async def test_two_simultaneous_refreshes_with_the_same_token_let_only_one_win(client: httpx.AsyncClient) -> None:
    tokens = await register_and_login(client)

    results = await asyncio.gather(
        *(client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}) for _ in range(2))
    )

    assert sorted(response.status_code for response in results) == [200, 401]


async def test_a_refresh_token_expires_after_its_lifetime(client: httpx.AsyncClient, clock: FakeClock) -> None:
    tokens = await register_and_login(client)
    clock.advance(days=31)

    response = await client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})

    assert response.status_code == 401


async def test_logout_ends_the_session(client: httpx.AsyncClient) -> None:
    tokens = await register_and_login(client)

    logout = await client.post("/v1/auth/logout", json={"refresh_token": tokens["refresh_token"]})
    refresh = await client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})

    assert logout.status_code == 204
    assert refresh.status_code == 401


async def test_refresh_tokens_are_stored_hashed(client: httpx.AsyncClient, db: Any) -> None:
    tokens = await register_and_login(client)

    stored = await db["refresh_tokens"].find({}).to_list(length=None)

    assert [document["token_hash"] for document in stored] != [tokens["refresh_token"]]
    assert all(tokens["refresh_token"] not in str(document) for document in stored)
