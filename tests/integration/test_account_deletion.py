from typing import Any

import httpx
from bson import ObjectId

from tests.conftest import NEW_USER, bearer, register_and_login


async def _delete(client: httpx.AsyncClient, tokens: dict[str, Any], password: str) -> httpx.Response:
    return await client.request("DELETE", "/v1/users/me", headers=bearer(tokens), json={"password": password})


async def test_deleting_the_account_removes_the_user_and_everything_linked_to_it(
    client: httpx.AsyncClient, db: Any
) -> None:
    tokens = await register_and_login(client)
    user_id = ObjectId(tokens["user"]["id"])
    await db["recognition_results"].insert_one({"user_id": user_id, "result": {}, "filename": "x", "timestamp": None})
    other = await register_and_login(client, {**NEW_USER, "email": "other@example.com"})

    response = await _delete(client, tokens, NEW_USER["password"])

    assert response.status_code == 204
    assert await db["users"].count_documents({"_id": user_id}) == 0
    assert await db["refresh_tokens"].count_documents({"user_id": tokens["user"]["id"]}) == 0
    assert await db["recognition_results"].count_documents({"user_id": user_id}) == 0
    assert (await client.get("/v1/users/me", headers=bearer(other))).status_code == 200


async def test_a_deleted_account_can_no_longer_use_its_tokens(client: httpx.AsyncClient) -> None:
    tokens = await register_and_login(client)
    await _delete(client, tokens, NEW_USER["password"])

    me = await client.get("/v1/users/me", headers=bearer(tokens))
    refresh = await client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})

    assert me.status_code == refresh.status_code == 401


async def test_a_wrong_password_deletes_nothing(client: httpx.AsyncClient, db: Any) -> None:
    tokens = await register_and_login(client)

    response = await _delete(client, tokens, "wrong-password")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_CREDENTIALS"
    assert await db["users"].count_documents({}) == 1


async def test_deleting_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.request("DELETE", "/v1/users/me", json={"password": "whatever"})

    assert response.status_code == 401
