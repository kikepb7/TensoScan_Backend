from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import APIRouter

from app.core.config import Settings
from app.main import create_app
from tests.conftest import NEW_USER
from tests.support.fakes import FakeClock


@pytest.fixture
async def strict_client(settings: Settings, db: Any, clock: FakeClock) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings.model_copy(update={"login_attempts_per_minute": 3}), database=db, clock=clock)
    async with LifespanManager(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            yield http


async def test_too_many_login_attempts_are_throttled_with_a_retry_delay(strict_client: httpx.AsyncClient) -> None:
    attempts = [
        await strict_client.post("/v1/auth/login", data={"username": f"user{i}@example.com", "password": "x" * 9})
        for i in range(3)
    ]
    throttled = await strict_client.post("/v1/auth/login", data={"username": "user9@example.com", "password": "x" * 9})

    assert [response.status_code for response in attempts] == [401, 401, 401]
    assert throttled.status_code == 429
    assert throttled.json()["error"]["code"] == "RATE_LIMITED"
    assert int(throttled.headers["Retry-After"]) >= 1


async def test_attempts_against_one_email_are_throttled_even_from_other_addresses(
    strict_client: httpx.AsyncClient,
) -> None:
    for _ in range(3):
        await strict_client.post("/v1/auth/login", data={"username": NEW_USER["email"], "password": "wrong-password"})

    throttled = await strict_client.post(
        "/v1/auth/login", data={"username": NEW_USER["email"].upper(), "password": "wrong-password"}
    )

    assert throttled.status_code == 429


async def test_unexpected_failures_answer_a_generic_500_without_internals(
    settings: Settings, db: Any, clock: FakeClock
) -> None:
    app = create_app(settings, database=db, clock=clock)
    boom = APIRouter()

    @boom.get("/boom")
    async def explode() -> None:
        raise RuntimeError("secret internal detail: password=hunter2")

    app.include_router(boom)
    async with LifespanManager(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
        ) as http:
            response = await http.get("/boom")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "hunter2" not in response.text


async def test_every_response_carries_a_request_id(client: httpx.AsyncClient) -> None:
    generated = await client.get("/health")
    propagated = await client.get("/health", headers={"X-Request-ID": "abc-123"})

    assert generated.headers["X-Request-ID"]
    assert propagated.headers["X-Request-ID"] == "abc-123"


async def test_health_reports_ok_when_the_database_answers(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_unknown_routes_use_the_error_envelope(client: httpx.AsyncClient) -> None:
    response = await client.get("/nope")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
