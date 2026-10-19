import os
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from app.core.config import Settings
from app.infrastructure.mongo.indexes import create_indexes, ensure_indexes
from app.main import create_app
from tests.support.fakes import FakeClock

LlmHandler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        mongodb_url="mongodb://unused",  # type: ignore[arg-type]
        secret_key="s" * 40,  # type: ignore[arg-type]
        environment="test",
        log_json=False,
        assistant_provider_api_key="test-key",  # type: ignore[arg-type]
        # Relaxed so ordinary tests never trip the limiter; the rate-limit tests build their own settings.
        login_attempts_per_minute=1000,
        register_per_hour=1000,
        refresh_per_minute=1000,
        assistant_per_minute=1000,
    )


@pytest.fixture
async def db() -> AsyncIterator[Any]:
    """A real MongoDB when ``TENSOSCAN_TEST_MONGODB_URL`` is set (CI), an in-memory double otherwise."""
    url = os.getenv("TENSOSCAN_TEST_MONGODB_URL")
    if url:
        from pymongo import AsyncMongoClient

        client: Any = AsyncMongoClient(url, tz_aware=True)
        name = f"tensoscan_test_{uuid.uuid4().hex}"
        database = client[name]
        await ensure_indexes(database)
        yield database
        await client.drop_database(name)
        await client.close()
    else:
        from mongomock_motor import AsyncMongoMockClient

        database = AsyncMongoMockClient(tz_aware=True)["tensoscan_test"]
        await create_indexes(database)
        yield database


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def llm_handler() -> list[LlmHandler]:
    """Replace the single element to control what the assistant provider answers."""

    def ok(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hello from the model"}}]})

    return [ok]


@pytest.fixture
async def app(settings: Settings, db: Any, clock: FakeClock, llm_handler: list[LlmHandler]) -> AsyncIterator[FastAPI]:
    provider = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: llm_handler[0](request)))
    application = create_app(settings, database=db, clock=clock, http_client=provider)
    async with LifespanManager(application):
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        yield http


NEW_USER = {"name": "Ana", "last_name": "García", "email": "ana@example.com", "password": "s3cret-pass"}


async def register_and_login(client: httpx.AsyncClient, user: dict[str, str] | None = None) -> dict[str, Any]:
    user = user or NEW_USER
    created = await client.post("/v1/auth/register", json=user)
    assert created.status_code == 201, created.text
    login = await client.post("/v1/auth/login", data={"username": user["email"], "password": user["password"]})
    assert login.status_code == 200, login.text
    return {**login.json(), "user": created.json()}


def bearer(tokens: dict[str, Any]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access_token']}"}
