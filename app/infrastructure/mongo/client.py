from typing import Any

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings

Database = AsyncDatabase[dict[str, Any]]


def create_client(settings: Settings) -> AsyncMongoClient[dict[str, Any]]:
    # tz_aware: every datetime read back is timezone-aware UTC, never a naive one.
    return AsyncMongoClient(settings.mongodb_url.get_secret_value(), tz_aware=True)
