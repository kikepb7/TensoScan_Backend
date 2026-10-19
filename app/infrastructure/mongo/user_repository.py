from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.domain.errors import EmailAlreadyRegistered
from app.domain.user import NewUser, User
from app.infrastructure.mongo.client import Database


class MongoUserRepository:
    def __init__(self, db: Database) -> None:
        self._users = db["users"]

    async def create(self, user: NewUser) -> User:
        if await self._users.find_one({"email": user.email}, projection={"_id": 1}) is not None:
            raise EmailAlreadyRegistered
        document: dict[str, Any] = {
            "email": user.email,
            "name": user.name,
            "last_name": user.last_name,
            "password_hash": user.password_hash,
            "created_at": datetime.now(UTC),
        }
        try:
            result = await self._users.insert_one(document)
        except DuplicateKeyError as error:
            # Lost a race against a simultaneous registration: the unique index is the real guard.
            raise EmailAlreadyRegistered from error
        return _to_user({**document, "_id": result.inserted_id})

    async def get_by_email(self, email: str) -> User | None:
        document = await self._users.find_one({"email": email})
        return _to_user(document) if document else None

    async def get_by_id(self, user_id: str) -> User | None:
        if not ObjectId.is_valid(user_id):
            return None
        document = await self._users.find_one({"_id": ObjectId(user_id)})
        return _to_user(document) if document else None

    async def update_password_hash(self, user_id: str, password_hash: str) -> None:
        await self._users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"password_hash": password_hash}, "$unset": {"hashed_password": ""}},
        )

    async def delete(self, user_id: str) -> None:
        await self._users.delete_one({"_id": ObjectId(user_id)})


def _to_user(document: dict[str, Any]) -> User:
    created_at = document.get("created_at")
    if isinstance(created_at, datetime) and created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return User(
        id=str(document["_id"]),
        email=document["email"],
        name=document.get("name", ""),
        last_name=document.get("last_name", ""),
        # Accounts created by the previous version stored the bcrypt hash under "hashed_password".
        password_hash=document.get("password_hash") or document.get("hashed_password") or "",
        created_at=created_at,
    )
