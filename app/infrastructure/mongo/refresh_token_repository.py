from datetime import UTC, datetime
from typing import Any

from app.domain.auth import RefreshTokenRecord
from app.infrastructure.mongo.client import Database


class MongoRefreshTokenRepository:
    def __init__(self, db: Database) -> None:
        self._tokens = db["refresh_tokens"]

    async def add(self, record: RefreshTokenRecord) -> None:
        await self._tokens.insert_one(
            {
                "token_hash": record.token_hash,
                "user_id": record.user_id,
                "family_id": record.family_id,
                "issued_at": record.issued_at,
                "expires_at": record.expires_at,
                "revoked_at": record.revoked_at,
                "replaced_by": record.replaced_by,
            }
        )

    async def find(self, token_hash: str) -> RefreshTokenRecord | None:
        document = await self._tokens.find_one({"token_hash": token_hash})
        return _to_record(document) if document else None

    async def revoke_if_active(self, token_hash: str, now: datetime, replaced_by: str) -> bool:
        previous = await self._tokens.find_one_and_update(
            {"token_hash": token_hash, "revoked_at": None, "expires_at": {"$gt": now}},
            {"$set": {"revoked_at": now, "replaced_by": replaced_by}},
        )
        return previous is not None

    async def revoke_family(self, family_id: str, now: datetime) -> None:
        await self._tokens.update_many({"family_id": family_id, "revoked_at": None}, {"$set": {"revoked_at": now}})

    async def delete_all_for_user(self, user_id: str) -> None:
        await self._tokens.delete_many({"user_id": user_id})


def _aware(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _to_record(document: dict[str, Any]) -> RefreshTokenRecord:
    issued_at = _aware(document["issued_at"])
    expires_at = _aware(document["expires_at"])
    assert issued_at is not None  # noqa: S101 - written by add(), never null
    assert expires_at is not None  # noqa: S101
    return RefreshTokenRecord(
        token_hash=document["token_hash"],
        user_id=document["user_id"],
        family_id=document["family_id"],
        issued_at=issued_at,
        expires_at=expires_at,
        revoked_at=_aware(document.get("revoked_at")),
        replaced_by=document.get("replaced_by"),
    )
