"""In-memory test doubles for the ports of ``app.domain.ports`` (no database, no network, no real hashing cost)."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import count

from app.domain.auth import PasswordCheck, RefreshTokenRecord
from app.domain.errors import EmailAlreadyRegistered
from app.domain.user import NewUser, User


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 10, 19, 8, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, **kwargs: float) -> None:
        self._now += timedelta(**kwargs)


class InMemoryUserRepository:
    def __init__(self) -> None:
        self.users: dict[str, User] = {}
        self._ids = count(1)

    async def create(self, user: NewUser) -> User:
        if any(existing.email == user.email for existing in self.users.values()):
            raise EmailAlreadyRegistered
        created = User(
            id=f"{next(self._ids):024x}",
            email=user.email,
            name=user.name,
            last_name=user.last_name,
            password_hash=user.password_hash,
        )
        self.users[created.id] = created
        return created

    async def get_by_email(self, email: str) -> User | None:
        return next((user for user in self.users.values() if user.email == email), None)

    async def get_by_id(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    async def update_password_hash(self, user_id: str, password_hash: str) -> None:
        self.users[user_id] = replace(self.users[user_id], password_hash=password_hash)

    async def delete(self, user_id: str) -> None:
        self.users.pop(user_id, None)


class InMemoryRefreshTokenRepository:
    def __init__(self) -> None:
        self.records: dict[str, RefreshTokenRecord] = {}

    async def add(self, record: RefreshTokenRecord) -> None:
        self.records[record.token_hash] = record

    async def find(self, token_hash: str) -> RefreshTokenRecord | None:
        return self.records.get(token_hash)

    async def revoke_if_active(self, token_hash: str, now: datetime, replaced_by: str) -> bool:
        record = self.records.get(token_hash)
        if record is None or record.revoked_at is not None or record.expires_at <= now:
            return False
        self.records[token_hash] = replace(record, revoked_at=now, replaced_by=replaced_by)
        return True

    async def revoke_family(self, family_id: str, now: datetime) -> None:
        for key, record in list(self.records.items()):
            if record.family_id == family_id and record.revoked_at is None:
                self.records[key] = replace(record, revoked_at=now)

    async def delete_all_for_user(self, user_id: str) -> None:
        self.records = {key: value for key, value in self.records.items() if value.user_id != user_id}


class FakePasswordHasher:
    """``hash`` prefixes the password; ``legacy:`` hashes verify but ask to be re-hashed."""

    def __init__(self) -> None:
        self.burned = 0

    def hash(self, password: str) -> str:
        return f"hashed:{password}"

    def check(self, password: str, password_hash: str) -> PasswordCheck:
        if password_hash == f"hashed:{password}":
            return PasswordCheck(valid=True)
        if password_hash == f"legacy:{password}":
            return PasswordCheck(valid=True, needs_rehash=True)
        return PasswordCheck(valid=False)

    def burn(self, password: str) -> None:
        self.burned += 1


class RecordingMeasurementEraser:
    def __init__(self) -> None:
        self.erased: list[str] = []

    async def delete_all_for_user(self, user_id: str) -> int:
        self.erased.append(user_id)
        return 0
