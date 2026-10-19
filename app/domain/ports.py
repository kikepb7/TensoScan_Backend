"""Ports (interfaces) the application layer depends on; infrastructure implements them."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from app.domain.assistant import ChatMessage
from app.domain.auth import PasswordCheck, RefreshTokenRecord
from app.domain.user import NewUser, User


class Clock(Protocol):
    def now(self) -> datetime:
        """Current time, timezone-aware (UTC)."""


class UserRepository(Protocol):
    async def create(self, user: NewUser) -> User:
        """Raises ``EmailAlreadyRegistered`` when the (normalised) email is taken."""

    async def get_by_email(self, email: str) -> User | None: ...

    async def get_by_id(self, user_id: str) -> User | None:
        """Returns None for unknown ids and for malformed ones."""

    async def update_password_hash(self, user_id: str, password_hash: str) -> None: ...

    async def delete(self, user_id: str) -> None: ...


class RefreshTokenRepository(Protocol):
    async def add(self, record: RefreshTokenRecord) -> None: ...

    async def find(self, token_hash: str) -> RefreshTokenRecord | None: ...

    async def revoke_if_active(self, token_hash: str, now: datetime, replaced_by: str) -> bool:
        """Atomically revokes the token if it is still active. Returns False if it was not (reuse or unknown)."""

    async def revoke_family(self, family_id: str, now: datetime) -> None: ...

    async def delete_all_for_user(self, user_id: str) -> None: ...


class MeasurementEraser(Protocol):
    async def delete_all_for_user(self, user_id: str) -> int:
        """Removes every measurement of a user (account deletion). Returns how many were removed."""


class PasswordHasher(Protocol):
    def hash(self, password: str) -> str: ...

    def check(self, password: str, password_hash: str) -> PasswordCheck: ...

    def burn(self, password: str) -> None:
        """Spends the time of a verification, to keep unknown-email logins as slow as real ones."""


class AccessTokenService(Protocol):
    def issue(self, user_id: str) -> tuple[str, int]:
        """Returns the signed access token and its lifetime in seconds."""

    def user_id_from(self, token: str) -> str:
        """Raises ``TokenExpired`` or ``Unauthorized``."""


class LlmClient(Protocol):
    async def complete(self, system_prompt: str, messages: Sequence[ChatMessage]) -> str:
        """Raises ``UpstreamUnavailable`` when the provider cannot answer."""
