"""Ports (interfaces) the application layer depends on; infrastructure implements them."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from app.domain.assistant import ChatMessage
from app.domain.auth import PasswordCheck, RefreshTokenRecord
from app.domain.measurements.cursor import PageCursor
from app.domain.measurements.model import Measurement, MeasurementPage
from app.domain.measurements.report import MeasurementsReport
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


class MeasurementRepository(Protocol):
    """Every method is scoped by ``user_id`` inside the repository (C-SEC-02)."""

    async def add_idempotent(self, measurement: Measurement, *, created_at: datetime) -> tuple[Measurement, bool]:
        """Stores a new measurement and returns ``(stored, True)``. If ``(user_id, client_id)`` already exists with
        the same canonical payload, returns ``(existing, False)``; with a different one raises ``ClientIdConflict``."""

    async def list_page(
        self,
        user_id: str,
        *,
        since: datetime | None,
        until: datetime | None,
        limit: int,
        after: PageCursor | None,
    ) -> MeasurementPage:
        """Newest first, ordered by ``(measured_at, id)`` descending; ``since`` and ``until`` are inclusive."""

    async def list_all(self, user_id: str, *, since: datetime | None, until: datetime | None) -> list[Measurement]:
        """Same filters and order as ``list_page`` without pagination."""

    async def delete(self, user_id: str, measurement_id: str) -> bool:
        """True if a measurement of that user was removed; False for unknown, malformed or foreign ids."""

    async def delete_all_for_user(self, user_id: str) -> int:
        """Removes every measurement of a user (account deletion). Returns how many were removed."""


class ReportRenderer(Protocol):
    def render(self, report: MeasurementsReport) -> bytes:
        """The report as a PDF document. Synchronous and CPU bound: callers run it off the event loop."""


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
