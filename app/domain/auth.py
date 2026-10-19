from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "bearer"  # noqa: S105 - OAuth2 token type, not a secret


@dataclass(frozen=True)
class RefreshTokenRecord:
    """Server-side state of a refresh token. Only the SHA-256 of the token is stored, never the token itself."""

    token_hash: str
    user_id: str
    family_id: str
    issued_at: datetime
    expires_at: datetime
    revoked_at: datetime | None = None
    replaced_by: str | None = None


@dataclass(frozen=True)
class PasswordCheck:
    valid: bool
    needs_rehash: bool = False
