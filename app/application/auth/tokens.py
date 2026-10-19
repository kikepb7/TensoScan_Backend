import hashlib
import secrets
from datetime import timedelta

from app.domain.auth import RefreshTokenRecord, TokenPair
from app.domain.ports import AccessTokenService, Clock, RefreshTokenRepository


def hash_refresh_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode()).hexdigest()


class SessionIssuer:
    """Creates access/refresh token pairs. Refresh tokens are opaque; only their hash is persisted."""

    def __init__(
        self,
        access_tokens: AccessTokenService,
        refresh_tokens: RefreshTokenRepository,
        clock: Clock,
        refresh_ttl: timedelta,
    ) -> None:
        self._access_tokens = access_tokens
        self._refresh_tokens = refresh_tokens
        self._clock = clock
        self._refresh_ttl = refresh_ttl

    @staticmethod
    def new_refresh_token() -> tuple[str, str]:
        """Returns ``(raw_token, token_hash)``."""
        raw = secrets.token_urlsafe(48)
        return raw, hash_refresh_token(raw)

    async def issue(
        self,
        user_id: str,
        family_id: str | None = None,
        refresh_token: tuple[str, str] | None = None,
    ) -> TokenPair:
        now = self._clock.now()
        raw, token_hash = refresh_token or self.new_refresh_token()
        await self._refresh_tokens.add(
            RefreshTokenRecord(
                token_hash=token_hash,
                user_id=user_id,
                family_id=family_id or secrets.token_hex(16),
                issued_at=now,
                expires_at=now + self._refresh_ttl,
            )
        )
        access_token, expires_in = self._access_tokens.issue(user_id)
        return TokenPair(access_token=access_token, refresh_token=raw, expires_in=expires_in)
