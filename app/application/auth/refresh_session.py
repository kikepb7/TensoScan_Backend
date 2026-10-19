from app.application.auth.tokens import SessionIssuer, hash_refresh_token
from app.domain.auth import TokenPair
from app.domain.errors import Unauthorized
from app.domain.ports import Clock, RefreshTokenRepository, UserRepository


class RefreshSession:
    """Rotates a refresh token. Presenting an already-used token revokes its whole family (token theft)."""

    def __init__(
        self,
        users: UserRepository,
        refresh_tokens: RefreshTokenRepository,
        issuer: SessionIssuer,
        clock: Clock,
    ) -> None:
        self._users = users
        self._refresh_tokens = refresh_tokens
        self._issuer = issuer
        self._clock = clock

    async def execute(self, *, refresh_token: str) -> TokenPair:
        now = self._clock.now()
        record = await self._refresh_tokens.find(hash_refresh_token(refresh_token))
        if record is None or record.expires_at <= now:
            raise Unauthorized

        if record.revoked_at is not None:
            await self._refresh_tokens.revoke_family(record.family_id, now)
            raise Unauthorized

        if await self._users.get_by_id(record.user_id) is None:
            raise Unauthorized

        successor = self._issuer.new_refresh_token()
        # Atomic: of two concurrent requests with the same token only one wins.
        if not await self._refresh_tokens.revoke_if_active(record.token_hash, now, replaced_by=successor[1]):
            await self._refresh_tokens.revoke_family(record.family_id, now)
            raise Unauthorized
        return await self._issuer.issue(record.user_id, family_id=record.family_id, refresh_token=successor)
