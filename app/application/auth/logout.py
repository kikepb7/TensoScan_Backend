from app.application.auth.tokens import hash_refresh_token
from app.domain.ports import Clock, RefreshTokenRepository


class Logout:
    """Ends the session of a refresh token. Idempotent: an unknown token is not an error."""

    def __init__(self, refresh_tokens: RefreshTokenRepository, clock: Clock) -> None:
        self._refresh_tokens = refresh_tokens
        self._clock = clock

    async def execute(self, *, refresh_token: str) -> None:
        record = await self._refresh_tokens.find(hash_refresh_token(refresh_token))
        if record is not None:
            await self._refresh_tokens.revoke_family(record.family_id, self._clock.now())
