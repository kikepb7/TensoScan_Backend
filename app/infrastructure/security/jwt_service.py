import secrets
from datetime import timedelta

import jwt

from app.domain.errors import TokenExpired, Unauthorized
from app.domain.ports import Clock

ALGORITHM = "HS256"
ACCESS_TOKEN_TYPE = "access"  # noqa: S105 - claim value, not a secret


class JwtAccessTokenService:
    def __init__(self, secret_key: str, ttl: timedelta, clock: Clock) -> None:
        self._secret_key = secret_key
        self._ttl = ttl
        self._clock = clock

    def issue(self, user_id: str) -> tuple[str, int]:
        now = self._clock.now()
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + self._ttl,
            "jti": secrets.token_hex(8),
            "type": ACCESS_TOKEN_TYPE,
        }
        return jwt.encode(payload, self._secret_key, algorithm=ALGORITHM), int(self._ttl.total_seconds())

    def user_id_from(self, token: str) -> str:
        try:
            payload = jwt.decode(
                token,
                self._secret_key,
                algorithms=[ALGORITHM],
                options={"require": ["exp", "iat", "sub"], "verify_exp": False, "verify_iat": False},
            )
        except jwt.InvalidTokenError as error:
            raise Unauthorized from error

        if payload.get("type") != ACCESS_TOKEN_TYPE or not isinstance(payload.get("sub"), str):
            raise Unauthorized
        # Expiry is checked against the injected clock (not the system one) so it stays testable.
        if payload["exp"] <= int(self._clock.now().timestamp()):
            raise TokenExpired
        return str(payload["sub"])
