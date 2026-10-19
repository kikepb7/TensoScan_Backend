from app.application.auth.tokens import SessionIssuer
from app.domain.auth import TokenPair
from app.domain.errors import InvalidCredentials
from app.domain.ports import PasswordHasher, UserRepository
from app.domain.user import normalize_email


class AuthenticateUser:
    def __init__(self, users: UserRepository, hasher: PasswordHasher, issuer: SessionIssuer) -> None:
        self._users = users
        self._hasher = hasher
        self._issuer = issuer

    async def execute(self, *, email: str, password: str) -> TokenPair:
        user = await self._users.get_by_email(normalize_email(email))
        if user is None:
            # Same cost as a real verification: the response time must not reveal whether the email exists.
            self._hasher.burn(password)
            raise InvalidCredentials
        check = self._hasher.check(password, user.password_hash)
        if not check.valid:
            raise InvalidCredentials
        if check.needs_rehash:
            await self._users.update_password_hash(user.id, self._hasher.hash(password))
        return await self._issuer.issue(user.id)
