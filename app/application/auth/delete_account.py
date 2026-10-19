from app.domain.errors import InvalidCredentials, Unauthorized
from app.domain.ports import MeasurementEraser, PasswordHasher, RefreshTokenRepository, UserRepository


class DeleteAccount:
    """Right to erasure (GDPR art. 17): removes the user and every piece of data linked to them."""

    def __init__(
        self,
        users: UserRepository,
        hasher: PasswordHasher,
        refresh_tokens: RefreshTokenRepository,
        measurements: MeasurementEraser,
    ) -> None:
        self._users = users
        self._hasher = hasher
        self._refresh_tokens = refresh_tokens
        self._measurements = measurements

    async def execute(self, *, user_id: str, password: str) -> None:
        user = await self._users.get_by_id(user_id)
        if user is None:
            raise Unauthorized
        if not self._hasher.check(password, user.password_hash).valid:
            raise InvalidCredentials
        # Dependent data first and the user last, so a failure half-way can simply be retried.
        await self._measurements.delete_all_for_user(user_id)
        await self._refresh_tokens.delete_all_for_user(user_id)
        await self._users.delete(user_id)
