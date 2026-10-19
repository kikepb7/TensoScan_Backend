from app.domain.errors import Unauthorized
from app.domain.ports import AccessTokenService, UserRepository
from app.domain.user import User


class GetCurrentUser:
    def __init__(self, users: UserRepository, access_tokens: AccessTokenService) -> None:
        self._users = users
        self._access_tokens = access_tokens

    async def execute(self, *, access_token: str) -> User:
        user = await self._users.get_by_id(self._access_tokens.user_id_from(access_token))
        if user is None:
            raise Unauthorized
        return user
