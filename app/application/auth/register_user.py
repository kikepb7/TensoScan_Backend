from app.domain.errors import FieldIssue, ValidationFailed
from app.domain.ports import PasswordHasher, UserRepository
from app.domain.user import NewUser, User, normalize_email

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128


class RegisterUser:
    def __init__(self, users: UserRepository, hasher: PasswordHasher) -> None:
        self._users = users
        self._hasher = hasher

    async def execute(self, *, name: str, last_name: str, email: str, password: str) -> User:
        if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
            raise ValidationFailed([FieldIssue(field="password", code="LENGTH_OUT_OF_RANGE")])
        return await self._users.create(
            NewUser(
                email=normalize_email(email),
                name=name.strip(),
                last_name=last_name.strip(),
                password_hash=self._hasher.hash(password),
            )
        )
