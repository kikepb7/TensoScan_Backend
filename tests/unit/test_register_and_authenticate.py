from datetime import timedelta

import pytest

from app.application.auth.authenticate_user import AuthenticateUser
from app.application.auth.register_user import RegisterUser
from app.application.auth.tokens import SessionIssuer
from app.domain.errors import EmailAlreadyRegistered, InvalidCredentials, ValidationFailed
from app.infrastructure.security.jwt_service import JwtAccessTokenService
from tests.support.fakes import FakeClock, FakePasswordHasher, InMemoryRefreshTokenRepository, InMemoryUserRepository


@pytest.fixture
def users() -> InMemoryUserRepository:
    return InMemoryUserRepository()


@pytest.fixture
def hasher() -> FakePasswordHasher:
    return FakePasswordHasher()


@pytest.fixture
def register(users: InMemoryUserRepository, hasher: FakePasswordHasher) -> RegisterUser:
    return RegisterUser(users, hasher)


@pytest.fixture
def authenticate(users: InMemoryUserRepository, hasher: FakePasswordHasher) -> AuthenticateUser:
    clock = FakeClock()
    issuer = SessionIssuer(
        JwtAccessTokenService("k" * 40, timedelta(minutes=30), clock),
        InMemoryRefreshTokenRepository(),
        clock,
        timedelta(days=30),
    )
    return AuthenticateUser(users, hasher, issuer)


async def test_register_normalizes_the_email_and_trims_names(register: RegisterUser) -> None:
    user = await register.execute(name=" Ana ", last_name=" Pérez ", email="  Ana@Example.COM ", password="s3cret-pass")

    assert (user.email, user.name, user.last_name) == ("ana@example.com", "Ana", "Pérez")


async def test_register_never_stores_the_plain_password(register: RegisterUser, users: InMemoryUserRepository) -> None:
    user = await register.execute(name="A", last_name="B", email="a@b.es", password="s3cret-pass")

    assert users.users[user.id].password_hash != "s3cret-pass"


@pytest.mark.parametrize("password", ["short", "x" * 129])
async def test_register_rejects_passwords_outside_the_allowed_length(register: RegisterUser, password: str) -> None:
    with pytest.raises(ValidationFailed) as error:
        await register.execute(name="A", last_name="B", email="a@b.es", password=password)

    assert [(issue.field, issue.code) for issue in error.value.details] == [("password", "LENGTH_OUT_OF_RANGE")]


async def test_register_rejects_an_email_already_in_use_ignoring_case(register: RegisterUser) -> None:
    await register.execute(name="A", last_name="B", email="a@b.es", password="s3cret-pass")

    with pytest.raises(EmailAlreadyRegistered):
        await register.execute(name="C", last_name="D", email="A@B.ES", password="another-pass")


async def test_authenticate_returns_tokens_for_valid_credentials(
    register: RegisterUser, authenticate: AuthenticateUser
) -> None:
    await register.execute(name="A", last_name="B", email="a@b.es", password="s3cret-pass")

    pair = await authenticate.execute(email=" A@b.es ", password="s3cret-pass")

    assert pair.access_token and pair.refresh_token and pair.expires_in == 1800


async def test_authenticate_rejects_a_wrong_password(register: RegisterUser, authenticate: AuthenticateUser) -> None:
    await register.execute(name="A", last_name="B", email="a@b.es", password="s3cret-pass")

    with pytest.raises(InvalidCredentials):
        await authenticate.execute(email="a@b.es", password="wrong-password")


async def test_authenticate_spends_hashing_time_when_the_email_is_unknown(
    authenticate: AuthenticateUser, hasher: FakePasswordHasher
) -> None:
    with pytest.raises(InvalidCredentials):
        await authenticate.execute(email="nobody@b.es", password="whatever-pass")

    assert hasher.burned == 1


async def test_authenticate_rehashes_a_legacy_hash_after_a_successful_login(
    authenticate: AuthenticateUser, users: InMemoryUserRepository
) -> None:
    from app.domain.user import NewUser

    created = await users.create(NewUser(email="a@b.es", name="A", last_name="B", password_hash="legacy:s3cret-pass"))

    await authenticate.execute(email="a@b.es", password="s3cret-pass")

    assert users.users[created.id].password_hash == "hashed:s3cret-pass"
