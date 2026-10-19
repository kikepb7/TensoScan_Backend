from datetime import timedelta

import pytest

from app.application.auth.logout import Logout
from app.application.auth.refresh_session import RefreshSession
from app.application.auth.tokens import SessionIssuer, hash_refresh_token
from app.domain.errors import Unauthorized
from app.domain.user import NewUser
from app.infrastructure.security.jwt_service import JwtAccessTokenService
from tests.support.fakes import FakeClock, InMemoryRefreshTokenRepository, InMemoryUserRepository


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def users() -> InMemoryUserRepository:
    return InMemoryUserRepository()


@pytest.fixture
def tokens() -> InMemoryRefreshTokenRepository:
    return InMemoryRefreshTokenRepository()


@pytest.fixture
def issuer(clock: FakeClock, tokens: InMemoryRefreshTokenRepository) -> SessionIssuer:
    return SessionIssuer(
        JwtAccessTokenService("k" * 40, timedelta(minutes=30), clock), tokens, clock, timedelta(days=30)
    )


@pytest.fixture
def sut(
    users: InMemoryUserRepository, tokens: InMemoryRefreshTokenRepository, issuer: SessionIssuer, clock: FakeClock
) -> RefreshSession:
    return RefreshSession(users, tokens, issuer, clock)


async def _user_with_session(users: InMemoryUserRepository, issuer: SessionIssuer) -> tuple[str, str]:
    user = await users.create(NewUser(email="a@b.es", name="A", last_name="B", password_hash="x"))
    pair = await issuer.issue(user.id)
    return user.id, pair.refresh_token


async def test_refresh_rotates_the_token_and_keeps_the_family(
    sut: RefreshSession, users: InMemoryUserRepository, issuer: SessionIssuer, tokens: InMemoryRefreshTokenRepository
) -> None:
    _, original = await _user_with_session(users, issuer)

    renewed = await sut.execute(refresh_token=original)

    old_record = tokens.records[hash_refresh_token(original)]
    new_record = tokens.records[hash_refresh_token(renewed.refresh_token)]
    assert renewed.refresh_token != original
    assert old_record.revoked_at is not None
    assert old_record.replaced_by == new_record.token_hash
    assert new_record.family_id == old_record.family_id


async def test_reusing_a_rotated_token_revokes_the_whole_family(
    sut: RefreshSession, users: InMemoryUserRepository, issuer: SessionIssuer
) -> None:
    _, original = await _user_with_session(users, issuer)
    renewed = await sut.execute(refresh_token=original)

    with pytest.raises(Unauthorized):
        await sut.execute(refresh_token=original)

    with pytest.raises(Unauthorized):  # the legitimate successor was revoked too: someone stole the old token
        await sut.execute(refresh_token=renewed.refresh_token)


async def test_an_unknown_token_is_rejected(sut: RefreshSession) -> None:
    with pytest.raises(Unauthorized):
        await sut.execute(refresh_token="never-issued")


async def test_an_expired_token_is_rejected(
    sut: RefreshSession, users: InMemoryUserRepository, issuer: SessionIssuer, clock: FakeClock
) -> None:
    _, original = await _user_with_session(users, issuer)
    clock.advance(days=31)

    with pytest.raises(Unauthorized):
        await sut.execute(refresh_token=original)


async def test_a_token_of_a_deleted_user_is_rejected(
    sut: RefreshSession, users: InMemoryUserRepository, issuer: SessionIssuer
) -> None:
    user_id, original = await _user_with_session(users, issuer)
    await users.delete(user_id)

    with pytest.raises(Unauthorized):
        await sut.execute(refresh_token=original)


async def test_logout_revokes_the_session_and_is_idempotent(
    sut: RefreshSession,
    users: InMemoryUserRepository,
    issuer: SessionIssuer,
    tokens: InMemoryRefreshTokenRepository,
    clock: FakeClock,
) -> None:
    _, original = await _user_with_session(users, issuer)
    logout = Logout(tokens, clock)

    await logout.execute(refresh_token=original)
    await logout.execute(refresh_token=original)
    await logout.execute(refresh_token="never-issued")

    with pytest.raises(Unauthorized):
        await sut.execute(refresh_token=original)
