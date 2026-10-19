from datetime import timedelta

import jwt
import pytest

from app.domain.errors import TokenExpired, Unauthorized
from app.infrastructure.security.jwt_service import JwtAccessTokenService
from tests.support.fakes import FakeClock

SECRET = "k" * 40


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sut(clock: FakeClock) -> JwtAccessTokenService:
    return JwtAccessTokenService(SECRET, timedelta(minutes=30), clock)


def test_a_fresh_token_resolves_to_its_user(sut: JwtAccessTokenService) -> None:
    token, expires_in = sut.issue("user-1")

    assert sut.user_id_from(token) == "user-1"
    assert expires_in == 1800


def test_the_token_carries_the_expected_claims(sut: JwtAccessTokenService) -> None:
    token, _ = sut.issue("user-1")

    claims = jwt.decode(token, SECRET, algorithms=["HS256"], options={"verify_exp": False, "verify_iat": False})

    assert {"sub", "iat", "exp", "jti", "type"} <= claims.keys()
    assert claims["type"] == "access"


def test_an_expired_token_is_reported_as_expired(sut: JwtAccessTokenService, clock: FakeClock) -> None:
    token, _ = sut.issue("user-1")
    clock.advance(minutes=31)

    with pytest.raises(TokenExpired):
        sut.user_id_from(token)


def test_a_token_signed_with_another_key_is_rejected(sut: JwtAccessTokenService, clock: FakeClock) -> None:
    other = JwtAccessTokenService("z" * 40, timedelta(minutes=30), clock)
    token, _ = other.issue("user-1")

    with pytest.raises(Unauthorized):
        sut.user_id_from(token)


def test_garbage_is_rejected(sut: JwtAccessTokenService) -> None:
    with pytest.raises(Unauthorized):
        sut.user_id_from("not-a-token")


def test_a_token_of_another_type_is_rejected(sut: JwtAccessTokenService, clock: FakeClock) -> None:
    now = clock.now()
    token = jwt.encode(
        {"sub": "user-1", "iat": now, "exp": now + timedelta(minutes=5), "type": "refresh"}, SECRET, algorithm="HS256"
    )

    with pytest.raises(Unauthorized):
        sut.user_id_from(token)


def test_a_token_using_the_none_algorithm_is_rejected(sut: JwtAccessTokenService, clock: FakeClock) -> None:
    now = clock.now()
    token = jwt.encode(
        {"sub": "user-1", "iat": now, "exp": now + timedelta(minutes=5), "type": "access"}, None, algorithm="none"
    )

    with pytest.raises(Unauthorized):
        sut.user_id_from(token)
