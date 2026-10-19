import pytest

from app.core.rate_limit import RateLimiter
from app.domain.errors import RateLimited


class Ticker:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_requests_under_the_limit_pass() -> None:
    sut = RateLimiter(limit=3, window_seconds=60, monotonic=Ticker())

    for _ in range(3):
        sut.check("ip")


def test_the_request_over_the_limit_is_rejected_with_a_retry_delay() -> None:
    ticker = Ticker()
    sut = RateLimiter(limit=2, window_seconds=60, monotonic=ticker)
    sut.check("ip")
    ticker.now += 10
    sut.check("ip")

    with pytest.raises(RateLimited) as error:
        sut.check("ip")

    assert 49 <= error.value.retry_after <= 51


def test_the_window_slides() -> None:
    ticker = Ticker()
    sut = RateLimiter(limit=1, window_seconds=60, monotonic=ticker)
    sut.check("ip")
    ticker.now += 61

    sut.check("ip")


def test_keys_are_independent() -> None:
    sut = RateLimiter(limit=1, window_seconds=60, monotonic=Ticker())
    sut.check("a")

    sut.check("b")
