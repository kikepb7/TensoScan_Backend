import time
from collections import defaultdict, deque
from collections.abc import Callable

from app.domain.errors import RateLimited


class RateLimiter:
    """Sliding-window limiter kept in memory.

    Enough for a single API process. If the API is scaled horizontally the counters must move to a shared store
    (Mongo or Redis); this is documented debt in spec B-001 (ADR-B001-07).
    """

    def __init__(self, limit: int, window_seconds: float, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._limit = limit
        self._window = window_seconds
        self._monotonic = monotonic
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        """Registers a hit for ``key``. Raises ``RateLimited`` when the key is over the limit."""
        now = self._monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] >= self._window:
            hits.popleft()
        if len(hits) >= self._limit:
            retry_after = max(1, int(self._window - (now - hits[0])) + 1)
            raise RateLimited(retry_after=retry_after)
        hits.append(now)
        if len(self._hits) > _MAX_TRACKED_KEYS:
            self._drop_idle_keys(now)

    def _drop_idle_keys(self, now: float) -> None:
        for key in [key for key, hits in self._hits.items() if not hits or now - hits[-1] >= self._window]:
            del self._hits[key]


_MAX_TRACKED_KEYS = 10_000
