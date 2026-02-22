"""Token bucket rate limiter for exchange API calls."""

import asyncio
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()

# Default per-exchange rate limits (requests per minute)
DEFAULT_EXCHANGE_LIMITS: dict[str, dict[str, float]] = {
    "binance": {"requests_per_second": 20.0, "burst_size": 40},
    "upbit": {"requests_per_second": 10.0, "burst_size": 20},
}


@dataclass
class RateLimitMetrics:
    """Tracks rate limiter performance metrics."""

    total_requests: int = 0
    throttled_requests: int = 0
    total_wait_time_ms: float = 0.0
    _recent_request_times: list[float] = field(default_factory=list)

    @property
    def avg_wait_ms(self) -> float:
        if self.throttled_requests == 0:
            return 0.0
        return self.total_wait_time_ms / self.throttled_requests

    @property
    def requests_per_second(self) -> float:
        """Compute actual requests/second over the last 60 seconds."""
        now = time.monotonic()
        cutoff = now - 60.0
        self._recent_request_times = [
            t for t in self._recent_request_times if t > cutoff
        ]
        count = len(self._recent_request_times)
        if count == 0:
            return 0.0
        window = now - self._recent_request_times[0]
        if window <= 0:
            return float(count)
        return count / window

    def record_request(self, wait_ms: float = 0.0) -> None:
        """Record a request with optional wait time."""
        self.total_requests += 1
        self._recent_request_times.append(time.monotonic())
        if wait_ms > 0:
            self.throttled_requests += 1
            self.total_wait_time_ms += wait_ms

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "throttled_requests": self.throttled_requests,
            "avg_wait_ms": round(self.avg_wait_ms, 2),
            "current_rps": round(self.requests_per_second, 2),
        }


class RateLimiter:
    """Token bucket rate limiter for exchange API calls.

    The bucket starts full (burst_size tokens). Tokens are consumed on each
    request and refilled at max_requests_per_second rate.  If no tokens are
    available, acquire() awaits until one becomes available.
    """

    def __init__(
        self,
        max_requests_per_second: float = 10.0,
        burst_size: int = 20,
        name: str = "default",
    ):
        self._rate = max_requests_per_second
        self._burst_size = burst_size
        self._name = name
        self._tokens = float(burst_size)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._metrics = RateLimitMetrics()

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_requests_per_second(self) -> float:
        return self._rate

    @property
    def burst_size(self) -> int:
        return self._burst_size

    @property
    def metrics(self) -> RateLimitMetrics:
        return self._metrics

    @property
    def tokens_remaining(self) -> float:
        """Current tokens available (approximate — does not lock)."""
        self._refill()
        return self._tokens

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._rate
        if new_tokens > 0:
            self._tokens = min(self._tokens + new_tokens, float(self._burst_size))
            self._last_refill = now

    async def acquire(self) -> float:
        """Acquire a token, waiting if necessary.

        Returns:
            Wait time in seconds (0.0 if no wait was needed).
        """
        async with self._lock:
            self._refill()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self._metrics.record_request(wait_ms=0.0)
                return 0.0

            # Calculate wait time for one token to become available
            deficit = 1.0 - self._tokens
            wait_seconds = deficit / self._rate

        # Wait outside the lock so other coroutines can check
        await asyncio.sleep(wait_seconds)
        wait_ms = wait_seconds * 1000.0

        async with self._lock:
            self._refill()
            self._tokens = max(0.0, self._tokens - 1.0)
            self._metrics.record_request(wait_ms=wait_ms)

            logger.debug(
                "rate_limiter_throttled",
                limiter=self._name,
                wait_ms=round(wait_ms, 1),
                tokens_remaining=round(self._tokens, 1),
            )

        return wait_seconds

    def to_dict(self) -> dict:
        """Return rate limiter state for API/dashboard."""
        self._refill()
        return {
            "name": self._name,
            "max_requests_per_second": self._rate,
            "burst_size": self._burst_size,
            "tokens_remaining": round(self._tokens, 1),
            "metrics": self._metrics.to_dict(),
        }
