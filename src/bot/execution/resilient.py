"""Resilient exchange wrapper with rate limiter, circuit breaker, and auto-reconnect."""

import asyncio
from typing import Any

import structlog

from bot.exchanges.base import ExchangeAdapter
from bot.exchanges.rate_limiter import DEFAULT_EXCHANGE_LIMITS, RateLimiter
from bot.execution.circuit_breaker import CircuitBreaker

logger = structlog.get_logger()


class ResilientExchange:
    """Wraps an ExchangeAdapter with rate limiter, circuit breaker, and retry logic.

    Call order: rate limiter → circuit breaker → exchange call → retry on failure.
    Rate limit waits happen before the circuit breaker check so that throttled
    requests do not count as failures.
    """

    def __init__(
        self,
        exchange: ExchangeAdapter,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limiter: RateLimiter | None = None,
        rate_limit_enabled: bool = True,
    ):
        self._exchange = exchange
        self._breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout,
            name=exchange.name,
        )
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # Create rate limiter: use provided, or auto-create from exchange defaults
        if rate_limiter is not None:
            self._rate_limiter = rate_limiter
        elif rate_limit_enabled:
            defaults = DEFAULT_EXCHANGE_LIMITS.get(
                exchange.name, {"requests_per_second": 10.0, "burst_size": 20}
            )
            self._rate_limiter = RateLimiter(
                max_requests_per_second=defaults["requests_per_second"],
                burst_size=int(defaults["burst_size"]),
                name=exchange.name,
            )
        else:
            self._rate_limiter = None

    @property
    def name(self) -> str:
        return self._exchange.name

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._breaker

    @property
    def rate_limiter(self) -> RateLimiter | None:
        return self._rate_limiter

    @property
    def is_available(self) -> bool:
        """Check if the exchange is available (circuit not open)."""
        return self._breaker.can_execute()

    async def _call_with_breaker(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute an exchange method with rate limiting and circuit breaker protection."""
        # Step 1: Rate limit (wait for token before checking circuit breaker)
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()

        # Step 2: Circuit breaker check
        if not self._breaker.can_execute():
            raise ConnectionError(
                f"Circuit breaker open for {self._exchange.name}"
            )

        method = getattr(self._exchange, method_name)

        for attempt in range(self._max_retries):
            try:
                result = await method(*args, **kwargs)
                self._breaker.record_success()
                return result
            except ConnectionError as e:
                self._breaker.record_failure()
                logger.warning(
                    "exchange_call_failed",
                    exchange=self._exchange.name,
                    method=method_name,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
            except (ValueError, RuntimeError):
                raise

        raise ConnectionError(
            f"Max retries exceeded for {self._exchange.name}.{method_name}"
        )

    async def get_ticker(self, symbol: str) -> dict:
        return await self._call_with_breaker("get_ticker", symbol)

    async def get_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> list:
        return await self._call_with_breaker(
            "get_ohlcv", symbol, timeframe, limit
        )

    async def get_balance(self) -> dict[str, float]:
        return await self._call_with_breaker("get_balance")

    async def create_order(self, **kwargs: Any) -> Any:
        return await self._call_with_breaker("create_order", **kwargs)

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        return await self._call_with_breaker("cancel_order", order_id, symbol)

    async def get_order_status(self, order_id: str, symbol: str) -> Any:
        return await self._call_with_breaker("get_order_status", order_id, symbol)

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        return await self._call_with_breaker("get_order_book", symbol, limit)

    async def close(self) -> None:
        await self._exchange.close()
