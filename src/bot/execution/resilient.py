"""Resilient exchange wrapper with circuit breaker and auto-reconnect."""

import asyncio
from typing import Any

import structlog

from bot.exchanges.base import ExchangeAdapter
from bot.execution.circuit_breaker import CircuitBreaker

logger = structlog.get_logger()


class ResilientExchange:
    """Wraps an ExchangeAdapter with circuit breaker and retry logic."""

    def __init__(
        self,
        exchange: ExchangeAdapter,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self._exchange = exchange
        self._breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout,
            name=exchange.name,
        )
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    @property
    def name(self) -> str:
        return self._exchange.name

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._breaker

    @property
    def is_available(self) -> bool:
        """Check if the exchange is available (circuit not open)."""
        return self._breaker.can_execute()

    async def _call_with_breaker(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute an exchange method with circuit breaker protection."""
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
