"""Tests for the token bucket rate limiter."""

import asyncio
from unittest.mock import AsyncMock, PropertyMock

import pytest

from bot.exchanges.rate_limiter import (
    DEFAULT_EXCHANGE_LIMITS,
    RateLimiter,
    RateLimitMetrics,
)
from bot.execution.resilient import ResilientExchange


class TestRateLimitMetrics:
    def test_initial_state(self):
        m = RateLimitMetrics()
        assert m.total_requests == 0
        assert m.throttled_requests == 0
        assert m.total_wait_time_ms == 0.0
        assert m.avg_wait_ms == 0.0

    def test_record_request_no_wait(self):
        m = RateLimitMetrics()
        m.record_request(wait_ms=0.0)
        assert m.total_requests == 1
        assert m.throttled_requests == 0

    def test_record_request_with_wait(self):
        m = RateLimitMetrics()
        m.record_request(wait_ms=50.0)
        m.record_request(wait_ms=30.0)
        assert m.total_requests == 2
        assert m.throttled_requests == 2
        assert m.total_wait_time_ms == 80.0
        assert m.avg_wait_ms == 40.0

    def test_to_dict(self):
        m = RateLimitMetrics()
        m.record_request(wait_ms=10.0)
        d = m.to_dict()
        assert "total_requests" in d
        assert "throttled_requests" in d
        assert "avg_wait_ms" in d
        assert "current_rps" in d

    def test_requests_per_second(self):
        m = RateLimitMetrics()
        # Record several requests
        for _ in range(5):
            m.record_request()
        # Should be > 0 since all recorded just now
        assert m.requests_per_second > 0


class TestRateLimiter:
    def test_creation_defaults(self):
        rl = RateLimiter()
        assert rl.name == "default"
        assert rl.max_requests_per_second == 10.0
        assert rl.burst_size == 20

    def test_creation_custom(self):
        rl = RateLimiter(
            max_requests_per_second=5.0,
            burst_size=10,
            name="test",
        )
        assert rl.name == "test"
        assert rl.max_requests_per_second == 5.0
        assert rl.burst_size == 10

    def test_tokens_remaining_starts_full(self):
        rl = RateLimiter(burst_size=15)
        assert rl.tokens_remaining == pytest.approx(15.0, abs=0.5)

    @pytest.mark.asyncio
    async def test_acquire_no_wait_when_tokens_available(self):
        rl = RateLimiter(max_requests_per_second=100.0, burst_size=10)
        wait = await rl.acquire()
        assert wait == 0.0
        assert rl.metrics.total_requests == 1
        assert rl.metrics.throttled_requests == 0

    @pytest.mark.asyncio
    async def test_burst_allowed(self):
        """Burst of requests up to burst_size should not wait."""
        rl = RateLimiter(max_requests_per_second=10.0, burst_size=5)
        for _ in range(5):
            wait = await rl.acquire()
            assert wait == 0.0
        assert rl.metrics.total_requests == 5
        assert rl.metrics.throttled_requests == 0

    @pytest.mark.asyncio
    async def test_throttle_after_burst_exhausted(self):
        """After exhausting burst, requests should be throttled (wait > 0)."""
        rl = RateLimiter(max_requests_per_second=100.0, burst_size=2)
        # Exhaust burst
        await rl.acquire()
        await rl.acquire()
        # Next request should throttle
        wait = await rl.acquire()
        assert wait > 0
        assert rl.metrics.throttled_requests >= 1

    @pytest.mark.asyncio
    async def test_tokens_refill_over_time(self):
        """Tokens should refill after some time passes."""
        rl = RateLimiter(max_requests_per_second=1000.0, burst_size=2)
        # Exhaust tokens
        await rl.acquire()
        await rl.acquire()
        # Wait briefly for refill
        await asyncio.sleep(0.01)
        # Should have some tokens now
        remaining = rl.tokens_remaining
        assert remaining > 0

    @pytest.mark.asyncio
    async def test_tokens_capped_at_burst_size(self):
        """Tokens should never exceed burst_size."""
        rl = RateLimiter(max_requests_per_second=1000.0, burst_size=5)
        # Wait long enough that many tokens would be generated
        await asyncio.sleep(0.05)
        assert rl.tokens_remaining <= 5.0

    @pytest.mark.asyncio
    async def test_concurrent_acquire(self):
        """Multiple concurrent acquire() calls should all be served."""
        rl = RateLimiter(max_requests_per_second=100.0, burst_size=10)
        results = await asyncio.gather(*[rl.acquire() for _ in range(10)])
        assert len(results) == 10
        assert rl.metrics.total_requests == 10

    def test_to_dict(self):
        rl = RateLimiter(
            max_requests_per_second=20.0,
            burst_size=40,
            name="binance",
        )
        d = rl.to_dict()
        assert d["name"] == "binance"
        assert d["max_requests_per_second"] == 20.0
        assert d["burst_size"] == 40
        assert "tokens_remaining" in d
        assert "metrics" in d
        assert isinstance(d["metrics"], dict)

    def test_metrics_property(self):
        rl = RateLimiter()
        assert isinstance(rl.metrics, RateLimitMetrics)


class TestDefaultExchangeLimits:
    def test_binance_defaults_exist(self):
        assert "binance" in DEFAULT_EXCHANGE_LIMITS
        limits = DEFAULT_EXCHANGE_LIMITS["binance"]
        assert limits["requests_per_second"] == 20.0
        assert limits["burst_size"] == 40

    def test_upbit_defaults_exist(self):
        assert "upbit" in DEFAULT_EXCHANGE_LIMITS
        limits = DEFAULT_EXCHANGE_LIMITS["upbit"]
        assert limits["requests_per_second"] == 10.0
        assert limits["burst_size"] == 20


class TestResilientExchangeWithRateLimiter:
    @pytest.fixture
    def mock_exchange(self):
        exchange = AsyncMock()
        type(exchange).name = PropertyMock(return_value="mock_exchange")
        exchange.get_ticker = AsyncMock(return_value={"last": 50000.0})
        exchange.get_balance = AsyncMock(return_value={"USDT": 10000.0})
        exchange.close = AsyncMock()
        return exchange

    def test_auto_creates_rate_limiter(self, mock_exchange):
        """ResilientExchange should auto-create a rate limiter when enabled."""
        re = ResilientExchange(mock_exchange, rate_limit_enabled=True)
        assert re.rate_limiter is not None
        # mock_exchange is not in DEFAULT_EXCHANGE_LIMITS, so should use defaults
        assert re.rate_limiter.max_requests_per_second == 10.0

    def test_no_rate_limiter_when_disabled(self, mock_exchange):
        """ResilientExchange should not have rate limiter when disabled."""
        re = ResilientExchange(mock_exchange, rate_limit_enabled=False)
        assert re.rate_limiter is None

    def test_custom_rate_limiter(self, mock_exchange):
        """Custom rate limiter should be used when provided."""
        custom = RateLimiter(max_requests_per_second=5.0, burst_size=8, name="custom")
        re = ResilientExchange(mock_exchange, rate_limiter=custom)
        assert re.rate_limiter is custom
        assert re.rate_limiter.max_requests_per_second == 5.0

    def test_binance_exchange_defaults(self):
        """Binance exchange should get Binance-specific rate limits."""
        exchange = AsyncMock()
        type(exchange).name = PropertyMock(return_value="binance")
        re = ResilientExchange(exchange, rate_limit_enabled=True)
        assert re.rate_limiter is not None
        assert re.rate_limiter.max_requests_per_second == 20.0
        assert re.rate_limiter.burst_size == 40

    def test_upbit_exchange_defaults(self):
        """Upbit exchange should get Upbit-specific rate limits."""
        exchange = AsyncMock()
        type(exchange).name = PropertyMock(return_value="upbit")
        re = ResilientExchange(exchange, rate_limit_enabled=True)
        assert re.rate_limiter is not None
        assert re.rate_limiter.max_requests_per_second == 10.0
        assert re.rate_limiter.burst_size == 20

    @pytest.mark.asyncio
    async def test_rate_limiter_applied_before_circuit_breaker(self, mock_exchange):
        """Rate limiter should be applied before the circuit breaker check."""
        rl = RateLimiter(max_requests_per_second=100.0, burst_size=10, name="test")
        re = ResilientExchange(
            mock_exchange,
            rate_limiter=rl,
            rate_limit_enabled=True,
        )
        result = await re.get_ticker("BTC/USDT")
        assert result["last"] == 50000.0
        assert rl.metrics.total_requests == 1

    @pytest.mark.asyncio
    async def test_rate_limiter_metrics_tracked_across_calls(self, mock_exchange):
        """Multiple calls should increment rate limiter metrics."""
        rl = RateLimiter(max_requests_per_second=100.0, burst_size=20, name="test")
        re = ResilientExchange(mock_exchange, rate_limiter=rl)
        await re.get_ticker("BTC/USDT")
        await re.get_balance()
        assert rl.metrics.total_requests == 2

    @pytest.mark.asyncio
    async def test_rate_limiter_not_called_when_disabled(self, mock_exchange):
        """When rate_limit_enabled=False, no rate limiter should be used."""
        re = ResilientExchange(mock_exchange, rate_limit_enabled=False)
        result = await re.get_ticker("BTC/USDT")
        assert result["last"] == 50000.0
        assert re.rate_limiter is None

    @pytest.mark.asyncio
    async def test_existing_behavior_preserved(self, mock_exchange):
        """Existing circuit breaker behavior should still work with rate limiter."""
        mock_exchange.get_ticker = AsyncMock(side_effect=ConnectionError("fail"))
        re = ResilientExchange(
            mock_exchange,
            failure_threshold=1,
            max_retries=1,
            retry_delay=0.01,
            rate_limit_enabled=True,
        )
        with pytest.raises(ConnectionError):
            await re.get_ticker("BTC/USDT")
        # Circuit should be open
        assert not re.is_available

    @pytest.mark.asyncio
    async def test_value_error_still_propagates_with_rate_limiter(self, mock_exchange):
        """ValueError should still propagate through rate limiter."""
        mock_exchange.get_ticker = AsyncMock(side_effect=ValueError("bad symbol"))
        re = ResilientExchange(mock_exchange, rate_limit_enabled=True)
        with pytest.raises(ValueError, match="bad symbol"):
            await re.get_ticker("INVALID")

    @pytest.mark.asyncio
    async def test_retry_works_with_rate_limiter(self, mock_exchange):
        """Retry logic should work correctly alongside rate limiter."""
        mock_exchange.get_ticker = AsyncMock(
            side_effect=[ConnectionError("timeout"), {"last": 50000.0}]
        )
        re = ResilientExchange(
            mock_exchange,
            retry_delay=0.01,
            rate_limit_enabled=True,
        )
        result = await re.get_ticker("BTC/USDT")
        assert result["last"] == 50000.0
        assert mock_exchange.get_ticker.call_count == 2


class TestRateLimiterDashboardStatus:
    """Test that rate limit info appears in /api/status when configured."""

    @pytest.fixture(autouse=True)
    def reset_dashboard(self):
        """Reset dashboard state before each test."""
        from bot.dashboard.app import _bot_state, set_settings

        original_state = dict(_bot_state)
        set_settings(None)
        yield
        _bot_state.update(original_state)
        _bot_state.pop("rate_limits", None)
        set_settings(None)

    @pytest.mark.asyncio
    async def test_status_includes_rate_limits(self):
        """When rate_limits is in bot_state, /api/status should include it."""
        from httpx import ASGITransport, AsyncClient

        from bot.dashboard.app import app, update_state

        update_state(
            status="running",
            rate_limits={
                "binance": {
                    "name": "binance",
                    "max_requests_per_second": 20.0,
                    "burst_size": 40,
                    "tokens_remaining": 38.0,
                    "metrics": {
                        "total_requests": 10,
                        "throttled_requests": 0,
                        "avg_wait_ms": 0.0,
                        "current_rps": 1.5,
                    },
                }
            },
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/status")
            assert resp.status_code == 200
            data = resp.json()
            assert "rate_limits" in data
            assert "binance" in data["rate_limits"]
            rl = data["rate_limits"]["binance"]
            assert rl["max_requests_per_second"] == 20.0
            assert rl["metrics"]["total_requests"] == 10

    @pytest.mark.asyncio
    async def test_status_no_rate_limits_when_not_set(self):
        """When rate_limits is not in bot_state, /api/status should omit it."""
        from httpx import ASGITransport, AsyncClient

        from bot.dashboard.app import app, update_state

        update_state(status="running")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/status")
            assert resp.status_code == 200
            data = resp.json()
            assert "rate_limits" not in data
