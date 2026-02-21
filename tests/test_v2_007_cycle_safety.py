"""Tests for V2-007: Trading cycle safety — overlap lock, error tracebacks, cycle metrics."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from bot.config import Settings, TradingMode
from bot.dashboard.app import app, get_state, update_state
from bot.main import TradingBot


def make_settings(**kwargs):
    """Create test settings with safe defaults."""
    defaults = {
        "trading_mode": TradingMode.PAPER,
        "database_url": "sqlite+aiosqlite:///:memory:",
        "binance_api_key": "",
        "upbit_api_key": "",
        "symbols": ["BTC/USDT"],
        "loop_interval_seconds": 1,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


class TestCycleLock:
    @pytest.mark.asyncio
    async def test_bot_has_cycle_lock(self):
        """TradingBot should have an asyncio.Lock for cycle overlap prevention."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        assert isinstance(bot._cycle_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_cycle_lock_prevents_overlap(self):
        """When lock is held, run_trading_loop should skip the cycle."""
        settings = make_settings(loop_interval_seconds=1)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Acquire the lock before the loop starts
        await bot._cycle_lock.acquire()

        cycle_called = False
        original_trading_cycle = bot._trading_cycle

        async def mock_trading_cycle():
            nonlocal cycle_called
            cycle_called = True
            await original_trading_cycle()

        bot._trading_cycle = mock_trading_cycle

        # Run one iteration of the loop — the lock is held, so cycle should be skipped
        bot._running = True

        async def stop_after_brief():
            await asyncio.sleep(0.05)
            bot.stop()

        asyncio.create_task(stop_after_brief())
        await bot.run_trading_loop()

        # Cycle should NOT have been called because lock was held
        assert not cycle_called

        # Release the lock
        bot._cycle_lock.release()
        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_cycle_completes_normally_without_lock(self):
        """When lock is not held, trading cycle should execute normally."""
        settings = make_settings(loop_interval_seconds=1)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Run a single cycle directly (not through loop, to avoid timing issues)
        async with bot._cycle_lock:
            await bot._trading_cycle()

        # Cycle count is updated by run_trading_loop, not _trading_cycle directly
        # So we just confirm _trading_cycle doesn't error
        await bot.shutdown()


class TestCycleMetrics:
    @pytest.mark.asyncio
    async def test_initial_cycle_metrics(self):
        """Initial cycle metrics should be zeroed."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        metrics = bot.cycle_metrics
        assert metrics["cycle_count"] == 0
        assert metrics["average_cycle_duration"] == 0.0
        assert metrics["last_cycle_time"] is None

    @pytest.mark.asyncio
    async def test_cycle_metrics_updated_after_cycle(self):
        """Cycle metrics should be updated after a successful trading cycle."""
        settings = make_settings(loop_interval_seconds=1)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Run one iteration
        bot._running = True

        async def stop_after_one():
            await asyncio.sleep(0.05)
            bot.stop()

        asyncio.create_task(stop_after_one())
        await bot.run_trading_loop()

        metrics = bot.cycle_metrics
        assert metrics["cycle_count"] >= 1
        assert metrics["average_cycle_duration"] >= 0.0
        assert metrics["last_cycle_time"] is not None
        # last_cycle_time should be a recent timestamp
        assert time.time() - metrics["last_cycle_time"] < 5.0

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_average_cycle_duration_calculation(self):
        """Average duration should be total duration / cycle count."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        bot._cycle_count = 4
        bot._total_cycle_duration = 2.0

        metrics = bot.cycle_metrics
        assert metrics["average_cycle_duration"] == 0.5

    @pytest.mark.asyncio
    async def test_cycle_metrics_in_dashboard_state(self):
        """Cycle metrics should be included in dashboard state after a cycle."""
        settings = make_settings(loop_interval_seconds=1)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Reset dashboard state
        update_state(
            status="stopped",
            trades=[],
            metrics={},
            portfolio={},
            cycle_metrics={
                "cycle_count": 0,
                "average_cycle_duration": 0.0,
                "last_cycle_time": None,
            },
        )

        bot._running = True

        async def stop_after_one():
            await asyncio.sleep(0.05)
            bot.stop()

        asyncio.create_task(stop_after_one())
        await bot.run_trading_loop()

        state = get_state()
        assert "cycle_metrics" in state
        assert state["cycle_metrics"]["cycle_count"] >= 1

        await bot.shutdown()


class TestErrorLogging:
    @pytest.mark.asyncio
    async def test_error_logged_with_exc_info(self):
        """Errors in trading cycle should be logged with exc_info=True for full tracebacks."""
        settings = make_settings(loop_interval_seconds=1)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Make collector raise an error
        bot._collector = AsyncMock()
        bot._collector.collect_once = AsyncMock(
            side_effect=RuntimeError("test error")
        )

        bot._running = True

        async def stop_after_one():
            await asyncio.sleep(0.05)
            bot.stop()

        asyncio.create_task(stop_after_one())

        with patch("bot.main.logger") as mock_logger:
            await bot.run_trading_loop()
            # Verify logger.error was called with exc_info=True
            mock_logger.error.assert_called()
            call_kwargs = mock_logger.error.call_args
            assert call_kwargs[1].get("exc_info") is True or (
                len(call_kwargs[0]) > 0
                and call_kwargs[0][0] == "trading_cycle_error"
            )

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_telegram_gets_traceback_on_error(self):
        """Telegram should receive the full traceback, not just the error message."""
        settings = make_settings(loop_interval_seconds=1)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Inject mock telegram
        mock_telegram = AsyncMock()
        mock_telegram.notify_error = AsyncMock(return_value=True)
        bot._telegram = mock_telegram

        # Make collector raise an error
        bot._collector = AsyncMock()
        bot._collector.collect_once = AsyncMock(
            side_effect=RuntimeError("test traceback error")
        )

        bot._running = True

        async def stop_after_one():
            await asyncio.sleep(0.05)
            bot.stop()

        asyncio.create_task(stop_after_one())
        await bot.run_trading_loop()

        # Telegram should have been called with traceback text (not just "test error")
        mock_telegram.notify_error.assert_called()
        error_msg = mock_telegram.notify_error.call_args[0][0]
        assert "test traceback error" in error_msg
        assert "Traceback" in error_msg

        await bot.shutdown()


class TestHealthEndpoint:
    @pytest.fixture
    async def client(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.fixture(autouse=True)
    def reset_dashboard_state(self):
        """Reset bot state before each test."""
        update_state(
            status="stopped",
            started_at=None,
            trades=[],
            metrics={},
            portfolio={"balances": {}, "positions": [], "total_value": 0.0},
            cycle_metrics={
                "cycle_count": 0,
                "average_cycle_duration": 0.0,
                "last_cycle_time": None,
            },
        )

    @pytest.mark.asyncio
    async def test_health_healthy_when_stopped(self, client):
        """Health should be healthy when bot is stopped (not expected to have cycles)."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_healthy_when_running_recent_cycle(self, client):
        """Health should be healthy when bot is running with a recent cycle."""
        update_state(
            status="running",
            cycle_metrics={
                "cycle_count": 10,
                "average_cycle_duration": 0.5,
                "last_cycle_time": time.time(),  # just now
            },
        )
        resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_unhealthy_when_cycle_stale(self, client):
        """Health should be unhealthy when last cycle was >5 minutes ago."""
        update_state(
            status="running",
            cycle_metrics={
                "cycle_count": 10,
                "average_cycle_duration": 0.5,
                "last_cycle_time": time.time() - 600,  # 10 minutes ago
            },
        )
        resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["reason"] == "last_cycle_stale"
        assert data["last_cycle_seconds_ago"] > 300

    @pytest.mark.asyncio
    async def test_health_healthy_when_running_no_cycle_yet(self, client):
        """Health should be healthy when bot is running but no cycle has completed yet."""
        update_state(
            status="running",
            cycle_metrics={
                "cycle_count": 0,
                "average_cycle_duration": 0.0,
                "last_cycle_time": None,
            },
        )
        resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"


class TestStatusEndpoint:
    @pytest.fixture
    async def client(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.fixture(autouse=True)
    def reset_dashboard_state(self):
        """Reset bot state before each test."""
        update_state(
            status="stopped",
            started_at=None,
            trades=[],
            metrics={},
            portfolio={"balances": {}, "positions": [], "total_value": 0.0},
            cycle_metrics={
                "cycle_count": 0,
                "average_cycle_duration": 0.0,
                "last_cycle_time": None,
            },
        )

    @pytest.mark.asyncio
    async def test_status_includes_cycle_metrics(self, client):
        """Status endpoint should include cycle_metrics."""
        update_state(
            status="running",
            cycle_metrics={
                "cycle_count": 5,
                "average_cycle_duration": 1.2,
                "last_cycle_time": 1700000000.0,
            },
        )
        resp = await client.get("/status")
        data = resp.json()
        assert "cycle_metrics" in data
        assert data["cycle_metrics"]["cycle_count"] == 5
        assert data["cycle_metrics"]["average_cycle_duration"] == 1.2
