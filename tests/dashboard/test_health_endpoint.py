"""Tests for enhanced health endpoint — /health and /api/health."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.config import Settings

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _mock_engine_manager(statuses: dict[str, str]) -> MagicMock:
    """Build a mock EngineManager with given engine statuses.

    Args:
        statuses: dict of engine_name -> status string (running/stopped/error/paused)
    """
    mgr = MagicMock()
    status_map = {}
    for name, st in statuses.items():
        status_map[name] = {"status": st, "name": name}
    mgr.get_status.return_value = status_map
    return mgr


# ──────────────────────────────────────────────────────────────
# Health status determination
# ──────────────────────────────────────────────────────────────


class TestHealthEndpointStatus:
    """Test /health status logic: healthy, degraded, unhealthy."""

    @pytest.mark.asyncio
    async def test_healthy_no_engine_manager(self):
        """Without engine manager, should be healthy."""
        from bot.dashboard.app import health_check, set_engine_manager

        set_engine_manager(None)
        result = await health_check()
        assert result["status"] == "healthy"
        assert result["engines"] == {}

    @pytest.mark.asyncio
    async def test_healthy_all_running(self):
        """All engines running → healthy."""
        from bot.dashboard.app import health_check, set_engine_manager

        mgr = _mock_engine_manager({
            "funding_rate_arb": "running",
            "grid_trading": "running",
            "stat_arb": "running",
        })
        set_engine_manager(mgr)
        result = await health_check()
        assert result["status"] == "healthy"
        assert result["engines"]["funding_rate_arb"] == "running"
        assert result["engines"]["grid_trading"] == "running"
        assert result["engines"]["stat_arb"] == "running"

    @pytest.mark.asyncio
    async def test_degraded_some_error(self):
        """Some engines in error state → degraded."""
        from bot.dashboard.app import health_check, set_engine_manager

        mgr = _mock_engine_manager({
            "funding_rate_arb": "running",
            "grid_trading": "error",
            "stat_arb": "running",
        })
        set_engine_manager(mgr)
        result = await health_check()
        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_unhealthy_all_stopped(self):
        """All engines stopped → unhealthy."""
        from bot.dashboard.app import health_check, set_engine_manager

        mgr = _mock_engine_manager({
            "funding_rate_arb": "stopped",
            "grid_trading": "stopped",
            "stat_arb": "stopped",
        })
        set_engine_manager(mgr)
        result = await health_check()
        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_degraded_error_and_stopped(self):
        """Mix of error and stopped → degraded (error takes priority)."""
        from bot.dashboard.app import health_check, set_engine_manager

        mgr = _mock_engine_manager({
            "funding_rate_arb": "stopped",
            "grid_trading": "error",
        })
        set_engine_manager(mgr)
        result = await health_check()
        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_healthy_paused_engines(self):
        """Paused engines (not error, not all stopped) → healthy."""
        from bot.dashboard.app import health_check, set_engine_manager

        mgr = _mock_engine_manager({
            "funding_rate_arb": "running",
            "grid_trading": "paused",
        })
        set_engine_manager(mgr)
        result = await health_check()
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_unhealthy_stale_cycle(self):
        """Running bot with stale cycle time → unhealthy."""
        from bot.dashboard.app import (
            _bot_state,
            health_check,
            set_engine_manager,
        )

        set_engine_manager(None)
        # Simulate stale cycle (more than 5 minutes ago)
        original_status = _bot_state["status"]
        original_cycle = _bot_state["cycle_metrics"].get("last_cycle_time")
        try:
            _bot_state["status"] = "running"
            _bot_state["cycle_metrics"]["last_cycle_time"] = (
                time.time() - 600
            )
            result = await health_check()
            assert result["status"] == "unhealthy"
        finally:
            _bot_state["status"] = original_status
            _bot_state["cycle_metrics"]["last_cycle_time"] = original_cycle


# ──────────────────────────────────────────────────────────────
# Health response fields
# ──────────────────────────────────────────────────────────────


class TestHealthResponseFields:
    """Test that response contains required fields."""

    @pytest.mark.asyncio
    async def test_basic_fields(self):
        from bot.dashboard.app import health_check, set_engine_manager

        set_engine_manager(None)
        result = await health_check()
        assert "status" in result
        assert "uptime_seconds" in result
        assert "engines" in result
        assert "database_connected" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_uptime_positive(self):
        from bot.dashboard.app import health_check, set_engine_manager

        set_engine_manager(None)
        result = await health_check()
        assert result["uptime_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_database_connected_false_by_default(self):
        from bot.dashboard.app import (
            health_check,
            set_engine_manager,
            set_store_ref,
        )

        set_engine_manager(None)
        set_store_ref(None)
        result = await health_check()
        assert result["database_connected"] is False

    @pytest.mark.asyncio
    async def test_database_connected_true(self):
        from bot.dashboard.app import (
            health_check,
            set_engine_manager,
            set_store_ref,
        )

        set_engine_manager(None)
        set_store_ref(MagicMock())  # Non-None store → connected
        result = await health_check()
        assert result["database_connected"] is True
        # Clean up
        set_store_ref(None)

    @pytest.mark.asyncio
    async def test_timestamp_is_iso(self):
        from datetime import datetime

        from bot.dashboard.app import health_check, set_engine_manager

        set_engine_manager(None)
        result = await health_check()
        ts = result["timestamp"]
        # Should not raise
        datetime.fromisoformat(ts.replace("Z", "+00:00"))


# ──────────────────────────────────────────────────────────────
# Detailed mode
# ──────────────────────────────────────────────────────────────


class TestHealthDetailed:
    """Test detailed=true adds system info."""

    @pytest.mark.asyncio
    async def test_no_detailed_fields_by_default(self):
        from bot.dashboard.app import health_check, set_engine_manager

        set_engine_manager(None)
        result = await health_check()
        assert "disk_space_mb" not in result
        assert "memory_usage_mb" not in result

    @pytest.mark.asyncio
    async def test_detailed_includes_disk_space(self):
        from bot.dashboard.app import health_check, set_engine_manager

        set_engine_manager(None)
        result = await health_check(detailed=True)
        assert "disk_space_mb" in result
        # Should be a number or None
        assert result["disk_space_mb"] is None or result["disk_space_mb"] > 0

    @pytest.mark.asyncio
    async def test_detailed_includes_memory_usage(self):
        from bot.dashboard.app import health_check, set_engine_manager

        set_engine_manager(None)
        result = await health_check(detailed=True)
        assert "memory_usage_mb" in result
        assert (
            result["memory_usage_mb"] is None
            or result["memory_usage_mb"] > 0
        )


# ──────────────────────────────────────────────────────────────
# Graceful shutdown
# ──────────────────────────────────────────────────────────────


class TestGracefulShutdown:
    """Test main.py shutdown sequence."""

    @pytest.mark.asyncio
    async def test_shutdown_stops_engines(self):
        """Shutdown calls engine_manager.stop_all()."""
        from bot.main import TradingBot

        bot = TradingBot.__new__(TradingBot)
        bot._settings = MagicMock()
        bot._settings.shutdown_timeout_seconds = 30.0
        bot._running = True
        bot._telegram = None
        bot._ws_feed = None
        bot._dashboard_task = None
        bot._exchanges = []
        bot._store = None
        bot._audit_logger = MagicMock()
        bot._audit_logger.log_bot_stopped = AsyncMock()

        mgr = MagicMock()
        mgr.stop_all = AsyncMock()
        mgr._metrics_persistence = None
        bot._engine_manager = mgr

        await bot.shutdown()

        assert bot._running is False
        mgr.stop_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_saves_metrics_snapshot(self):
        """Shutdown saves final metrics snapshot before closing."""
        from bot.main import TradingBot

        bot = TradingBot.__new__(TradingBot)
        bot._settings = MagicMock()
        bot._settings.shutdown_timeout_seconds = 30.0
        bot._settings.metrics_retention_days = 90
        bot._running = True
        bot._telegram = None
        bot._ws_feed = None
        bot._dashboard_task = None
        bot._exchanges = []
        bot._store = MagicMock()
        bot._store.close = AsyncMock()
        bot._audit_logger = MagicMock()
        bot._audit_logger.log_bot_stopped = AsyncMock()

        persistence = MagicMock()
        persistence.save_metrics_snapshot = AsyncMock()
        persistence.cleanup = AsyncMock()

        mgr = MagicMock()
        mgr.stop_all = AsyncMock()
        mgr._metrics_persistence = persistence
        bot._engine_manager = mgr

        await bot.shutdown()

        persistence.save_metrics_snapshot.assert_awaited_once()
        persistence.cleanup.assert_awaited_once_with(max_days=90)

    @pytest.mark.asyncio
    async def test_shutdown_timeout(self):
        """Shutdown respects timeout and logs warning on timeout."""
        from bot.main import TradingBot

        bot = TradingBot.__new__(TradingBot)
        bot._settings = MagicMock()
        bot._settings.shutdown_timeout_seconds = 0.1  # Very short
        bot._running = True
        bot._telegram = None
        bot._ws_feed = None
        bot._dashboard_task = None
        bot._exchanges = []
        bot._store = None
        bot._audit_logger = MagicMock()
        bot._audit_logger.log_bot_stopped = AsyncMock()

        # Engine manager that takes too long to stop
        async def slow_stop():
            await asyncio.sleep(10)

        mgr = MagicMock()
        mgr.stop_all = slow_stop
        mgr._metrics_persistence = None
        bot._engine_manager = mgr

        # Should not hang — times out
        await bot.shutdown()
        assert bot._running is False

    @pytest.mark.asyncio
    async def test_shutdown_no_engine_manager(self):
        """Shutdown works when no engine manager is set."""
        from bot.main import TradingBot

        bot = TradingBot.__new__(TradingBot)
        bot._settings = MagicMock()
        bot._settings.shutdown_timeout_seconds = 5.0
        bot._running = True
        bot._telegram = None
        bot._ws_feed = None
        bot._dashboard_task = None
        bot._exchanges = []
        bot._store = None
        bot._engine_manager = None
        bot._audit_logger = MagicMock()
        bot._audit_logger.log_bot_stopped = AsyncMock()

        await bot.shutdown()
        assert bot._running is False

    @pytest.mark.asyncio
    async def test_shutdown_closes_store(self):
        """Shutdown closes the data store."""
        from bot.main import TradingBot

        bot = TradingBot.__new__(TradingBot)
        bot._settings = MagicMock()
        bot._settings.shutdown_timeout_seconds = 5.0
        bot._running = True
        bot._telegram = None
        bot._ws_feed = None
        bot._dashboard_task = None
        bot._exchanges = []
        bot._engine_manager = None
        bot._audit_logger = MagicMock()
        bot._audit_logger.log_bot_stopped = AsyncMock()

        store = MagicMock()
        store.close = AsyncMock()
        bot._store = store

        await bot.shutdown()
        store.close.assert_awaited_once()


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────


class TestHealthConfig:
    def test_shutdown_timeout_default(self):
        s = Settings()
        assert s.shutdown_timeout_seconds == 30.0

    def test_shutdown_timeout_in_metadata(self):
        from bot.config import SETTINGS_METADATA

        assert "shutdown_timeout_seconds" in SETTINGS_METADATA
        meta = SETTINGS_METADATA["shutdown_timeout_seconds"]
        assert meta["section"] == "Shutdown"
        assert meta["type"] == "float"
        assert meta["requires_restart"] is True
