"""Tests for Heatmap API endpoints — /api/analytics/heatmap."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bot.engines.tracker import EngineTracker, TradeRecord

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _make_tracker_with_trades() -> EngineTracker:
    """Build an EngineTracker populated with sample trades at various times."""
    tracker = EngineTracker()

    trades = [
        # Monday 10:00 UTC (dow=0, hour=10)
        TradeRecord(
            engine_name="funding_rate_arb",
            symbol="BTC/USDT",
            side="long",
            entry_price=40000,
            exit_price=40500,
            quantity=0.1,
            pnl=50.0,
            cost=2.0,
            net_pnl=48.0,
            entry_time="2026-02-16T08:00:00Z",
            exit_time="2026-02-16T10:00:00Z",  # Monday
            hold_time_seconds=7200,
        ),
        # Monday 14:00 UTC (dow=0, hour=14)
        TradeRecord(
            engine_name="funding_rate_arb",
            symbol="ETH/USDT",
            side="long",
            entry_price=2500,
            exit_price=2450,
            quantity=1.0,
            pnl=-50.0,
            cost=1.5,
            net_pnl=-51.5,
            entry_time="2026-02-16T12:00:00Z",
            exit_time="2026-02-16T14:00:00Z",  # Monday
            hold_time_seconds=7200,
        ),
        # Wednesday 08:00 UTC (dow=2, hour=8)
        TradeRecord(
            engine_name="grid_trading",
            symbol="BTC/USDT",
            side="buy",
            entry_price=39000,
            exit_price=39800,
            quantity=0.05,
            pnl=40.0,
            cost=1.0,
            net_pnl=39.0,
            entry_time="2026-02-18T06:00:00Z",
            exit_time="2026-02-18T08:00:00Z",  # Wednesday
            hold_time_seconds=7200,
        ),
        # Trade in January (different month)
        TradeRecord(
            engine_name="stat_arb",
            symbol="SOL/USDT",
            side="long",
            entry_price=100,
            exit_price=105,
            quantity=10.0,
            pnl=50.0,
            cost=3.0,
            net_pnl=47.0,
            entry_time="2026-01-15T12:00:00Z",
            exit_time="2026-01-15T16:00:00Z",  # Thursday
            hold_time_seconds=14400,
        ),
    ]

    for t in trades:
        tracker.record_trade(t.engine_name, t)

    return tracker


def _mock_engine_manager() -> MagicMock:
    """Create a mock EngineManager with populated tracker."""
    mgr = MagicMock()
    mgr.tracker = _make_tracker_with_trades()
    return mgr


# ──────────────────────────────────────────────────────────────
# hourly_dow heatmap
# ──────────────────────────────────────────────────────────────


class TestHeatmapHourlyDow:
    @pytest.mark.asyncio
    async def test_no_engine_manager(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        set_engine_manager(None)
        result = await get_heatmap(type="hourly_dow")
        assert result["data"] == []

    @pytest.mark.asyncio
    async def test_hourly_dow_grid_size(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_heatmap(type="hourly_dow")
        # 7 days x 24 hours = 168 cells
        assert len(result["data"]) == 168

    @pytest.mark.asyncio
    async def test_hourly_dow_fields(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_heatmap(type="hourly_dow")
        cell = result["data"][0]
        assert "hour" in cell
        assert "dow" in cell
        assert "pnl" in cell
        assert "trade_count" in cell
        assert "win_rate" in cell

    @pytest.mark.asyncio
    async def test_hourly_dow_pnl_values(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_heatmap(type="hourly_dow")
        # Monday hour=10 should have BTC win (+48.0)
        mon_10 = next(c for c in result["data"] if c["dow"] == 0 and c["hour"] == 10)
        assert mon_10["pnl"] == 48.0
        assert mon_10["trade_count"] == 1
        assert mon_10["win_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_hourly_dow_loss_cell(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_heatmap(type="hourly_dow")
        # Monday hour=14 should have ETH loss (-51.5)
        mon_14 = next(c for c in result["data"] if c["dow"] == 0 and c["hour"] == 14)
        assert mon_14["pnl"] == -51.5
        assert mon_14["trade_count"] == 1
        assert mon_14["win_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_hourly_dow_empty_cell(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_heatmap(type="hourly_dow")
        # Tuesday hour=0 should have no trades
        tue_0 = next(c for c in result["data"] if c["dow"] == 1 and c["hour"] == 0)
        assert tue_0["pnl"] == 0.0
        assert tue_0["trade_count"] == 0
        assert tue_0["win_rate"] == 0.0


# ──────────────────────────────────────────────────────────────
# engine_symbol heatmap
# ──────────────────────────────────────────────────────────────


class TestHeatmapEngineSymbol:
    @pytest.mark.asyncio
    async def test_engine_symbol_data(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_heatmap(type="engine_symbol")
        data = result["data"]
        # We have 4 trades across 3 engine-symbol combos:
        # funding_rate_arb/BTC, funding_rate_arb/ETH, grid_trading/BTC, stat_arb/SOL
        assert len(data) == 4

    @pytest.mark.asyncio
    async def test_engine_symbol_fields(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_heatmap(type="engine_symbol")
        cell = result["data"][0]
        assert "engine" in cell
        assert "symbol" in cell
        assert "pnl" in cell
        assert "trade_count" in cell
        assert "win_rate" in cell

    @pytest.mark.asyncio
    async def test_engine_symbol_aggregation(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_heatmap(type="engine_symbol")
        # Find funding_rate_arb/BTC
        farb_btc = next(
            (
                c for c in result["data"]
                if c["engine"] == "funding_rate_arb"
                and c["symbol"] == "BTC/USDT"
            ),
            None,
        )
        assert farb_btc is not None
        assert farb_btc["pnl"] == 48.0
        assert farb_btc["trade_count"] == 1

    @pytest.mark.asyncio
    async def test_engine_filter(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_heatmap(type="engine_symbol", engine="grid_trading")
        assert len(result["data"]) == 1
        assert result["data"][0]["engine"] == "grid_trading"


# ──────────────────────────────────────────────────────────────
# monthly heatmap
# ──────────────────────────────────────────────────────────────


class TestHeatmapMonthly:
    @pytest.mark.asyncio
    async def test_monthly_data(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_heatmap(type="monthly")
        # Trades in Jan 2026 and Feb 2026
        assert len(result["data"]) == 2

    @pytest.mark.asyncio
    async def test_monthly_fields(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_heatmap(type="monthly")
        cell = result["data"][0]
        assert "year" in cell
        assert "month" in cell
        assert "pnl" in cell
        assert "trade_count" in cell
        assert "win_rate" in cell

    @pytest.mark.asyncio
    async def test_monthly_sorted_chronologically(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_heatmap(type="monthly")
        data = result["data"]
        # January should come before February
        assert data[0]["month"] == 1
        assert data[1]["month"] == 2

    @pytest.mark.asyncio
    async def test_monthly_pnl_aggregation(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_heatmap(type="monthly")
        # Feb 2026: 48.0 + (-51.5) + 39.0 = 35.5
        feb = next(c for c in result["data"] if c["month"] == 2)
        assert feb["pnl"] == 35.5
        assert feb["trade_count"] == 3
        # Jan 2026: 47.0
        jan = next(c for c in result["data"] if c["month"] == 1)
        assert jan["pnl"] == 47.0
        assert jan["trade_count"] == 1


# ──────────────────────────────────────────────────────────────
# Unknown type
# ──────────────────────────────────────────────────────────────


class TestHeatmapUnknownType:
    @pytest.mark.asyncio
    async def test_unknown_type(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_heatmap(type="invalid_type")
        assert result["data"] == []
        assert "error" in result


# ──────────────────────────────────────────────────────────────
# Empty tracker
# ──────────────────────────────────────────────────────────────


class TestHeatmapEmptyTracker:
    @pytest.mark.asyncio
    async def test_empty_hourly_dow(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = MagicMock()
        mgr.tracker = EngineTracker()
        set_engine_manager(mgr)
        result = await get_heatmap(type="hourly_dow")
        assert len(result["data"]) == 168
        assert all(c["trade_count"] == 0 for c in result["data"])

    @pytest.mark.asyncio
    async def test_empty_engine_symbol(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = MagicMock()
        mgr.tracker = EngineTracker()
        set_engine_manager(mgr)
        result = await get_heatmap(type="engine_symbol")
        assert result["data"] == []

    @pytest.mark.asyncio
    async def test_empty_monthly(self):
        from bot.dashboard.app import get_heatmap, set_engine_manager

        mgr = MagicMock()
        mgr.tracker = EngineTracker()
        set_engine_manager(mgr)
        result = await get_heatmap(type="monthly")
        assert result["data"] == []
