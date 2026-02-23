"""Tests for Trade Explorer API endpoints — /api/trades/detail and /api/trades/export."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bot.engines.tracker import EngineTracker, TradeRecord

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _make_tracker_with_trades() -> EngineTracker:
    """Build an EngineTracker populated with sample trades."""
    tracker = EngineTracker()

    trades = [
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
            entry_time="2026-02-22T10:00:00Z",
            exit_time="2026-02-22T14:00:00Z",
            hold_time_seconds=14400,
        ),
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
            entry_time="2026-02-22T12:00:00Z",
            exit_time="2026-02-22T18:00:00Z",
            hold_time_seconds=21600,
        ),
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
            entry_time="2026-02-23T08:00:00Z",
            exit_time="2026-02-23T10:00:00Z",
            hold_time_seconds=7200,
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
# /api/trades/detail
# ──────────────────────────────────────────────────────────────


class TestTradeDetail:
    @pytest.mark.asyncio
    async def test_no_engine_manager(self):
        from bot.dashboard.app import get_trade_detail, set_engine_manager

        set_engine_manager(None)
        result = await get_trade_detail()
        assert result["trades"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_all_trades(self):
        from bot.dashboard.app import get_trade_detail, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_trade_detail(limit=50)
        assert result["total"] == 3
        assert len(result["trades"]) == 3
        # Newest first
        assert result["trades"][0]["exit_time"] == "2026-02-23T10:00:00Z"

    @pytest.mark.asyncio
    async def test_filter_by_engine(self):
        from bot.dashboard.app import get_trade_detail, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_trade_detail(engine="grid_trading", limit=50)
        assert result["total"] == 1
        assert result["trades"][0]["engine_name"] == "grid_trading"

    @pytest.mark.asyncio
    async def test_filter_by_symbol(self):
        from bot.dashboard.app import get_trade_detail, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_trade_detail(symbol="BTC/USDT", limit=50)
        assert result["total"] == 2
        for t in result["trades"]:
            assert t["symbol"] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_filter_win_only(self):
        from bot.dashboard.app import get_trade_detail, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_trade_detail(win_only=True, limit=50)
        assert result["total"] == 2
        for t in result["trades"]:
            assert t["net_pnl"] > 0

    @pytest.mark.asyncio
    async def test_filter_loss_only(self):
        from bot.dashboard.app import get_trade_detail, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_trade_detail(win_only=False, limit=50)
        assert result["total"] == 1
        assert result["trades"][0]["net_pnl"] < 0

    @pytest.mark.asyncio
    async def test_pagination(self):
        from bot.dashboard.app import get_trade_detail, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_trade_detail(limit=2, offset=0)
        assert result["total"] == 3
        assert len(result["trades"]) == 2

        result2 = await get_trade_detail(limit=2, offset=2)
        assert len(result2["trades"]) == 1

    @pytest.mark.asyncio
    async def test_filter_by_date_range(self):
        from bot.dashboard.app import get_trade_detail, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_trade_detail(
            start="2026-02-23T00:00:00Z",
            limit=50,
        )
        assert result["total"] == 1
        assert result["trades"][0]["engine_name"] == "grid_trading"

    @pytest.mark.asyncio
    async def test_trade_fields(self):
        from bot.dashboard.app import get_trade_detail, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await get_trade_detail(limit=1)
        trade = result["trades"][0]
        expected_fields = {
            "engine_name", "symbol", "side", "entry_price",
            "exit_price", "quantity", "pnl", "cost", "net_pnl",
            "entry_time", "exit_time", "hold_time_seconds",
        }
        assert expected_fields.issubset(set(trade.keys()))


# ──────────────────────────────────────────────────────────────
# /api/trades/export
# ──────────────────────────────────────────────────────────────


class TestTradeExport:
    @pytest.mark.asyncio
    async def test_csv_export_no_manager(self):
        from bot.dashboard.app import export_trades, set_engine_manager

        set_engine_manager(None)
        result = await export_trades()
        assert result.body == b""

    @pytest.mark.asyncio
    async def test_csv_export_format(self):
        from bot.dashboard.app import export_trades, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await export_trades()
        content = result.body.decode("utf-8")
        lines = content.strip().split("\n")

        # Header + 3 data rows
        assert len(lines) == 4
        assert lines[0].startswith("timestamp,engine,symbol,side")

    @pytest.mark.asyncio
    async def test_csv_export_filter(self):
        from bot.dashboard.app import export_trades, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await export_trades(engine="grid_trading")
        content = result.body.decode("utf-8")
        lines = content.strip().split("\n")
        # Header + 1 data row
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_csv_content_type(self):
        from bot.dashboard.app import export_trades, set_engine_manager

        mgr = _mock_engine_manager()
        set_engine_manager(mgr)
        result = await export_trades()
        assert result.media_type == "text/csv"


# ──────────────────────────────────────────────────────────────
# _format_hold_time helper
# ──────────────────────────────────────────────────────────────


class TestFormatHoldTime:
    def test_zero(self):
        from bot.dashboard.app import _format_hold_time

        assert _format_hold_time(0) == "0m"

    def test_minutes_only(self):
        from bot.dashboard.app import _format_hold_time

        assert _format_hold_time(1800) == "30m"

    def test_hours_and_minutes(self):
        from bot.dashboard.app import _format_hold_time

        assert _format_hold_time(5400) == "1h 30m"

    def test_negative(self):
        from bot.dashboard.app import _format_hold_time

        assert _format_hold_time(-100) == "0m"
