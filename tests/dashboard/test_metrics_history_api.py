"""Tests for Metrics History API — /api/metrics/history, /compare, /daily-summary."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.engines.tracker import EngineTracker, TradeRecord

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _sample_trade(
    engine: str = "funding_rate_arb",
    symbol: str = "BTC/USDT",
    pnl: float = 50.0,
    exit_time: str = "2026-02-23T10:00:00Z",
) -> TradeRecord:
    return TradeRecord(
        engine_name=engine,
        symbol=symbol,
        side="long",
        entry_price=40000,
        exit_price=40500,
        quantity=0.1,
        pnl=pnl + 2.0,
        cost=2.0,
        net_pnl=pnl,
        entry_time="2026-02-23T08:00:00Z",
        exit_time=exit_time,
        hold_time_seconds=7200,
    )


def _mock_engine_manager_with_persistence(snapshots=None, trades=None):
    """Build a mock EngineManager with MetricsPersistence."""
    mgr = MagicMock()
    tracker = EngineTracker()
    mgr.tracker = tracker

    persistence = MagicMock()
    store = MagicMock()

    # Mock get_engine_metric_snapshots
    if snapshots is not None:
        store.get_engine_metric_snapshots = AsyncMock(
            return_value=snapshots
        )
    else:
        store.get_engine_metric_snapshots = AsyncMock(return_value=[])

    # Mock get_engine_trades
    if trades is not None:
        store.get_engine_trades = AsyncMock(return_value=trades)
    else:
        store.get_engine_trades = AsyncMock(return_value=[])

    persistence._data_store = store
    mgr._metrics_persistence = persistence
    return mgr


def _sample_snapshots(
    engine: str = "funding_rate_arb", count: int = 3
) -> list[dict]:
    """Generate sample metric snapshot dicts."""
    result = []
    for i in range(count):
        result.append({
            "engine_name": engine,
            "timestamp": f"2026-02-{20 + i:02d}T12:00:00",
            "total_trades": 10 + i,
            "winning_trades": 6 + i,
            "losing_trades": 4,
            "win_rate": round((6 + i) / (10 + i), 4),
            "total_pnl": 100.0 + i * 50,
            "sharpe_ratio": 1.2 + i * 0.1,
            "max_drawdown": 0.05 + i * 0.01,
            "profit_factor": 1.5 + i * 0.1,
            "cost_ratio": 0.02,
        })
    return result


def _sample_db_trades(count: int = 3) -> list[dict]:
    """Generate sample engine trade dicts (as returned from DataStore)."""
    result = []
    for i in range(count):
        day = 20 + i
        result.append({
            "engine_name": "funding_rate_arb",
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_price": 40000,
            "exit_price": 40500,
            "quantity": 0.1,
            "pnl": 52.0,
            "cost": 2.0,
            "net_pnl": 50.0 if i % 2 == 0 else -10.0,
            "entry_time": f"2026-02-{day:02d}T08:00:00Z",
            "exit_time": f"2026-02-{day:02d}T10:00:00Z",
            "hold_time_seconds": 7200,
        })
    return result


# ──────────────────────────────────────────────────────────────
# /api/metrics/history
# ──────────────────────────────────────────────────────────────


class TestMetricsHistory:
    @pytest.mark.asyncio
    async def test_no_engine_manager(self):
        from bot.dashboard.app import get_metrics_history, set_engine_manager

        set_engine_manager(None)
        result = await get_metrics_history(engine="test", days=30)
        assert result["timestamps"] == []
        assert result["sharpe"] == []

    @pytest.mark.asyncio
    async def test_no_persistence(self):
        from bot.dashboard.app import get_metrics_history, set_engine_manager

        mgr = MagicMock(spec=[])
        set_engine_manager(mgr)
        result = await get_metrics_history(engine="test", days=30)
        assert result["timestamps"] == []

    @pytest.mark.asyncio
    async def test_with_snapshots(self):
        from bot.dashboard.app import get_metrics_history, set_engine_manager

        snapshots = _sample_snapshots("engine_a", 3)
        mgr = _mock_engine_manager_with_persistence(snapshots=snapshots)
        set_engine_manager(mgr)

        result = await get_metrics_history(engine="engine_a", days=30)
        assert len(result["timestamps"]) == 3
        assert len(result["sharpe"]) == 3
        assert len(result["win_rate"]) == 3
        assert len(result["total_pnl"]) == 3
        assert len(result["max_drawdown"]) == 3

    @pytest.mark.asyncio
    async def test_response_values(self):
        from bot.dashboard.app import get_metrics_history, set_engine_manager

        snapshots = _sample_snapshots("engine_a", 2)
        mgr = _mock_engine_manager_with_persistence(snapshots=snapshots)
        set_engine_manager(mgr)

        result = await get_metrics_history(engine="engine_a", days=7)
        assert result["sharpe"][0] == 1.2
        assert result["total_pnl"][1] == 150.0

    @pytest.mark.asyncio
    async def test_empty_snapshots(self):
        from bot.dashboard.app import get_metrics_history, set_engine_manager

        mgr = _mock_engine_manager_with_persistence(snapshots=[])
        set_engine_manager(mgr)

        result = await get_metrics_history(engine="engine_a", days=30)
        assert result["timestamps"] == []


# ──────────────────────────────────────────────────────────────
# /api/metrics/compare
# ──────────────────────────────────────────────────────────────


class TestMetricsCompare:
    @pytest.mark.asyncio
    async def test_no_engine_manager(self):
        from bot.dashboard.app import get_metrics_compare, set_engine_manager

        set_engine_manager(None)
        result = await get_metrics_compare(
            engines="a,b", metric="sharpe", days=30
        )
        assert result["engines"] == {}

    @pytest.mark.asyncio
    async def test_no_persistence(self):
        from bot.dashboard.app import get_metrics_compare, set_engine_manager

        mgr = MagicMock(spec=[])
        set_engine_manager(mgr)
        result = await get_metrics_compare(
            engines="a,b", metric="sharpe", days=30
        )
        assert result["engines"] == {}

    @pytest.mark.asyncio
    async def test_compare_multiple_engines(self):
        from bot.dashboard.app import get_metrics_compare, set_engine_manager

        snapshots_a = _sample_snapshots("engine_a", 2)
        snapshots_b = _sample_snapshots("engine_b", 2)

        mgr = _mock_engine_manager_with_persistence()
        store = mgr._metrics_persistence._data_store

        async def mock_get_snapshots(engine_name=None, **kwargs):
            if engine_name == "engine_a":
                return snapshots_a
            elif engine_name == "engine_b":
                return snapshots_b
            return []

        store.get_engine_metric_snapshots = mock_get_snapshots
        set_engine_manager(mgr)

        result = await get_metrics_compare(
            engines="engine_a,engine_b", metric="sharpe", days=30
        )
        assert "engine_a" in result["engines"]
        assert "engine_b" in result["engines"]
        assert len(result["engines"]["engine_a"]["timestamps"]) == 2
        assert len(result["engines"]["engine_b"]["values"]) == 2

    @pytest.mark.asyncio
    async def test_compare_metric_field_mapping(self):
        from bot.dashboard.app import get_metrics_compare, set_engine_manager

        snapshots = _sample_snapshots("engine_a", 1)
        mgr = _mock_engine_manager_with_persistence(snapshots=snapshots)
        set_engine_manager(mgr)

        result = await get_metrics_compare(
            engines="engine_a", metric="total_pnl", days=30
        )
        # total_pnl maps to total_pnl field
        vals = result["engines"]["engine_a"]["values"]
        assert len(vals) == 1
        assert vals[0] == 100.0

    @pytest.mark.asyncio
    async def test_compare_invalid_metric_defaults(self):
        from bot.dashboard.app import get_metrics_compare, set_engine_manager

        snapshots = _sample_snapshots("engine_a", 1)
        mgr = _mock_engine_manager_with_persistence(snapshots=snapshots)
        set_engine_manager(mgr)

        result = await get_metrics_compare(
            engines="engine_a", metric="nonexistent", days=30
        )
        # Falls back to sharpe_ratio
        vals = result["engines"]["engine_a"]["values"]
        assert len(vals) == 1
        assert vals[0] == 1.2


# ──────────────────────────────────────────────────────────────
# /api/metrics/daily-summary
# ──────────────────────────────────────────────────────────────


class TestDailySummary:
    @pytest.mark.asyncio
    async def test_no_engine_manager(self):
        from bot.dashboard.app import get_daily_summary, set_engine_manager

        set_engine_manager(None)
        result = await get_daily_summary(days=30)
        assert result["daily"] == []

    @pytest.mark.asyncio
    async def test_with_db_trades(self):
        from bot.dashboard.app import get_daily_summary, set_engine_manager

        trades = _sample_db_trades(3)
        mgr = _mock_engine_manager_with_persistence(trades=trades)
        set_engine_manager(mgr)

        result = await get_daily_summary(days=30)
        daily = result["daily"]
        assert len(daily) == 3
        # Each entry has the expected fields
        for entry in daily:
            assert "date" in entry
            assert "total_pnl" in entry
            assert "total_trades" in entry
            assert "winning_trades" in entry
            assert "total_cost" in entry
            assert "avg_win_rate" in entry

    @pytest.mark.asyncio
    async def test_daily_aggregation(self):
        from bot.dashboard.app import get_daily_summary, set_engine_manager

        # Two trades on the same day
        trades = [
            {
                "engine_name": "engine_a",
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 40000,
                "exit_price": 40500,
                "quantity": 0.1,
                "pnl": 52.0,
                "cost": 2.0,
                "net_pnl": 50.0,
                "entry_time": "2026-02-20T08:00:00Z",
                "exit_time": "2026-02-20T10:00:00Z",
                "hold_time_seconds": 7200,
            },
            {
                "engine_name": "engine_a",
                "symbol": "ETH/USDT",
                "side": "long",
                "entry_price": 2500,
                "exit_price": 2400,
                "quantity": 1.0,
                "pnl": -98.0,
                "cost": 2.0,
                "net_pnl": -100.0,
                "entry_time": "2026-02-20T11:00:00Z",
                "exit_time": "2026-02-20T13:00:00Z",
                "hold_time_seconds": 7200,
            },
        ]
        mgr = _mock_engine_manager_with_persistence(trades=trades)
        set_engine_manager(mgr)

        result = await get_daily_summary(days=30)
        daily = result["daily"]
        assert len(daily) == 1  # Same day aggregated
        assert daily[0]["date"] == "2026-02-20"
        assert daily[0]["total_trades"] == 2
        assert daily[0]["winning_trades"] == 1
        assert daily[0]["total_pnl"] == -50.0
        assert daily[0]["total_cost"] == 4.0
        assert daily[0]["avg_win_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_daily_sorted_chronologically(self):
        from bot.dashboard.app import get_daily_summary, set_engine_manager

        trades = [
            {
                "exit_time": "2026-02-22T10:00:00Z",
                "net_pnl": 20.0,
                "cost": 1.0,
            },
            {
                "exit_time": "2026-02-20T10:00:00Z",
                "net_pnl": 10.0,
                "cost": 1.0,
            },
        ]
        mgr = _mock_engine_manager_with_persistence(trades=trades)
        set_engine_manager(mgr)

        result = await get_daily_summary(days=30)
        daily = result["daily"]
        assert daily[0]["date"] < daily[1]["date"]

    @pytest.mark.asyncio
    async def test_empty_trades(self):
        from bot.dashboard.app import get_daily_summary, set_engine_manager

        mgr = _mock_engine_manager_with_persistence(trades=[])
        # Need tracker with no trades for fallback
        mgr.tracker = EngineTracker()
        set_engine_manager(mgr)

        result = await get_daily_summary(days=30)
        assert result["daily"] == []


# ──────────────────────────────────────────────────────────────
# Fallback to in-memory tracker
# ──────────────────────────────────────────────────────────────


class TestDailySummaryFallback:
    @pytest.mark.asyncio
    async def test_fallback_no_persistence(self):
        from bot.dashboard.app import get_daily_summary, set_engine_manager

        mgr = MagicMock()
        tracker = EngineTracker()
        tracker.record_trade(
            "engine_a",
            _sample_trade("engine_a", pnl=30.0, exit_time="2026-02-20T10:00:00Z"),
        )
        mgr.tracker = tracker
        # No _metrics_persistence
        mgr._metrics_persistence = None
        set_engine_manager(mgr)

        result = await get_daily_summary(days=30)
        daily = result["daily"]
        assert len(daily) == 1
        assert daily[0]["date"] == "2026-02-20"
        assert daily[0]["total_pnl"] == 30.0


# ──────────────────────────────────────────────────────────────
# _aggregate_daily helper
# ──────────────────────────────────────────────────────────────


class TestAggregateDaily:
    def test_empty(self):
        from bot.dashboard.app import _aggregate_daily

        result = _aggregate_daily([])
        assert result == []

    def test_multiple_days(self):
        from bot.dashboard.app import _aggregate_daily

        trades = [
            {"exit_time": "2026-02-20T10:00:00Z", "net_pnl": 50.0, "cost": 2.0},
            {"exit_time": "2026-02-20T12:00:00Z", "net_pnl": -10.0, "cost": 1.0},
            {"exit_time": "2026-02-21T09:00:00Z", "net_pnl": 30.0, "cost": 1.5},
        ]
        result = _aggregate_daily(trades)
        assert len(result) == 2
        assert result[0]["date"] == "2026-02-20"
        assert result[0]["total_trades"] == 2
        assert result[0]["total_pnl"] == 40.0
        assert result[0]["winning_trades"] == 1
        assert result[1]["date"] == "2026-02-21"
        assert result[1]["total_trades"] == 1

    def test_no_exit_time(self):
        from bot.dashboard.app import _aggregate_daily

        trades = [{"exit_time": "", "net_pnl": 50.0, "cost": 1.0}]
        result = _aggregate_daily(trades)
        assert result == []


# ──────────────────────────────────────────────────────────────
# DataStore queries
# ──────────────────────────────────────────────────────────────


class TestDataStoreQueries:
    @pytest.mark.asyncio
    async def test_get_engine_metric_snapshots_import(self):
        from bot.data.store import DataStore

        assert hasattr(DataStore, "get_engine_metric_snapshots")

    @pytest.mark.asyncio
    async def test_get_engine_trades_import(self):
        from bot.data.store import DataStore

        assert hasattr(DataStore, "get_engine_trades")
