"""Tests for V4-006: Equity curve and performance analytics page."""

from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from bot.dashboard.app import (
    _compute_analytics_stats,
    _compute_drawdown_series,
    _compute_monthly_returns,
    _compute_trade_markers,
    _filter_by_range,
    app,
    update_state,
)


@pytest.fixture(autouse=True)
def reset_state():
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
        strategy_stats={},
        equity_curve=[],
        open_positions=[],
        regime=None,
    )


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _make_equity_curve(n: int = 10, start_value: float = 10000.0) -> list[dict]:
    """Create sample equity curve data."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    points = []
    value = start_value
    for i in range(n):
        ts = (base + timedelta(hours=i)).isoformat()
        value += (i % 3 - 1) * 50  # fluctuating values
        points.append({"timestamp": ts, "total_value": round(value, 2)})
    return points


def _make_trades(n: int = 5) -> list[dict]:
    """Create sample trades."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trades = []
    for i in range(n):
        ts = (base + timedelta(hours=i)).isoformat()
        side = "BUY" if i % 2 == 0 else "SELL"
        pnl = (i + 1) * 10.0 * (1 if i % 3 != 0 else -1)
        trades.append({
            "timestamp": ts,
            "symbol": "BTC/USDT",
            "side": side,
            "quantity": 0.1,
            "price": 50000 + i * 100,
            "pnl": pnl,
            "strategy": "ma_crossover",
        })
    return trades


class TestAnalyticsEndpoint:
    """Test GET /api/analytics."""

    @pytest.mark.asyncio
    async def test_analytics_empty_state(self, client):
        resp = await client.get("/api/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["equity_curve"] == []
        assert data["drawdown"] == []
        assert data["trade_markers"] == []
        assert data["monthly_returns"] == []
        assert "stats" in data
        assert data["range"] == "all"

    @pytest.mark.asyncio
    async def test_analytics_with_equity_curve(self, client):
        curve = _make_equity_curve(5)
        update_state(equity_curve=curve)
        resp = await client.get("/api/analytics")
        data = resp.json()
        assert len(data["equity_curve"]) == 5
        assert len(data["drawdown"]) == 5
        assert data["equity_curve"][0]["total_value"] == curve[0]["total_value"]

    @pytest.mark.asyncio
    async def test_analytics_with_trades(self, client):
        trades = _make_trades(4)
        curve = _make_equity_curve(4)
        update_state(trades=trades, equity_curve=curve)
        resp = await client.get("/api/analytics")
        data = resp.json()
        assert len(data["trade_markers"]) > 0
        # Should have BUY and SELL markers
        sides = {m["side"] for m in data["trade_markers"]}
        assert "BUY" in sides
        assert "SELL" in sides

    @pytest.mark.asyncio
    async def test_analytics_stats_computed(self, client):
        trades = _make_trades(5)
        metrics = {
            "sharpe_ratio": 1.25,
            "max_drawdown_pct": 5.5,
            "win_rate": 65.0,
            "total_return_pct": 12.5,
            "total_trades": 5,
            "winning_trades": 3,
            "losing_trades": 2,
        }
        update_state(trades=trades, metrics=metrics)
        resp = await client.get("/api/analytics")
        data = resp.json()
        stats = data["stats"]
        assert stats["sharpe_ratio"] == 1.25
        assert stats["max_drawdown_pct"] == 5.5
        assert stats["win_rate"] == 65.0
        assert stats["total_trades"] == 5
        assert "sortino_ratio" in stats
        assert "profit_factor" in stats
        assert "avg_trade_pnl" in stats
        assert "best_trade" in stats
        assert "worst_trade" in stats

    @pytest.mark.asyncio
    async def test_analytics_range_param(self, client):
        resp = await client.get("/api/analytics?range=30d")
        assert resp.status_code == 200
        data = resp.json()
        assert data["range"] == "30d"

    @pytest.mark.asyncio
    async def test_analytics_range_filters_data(self, client):
        # Create data with old timestamps
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=60)
        curve = [
            {"timestamp": old.isoformat(), "total_value": 10000.0},
            {"timestamp": (old + timedelta(hours=1)).isoformat(), "total_value": 10050.0},
            {"timestamp": now.isoformat(), "total_value": 11000.0},
        ]
        update_state(equity_curve=curve)

        # "all" returns everything
        resp = await client.get("/api/analytics?range=all")
        assert len(resp.json()["equity_curve"]) == 3

        # "30d" filters out old data
        resp = await client.get("/api/analytics?range=30d")
        assert len(resp.json()["equity_curve"]) == 1


class TestDrawdownComputation:
    """Test drawdown series calculation."""

    def test_drawdown_empty(self):
        assert _compute_drawdown_series([]) == []

    def test_drawdown_increasing(self):
        curve = [
            {"timestamp": "2026-01-01T00:00:00", "total_value": 10000},
            {"timestamp": "2026-01-01T01:00:00", "total_value": 10500},
            {"timestamp": "2026-01-01T02:00:00", "total_value": 11000},
        ]
        dd = _compute_drawdown_series(curve)
        assert len(dd) == 3
        # No drawdown when always increasing
        assert all(d["drawdown_pct"] == 0.0 for d in dd)

    def test_drawdown_with_decline(self):
        curve = [
            {"timestamp": "2026-01-01T00:00:00", "total_value": 10000},
            {"timestamp": "2026-01-01T01:00:00", "total_value": 9000},
            {"timestamp": "2026-01-01T02:00:00", "total_value": 10000},
        ]
        dd = _compute_drawdown_series(curve)
        assert dd[0]["drawdown_pct"] == 0.0
        assert dd[1]["drawdown_pct"] == 10.0  # 10% drawdown
        assert dd[2]["drawdown_pct"] == 0.0  # recovered

    def test_drawdown_preserves_timestamps(self):
        curve = [{"timestamp": "2026-01-15T12:00:00", "total_value": 5000}]
        dd = _compute_drawdown_series(curve)
        assert dd[0]["timestamp"] == "2026-01-15T12:00:00"


class TestMonthlyReturns:
    """Test monthly returns calculation."""

    def test_monthly_returns_empty(self):
        assert _compute_monthly_returns([]) == []

    def test_monthly_returns_single_point(self):
        curve = [{"timestamp": "2026-01-01T00:00:00", "total_value": 10000}]
        assert _compute_monthly_returns(curve) == []

    def test_monthly_returns_same_month(self):
        curve = [
            {"timestamp": "2026-01-01T00:00:00", "total_value": 10000},
            {"timestamp": "2026-01-15T00:00:00", "total_value": 10500},
            {"timestamp": "2026-01-31T00:00:00", "total_value": 11000},
        ]
        result = _compute_monthly_returns(curve)
        assert len(result) == 1
        assert result[0]["month"] == "2026-01"
        assert result[0]["return_pct"] == 10.0  # 10000 -> 11000 = 10%

    def test_monthly_returns_multiple_months(self):
        curve = [
            {"timestamp": "2026-01-01T00:00:00", "total_value": 10000},
            {"timestamp": "2026-01-31T00:00:00", "total_value": 11000},
            {"timestamp": "2026-02-01T00:00:00", "total_value": 11000},
            {"timestamp": "2026-02-28T00:00:00", "total_value": 10450},
        ]
        result = _compute_monthly_returns(curve)
        assert len(result) == 2
        assert result[0]["month"] == "2026-01"
        assert result[0]["return_pct"] == 10.0
        assert result[1]["month"] == "2026-02"
        assert result[1]["return_pct"] == -5.0  # 11000 -> 10450 = -5%


class TestAnalyticsStats:
    """Test extended analytics stats computation."""

    def test_stats_empty_trades(self):
        stats = _compute_analytics_stats({}, [])
        assert stats["sharpe_ratio"] == 0.0
        assert stats["sortino_ratio"] == 0.0
        assert stats["profit_factor"] == 0.0
        assert stats["avg_trade_pnl"] == 0.0
        assert stats["best_trade"] == 0.0
        assert stats["worst_trade"] == 0.0

    def test_stats_with_trades(self):
        metrics = {
            "sharpe_ratio": 1.5,
            "max_drawdown_pct": 8.0,
            "win_rate": 60.0,
            "total_return_pct": 15.0,
            "total_trades": 5,
            "winning_trades": 3,
            "losing_trades": 2,
        }
        pnls = [100.0, -50.0, 75.0, -25.0, 200.0]
        stats = _compute_analytics_stats(metrics, pnls)

        assert stats["sharpe_ratio"] == 1.5
        assert stats["max_drawdown_pct"] == 8.0
        assert stats["best_trade"] == 200.0
        assert stats["worst_trade"] == -50.0
        assert stats["avg_trade_pnl"] == 60.0  # 300/5
        assert stats["total_trades"] == 5

    def test_stats_profit_factor(self):
        pnls = [100.0, -50.0, 200.0, -100.0]
        stats = _compute_analytics_stats({}, pnls)
        # gross profit = 300, gross loss = 150
        assert stats["profit_factor"] == 2.0

    def test_stats_sortino_ratio(self):
        pnls = [100.0, -50.0, 75.0, -25.0, 200.0]
        stats = _compute_analytics_stats({}, pnls)
        assert stats["sortino_ratio"] != 0.0
        # Sortino should be positive when mean return is positive
        assert stats["sortino_ratio"] > 0

    def test_stats_all_positive_trades(self):
        pnls = [10.0, 20.0, 30.0]
        stats = _compute_analytics_stats({}, pnls)
        # No downside deviation → sortino = 0
        assert stats["sortino_ratio"] == 0.0
        assert stats["profit_factor"] == 0.0  # no losses → 0


class TestTradeMarkers:
    """Test trade marker computation."""

    def test_markers_empty(self):
        assert _compute_trade_markers([], []) == []

    def test_markers_no_curve(self):
        trades = [{"timestamp": "2026-01-01", "side": "BUY"}]
        assert _compute_trade_markers(trades, []) == []

    def test_markers_mapped_correctly(self):
        curve = [
            {"timestamp": "2026-01-01T00:00:00", "total_value": 10000},
            {"timestamp": "2026-01-01T01:00:00", "total_value": 10100},
        ]
        trades = [
            {
                "timestamp": "2026-01-01T00:00:00",
                "side": "BUY",
                "symbol": "BTC/USDT",
                "price": 50000,
            },
        ]
        markers = _compute_trade_markers(trades, curve)
        assert len(markers) == 1
        assert markers[0]["side"] == "BUY"
        assert markers[0]["value"] == 10000
        assert markers[0]["symbol"] == "BTC/USDT"


class TestFilterByRange:
    """Test date range filtering."""

    def test_filter_all(self):
        data = [{"timestamp": "2020-01-01T00:00:00"}]
        assert _filter_by_range(data, "all") == data

    def test_filter_empty(self):
        assert _filter_by_range([], "7d") == []

    def test_filter_invalid_range(self):
        data = [{"timestamp": "2020-01-01T00:00:00"}]
        assert _filter_by_range(data, "invalid") == data

    def test_filter_7d(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=10)
        recent = now - timedelta(hours=1)
        data = [
            {"timestamp": old.isoformat()},
            {"timestamp": recent.isoformat()},
        ]
        result = _filter_by_range(data, "7d")
        assert len(result) == 1
