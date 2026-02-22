"""Tests for V4-005: Strategy management page with live toggle and stats."""

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from bot.dashboard.app import app, set_strategy_registry, update_state
from bot.monitoring.strategy_tracker import StrategyStats, StrategyTracker


@pytest.fixture(autouse=True)
def reset_state():
    """Reset bot state and registry before each test."""
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
    set_strategy_registry(None)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _make_strategy_stats() -> dict:
    """Create sample strategy stats dict (as returned by to_dict)."""
    return {
        "total_trades": 25,
        "wins": 15,
        "losses": 10,
        "total_pnl": 350.50,
        "win_rate": 60.0,
        "avg_pnl": 14.02,
        "consecutive_losses": 1,
        "sharpe_ratio": 0.85,
        "profit_factor": 1.75,
        "disabled": False,
        "disabled_reason": None,
        "pnl_history": [10.0, -5.0, 20.0, -3.0, 15.0],
    }


class TestStrategiesEndpoint:
    """Test GET /api/strategies with active status from registry."""

    @pytest.mark.asyncio
    async def test_strategies_empty(self, client):
        resp = await client.get("/api/strategies")
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategies"] == {}

    @pytest.mark.asyncio
    async def test_strategies_returns_stats(self, client):
        stats = _make_strategy_stats()
        update_state(strategy_stats={"ma_crossover": stats, "rsi": stats})
        resp = await client.get("/api/strategies")
        data = resp.json()
        assert "ma_crossover" in data["strategies"]
        assert "rsi" in data["strategies"]
        assert data["strategies"]["ma_crossover"]["total_trades"] == 25

    @pytest.mark.asyncio
    async def test_strategies_includes_pnl_history(self, client):
        stats = _make_strategy_stats()
        update_state(strategy_stats={"rsi": stats})
        resp = await client.get("/api/strategies")
        data = resp.json()
        assert "pnl_history" in data["strategies"]["rsi"]
        assert data["strategies"]["rsi"]["pnl_history"] == [10.0, -5.0, 20.0, -3.0, 15.0]

    @pytest.mark.asyncio
    async def test_strategies_includes_disabled_reason(self, client):
        stats = _make_strategy_stats()
        stats["disabled"] = True
        stats["disabled_reason"] = "consecutive_losses (5 >= 5)"
        update_state(strategy_stats={"rsi": stats})
        resp = await client.get("/api/strategies")
        data = resp.json()
        assert data["strategies"]["rsi"]["disabled"] is True
        assert "consecutive_losses" in data["strategies"]["rsi"]["disabled_reason"]

    @pytest.mark.asyncio
    async def test_strategies_merges_active_from_registry(self, client):
        """Active status is enriched from registry when available."""
        mock_registry = MagicMock()
        mock_registry.is_active.side_effect = lambda n: n == "ma_crossover"
        set_strategy_registry(mock_registry)

        stats = _make_strategy_stats()
        update_state(strategy_stats={"ma_crossover": stats, "rsi": stats})

        resp = await client.get("/api/strategies")
        data = resp.json()
        assert data["strategies"]["ma_crossover"]["active"] is True
        assert data["strategies"]["rsi"]["active"] is False

    @pytest.mark.asyncio
    async def test_strategies_without_registry_returns_raw(self, client):
        """Without registry, returns stats as-is."""
        stats = _make_strategy_stats()
        update_state(strategy_stats={"rsi": stats})
        resp = await client.get("/api/strategies")
        data = resp.json()
        assert data["strategies"]["rsi"]["total_pnl"] == 350.50


class TestStrategyToggleWithPositions:
    """Test POST /api/strategies/{name}/toggle with open position checks."""

    @pytest.mark.asyncio
    async def test_toggle_disable_warns_open_positions(self, client):
        """Disabling a strategy with open positions returns warning."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock()
        mock_registry.is_active.return_value = True
        set_strategy_registry(mock_registry)

        update_state(open_positions=[
            {"symbol": "BTC/USDT", "strategy": "ma_crossover", "quantity": 0.5},
            {"symbol": "ETH/USDT", "strategy": "ma_crossover", "quantity": 1.0},
        ])

        resp = await client.post("/api/strategies/ma_crossover/toggle")
        data = resp.json()
        assert data["success"] is False
        assert data["has_open_positions"] is True
        assert data["open_position_count"] == 2
        assert "warning" in data
        # Strategy should NOT have been disabled
        mock_registry.disable.assert_not_called()

    @pytest.mark.asyncio
    async def test_toggle_disable_force_with_open_positions(self, client):
        """Force disable bypasses open position check."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock()
        mock_registry.is_active.return_value = True
        set_strategy_registry(mock_registry)

        update_state(open_positions=[
            {"symbol": "BTC/USDT", "strategy": "ma_crossover", "quantity": 0.5},
        ])

        resp = await client.post("/api/strategies/ma_crossover/toggle?force=true")
        data = resp.json()
        assert data["success"] is True
        assert data["active"] is False
        mock_registry.disable.assert_called_once_with("ma_crossover")

    @pytest.mark.asyncio
    async def test_toggle_disable_no_open_positions(self, client):
        """Disabling without open positions works normally."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock()
        mock_registry.is_active.return_value = True
        set_strategy_registry(mock_registry)

        resp = await client.post("/api/strategies/ma_crossover/toggle")
        data = resp.json()
        assert data["success"] is True
        assert data["active"] is False

    @pytest.mark.asyncio
    async def test_toggle_disable_other_strategy_positions(self, client):
        """Positions from other strategies don't block toggle."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock()
        mock_registry.is_active.return_value = True
        set_strategy_registry(mock_registry)

        update_state(open_positions=[
            {"symbol": "BTC/USDT", "strategy": "rsi", "quantity": 0.5},
        ])

        resp = await client.post("/api/strategies/ma_crossover/toggle")
        data = resp.json()
        assert data["success"] is True
        assert data["active"] is False

    @pytest.mark.asyncio
    async def test_toggle_enable_no_position_check(self, client):
        """Enabling a strategy skips position check."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock()
        mock_registry.is_active.return_value = False
        set_strategy_registry(mock_registry)

        resp = await client.post("/api/strategies/rsi/toggle")
        data = resp.json()
        assert data["success"] is True
        assert data["active"] is True


class TestStrategyStatsEnhancements:
    """Test StrategyStats enhancements for V4-005."""

    def test_disabled_reason_stored_on_consecutive_losses(self):
        tracker = StrategyTracker(max_consecutive_losses=3)
        for _ in range(3):
            tracker.record_trade("rsi", -10.0)
        stats = tracker.get_stats("rsi")
        assert stats.disabled is True
        assert stats.disabled_reason is not None
        assert "consecutive_losses" in stats.disabled_reason

    def test_disabled_reason_stored_on_low_win_rate(self):
        tracker = StrategyTracker(
            min_win_rate_pct=50.0,
            min_trades_for_evaluation=5,
            max_consecutive_losses=100,
        )
        tracker.record_trade("rsi", 10.0)
        for _ in range(4):
            tracker.record_trade("rsi", -5.0)
        stats = tracker.get_stats("rsi")
        assert stats.disabled is True
        assert stats.disabled_reason is not None
        assert "low_win_rate" in stats.disabled_reason

    def test_disabled_reason_cleared_on_re_enable(self):
        from bot.strategies.regime import MarketRegime

        tracker = StrategyTracker(max_consecutive_losses=2)
        tracker.update_regime(MarketRegime.RANGING)
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -10.0)
        assert tracker.get_stats("rsi").disabled_reason is not None

        tracker.update_regime(MarketRegime.TRENDING_UP)
        stats = tracker.get_stats("rsi")
        assert stats.disabled is False
        assert stats.disabled_reason is None

    def test_to_dict_includes_pnl_history(self):
        stats = StrategyStats()
        stats.trade_pnls = [10.0, -5.0, 20.0, -3.0, 15.0]
        d = stats.to_dict()
        assert "pnl_history" in d
        assert d["pnl_history"] == [10.0, -5.0, 20.0, -3.0, 15.0]

    def test_to_dict_pnl_history_capped_at_50(self):
        stats = StrategyStats()
        stats.trade_pnls = list(range(100))
        d = stats.to_dict()
        assert len(d["pnl_history"]) == 50
        # Should be last 50
        assert d["pnl_history"][0] == 50.0
        assert d["pnl_history"][-1] == 99.0

    def test_to_dict_includes_disabled_reason(self):
        stats = StrategyStats()
        d = stats.to_dict()
        assert "disabled_reason" in d
        assert d["disabled_reason"] is None

    def test_to_dict_disabled_reason_with_value(self):
        stats = StrategyStats()
        stats.disabled = True
        stats.disabled_reason = "consecutive_losses (5 >= 5)"
        d = stats.to_dict()
        assert d["disabled_reason"] == "consecutive_losses (5 >= 5)"

    def test_initial_disabled_reason_is_none(self):
        stats = StrategyStats()
        assert stats.disabled_reason is None
