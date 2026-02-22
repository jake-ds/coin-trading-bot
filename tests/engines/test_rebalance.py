"""Tests for V5-009: Capital rebalance + tuner/rebalance integration."""

import pytest

from bot.engines.portfolio_manager import PortfolioManager
from bot.engines.tracker import EngineMetrics

# ------------------------------------------------------------------ #
# PortfolioManager.rebalance_allocations
# ------------------------------------------------------------------ #


class TestRebalanceAllocations:
    def test_equal_sharpe_equal_allocation(self):
        pm = PortfolioManager(
            total_capital=10000,
            engine_allocations={
                "a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25,
            },
        )
        metrics = {
            "a": EngineMetrics(sharpe_ratio=1.0),
            "b": EngineMetrics(sharpe_ratio=1.0),
            "c": EngineMetrics(sharpe_ratio=1.0),
            "d": EngineMetrics(sharpe_ratio=1.0),
        }
        result = pm.rebalance_allocations(metrics)
        assert len(result) == 4
        for v in result.values():
            assert v == pytest.approx(0.25, abs=0.01)

    def test_higher_sharpe_gets_more(self):
        pm = PortfolioManager(
            total_capital=10000,
            engine_allocations={"a": 0.50, "b": 0.50},
        )
        metrics = {
            "a": EngineMetrics(sharpe_ratio=3.0),
            "b": EngineMetrics(sharpe_ratio=0.5),
        }
        result = pm.rebalance_allocations(metrics)
        assert result["a"] > result["b"]

    def test_negative_sharpe_gets_min(self):
        pm = PortfolioManager(
            total_capital=10000,
            engine_allocations={"a": 0.50, "b": 0.50},
        )
        metrics = {
            "a": EngineMetrics(sharpe_ratio=-2.0),
            "b": EngineMetrics(sharpe_ratio=2.0),
        }
        result = pm.rebalance_allocations(metrics)
        # Negative sharpe → weight = 0.1 (floor)
        # So a's weight is much less than b's
        assert result["a"] < result["b"]

    def test_clamps_min_10_pct(self):
        pm = PortfolioManager(
            total_capital=10000,
            engine_allocations={"a": 0.50, "b": 0.50},
        )
        metrics = {
            "a": EngineMetrics(sharpe_ratio=-5.0),  # Very bad → min weight
            "b": EngineMetrics(sharpe_ratio=10.0),  # Very good
        }
        result = pm.rebalance_allocations(metrics)
        assert result["a"] >= 0.10 - 0.001

    def test_clamps_max_40_pct(self):
        pm = PortfolioManager(
            total_capital=10000,
            engine_allocations={"a": 0.34, "b": 0.33, "c": 0.33},
        )
        metrics = {
            "a": EngineMetrics(sharpe_ratio=100.0),
            "b": EngineMetrics(sharpe_ratio=-5.0),
            "c": EngineMetrics(sharpe_ratio=-5.0),
        }
        result = pm.rebalance_allocations(metrics)
        assert result["a"] <= 0.40 + 0.001

    def test_allocations_sum_to_one(self):
        pm = PortfolioManager(
            total_capital=10000,
            engine_allocations={
                "a": 0.30, "b": 0.25, "c": 0.20, "d": 0.25,
            },
        )
        metrics = {
            "a": EngineMetrics(sharpe_ratio=2.0),
            "b": EngineMetrics(sharpe_ratio=0.3),
            "c": EngineMetrics(sharpe_ratio=-1.0),
            "d": EngineMetrics(sharpe_ratio=1.5),
        }
        result = pm.rebalance_allocations(metrics)
        assert sum(result.values()) == pytest.approx(1.0, abs=0.001)

    def test_empty_allocations(self):
        pm = PortfolioManager(total_capital=10000)
        result = pm.rebalance_allocations({})
        assert result == {}

    def test_missing_metrics_uses_floor(self):
        pm = PortfolioManager(
            total_capital=10000,
            engine_allocations={"a": 0.50, "b": 0.50},
        )
        # Only provide metrics for 'a'
        metrics = {"a": EngineMetrics(sharpe_ratio=2.0)}
        result = pm.rebalance_allocations(metrics)
        assert "b" in result
        assert result["a"] > result["b"]  # 'a' has better metrics


# ------------------------------------------------------------------ #
# EngineManager tuner/rebalance integration
# ------------------------------------------------------------------ #


class TestManagerTunerIntegration:
    def test_manager_has_tuner(self):
        from bot.engines.manager import EngineManager

        pm = PortfolioManager(total_capital=10000)
        mgr = EngineManager(portfolio_manager=pm)
        assert mgr.tuner is not None

    def test_manager_has_rebalance_history(self):
        from bot.engines.manager import EngineManager

        pm = PortfolioManager(total_capital=10000)
        mgr = EngineManager(portfolio_manager=pm)
        assert mgr._rebalance_history == []

    def test_get_engine_params(self):
        from bot.config import load_settings
        from bot.engines.manager import EngineManager

        pm = PortfolioManager(total_capital=10000)
        settings = load_settings()
        mgr = EngineManager(portfolio_manager=pm, settings=settings)

        params = mgr._get_engine_params("funding_rate_arb")
        assert "funding_arb_min_rate" in params

    def test_get_engine_params_unknown(self):
        from bot.engines.manager import EngineManager

        pm = PortfolioManager(total_capital=10000)
        mgr = EngineManager(portfolio_manager=pm)
        params = mgr._get_engine_params("nonexistent")
        assert params == {}


# ------------------------------------------------------------------ #
# Config fields
# ------------------------------------------------------------------ #


class TestConfigFields:
    def test_tuner_enabled_default(self):
        from bot.config import load_settings

        s = load_settings()
        assert s.tuner_enabled is True
        assert s.tuner_interval_hours == 24

    def test_rebalance_enabled_default(self):
        from bot.config import load_settings

        s = load_settings()
        assert s.engine_rebalance_enabled is True
        assert s.engine_rebalance_interval_hours == 24

    def test_settings_metadata_exists(self):
        from bot.config import SETTINGS_METADATA

        assert "tuner_enabled" in SETTINGS_METADATA
        assert "tuner_interval_hours" in SETTINGS_METADATA
        assert "engine_rebalance_enabled" in SETTINGS_METADATA
        assert "engine_rebalance_interval_hours" in SETTINGS_METADATA
