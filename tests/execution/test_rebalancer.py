"""Tests for portfolio rebalancer."""

import numpy as np

from bot.execution.rebalancer import PortfolioRebalancer


def _make_returns(n_periods=200, n_assets=3, seed=42):
    np.random.seed(seed)
    returns = {}
    for i in range(n_assets):
        returns[f"ASSET{i}/USDT"] = np.random.randn(n_periods) * 0.02
    return returns


class TestPortfolioRebalancer:
    def test_compute_target_weights(self):
        rebalancer = PortfolioRebalancer()
        returns = _make_returns()
        weights = rebalancer.compute_target_weights(returns)
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_weights_single_asset(self):
        rebalancer = PortfolioRebalancer()
        returns = {"A/USDT": np.random.randn(100)}
        weights = rebalancer.compute_target_weights(returns)
        assert weights["A/USDT"] == 1.0

    def test_check_rebalance_no_targets(self):
        rebalancer = PortfolioRebalancer()
        orders = rebalancer.check_rebalance_needed({}, 10000)
        assert orders == []

    def test_rebalance_on_drift(self):
        rebalancer = PortfolioRebalancer(
            rebalance_threshold_pct=5.0,
            min_rebalance_interval=0,
        )
        rebalancer._target_weights = {
            "A/USDT": 0.5,
            "B/USDT": 0.5,
        }
        # Current allocation drifted significantly
        orders = rebalancer.check_rebalance_needed(
            {"A/USDT": 7000, "B/USDT": 3000},
            portfolio_value=10000,
        )
        assert len(orders) > 0
        # Should have both buy and sell
        actions = [o["action"] for o in orders]
        assert "BUY" in actions or "SELL" in actions

    def test_no_rebalance_within_interval(self):
        rebalancer = PortfolioRebalancer(
            min_rebalance_interval=24,
        )
        rebalancer._target_weights = {"A/USDT": 0.5, "B/USDT": 0.5}
        rebalancer._hours_since_rebalance = 5  # Not enough time
        orders = rebalancer.check_rebalance_needed(
            {"A/USDT": 8000, "B/USDT": 2000}, 10000,
        )
        assert orders == []

    def test_max_sharpe_method(self):
        rebalancer = PortfolioRebalancer(method="max_sharpe")
        returns = _make_returns()
        weights = rebalancer.compute_target_weights(returns)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_reset_timer(self):
        rebalancer = PortfolioRebalancer()
        rebalancer._hours_since_rebalance = 100
        rebalancer.reset_timer()
        assert rebalancer._hours_since_rebalance == 0

    def test_target_weights_property(self):
        rebalancer = PortfolioRebalancer()
        rebalancer._target_weights = {"A/USDT": 0.6, "B/USDT": 0.4}
        tw = rebalancer.target_weights
        assert tw == {"A/USDT": 0.6, "B/USDT": 0.4}
        # Should be a copy
        tw["C/USDT"] = 0.0
        assert "C/USDT" not in rebalancer.target_weights

    def test_rebalance_order_fields(self):
        rebalancer = PortfolioRebalancer(
            rebalance_threshold_pct=5.0,
            min_rebalance_interval=0,
        )
        rebalancer._target_weights = {"A/USDT": 0.5, "B/USDT": 0.5}
        orders = rebalancer.check_rebalance_needed(
            {"A/USDT": 7000, "B/USDT": 3000},
            portfolio_value=10000,
        )
        for order in orders:
            assert "symbol" in order
            assert "action" in order
            assert "target_pct" in order
            assert "current_pct" in order
            assert "drift_pct" in order
            assert "target_value" in order
            assert "current_value" in order

    def test_rebalance_sell_overweight(self):
        rebalancer = PortfolioRebalancer(
            rebalance_threshold_pct=5.0,
            min_rebalance_interval=0,
        )
        rebalancer._target_weights = {"A/USDT": 0.5, "B/USDT": 0.5}
        orders = rebalancer.check_rebalance_needed(
            {"A/USDT": 7000, "B/USDT": 3000},
            portfolio_value=10000,
        )
        a_orders = [o for o in orders if o["symbol"] == "A/USDT"]
        b_orders = [o for o in orders if o["symbol"] == "B/USDT"]
        assert len(a_orders) == 1
        assert a_orders[0]["action"] == "SELL"
        assert len(b_orders) == 1
        assert b_orders[0]["action"] == "BUY"

    def test_no_rebalance_zero_portfolio(self):
        rebalancer = PortfolioRebalancer(min_rebalance_interval=0)
        rebalancer._target_weights = {"A/USDT": 0.5}
        rebalancer._hours_since_rebalance = 100
        orders = rebalancer.check_rebalance_needed({"A/USDT": 0}, 0)
        assert orders == []

    def test_insufficient_returns(self):
        rebalancer = PortfolioRebalancer()
        returns = {
            "A/USDT": np.random.randn(10),
            "B/USDT": np.random.randn(10),
        }
        weights = rebalancer.compute_target_weights(returns)
        # Should return equal weights for insufficient data
        assert abs(weights["A/USDT"] - 0.5) < 0.01
        assert abs(weights["B/USDT"] - 0.5) < 0.01

    def test_rebalance_resets_timer(self):
        rebalancer = PortfolioRebalancer(
            rebalance_threshold_pct=5.0,
            min_rebalance_interval=0,
        )
        rebalancer._target_weights = {"A/USDT": 0.5, "B/USDT": 0.5}
        rebalancer._hours_since_rebalance = 50
        orders = rebalancer.check_rebalance_needed(
            {"A/USDT": 7000, "B/USDT": 3000},
            portfolio_value=10000,
        )
        assert len(orders) > 0
        assert rebalancer._hours_since_rebalance == 0
