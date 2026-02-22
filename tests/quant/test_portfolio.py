"""Tests for portfolio optimization."""

import numpy as np

from bot.quant.portfolio import (
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    risk_parity_portfolio,
)


def _make_returns(n_periods=200, n_assets=3, seed=42):
    np.random.seed(seed)
    # Correlated returns
    cov = np.eye(n_assets) * 0.04
    for i in range(n_assets):
        for j in range(n_assets):
            if i != j:
                cov[i, j] = 0.01
    returns = np.random.multivariate_normal(
        mean=[0.001] * n_assets, cov=cov, size=n_periods
    )
    return returns


class TestMinVariance:
    def test_basic(self):
        returns = _make_returns()
        result = min_variance_portfolio(returns)
        assert len(result["weights"]) == 3
        assert abs(sum(result["weights"]) - 1.0) < 1e-6
        assert all(w >= -0.01 for w in result["weights"])
        assert result["volatility"] >= 0

    def test_insufficient_data(self):
        result = min_variance_portfolio(np.random.randn(5, 3))
        assert result["volatility"] == 0.0

    def test_single_asset(self):
        result = min_variance_portfolio(np.random.randn(100, 1))
        assert result["volatility"] == 0.0


class TestMaxSharpe:
    def test_basic(self):
        returns = _make_returns()
        result = max_sharpe_portfolio(returns)
        assert len(result["weights"]) == 3
        assert abs(sum(result["weights"]) - 1.0) < 1e-6
        assert "sharpe" in result

    def test_with_risk_free(self):
        returns = _make_returns()
        result = max_sharpe_portfolio(returns, risk_free_rate=0.02)
        assert "sharpe" in result


class TestRiskParity:
    def test_basic(self):
        returns = _make_returns()
        result = risk_parity_portfolio(returns)
        assert len(result["weights"]) == 3
        assert abs(sum(result["weights"]) - 1.0) < 1e-6
        assert "risk_contributions" in result
        # Risk contributions should be approximately equal
        rc = result["risk_contributions"]
        assert max(rc) - min(rc) < 0.3  # Rough check

    def test_equal_vol_assets(self):
        np.random.seed(42)
        # Equal volatility, zero correlation -> should give ~equal weights
        returns = np.column_stack([
            np.random.randn(200) * 0.02,
            np.random.randn(200) * 0.02,
        ])
        result = risk_parity_portfolio(returns)
        # Weights should be approximately equal
        assert abs(result["weights"][0] - result["weights"][1]) < 0.2


class TestEfficientFrontier:
    def test_basic(self):
        returns = _make_returns()
        frontier = efficient_frontier(returns, n_points=10)
        assert len(frontier) > 0
        # Points should have increasing returns (approximately)
        for point in frontier:
            assert "weights" in point
            assert "expected_return" in point
            assert "volatility" in point

    def test_insufficient_data(self):
        frontier = efficient_frontier(np.random.randn(5, 3))
        assert frontier == []
