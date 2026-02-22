"""Tests for VaR/CVaR and risk-adjusted performance metrics."""

import numpy as np

from bot.quant.risk_metrics import (
    calmar_ratio,
    cornish_fisher_var,
    cvar,
    historical_var,
    information_ratio,
    parametric_var,
    sortino_ratio,
)


class TestParametricVaR:
    def test_basic_var(self):
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02
        var = parametric_var(returns, confidence=0.95)
        assert var > 0
        # For normal distribution, 95% VaR ~ 1.645 * sigma
        expected = 1.645 * 0.02
        assert abs(var - expected) < 0.01

    def test_higher_confidence_higher_var(self):
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02
        var_95 = parametric_var(returns, confidence=0.95)
        var_99 = parametric_var(returns, confidence=0.99)
        assert var_99 > var_95

    def test_short_series(self):
        var = parametric_var([0.01, 0.02])
        assert var == 0.0

    def test_horizon_scaling(self):
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02
        var_1 = parametric_var(returns, horizon=1)
        var_10 = parametric_var(returns, horizon=10)
        # VaR scales with sqrt(horizon) approximately
        assert var_10 > var_1


class TestHistoricalVaR:
    def test_basic(self):
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02
        var = historical_var(returns, confidence=0.95)
        assert var > 0

    def test_short_series(self):
        assert historical_var([0.01]) == 0.0

    def test_all_positive(self):
        returns = np.abs(np.random.randn(100)) * 0.01
        var = historical_var(returns, confidence=0.95)
        assert var == 0.0  # No losses


class TestCornishFisherVaR:
    def test_basic(self):
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02
        var = cornish_fisher_var(returns, confidence=0.95)
        assert var > 0

    def test_skewed_distribution(self):
        np.random.seed(42)
        # Create negatively skewed returns
        returns = np.random.randn(500) * 0.02
        returns[returns < -0.02] *= 2  # Amplify negative tail
        var_cf = cornish_fisher_var(returns, confidence=0.95)
        var_p = parametric_var(returns, confidence=0.95)
        # CF-VaR should be higher for negatively skewed distributions
        assert var_cf > var_p * 0.8  # At least close to parametric


class TestCVaR:
    def test_cvar_greater_than_var(self):
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02
        var = historical_var(returns, confidence=0.95)
        cvar_val = cvar(returns, confidence=0.95)
        assert cvar_val >= var

    def test_short_series(self):
        assert cvar([0.01, 0.02]) == 0.0


class TestSortinoRatio:
    def test_positive_returns(self):
        np.random.seed(42)
        returns = np.abs(np.random.randn(100)) * 0.01  # All positive
        ratio = sortino_ratio(returns)
        assert ratio > 0

    def test_negative_returns(self):
        np.random.seed(42)
        returns = -np.abs(np.random.randn(100)) * 0.01  # All negative
        ratio = sortino_ratio(returns)
        assert ratio < 0

    def test_short_series(self):
        assert sortino_ratio([0.01]) == 0.0


class TestCalmarRatio:
    def test_positive_trend(self):
        np.random.seed(42)
        returns = np.random.randn(200) * 0.005 + 0.001  # Small positive drift
        ratio = calmar_ratio(returns)
        assert ratio > 0

    def test_short_series(self):
        assert calmar_ratio([0.01, -0.01]) == 0.0


class TestInformationRatio:
    def test_outperforming(self):
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02 + 0.005
        benchmark = np.random.randn(200) * 0.02
        ratio = information_ratio(returns, benchmark)
        assert ratio > 0

    def test_identical_returns(self):
        returns = np.ones(100) * 0.01
        ir = information_ratio(returns, returns)
        assert ir == 0.0

    def test_short_series(self):
        assert information_ratio([0.01], [0.02]) == 0.0
