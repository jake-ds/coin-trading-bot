"""Tests for quant statistics utilities."""

import numpy as np

from bot.quant.statistics import (
    adf_test,
    calculate_half_life,
    calculate_zscore,
    engle_granger_cointegration,
    estimate_ou_params,
    rolling_ols_hedge_ratio,
)


class TestADFTest:
    def test_stationary_series(self):
        np.random.seed(42)
        # White noise is stationary
        series = np.random.randn(200)
        result = adf_test(series)
        assert result["is_stationary"] is True
        assert result["pvalue"] < 0.05
        assert "statistic" in result
        assert "critical_values" in result

    def test_random_walk_not_stationary(self):
        np.random.seed(42)
        # Random walk = non-stationary
        series = np.cumsum(np.random.randn(200))
        result = adf_test(series)
        assert result["is_stationary"] is False
        assert result["pvalue"] > 0.05

    def test_short_series(self):
        result = adf_test([1, 2, 3])
        assert result["is_stationary"] is False
        assert result["pvalue"] == 1.0


class TestCointegration:
    def test_cointegrated_pair(self):
        np.random.seed(42)
        n = 300
        # Create cointegrated pair: b = 2*a + noise
        a = np.cumsum(np.random.randn(n)) + 100
        b = 2 * a + np.random.randn(n) * 0.5
        result = engle_granger_cointegration(a, b)
        assert result["is_cointegrated"] is True
        assert result["pvalue"] < 0.05

    def test_non_cointegrated_pair(self):
        np.random.seed(42)
        n = 300
        # Two independent random walks
        a = np.cumsum(np.random.randn(n)) + 100
        b = np.cumsum(np.random.randn(n)) + 100
        result = engle_granger_cointegration(a, b)
        assert result["is_cointegrated"] is False

    def test_short_series(self):
        result = engle_granger_cointegration([1, 2, 3], [4, 5, 6])
        assert result["is_cointegrated"] is False


class TestZScore:
    def test_basic_zscore(self):
        np.random.seed(42)
        series = np.random.randn(100)
        zscores = calculate_zscore(series, window=20)
        assert len(zscores) == 100
        assert np.isnan(zscores[0])  # Not enough data
        assert not np.isnan(zscores[19])  # First valid point

    def test_zscore_normalization(self):
        # Constant series should give 0 z-score... but std=0 gives nan
        series = np.ones(50)
        zscores = calculate_zscore(series, window=20)
        # With constant input, std is 0, so z-score is NaN
        assert np.isnan(zscores[-1])

    def test_spike_detection(self):
        series = np.zeros(50)
        series[-1] = 10.0  # Spike
        zscores = calculate_zscore(series, window=20)
        # Last z-score should be very high
        assert zscores[-1] > 2.0


class TestHalfLife:
    def test_mean_reverting_series(self):
        np.random.seed(42)
        # OU process with known mean-reversion
        n = 500
        kappa = 0.1
        theta = 0.0
        sigma = 0.1
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = x[i - 1] + kappa * (theta - x[i - 1]) + sigma * np.random.randn()

        hl = calculate_half_life(x)
        assert 1.0 < hl < 50.0  # Should be finite

    def test_random_walk_large_half_life(self):
        np.random.seed(7)
        # Use a longer random walk with a seed that clearly shows no mean reversion
        series = np.cumsum(np.random.randn(500) * 0.01)
        hl = calculate_half_life(series)
        # Random walks should have very large or infinite half-life
        assert hl > 10 or hl == float("inf")

    def test_short_series(self):
        hl = calculate_half_life([1, 2, 3])
        assert hl == float("inf")


class TestOUParams:
    def test_ou_estimation(self):
        np.random.seed(42)
        n = 500
        kappa = 0.1
        theta = 5.0
        sigma = 0.2
        x = np.zeros(n)
        x[0] = 5.0
        for i in range(1, n):
            x[i] = x[i - 1] + kappa * (theta - x[i - 1]) + sigma * np.random.randn()

        params = estimate_ou_params(x)
        assert params["kappa"] > 0
        assert abs(params["theta"] - theta) < 2.0
        assert params["half_life"] < 100

    def test_short_series(self):
        params = estimate_ou_params([1, 2])
        assert params["half_life"] == float("inf")


class TestRollingHedgeRatio:
    def test_known_relationship(self):
        np.random.seed(42)
        n = 200
        b = np.cumsum(np.random.randn(n)) + 100
        a = 1.5 * b + np.random.randn(n)  # Known hedge ratio ~1.5

        ratios = rolling_ols_hedge_ratio(a, b, window=60)
        assert len(ratios) == n
        assert np.isnan(ratios[0])  # Not enough data
        # Last ratio should be close to 1.5
        assert abs(ratios[-1] - 1.5) < 0.2

    def test_different_lengths(self):
        a = np.random.randn(100)
        b = np.random.randn(80)
        ratios = rolling_ols_hedge_ratio(a, b, window=20)
        assert len(ratios) == 80
