"""Tests for GARCH volatility modeling."""

import numpy as np

from bot.quant.volatility import (
    GARCHModel,
    VolatilityRegime,
    classify_volatility_regime,
)


class TestGARCHModel:
    def test_fit_basic(self):
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02
        model = GARCHModel()
        result = model.fit(returns)
        assert result["success"] is True
        assert result["alpha"] >= 0
        assert result["beta"] >= 0
        assert result["persistence"] <= 1.1  # alpha + beta

    def test_fit_insufficient_data(self):
        model = GARCHModel()
        result = model.fit(np.random.randn(5))
        assert result["success"] is False

    def test_forecast(self):
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02
        model = GARCHModel()
        model.fit(returns)

        forecasts = model.forecast(horizon=5)
        assert len(forecasts) == 5
        assert all(f > 0 for f in forecasts)

    def test_forecast_unfitted(self):
        model = GARCHModel()
        forecasts = model.forecast(horizon=3)
        assert all(np.isnan(f) for f in forecasts)

    def test_dynamic_stop_loss(self):
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02
        model = GARCHModel()
        model.fit(returns)

        stop = model.dynamic_stop_loss(100.0, multiplier=2.0)
        assert stop < 100.0
        assert stop > 0.0

    def test_dynamic_stop_loss_unfitted(self):
        model = GARCHModel()
        stop = model.dynamic_stop_loss(100.0)
        assert stop == 97.0  # Default 3% fallback

    def test_conditional_volatility(self):
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02
        model = GARCHModel()
        model.fit(returns)

        cv = model.conditional_volatility
        assert cv is not None
        assert len(cv) > 0
        assert all(v >= 0 for v in cv)

    def test_is_fitted_property(self):
        model = GARCHModel()
        assert model.is_fitted is False
        np.random.seed(42)
        model.fit(np.random.randn(200) * 0.02)
        assert model.is_fitted is True


class TestVolatilityRegime:
    def test_normal_regime(self):
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        regime = classify_volatility_regime(returns)
        assert regime == VolatilityRegime.NORMAL

    def test_high_regime(self):
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01
        # Make recent returns much more volatile
        returns[-30:] = np.random.randn(30) * 0.05
        regime = classify_volatility_regime(returns, window=30)
        assert regime == VolatilityRegime.HIGH

    def test_low_regime(self):
        np.random.seed(42)
        returns = np.random.randn(100) * 0.05
        # Make recent returns very calm
        returns[-30:] = np.random.randn(30) * 0.005
        regime = classify_volatility_regime(returns, window=30)
        assert regime == VolatilityRegime.LOW

    def test_insufficient_data(self):
        returns = np.random.randn(10)
        regime = classify_volatility_regime(returns)
        assert regime == VolatilityRegime.NORMAL
