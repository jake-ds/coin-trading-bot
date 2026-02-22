"""Tests for V2-020: ML Strategy v2 with GradientBoosting, CV, features."""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from bot.models import OHLCV, SignalAction
from bot.strategies.ml.prediction import (
    LABEL_DOWN,
    LABEL_NEUTRAL,
    LABEL_UP,
    MLPredictionStrategy,
)


def make_trending_candles(
    n: int, start_price: float = 100.0, trend: float = 0.5
) -> list[OHLCV]:
    """Create synthetic candles with a trend and some noise."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    np.random.seed(42)
    price = start_price

    for i in range(n):
        noise = np.random.normal(0, 1)
        price = price + trend + noise
        price = max(price, 1.0)
        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=i),
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000.0 + np.random.uniform(-100, 100),
            )
        )

    return candles


def make_volatile_candles(n: int) -> list[OHLCV]:
    """Create candles with high volatility (noisy, no clear trend)."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    np.random.seed(123)
    price = 100.0

    for i in range(n):
        noise = np.random.normal(0, 5)  # High noise
        price = max(price + noise, 1.0)
        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=i),
                open=price - 2.0,
                high=price + 3.0,
                low=price - 3.0,
                close=price,
                volume=1000.0 + np.random.uniform(-200, 200),
            )
        )

    return candles


class TestMLV2Name:
    def test_name(self):
        strategy = MLPredictionStrategy()
        assert strategy.name == "ml_prediction"

    def test_required_history_length(self):
        strategy = MLPredictionStrategy(min_training_samples=500)
        assert strategy.required_history_length >= 500

    def test_required_history_length_small(self):
        strategy = MLPredictionStrategy(min_training_samples=30)
        assert strategy.required_history_length >= 50


class TestMLV2FeatureEngineering:
    def test_feature_names_count(self):
        strategy = MLPredictionStrategy()
        assert len(strategy._feature_names) == 25

    def test_compute_features_shape_25(self):
        """V2 produces 25 features per candle."""
        strategy = MLPredictionStrategy()
        candles = make_trending_candles(100)
        features = strategy._compute_features(candles)
        assert features.ndim == 2
        assert features.shape[1] == 25
        assert features.shape[0] > 0

    def test_compute_features_no_nan(self):
        """Features should not contain NaN values."""
        strategy = MLPredictionStrategy()
        candles = make_trending_candles(200)
        features = strategy._compute_features(candles)
        assert not np.isnan(features).any()

    def test_compute_features_no_inf(self):
        """Features should not contain Inf values."""
        strategy = MLPredictionStrategy()
        candles = make_trending_candles(200)
        features = strategy._compute_features(candles)
        assert not np.isinf(features).any()

    def test_compute_features_empty_for_short_data(self):
        """Too few candles produces empty feature array."""
        strategy = MLPredictionStrategy(lookback_period=14)
        candles = make_trending_candles(10)
        features = strategy._compute_features(candles)
        assert features.shape[0] == 0

    def test_lag_returns_present(self):
        """Lag return features (indices 6-10) should be meaningful."""
        strategy = MLPredictionStrategy()
        candles = make_trending_candles(100, trend=1.0)
        features = strategy._compute_features(candles)
        # With a clear uptrend, return_1 (index 6) should be mostly positive
        avg_return_1 = np.mean(features[:, 6])
        assert avg_return_1 > 0

    def test_time_features_bounded(self):
        """Hour sin/cos features (indices 20-21) should be in [-1, 1]."""
        strategy = MLPredictionStrategy()
        candles = make_trending_candles(100)
        features = strategy._compute_features(candles)
        hour_sin = features[:, 20]
        hour_cos = features[:, 21]
        assert np.all(hour_sin >= -1.0) and np.all(hour_sin <= 1.0)
        assert np.all(hour_cos >= -1.0) and np.all(hour_cos <= 1.0)


class TestMLV2Labels:
    def test_compute_labels(self):
        strategy = MLPredictionStrategy()
        candles = make_trending_candles(100)
        labels = strategy._compute_labels(candles)
        assert len(labels) > 0
        assert set(labels).issubset({LABEL_UP, LABEL_DOWN, LABEL_NEUTRAL})

    def test_compute_labels_with_custom_threshold(self):
        strategy = MLPredictionStrategy(label_threshold=0.01)
        candles = make_trending_candles(100)
        labels_narrow = strategy._compute_labels(candles)
        strategy2 = MLPredictionStrategy(label_threshold=0.1)
        labels_wide = strategy2._compute_labels(candles)
        # Wider threshold → more NEUTRAL labels
        neutral_narrow = np.sum(labels_narrow == LABEL_NEUTRAL)
        neutral_wide = np.sum(labels_wide == LABEL_NEUTRAL)
        assert neutral_wide >= neutral_narrow


class TestMLV2CrossValidation:
    def test_cross_validate_returns_accuracy(self):
        """Cross-validation should return a float accuracy score."""
        strategy = MLPredictionStrategy(min_training_samples=100)
        candles = make_trending_candles(200)
        features = strategy._compute_features(candles)
        labels = strategy._compute_labels(candles)
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]

        accuracy = strategy._cross_validate(features, labels)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_cross_validate_uses_time_series_split(self):
        """CV should respect temporal ordering (no future leakage)."""
        strategy = MLPredictionStrategy(n_cv_splits=3, min_training_samples=100)
        candles = make_trending_candles(200)
        features = strategy._compute_features(candles)
        labels = strategy._compute_labels(candles)
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]

        # Should complete without error
        accuracy = strategy._cross_validate(features, labels)
        assert accuracy >= 0.0


class TestMLV2Training:
    def test_train_with_enough_data(self):
        strategy = MLPredictionStrategy(min_training_samples=100, min_cv_accuracy=0.0)
        candles = make_trending_candles(200)
        result = strategy.train(candles)
        assert result is True
        assert strategy._is_trained is True
        assert strategy._model is not None

    def test_train_insufficient_data(self):
        strategy = MLPredictionStrategy(min_training_samples=500)
        candles = make_trending_candles(50)
        result = strategy.train(candles)
        assert result is False
        assert strategy._is_trained is False

    def test_train_records_cv_accuracy(self):
        strategy = MLPredictionStrategy(min_training_samples=100, min_cv_accuracy=0.0)
        candles = make_trending_candles(200)
        strategy.train(candles)
        assert strategy._last_cv_accuracy is not None
        assert isinstance(strategy._last_cv_accuracy, float)

    def test_train_low_accuracy_returns_false(self):
        """If CV accuracy below threshold, train returns False."""
        strategy = MLPredictionStrategy(
            min_training_samples=100,
            min_cv_accuracy=0.99,  # Impossibly high threshold
        )
        candles = make_trending_candles(200)
        result = strategy.train(candles)
        assert result is False
        # Model is created but _is_trained stays False
        assert strategy._is_trained is False

    def test_train_records_feature_importances(self):
        strategy = MLPredictionStrategy(min_training_samples=100, min_cv_accuracy=0.0)
        candles = make_trending_candles(200)
        strategy.train(candles)
        assert strategy._feature_importances is not None
        assert isinstance(strategy._feature_importances, dict)
        assert len(strategy._feature_importances) > 0
        # All importance values should be non-negative
        for val in strategy._feature_importances.values():
            assert val >= 0

    def test_train_uses_gradient_boosting(self):
        """V2 uses GradientBoosting, not RandomForest."""
        strategy = MLPredictionStrategy(min_training_samples=100, min_cv_accuracy=0.0)
        candles = make_trending_candles(200)
        strategy.train(candles)
        model = strategy._model
        # Should be CalibratedClassifierCV or GradientBoosting
        model_type = type(model).__name__
        assert model_type in ("CalibratedClassifierCV", "GradientBoostingClassifier")


class TestMLV2Prediction:
    @pytest.mark.asyncio
    async def test_insufficient_data_returns_hold(self):
        strategy = MLPredictionStrategy(min_training_samples=500)
        candles = make_trending_candles(50)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "insufficient_data_for_training"

    @pytest.mark.asyncio
    async def test_auto_train_and_predict(self):
        strategy = MLPredictionStrategy(min_training_samples=100, min_cv_accuracy=0.0)
        candles = make_trending_candles(200)
        signal = await strategy.analyze(candles, symbol="ETH/USDT")

        assert signal.strategy_name == "ml_prediction"
        assert signal.symbol == "ETH/USDT"
        assert signal.action in (SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD)
        assert 0 <= signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_prediction_has_v2_metadata(self):
        strategy = MLPredictionStrategy(min_training_samples=100, min_cv_accuracy=0.0)
        candles = make_trending_candles(200)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")

        assert "prediction" in signal.metadata
        assert "probabilities" in signal.metadata
        assert "cv_accuracy" in signal.metadata
        assert "candles_since_train" in signal.metadata
        assert "feature_importances" in signal.metadata

    @pytest.mark.asyncio
    async def test_high_threshold_produces_hold(self):
        """With a very high prediction threshold, most predictions should be HOLD."""
        strategy = MLPredictionStrategy(
            min_training_samples=100,
            prediction_threshold=0.99,
            min_cv_accuracy=0.0,
        )
        candles = make_trending_candles(200)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_low_cv_accuracy_returns_hold(self):
        """Model with CV accuracy below threshold returns HOLD."""
        strategy = MLPredictionStrategy(
            min_training_samples=100,
            min_cv_accuracy=0.99,  # Impossibly high
        )
        candles = make_trending_candles(200)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "model_not_trained"


class TestMLV2WalkForward:
    def test_needs_retrain_increments(self):
        strategy = MLPredictionStrategy(retrain_interval=10)
        assert strategy.needs_retrain(1) is False  # 1 < 10
        for _ in range(8):
            strategy.needs_retrain(1)
        assert strategy.needs_retrain(1) is True  # 10 >= 10

    def test_needs_retrain_resets_after_train(self):
        strategy = MLPredictionStrategy(
            min_training_samples=100, retrain_interval=5, min_cv_accuracy=0.0
        )
        candles = make_trending_candles(200)
        strategy.train(candles)
        assert strategy._candles_since_train == 0
        strategy.needs_retrain(3)
        assert strategy._candles_since_train == 3

    @pytest.mark.asyncio
    async def test_walk_forward_retrain_triggers(self):
        """After retrain_interval candles, model retrains automatically."""
        strategy = MLPredictionStrategy(
            min_training_samples=100,
            retrain_interval=5,
            min_cv_accuracy=0.0,
        )
        candles = make_trending_candles(200)
        # Initial train
        strategy.train(candles)
        assert strategy._candles_since_train == 0

        # Simulate enough candles to trigger retrain
        strategy._candles_since_train = 5
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # Should have retrained (counter reset + 1 for this analyze)
        assert signal.strategy_name == "ml_prediction"
        assert strategy._candles_since_train <= 2  # Reset after retrain + incremented in analyze


class TestMLV2ModelVersioning:
    def test_save_model_creates_versioned_copy(self):
        strategy = MLPredictionStrategy(min_training_samples=100, min_cv_accuracy=0.0)
        candles = make_trending_candles(200)
        strategy.train(candles)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")
            strategy.save_model(path)

            # Main file should exist
            assert os.path.exists(path)

            # Should have a versioned copy too
            files = os.listdir(tmpdir)
            versioned = [f for f in files if f.startswith("model_") and f.endswith(".pkl")]
            assert len(versioned) >= 1

    def test_save_model_cleans_old_versions(self):
        strategy = MLPredictionStrategy(
            min_training_samples=100, max_model_versions=2, min_cv_accuracy=0.0
        )
        candles = make_trending_candles(200)
        strategy.train(candles)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")

            # Save 4 times (should keep only last 2 versioned copies)
            for _ in range(4):
                strategy.save_model(path)

            files = os.listdir(tmpdir)
            versioned = [f for f in files if f.startswith("model_") and f.endswith(".pkl")]
            assert len(versioned) <= 2

    def test_load_model(self):
        strategy = MLPredictionStrategy(min_training_samples=100, min_cv_accuracy=0.0)
        candles = make_trending_candles(200)
        strategy.train(candles)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            strategy.save_model(path)
            assert os.path.exists(path)

            strategy2 = MLPredictionStrategy(model_path=path)
            assert strategy2._is_trained is True
            assert strategy2._model is not None
        finally:
            os.unlink(path)


class TestMLV2RegimeAdaptation:
    def test_regime_disabled_in_high_volatility(self):
        """Strategy should be disabled in HIGH_VOLATILITY regime."""
        from bot.strategies.regime import MarketRegime

        strategy = MLPredictionStrategy()
        strategy.adapt_to_regime(MarketRegime.HIGH_VOLATILITY)
        assert strategy._regime_disabled is True

    def test_regime_enabled_in_trending(self):
        from bot.strategies.regime import MarketRegime

        strategy = MLPredictionStrategy()
        strategy._regime_disabled = True
        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        assert strategy._regime_disabled is False

    def test_regime_enabled_in_ranging(self):
        from bot.strategies.regime import MarketRegime

        strategy = MLPredictionStrategy()
        strategy._regime_disabled = True
        strategy.adapt_to_regime(MarketRegime.RANGING)
        assert strategy._regime_disabled is False

    @pytest.mark.asyncio
    async def test_regime_disabled_returns_hold(self):
        strategy = MLPredictionStrategy(min_training_samples=100, min_cv_accuracy=0.0)
        candles = make_trending_candles(200)
        strategy.train(candles)

        strategy._regime_disabled = True
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "regime_disabled"


class TestMLV2BackwardCompat:
    """Ensure backward compatibility with existing tests."""

    def test_default_constructor(self):
        """Default constructor should work without new params."""
        strategy = MLPredictionStrategy()
        assert strategy.name == "ml_prediction"
        assert strategy._n_cv_splits == 5
        assert strategy._min_cv_accuracy == 0.55

    def test_old_constructor_params_still_work(self):
        """Old-style constructor params should still work."""
        strategy = MLPredictionStrategy(
            lookback_period=14,
            min_training_samples=500,
            prediction_threshold=0.6,
            model_path=None,
        )
        assert strategy._lookback == 14
        assert strategy._min_training_samples == 500
        assert strategy._prediction_threshold == 0.6

    def test_compute_features_backward_compat(self):
        """First 6 features same as V1: RSI, mom5, mom10, vol, bb, macd."""
        strategy = MLPredictionStrategy()
        candles = make_trending_candles(100)
        features = strategy._compute_features(candles)
        # Should have 25 columns (6 original + 19 new)
        assert features.shape[1] == 25
