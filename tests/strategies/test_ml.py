"""Tests for ML price prediction strategy."""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from bot.models import OHLCV, SignalAction
from bot.strategies.ml.prediction import MLPredictionStrategy


def make_trending_candles(n: int, start_price: float = 100.0, trend: float = 0.5) -> list[OHLCV]:
    """Create synthetic candles with a trend and some noise."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    np.random.seed(42)
    price = start_price

    for i in range(n):
        noise = np.random.normal(0, 1)
        price = price + trend + noise
        price = max(price, 1.0)
        candles.append(OHLCV(
            timestamp=base + timedelta(hours=i),
            open=price - 0.5,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=1000.0 + np.random.uniform(-100, 100),
        ))

    return candles


class TestMLPredictionStrategy:
    def test_name(self):
        strategy = MLPredictionStrategy()
        assert strategy.name == "ml_prediction"

    def test_required_history_length(self):
        strategy = MLPredictionStrategy(min_training_samples=500)
        assert strategy.required_history_length >= 500

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_hold(self):
        strategy = MLPredictionStrategy(min_training_samples=500)
        candles = make_trending_candles(50)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "insufficient_data_for_training"

    def test_train_with_enough_data(self):
        strategy = MLPredictionStrategy(min_training_samples=100)
        candles = make_trending_candles(200)
        result = strategy.train(candles)
        assert result is True
        assert strategy._is_trained is True

    def test_train_insufficient_data(self):
        strategy = MLPredictionStrategy(min_training_samples=500)
        candles = make_trending_candles(50)
        result = strategy.train(candles)
        assert result is False
        assert strategy._is_trained is False

    @pytest.mark.asyncio
    async def test_auto_train_and_predict(self):
        strategy = MLPredictionStrategy(min_training_samples=100)
        candles = make_trending_candles(200)
        signal = await strategy.analyze(candles, symbol="ETH/USDT")

        assert signal.strategy_name == "ml_prediction"
        assert signal.symbol == "ETH/USDT"
        assert signal.action in (SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD)
        assert 0 <= signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_prediction_has_metadata(self):
        strategy = MLPredictionStrategy(min_training_samples=100)
        candles = make_trending_candles(200)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")

        assert "prediction" in signal.metadata
        assert "probabilities" in signal.metadata

    def test_compute_features_shape(self):
        strategy = MLPredictionStrategy()
        candles = make_trending_candles(100)
        features = strategy._compute_features(candles)

        assert features.ndim == 2
        assert features.shape[1] == 6  # 6 features
        assert features.shape[0] > 0

    def test_save_and_load_model(self):
        strategy = MLPredictionStrategy(min_training_samples=100)
        candles = make_trending_candles(200)
        strategy.train(candles)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            strategy.save_model(path)
            assert os.path.exists(path)

            # Load into new strategy
            strategy2 = MLPredictionStrategy(model_path=path)
            assert strategy2._is_trained is True
            assert strategy2._model is not None
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_high_threshold_produces_hold(self):
        """With a very high prediction threshold, most predictions should be HOLD."""
        strategy = MLPredictionStrategy(
            min_training_samples=100,
            prediction_threshold=0.99,
        )
        candles = make_trending_candles(200)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # With 0.99 threshold, most predictions won't be confident enough
        assert signal.action == SignalAction.HOLD
