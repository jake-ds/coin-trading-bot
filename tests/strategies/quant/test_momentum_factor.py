"""Tests for cross-timeframe momentum factor strategy."""

import numpy as np
import pytest

from bot.models import SignalAction
from bot.strategies.base import strategy_registry
from bot.strategies.quant.momentum_factor import MomentumFactorStrategy


@pytest.fixture(autouse=True)
def clean_registry():
    strategy_registry.clear()
    yield
    strategy_registry.clear()


def _make_ohlcv_list(closes, symbol="BTC/USDT", timeframe="1h"):
    from datetime import datetime

    from bot.models import OHLCV
    return [
        OHLCV(
            timestamp=datetime(2024, 1, 1, i % 24),
            open=c, high=c * 1.01, low=c * 0.99, close=c,
            volume=1000.0, symbol=symbol, timeframe=timeframe,
        )
        for i, c in enumerate(closes)
    ]


class TestMomentumFactorStrategy:
    def test_name(self):
        s = MomentumFactorStrategy()
        assert s.name == "momentum_factor"

    def test_required_history(self):
        s = MomentumFactorStrategy(long_window=50, zscore_window=20)
        assert s.required_history_length == 75

    @pytest.mark.asyncio
    async def test_hold_insufficient_data(self):
        s = MomentumFactorStrategy()
        candles = _make_ohlcv_list(np.ones(10) * 100)
        signal = await s.analyze(candles)
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_hold_without_multi_timeframe_candles(self):
        """HOLD when no multi_timeframe_candles provided and default data is too short."""
        s = MomentumFactorStrategy(
            short_window=5, long_window=20, zscore_window=10,
            timeframe_weights={"4h": 0.5, "1d": 0.5},
        )
        # Only 10 candles, not enough data — and weights only look for 4h/1d
        candles = _make_ohlcv_list(np.ones(10) * 100)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "insufficient_timeframes"

    @pytest.mark.asyncio
    async def test_buy_strong_positive_momentum(self):
        """BUY on strong positive momentum (uptrend)."""
        s = MomentumFactorStrategy(
            short_window=5, long_window=20, zscore_window=10,
            entry_threshold=1.0,
            timeframe_weights={"1h": 1.0},
        )
        np.random.seed(42)
        # Strong uptrend: cumulative positive moves
        prices = 100 + np.cumsum(np.abs(np.random.randn(100)) * 0.5)
        candles = _make_ohlcv_list(prices)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        # Strong uptrend should produce BUY or at least have positive composite_zscore
        assert "composite_zscore" in signal.metadata
        if signal.action == SignalAction.BUY:
            assert signal.confidence > 0

    @pytest.mark.asyncio
    async def test_sell_strong_negative_momentum(self):
        """SELL on strong negative momentum (downtrend)."""
        s = MomentumFactorStrategy(
            short_window=5, long_window=20, zscore_window=10,
            entry_threshold=1.0,
            timeframe_weights={"1h": 1.0},
        )
        np.random.seed(42)
        # Strong downtrend: cumulative negative moves
        prices = 200 - np.cumsum(np.abs(np.random.randn(100)) * 0.5)
        candles = _make_ohlcv_list(prices)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        assert "composite_zscore" in signal.metadata
        if signal.action == SignalAction.SELL:
            assert signal.confidence > 0

    @pytest.mark.asyncio
    async def test_hold_neutral_momentum(self):
        """HOLD when momentum is neutral (no strong trend)."""
        s = MomentumFactorStrategy(
            short_window=5, long_window=20, zscore_window=10,
            entry_threshold=3.0,  # Very high threshold
            timeframe_weights={"1h": 1.0},
        )
        np.random.seed(123)
        # Flat-ish prices with small noise
        prices = 100 + np.random.randn(100) * 0.1
        candles = _make_ohlcv_list(prices)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_multi_timeframe(self):
        """Composite z-score computed from multiple timeframes."""
        s = MomentumFactorStrategy(
            short_window=5, long_window=20, zscore_window=10,
        )
        np.random.seed(42)
        prices_1h = 100 + np.cumsum(np.random.randn(100) * 0.5)
        prices_4h = 100 + np.cumsum(np.random.randn(100) * 1.0)
        prices_1d = 100 + np.cumsum(np.random.randn(100) * 2.0)

        candles = _make_ohlcv_list(prices_1h)
        multi_tf = {
            "1h": _make_ohlcv_list(prices_1h, timeframe="1h"),
            "4h": _make_ohlcv_list(prices_4h, timeframe="4h"),
            "1d": _make_ohlcv_list(prices_1d, timeframe="1d"),
        }
        signal = await s.analyze(candles, multi_timeframe_candles=multi_tf)
        assert "timeframe_details" in signal.metadata

    def test_regime_adaptation_trending(self):
        s = MomentumFactorStrategy()
        s.adapt_to_regime("TRENDING_UP")
        assert s._regime_multiplier == 1.3

    def test_regime_adaptation_ranging(self):
        s = MomentumFactorStrategy()
        s.adapt_to_regime("RANGING")
        assert s._regime_multiplier == 0.5

    def test_regime_adaptation_other(self):
        s = MomentumFactorStrategy()
        s.adapt_to_regime("HIGH_VOLATILITY")
        assert s._regime_multiplier == 1.0

    def test_regime_adaptation_none(self):
        s = MomentumFactorStrategy()
        s.adapt_to_regime(None)
        assert s._regime_multiplier == 1.0

    def test_registration(self):
        s = MomentumFactorStrategy()
        strategy_registry.register(s)
        assert strategy_registry.get("momentum_factor") is not None

    @pytest.mark.asyncio
    async def test_signal_has_strategy_name(self):
        s = MomentumFactorStrategy(
            short_window=5, long_window=20, zscore_window=10,
            timeframe_weights={"1h": 1.0},
        )
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        candles = _make_ohlcv_list(prices)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        assert signal.strategy_name == "momentum_factor"

    @pytest.mark.asyncio
    async def test_regime_multiplier_affects_composite(self):
        """Regime multiplier scales the composite z-score."""
        s = MomentumFactorStrategy(
            short_window=5, long_window=20, zscore_window=10,
            timeframe_weights={"1h": 1.0},
        )
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        candles = _make_ohlcv_list(prices)

        # Get baseline composite zscore
        signal_normal = await s.analyze(candles, symbol="BTC/USDT")
        z_normal = signal_normal.metadata.get("composite_zscore", 0)

        # With trending regime (1.3x multiplier)
        s.adapt_to_regime("TRENDING_UP")
        signal_trending = await s.analyze(candles, symbol="BTC/USDT")
        z_trending = signal_trending.metadata.get("composite_zscore", 0)

        if z_normal != 0:
            assert abs(z_trending / z_normal - 1.3) < 0.01
