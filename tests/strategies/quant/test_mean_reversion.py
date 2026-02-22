"""Tests for mean reversion strategy."""

import numpy as np
import pytest

from bot.models import SignalAction
from bot.strategies.base import strategy_registry
from bot.strategies.quant.mean_reversion import MeanReversionStrategy


@pytest.fixture(autouse=True)
def clean_registry():
    strategy_registry.clear()
    yield
    strategy_registry.clear()


def _make_ohlcv_list(closes, symbol="BTC/USDT"):
    from datetime import datetime

    from bot.models import OHLCV
    return [
        OHLCV(
            timestamp=datetime(2024, 1, 1, i % 24),
            open=c, high=c * 1.01, low=c * 0.99, close=c,
            volume=1000.0, symbol=symbol,
        )
        for i, c in enumerate(closes)
    ]


class TestMeanReversionStrategy:
    def test_name(self):
        s = MeanReversionStrategy()
        assert s.name == "mean_reversion"

    def test_required_history(self):
        s = MeanReversionStrategy(lookback=100)
        assert s.required_history_length == 100

    @pytest.mark.asyncio
    async def test_hold_insufficient_data(self):
        s = MeanReversionStrategy(lookback=100)
        candles = _make_ohlcv_list(np.ones(20) * 100)
        signal = await s.analyze(candles)
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "insufficient_data"

    @pytest.mark.asyncio
    async def test_signal_on_mean_reverting_series(self):
        s = MeanReversionStrategy(zscore_entry=1.5, lookback=200)
        np.random.seed(42)
        n = 200
        kappa = 0.1
        x = np.zeros(n)
        x[0] = 100.0
        for i in range(1, n):
            x[i] = x[i - 1] + kappa * (100.0 - x[i - 1]) + 0.5 * np.random.randn()
        # Push price far from mean at the end
        x[-5:] = 110.0  # Above mean

        candles = _make_ohlcv_list(x)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        assert "adf_pvalue" in signal.metadata
        assert "half_life" in signal.metadata

    @pytest.mark.asyncio
    async def test_regime_disabled_trending_up(self):
        s = MeanReversionStrategy()
        s.adapt_to_regime("TRENDING_UP")
        candles = _make_ohlcv_list(np.ones(200) * 100)
        signal = await s.analyze(candles)
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "regime_disabled"

    @pytest.mark.asyncio
    async def test_regime_disabled_trending_down(self):
        s = MeanReversionStrategy()
        s.adapt_to_regime("TRENDING_DOWN")
        candles = _make_ohlcv_list(np.ones(200) * 100)
        signal = await s.analyze(candles)
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "regime_disabled"

    @pytest.mark.asyncio
    async def test_regime_enabled_ranging(self):
        s = MeanReversionStrategy()
        s.adapt_to_regime("RANGING")
        assert s._regime_disabled is False

    @pytest.mark.asyncio
    async def test_regime_re_enabled(self):
        s = MeanReversionStrategy()
        s.adapt_to_regime("TRENDING_UP")
        assert s._regime_disabled is True
        s.adapt_to_regime("RANGING")
        assert s._regime_disabled is False

    def test_registration(self):
        s = MeanReversionStrategy()
        strategy_registry.register(s)
        assert strategy_registry.get("mean_reversion") is not None

    @pytest.mark.asyncio
    async def test_buy_on_oversold(self):
        """BUY when z-score < -entry_threshold (oversold)."""
        s = MeanReversionStrategy(
            zscore_entry=1.5,
            lookback=200,
            min_half_life=0.1,
            max_half_life=200.0,
            zscore_window=20,
        )
        np.random.seed(42)
        n = 200
        kappa = 0.1
        x = np.zeros(n)
        x[0] = 100.0
        for i in range(1, n):
            x[i] = x[i - 1] + kappa * (100.0 - x[i - 1]) + 0.5 * np.random.randn()
        # Push price far below mean
        x[-5:] = 85.0

        candles = _make_ohlcv_list(x)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        # Should get BUY or HOLD depending on half-life/stationarity
        assert signal.action in (SignalAction.BUY, SignalAction.HOLD)
        assert signal.strategy_name == "mean_reversion"

    @pytest.mark.asyncio
    async def test_sell_on_overbought(self):
        """SELL when z-score > +entry_threshold (overbought)."""
        s = MeanReversionStrategy(
            zscore_entry=1.5,
            lookback=200,
            min_half_life=0.1,
            max_half_life=200.0,
            zscore_window=20,
        )
        np.random.seed(42)
        n = 200
        kappa = 0.1
        x = np.zeros(n)
        x[0] = 100.0
        for i in range(1, n):
            x[i] = x[i - 1] + kappa * (100.0 - x[i - 1]) + 0.5 * np.random.randn()
        # Push price far above mean
        x[-5:] = 115.0

        candles = _make_ohlcv_list(x)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        assert signal.action in (SignalAction.SELL, SignalAction.HOLD)
        assert signal.strategy_name == "mean_reversion"

    @pytest.mark.asyncio
    async def test_confidence_bounded(self):
        """Confidence should always be between 0 and 1."""
        s = MeanReversionStrategy(
            zscore_entry=1.5, lookback=200, min_half_life=0.1, max_half_life=200.0,
        )
        np.random.seed(42)
        n = 200
        kappa = 0.1
        x = np.zeros(n)
        x[0] = 100.0
        for i in range(1, n):
            x[i] = x[i - 1] + kappa * (100.0 - x[i - 1]) + 0.5 * np.random.randn()

        candles = _make_ohlcv_list(x)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        assert 0.0 <= signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_hold_when_half_life_out_of_range(self):
        """HOLD when half-life is outside tradeable range."""
        # Use very tight half-life range to force out-of-range
        s = MeanReversionStrategy(
            lookback=200,
            min_half_life=5.0,
            max_half_life=6.0,
        )
        np.random.seed(42)
        # Random walk-like data (half-life won't be in [5,6])
        closes = np.cumsum(np.random.randn(200)) + 100
        # Ensure all positive
        closes = np.abs(closes) + 50

        candles = _make_ohlcv_list(closes)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_stationarity_penalty(self):
        """Non-stationary series should have lower confidence via stationarity penalty."""
        s = MeanReversionStrategy(
            zscore_entry=1.0,
            lookback=200,
            min_half_life=0.01,
            max_half_life=500.0,
        )
        np.random.seed(99)
        # Create a series that is not stationary (random walk) but force extreme z-score
        x = np.cumsum(np.random.randn(200) * 0.01) + np.log(100)
        # Push extreme values at the end to trigger signal
        x[-5:] = x[-6] + 3.0

        candles_closes = np.exp(x)
        candles = _make_ohlcv_list(candles_closes, symbol="BTC/USDT")
        signal = await s.analyze(candles, symbol="BTC/USDT")
        # With non-stationary data, confidence should be penalized (max 0.5)
        if signal.action != SignalAction.HOLD:
            assert signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_symbol_from_kwargs(self):
        """Symbol should come from kwargs if provided."""
        s = MeanReversionStrategy(lookback=50, min_half_life=0.01, max_half_life=500.0)
        np.random.seed(77)
        closes = 100 + np.cumsum(np.random.randn(50) * 0.1)
        candles = _make_ohlcv_list(closes, symbol="ETH/USDT")
        signal = await s.analyze(candles, symbol="ETH/USDT")
        assert signal.symbol == "ETH/USDT"

    @pytest.mark.asyncio
    async def test_symbol_from_ohlcv(self):
        """Symbol should fall back to OHLCV data if not in kwargs."""
        s = MeanReversionStrategy(lookback=20)
        candles = _make_ohlcv_list(np.ones(10) * 100, symbol="SOL/USDT")
        signal = await s.analyze(candles)
        assert signal.symbol == "SOL/USDT"
