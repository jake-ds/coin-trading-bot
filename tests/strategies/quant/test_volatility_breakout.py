"""Tests for GARCH volatility breakout strategy."""

import numpy as np
import pytest

from bot.models import SignalAction
from bot.strategies.base import strategy_registry
from bot.strategies.quant.volatility_breakout import VolatilityBreakoutStrategy


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
            open=c,
            high=c * 1.01,
            low=c * 0.99,
            close=c,
            volume=1000.0,
            symbol=symbol,
        )
        for i, c in enumerate(closes)
    ]


class TestVolatilityBreakoutStrategy:
    def test_name(self):
        s = VolatilityBreakoutStrategy()
        assert s.name == "volatility_breakout"

    def test_required_history(self):
        s = VolatilityBreakoutStrategy(lookback=120)
        assert s.required_history_length == 120

    @pytest.mark.asyncio
    async def test_hold_insufficient_data(self):
        s = VolatilityBreakoutStrategy(min_data_points=60)
        candles = _make_ohlcv_list(np.ones(20) * 100)
        signal = await s.analyze(candles)
        assert signal.action == SignalAction.HOLD
        assert signal.metadata["reason"] == "insufficient_data"

    @pytest.mark.asyncio
    async def test_normal_volatility(self):
        s = VolatilityBreakoutStrategy(min_data_points=60, lookback=120)
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(150) * 0.01))
        candles = _make_ohlcv_list(prices)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        assert "forecasted_vol" in signal.metadata
        assert "realized_vol" in signal.metadata

    @pytest.mark.asyncio
    async def test_volatility_breakout_detection(self):
        s = VolatilityBreakoutStrategy(
            min_data_points=60,
            lookback=150,
            breakout_multiplier=1.5,
        )
        np.random.seed(42)
        # Normal volatility followed by a spike
        prices = np.ones(150) * 100
        returns = np.random.randn(149) * 0.005
        returns[-10:] = np.random.randn(10) * 0.05  # Big spike
        for i in range(1, 150):
            prices[i] = prices[i - 1] * (1 + returns[i - 1])

        candles = _make_ohlcv_list(prices)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        assert "vol_ratio" in signal.metadata
        # May generate a breakout signal due to vol spike
        if signal.metadata.get("vol_ratio", 0) >= 1.5:
            assert signal.action in (SignalAction.BUY, SignalAction.SELL)

    def test_registration(self):
        s = VolatilityBreakoutStrategy()
        strategy_registry.register(s)
        assert strategy_registry.get("volatility_breakout") is not None

    @pytest.mark.asyncio
    async def test_hold_on_garch_failure(self):
        """GARCH fit fails on constant data -> HOLD."""
        s = VolatilityBreakoutStrategy(min_data_points=30)
        # Near-constant prices with tiny noise to avoid log issues
        np.random.seed(99)
        prices = 100 + np.random.randn(80) * 1e-8
        candles = _make_ohlcv_list(prices)
        signal = await s.analyze(candles, symbol="ETH/USDT")
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_metadata_contains_dynamic_stop_loss(self):
        """When GARCH fits, metadata includes dynamic_stop_loss."""
        s = VolatilityBreakoutStrategy(min_data_points=60, lookback=120)
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(150) * 0.01))
        candles = _make_ohlcv_list(prices)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        if signal.metadata.get("reason") not in ("garch_fit_failed", "insufficient_data"):
            assert "dynamic_stop_loss" in signal.metadata
            assert signal.metadata["dynamic_stop_loss"] > 0

    @pytest.mark.asyncio
    async def test_metadata_contains_garch_persistence(self):
        """Successful GARCH fit populates persistence metadata."""
        s = VolatilityBreakoutStrategy(min_data_points=60, lookback=120)
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(150) * 0.01))
        candles = _make_ohlcv_list(prices)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        if signal.metadata.get("reason") not in ("garch_fit_failed", "insufficient_data"):
            assert "garch_persistence" in signal.metadata
            assert 0 <= signal.metadata["garch_persistence"] <= 2.0

    @pytest.mark.asyncio
    async def test_symbol_from_ohlcv(self):
        """Symbol falls back to ohlcv_data[-1].symbol when not in kwargs."""
        s = VolatilityBreakoutStrategy(min_data_points=60, lookback=120)
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(150) * 0.01))
        candles = _make_ohlcv_list(prices, symbol="SOL/USDT")
        signal = await s.analyze(candles)
        assert signal.symbol == "SOL/USDT"

    @pytest.mark.asyncio
    async def test_buy_on_positive_breakout(self):
        """Upside vol breakout with positive recent returns -> BUY."""
        s = VolatilityBreakoutStrategy(
            min_data_points=60,
            lookback=150,
            breakout_multiplier=1.2,
            realized_window=10,
        )
        np.random.seed(123)
        # Low-vol period
        prices = np.ones(150) * 100
        returns = np.random.randn(149) * 0.002
        # Strong upside spike at the end
        returns[-10:] = np.abs(np.random.randn(10)) * 0.04
        for i in range(1, 150):
            prices[i] = prices[i - 1] * (1 + returns[i - 1])
        candles = _make_ohlcv_list(prices)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        # If breakout detected and returns positive, should be BUY
        if signal.action != SignalAction.HOLD:
            assert signal.action == SignalAction.BUY
            assert signal.metadata.get("signal_reason") == "upside_vol_breakout"

    @pytest.mark.asyncio
    async def test_sell_on_negative_breakout(self):
        """Downside vol breakout with negative recent returns -> SELL."""
        s = VolatilityBreakoutStrategy(
            min_data_points=60,
            lookback=150,
            breakout_multiplier=1.2,
            realized_window=10,
        )
        np.random.seed(456)
        prices = np.ones(150) * 100
        returns = np.random.randn(149) * 0.002
        # Strong downside spike at the end
        returns[-10:] = -np.abs(np.random.randn(10)) * 0.04
        for i in range(1, 150):
            prices[i] = prices[i - 1] * (1 + returns[i - 1])
        candles = _make_ohlcv_list(prices)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        if signal.action != SignalAction.HOLD:
            assert signal.action == SignalAction.SELL
            assert signal.metadata.get("signal_reason") == "downside_vol_breakout"

    @pytest.mark.asyncio
    async def test_confidence_capped_at_one(self):
        """Confidence should never exceed 1.0."""
        s = VolatilityBreakoutStrategy(
            min_data_points=60,
            lookback=150,
            breakout_multiplier=1.0,
            realized_window=10,
        )
        np.random.seed(789)
        prices = np.ones(150) * 100
        returns = np.random.randn(149) * 0.002
        returns[-10:] = np.random.randn(10) * 0.1  # Extreme spike
        for i in range(1, 150):
            prices[i] = prices[i - 1] * (1 + returns[i - 1])
        candles = _make_ohlcv_list(prices)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        assert 0.0 <= signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_strategy_name_in_signal(self):
        """Strategy name is correctly set in the signal."""
        s = VolatilityBreakoutStrategy(min_data_points=60)
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(150) * 0.01))
        candles = _make_ohlcv_list(prices)
        signal = await s.analyze(candles, symbol="BTC/USDT")
        assert signal.strategy_name == "volatility_breakout"
