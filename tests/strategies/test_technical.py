"""Tests for technical analysis strategies."""

from datetime import datetime, timedelta

import pytest

from bot.models import OHLCV, SignalAction
from bot.strategies.technical.bollinger import BollingerStrategy
from bot.strategies.technical.ma_crossover import MACrossoverStrategy
from bot.strategies.technical.macd import MACDStrategy
from bot.strategies.technical.rsi import RSIStrategy


def make_candles(prices: list[float], symbol: str = "BTC/USDT") -> list[OHLCV]:
    """Create OHLCV candles from a list of close prices."""
    base = datetime(2024, 1, 1)
    return [
        OHLCV(
            timestamp=base + timedelta(hours=i),
            open=p * 0.99,
            high=p * 1.01,
            low=p * 0.98,
            close=p,
            volume=1000.0,
            symbol=symbol,
            timeframe="1h",
        )
        for i, p in enumerate(prices)
    ]


class TestMACrossoverStrategy:
    @pytest.fixture
    def strategy(self):
        return MACrossoverStrategy(short_period=3, long_period=7)

    def test_name(self, strategy):
        assert strategy.name == "ma_crossover"

    def test_required_history_length(self, strategy):
        assert strategy.required_history_length == 8  # long_period + 1

    @pytest.mark.asyncio
    async def test_insufficient_data(self, strategy):
        candles = make_candles([100.0] * 5)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.confidence == 0.0

    @pytest.mark.asyncio
    async def test_bullish_crossover(self, strategy):
        # Decline then sharp rise: short MA crosses above long MA
        prices = [100, 95, 90, 85, 80, 75, 70, 65, 80, 100, 130, 160, 200]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # After sharp rise, short MA > long MA
        assert signal.metadata["short_ma"] > signal.metadata["long_ma"]

    @pytest.mark.asyncio
    async def test_bearish_crossover(self, strategy):
        # Rise then sharp decline: short MA crosses below long MA
        prices = [50, 55, 60, 65, 70, 75, 80, 85, 70, 55, 35, 20, 10]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # After sharp decline, short MA < long MA
        assert signal.metadata["short_ma"] < signal.metadata["long_ma"]

    @pytest.mark.asyncio
    async def test_no_crossover_hold(self, strategy):
        prices = [100.0] * 15
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_metadata_contains_ma_values(self, strategy):
        prices = [100.0 + i for i in range(15)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert "short_ma" in signal.metadata
        assert "long_ma" in signal.metadata


class TestRSIStrategy:
    @pytest.fixture
    def strategy(self):
        return RSIStrategy(period=7, overbought=70.0, oversold=30.0)

    def test_name(self, strategy):
        assert strategy.name == "rsi"

    def test_required_history_length(self, strategy):
        assert strategy.required_history_length == 9

    @pytest.mark.asyncio
    async def test_insufficient_data(self, strategy):
        candles = make_candles([100.0] * 5)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_oversold_buy_signal(self, strategy):
        # Consecutive drops -> RSI very low
        prices = [100 - i * 3 for i in range(15)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY
        assert signal.metadata["rsi"] < 30.0

    @pytest.mark.asyncio
    async def test_overbought_sell_signal(self, strategy):
        # Consecutive rises -> RSI very high
        prices = [50 + i * 3 for i in range(15)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.SELL
        assert signal.metadata["rsi"] > 70.0

    @pytest.mark.asyncio
    async def test_neutral_hold(self, strategy):
        # Alternating up/down should keep RSI around 50
        prices = [100 + (1 if i % 2 == 0 else -1) for i in range(15)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_metadata_contains_rsi(self, strategy):
        prices = [100.0 + i for i in range(15)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert "rsi" in signal.metadata


class TestMACDStrategy:
    @pytest.fixture
    def strategy(self):
        return MACDStrategy(fast_period=5, slow_period=10, signal_period=4)

    def test_name(self, strategy):
        assert strategy.name == "macd"

    def test_required_history_length(self, strategy):
        assert strategy.required_history_length == 15  # 10 + 4 + 1

    @pytest.mark.asyncio
    async def test_insufficient_data(self, strategy):
        candles = make_candles([100.0] * 10)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_bullish_trend(self, strategy):
        # Decline then sharp rise — MACD should be positive
        prices = [100 - i * 2 for i in range(15)]
        prices.extend([prices[-1] + i * 5 for i in range(1, 10)])
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.metadata["histogram"] > 0

    @pytest.mark.asyncio
    async def test_bearish_trend(self, strategy):
        # Rise then sharp decline — MACD should be negative
        prices = [50 + i * 2 for i in range(15)]
        prices.extend([prices[-1] - i * 5 for i in range(1, 10)])
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.metadata["histogram"] < 0

    @pytest.mark.asyncio
    async def test_metadata_contains_macd_values(self, strategy):
        prices = [100.0 + i for i in range(20)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert "macd" in signal.metadata
        assert "signal" in signal.metadata
        assert "histogram" in signal.metadata


class TestBollingerStrategy:
    @pytest.fixture
    def strategy(self):
        return BollingerStrategy(period=10, std_dev=2.0)

    def test_name(self, strategy):
        assert strategy.name == "bollinger"

    def test_required_history_length(self, strategy):
        assert strategy.required_history_length == 11

    @pytest.mark.asyncio
    async def test_insufficient_data(self, strategy):
        candles = make_candles([100.0] * 5)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_price_below_lower_band_buy(self, strategy):
        # Stable prices then sudden drop
        prices = [100.0] * 12 + [60.0]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY

    @pytest.mark.asyncio
    async def test_price_above_upper_band_sell(self, strategy):
        # Stable prices then sudden spike
        prices = [100.0] * 12 + [140.0]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.SELL

    @pytest.mark.asyncio
    async def test_price_within_bands_hold(self, strategy):
        # Slightly varying prices stay within bands
        prices = [100 + (i % 3) * 0.5 for i in range(15)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_metadata_contains_bands(self, strategy):
        prices = [100.0 + i * 0.5 for i in range(15)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert "upper_band" in signal.metadata
        assert "lower_band" in signal.metadata
        assert "middle_band" in signal.metadata
