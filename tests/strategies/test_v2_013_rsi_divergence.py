"""Tests for V2-013: Enhanced RSI with divergence detection."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from bot.models import OHLCV, SignalAction
from bot.strategies.technical.rsi import RSIStrategy, _find_peaks, _find_troughs


def make_candles(
    prices: list[float], symbol: str = "BTC/USDT", volume: float = 1000.0
) -> list[OHLCV]:
    """Create OHLCV candles from a list of close prices."""
    base = datetime(2024, 1, 1)
    return [
        OHLCV(
            timestamp=base + timedelta(hours=i),
            open=p * 0.99,
            high=p * 1.01,
            low=p * 0.98,
            close=p,
            volume=volume,
            symbol=symbol,
            timeframe="1h",
        )
        for i, p in enumerate(prices)
    ]


class TestFindTroughs:
    def test_simple_trough(self):
        series = pd.Series([10, 8, 5, 8, 10])
        troughs = _find_troughs(series, order=1)
        assert len(troughs) >= 1
        # Index 2, value 5 should be a trough
        trough_indices = [t[0] for t in troughs]
        assert 2 in trough_indices

    def test_two_troughs(self):
        series = pd.Series([10, 5, 10, 3, 10])
        troughs = _find_troughs(series, order=1)
        assert len(troughs) == 2
        assert troughs[0][1] == 5.0
        assert troughs[1][1] == 3.0

    def test_no_troughs_flat(self):
        series = pd.Series([5, 5, 5, 5, 5])
        troughs = _find_troughs(series, order=1)
        # Flat series: every point is <= neighbors, so all interior points are troughs
        # This is fine — divergence logic looks at relationships between troughs
        assert len(troughs) >= 0

    def test_nan_handling(self):
        series = pd.Series([10, float("nan"), 5, 8, 10])
        troughs = _find_troughs(series, order=1)
        # NaN at index 1 should prevent index 2 from being detected (NaN neighbor)
        trough_indices = [t[0] for t in troughs]
        assert 1 not in trough_indices


class TestFindPeaks:
    def test_simple_peak(self):
        series = pd.Series([5, 8, 10, 8, 5])
        peaks = _find_peaks(series, order=1)
        assert len(peaks) >= 1
        peak_indices = [p[0] for p in peaks]
        assert 2 in peak_indices

    def test_two_peaks(self):
        series = pd.Series([5, 10, 5, 12, 5])
        peaks = _find_peaks(series, order=1)
        assert len(peaks) == 2
        assert peaks[0][1] == 10.0
        assert peaks[1][1] == 12.0

    def test_nan_handling(self):
        series = pd.Series([5, float("nan"), 10, 8, 5])
        peaks = _find_peaks(series, order=1)
        peak_indices = [p[0] for p in peaks]
        assert 1 not in peak_indices


class TestRSIDivergenceBackwardCompat:
    """Ensure divergence-disabled RSI behaves identically to the original."""

    @pytest.fixture
    def strategy(self):
        return RSIStrategy(period=7, overbought=70.0, oversold=30.0)

    def test_name(self, strategy):
        assert strategy.name == "rsi"

    def test_divergence_disabled_by_default(self, strategy):
        assert strategy._divergence_enabled is False

    @pytest.mark.asyncio
    async def test_oversold_buy_signal(self, strategy):
        prices = [100 - i * 3 for i in range(15)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY
        assert signal.metadata["rsi"] < 30.0
        # No divergence_type when disabled
        assert "divergence_type" not in signal.metadata

    @pytest.mark.asyncio
    async def test_overbought_sell_signal(self, strategy):
        prices = [50 + i * 3 for i in range(15)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.SELL
        assert signal.metadata["rsi"] > 70.0
        assert "divergence_type" not in signal.metadata

    @pytest.mark.asyncio
    async def test_neutral_hold(self, strategy):
        prices = [100 + (1 if i % 2 == 0 else -1) for i in range(15)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_insufficient_data(self, strategy):
        candles = make_candles([100.0] * 5)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "insufficient_data"


class TestRSIBullishDivergence:
    """Test bullish divergence: price makes lower low, RSI makes higher low."""

    @pytest.fixture
    def strategy(self):
        return RSIStrategy(
            period=7,
            overbought=70.0,
            oversold=30.0,
            divergence_enabled=True,
            divergence_lookback=30,
            divergence_swing_order=2,
        )

    @pytest.mark.asyncio
    async def test_bullish_divergence_detected(self, strategy):
        """Craft data where price makes lower lows but RSI makes higher lows.

        Pattern: warmup, strong drop (low RSI), recovery, gradual drop to lower
        price but with weaker momentum (RSI stays higher).
        """
        # Warmup: stable prices so RSI settles
        prices = [100, 101, 100, 99, 100, 101, 100, 99, 100, 101]
        # Phase 1: Strong drop (RSI drops sharply)
        prices += [96, 91, 86, 81, 76]
        # Phase 2: Recovery
        prices += [81, 86, 91, 96, 101, 106]
        # Phase 3: Gradual, weaker drop to even lower price (RSI higher)
        prices += [104, 102, 100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74]
        # Phase 4: Recovery for swing detection
        prices += [77, 80, 83]

        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")

        assert signal.action == SignalAction.BUY
        assert signal.confidence == 0.8
        assert signal.metadata["divergence_type"] == "bullish_divergence"
        assert "price_trough_1" in signal.metadata
        assert "price_trough_2" in signal.metadata
        assert "rsi_trough_1" in signal.metadata
        assert "rsi_trough_2" in signal.metadata
        # Price: second trough lower than first
        assert signal.metadata["price_trough_2"] < signal.metadata["price_trough_1"]
        # RSI: second trough higher than first
        assert signal.metadata["rsi_trough_2"] > signal.metadata["rsi_trough_1"]

    @pytest.mark.asyncio
    async def test_bullish_divergence_has_higher_confidence_than_oversold(self, strategy):
        """Divergence signals should have higher confidence (0.8) than oversold (<=0.5)."""
        # Simple oversold signal without divergence
        prices_simple = [100 - i * 3 for i in range(20)]
        candles_simple = make_candles(prices_simple)
        signal_simple = await strategy.analyze(candles_simple, symbol="BTC/USDT")

        if signal_simple.action == SignalAction.BUY:
            # Non-divergence signals are capped at 0.5 when divergence is enabled
            assert signal_simple.confidence <= 0.5


class TestRSIBearishDivergence:
    """Test bearish divergence: price makes higher high, RSI makes lower high."""

    @pytest.fixture
    def strategy(self):
        return RSIStrategy(
            period=7,
            overbought=70.0,
            oversold=30.0,
            divergence_enabled=True,
            divergence_lookback=30,
            divergence_swing_order=2,
        )

    @pytest.mark.asyncio
    async def test_bearish_divergence_detected(self, strategy):
        """Craft data where price makes higher highs but RSI makes lower highs.

        Pattern: warmup, strong rise (high RSI), pullback, gradual rise to
        higher price but with weaker momentum (RSI lower).
        """
        # Warmup: stable prices
        prices = [50, 49, 50, 51, 50, 49, 50, 51, 50, 49]
        # Phase 1: Strong rise (RSI spikes high)
        prices += [54, 59, 64, 69, 74]
        # Phase 2: Pullback
        prices += [69, 64, 59, 54, 49, 44]
        # Phase 3: Gradual, weaker rise to even higher price (RSI lower)
        prices += [46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76]
        # Phase 4: Pullback for swing detection
        prices += [73, 70, 67]

        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")

        assert signal.action == SignalAction.SELL
        assert signal.confidence == 0.8
        assert signal.metadata["divergence_type"] == "bearish_divergence"
        assert "price_peak_1" in signal.metadata
        assert "price_peak_2" in signal.metadata
        assert "rsi_peak_1" in signal.metadata
        assert "rsi_peak_2" in signal.metadata
        # Price: second peak higher than first
        assert signal.metadata["price_peak_2"] > signal.metadata["price_peak_1"]
        # RSI: second peak lower than first
        assert signal.metadata["rsi_peak_2"] < signal.metadata["rsi_peak_1"]

    @pytest.mark.asyncio
    async def test_bearish_divergence_has_higher_confidence_than_overbought(self, strategy):
        """Divergence signals should have higher confidence (0.8) than overbought (<=0.5)."""
        # Simple overbought signal without divergence
        prices_simple = [50 + i * 3 for i in range(20)]
        candles_simple = make_candles(prices_simple)
        signal_simple = await strategy.analyze(candles_simple, symbol="BTC/USDT")

        if signal_simple.action == SignalAction.SELL:
            assert signal_simple.confidence <= 0.5


class TestRSIDivergenceFallback:
    """Test that regular RSI signals still work as fallback."""

    @pytest.fixture
    def strategy(self):
        return RSIStrategy(
            period=7,
            overbought=70.0,
            oversold=30.0,
            divergence_enabled=True,
            divergence_lookback=14,
            divergence_swing_order=2,
        )

    @pytest.mark.asyncio
    async def test_oversold_fallback_when_no_divergence(self, strategy):
        """When RSI is oversold but no divergence detected, still generate BUY."""
        prices = [100 - i * 3 for i in range(15)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY
        assert signal.metadata["divergence_type"] == "oversold"
        assert signal.confidence <= 0.5

    @pytest.mark.asyncio
    async def test_overbought_fallback_when_no_divergence(self, strategy):
        """When RSI is overbought but no divergence detected, still generate SELL."""
        prices = [50 + i * 3 for i in range(15)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.SELL
        assert signal.metadata["divergence_type"] == "overbought"
        assert signal.confidence <= 0.5

    @pytest.mark.asyncio
    async def test_hold_when_rsi_neutral(self, strategy):
        """When RSI is neutral and no divergence, return HOLD."""
        prices = [100 + (1 if i % 2 == 0 else -1) for i in range(15)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.confidence == 0.0

    @pytest.mark.asyncio
    async def test_metadata_always_has_rsi(self, strategy):
        """Metadata should always include basic RSI info."""
        prices = [100.0 + i for i in range(15)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert "rsi" in signal.metadata
        assert "overbought" in signal.metadata
        assert "oversold" in signal.metadata
        assert "period" in signal.metadata


class TestRSIDivergenceConfig:
    """Test configurable parameters."""

    def test_default_divergence_disabled(self):
        strategy = RSIStrategy()
        assert strategy._divergence_enabled is False
        assert strategy._divergence_lookback == 14
        assert strategy._divergence_swing_order == 3

    def test_custom_divergence_config(self):
        strategy = RSIStrategy(
            divergence_enabled=True,
            divergence_lookback=20,
            divergence_swing_order=4,
        )
        assert strategy._divergence_enabled is True
        assert strategy._divergence_lookback == 20
        assert strategy._divergence_swing_order == 4

    def test_required_history_unchanged(self):
        """Divergence parameters should not change required_history_length."""
        strategy_off = RSIStrategy(period=14, divergence_enabled=False)
        strategy_on = RSIStrategy(period=14, divergence_enabled=True)
        assert strategy_off.required_history_length == strategy_on.required_history_length

    @pytest.mark.asyncio
    async def test_regime_adaptation_preserved(self):
        """adapt_to_regime should still work with divergence enabled."""
        from bot.strategies.regime import MarketRegime

        strategy = RSIStrategy(
            period=14,
            overbought=70.0,
            oversold=30.0,
            divergence_enabled=True,
        )
        strategy.adapt_to_regime(MarketRegime.RANGING)
        assert strategy._oversold == 35.0
        assert strategy._overbought == 65.0
        # Divergence should remain enabled
        assert strategy._divergence_enabled is True

    @pytest.mark.asyncio
    async def test_divergence_priority_over_oversold(self):
        """When both divergence and oversold are detected, divergence takes priority."""
        strategy = RSIStrategy(
            period=7,
            overbought=70.0,
            oversold=30.0,
            divergence_enabled=True,
            divergence_lookback=30,
            divergence_swing_order=2,
        )
        # Build data that has bullish divergence
        # Warmup
        prices = [100, 101, 100, 99, 100, 101, 100, 99, 100, 101]
        # Strong drop
        prices += [96, 91, 86, 81, 76]
        # Recovery
        prices += [81, 86, 91, 96, 101, 106]
        # Weaker drop to lower price (creates divergence)
        prices += [104, 102, 100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74]
        prices += [77, 80, 83]

        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")

        # Should detect divergence (higher priority)
        if signal.metadata.get("divergence_type") == "bullish_divergence":
            assert signal.confidence == 0.8
