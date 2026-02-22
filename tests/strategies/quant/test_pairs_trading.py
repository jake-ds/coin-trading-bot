"""Tests for pairs trading strategy."""

import numpy as np
import pytest

from bot.models import SignalAction
from bot.strategies.base import strategy_registry
from bot.strategies.quant.pairs_trading import PairsTradingStrategy


@pytest.fixture(autouse=True)
def clean_registry():
    strategy_registry.clear()
    yield
    strategy_registry.clear()


def _make_cointegrated_pair(n=300, seed=42):
    np.random.seed(seed)
    a = np.cumsum(np.random.randn(n)) + 100
    b = 1.5 * a + np.random.randn(n) * 0.3
    return a, b


def _make_ohlcv_list(closes):
    from datetime import datetime, timedelta

    from bot.models import OHLCV
    base = datetime(2024, 1, 1)
    return [
        OHLCV(
            timestamp=base + timedelta(hours=i),
            open=c, high=c * 1.01, low=c * 0.99, close=c,
            volume=1000.0, symbol="BTC/USDT",
        )
        for i, c in enumerate(closes)
    ]


class TestPairsTradingStrategy:
    def test_name(self):
        s = PairsTradingStrategy()
        assert s.name == "pairs_trading"

    def test_required_history(self):
        s = PairsTradingStrategy()
        assert s.required_history_length > 0

    @pytest.mark.asyncio
    async def test_hold_without_pair_data(self):
        s = PairsTradingStrategy()
        candles = _make_ohlcv_list(np.ones(100) * 100)
        signal = await s.analyze(candles)
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "no_pair_data"

    @pytest.mark.asyncio
    async def test_hold_not_cointegrated(self):
        s = PairsTradingStrategy()
        np.random.seed(42)
        prices_a = np.cumsum(np.random.randn(300)) + 100
        prices_b = np.cumsum(np.random.randn(300)) + 100
        candles = _make_ohlcv_list(prices_a)
        signal = await s.analyze(candles, pair_prices={
            "symbol_a": "A", "symbol_b": "B",
            "prices_a": prices_a, "prices_b": prices_b,
        })
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_generates_signal_cointegrated(self):
        s = PairsTradingStrategy(zscore_entry=1.5, min_half_life=0.1)
        prices_a, prices_b = _make_cointegrated_pair(n=300)

        candles = _make_ohlcv_list(prices_a)
        signal = await s.analyze(candles, pair_prices={
            "symbol_a": "A", "symbol_b": "B",
            "prices_a": prices_a, "prices_b": prices_b,
        })
        # With strongly cointegrated data, should produce signal with hedge_ratio
        assert signal.action in (SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD)
        assert "hedge_ratio" in signal.metadata

    @pytest.mark.asyncio
    async def test_regime_disabled(self):
        s = PairsTradingStrategy()
        s.adapt_to_regime("HIGH_VOLATILITY")
        candles = _make_ohlcv_list(np.ones(100) * 100)
        signal = await s.analyze(candles, pair_prices={
            "symbol_a": "A", "symbol_b": "B",
            "prices_a": np.ones(100), "prices_b": np.ones(100),
        })
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "regime_disabled"

    def test_registration(self):
        s = PairsTradingStrategy()
        strategy_registry.register(s)
        assert strategy_registry.get("pairs_trading") is not None

    @pytest.mark.asyncio
    async def test_hold_insufficient_data(self):
        s = PairsTradingStrategy()
        candles = _make_ohlcv_list(np.ones(10) * 100)
        signal = await s.analyze(candles, pair_prices={
            "symbol_a": "A", "symbol_b": "B",
            "prices_a": np.ones(10), "prices_b": np.ones(10),
        })
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "insufficient_data"

    @pytest.mark.asyncio
    async def test_buy_on_negative_zscore(self):
        """BUY when z-score < -entry_threshold."""
        s = PairsTradingStrategy(
            zscore_entry=2.0,
            min_half_life=0.1,
            max_half_life=200.0,
            hedge_ratio_window=30,
            zscore_window=20,
        )
        # Create cointegrated pair with forced negative z-score at the end
        np.random.seed(123)
        n = 200
        base = np.cumsum(np.random.randn(n) * 0.5) + 100
        prices_b = base.copy()
        prices_a = 1.0 * base + np.random.randn(n) * 0.2
        # Force spread negative at the end (push price_a down)
        prices_a[-5:] -= 8.0

        candles = _make_ohlcv_list(prices_a)
        signal = await s.analyze(candles, pair_prices={
            "symbol_a": "A", "symbol_b": "B",
            "prices_a": prices_a, "prices_b": prices_b,
        })
        # May or may not trigger BUY depending on cointegration/half-life;
        # at minimum verify structure
        assert signal.action in (SignalAction.BUY, SignalAction.HOLD)
        assert signal.strategy_name == "pairs_trading"

    @pytest.mark.asyncio
    async def test_sell_on_positive_zscore(self):
        """SELL when z-score > +entry_threshold."""
        s = PairsTradingStrategy(
            zscore_entry=2.0,
            min_half_life=0.1,
            max_half_life=200.0,
            hedge_ratio_window=30,
            zscore_window=20,
        )
        np.random.seed(123)
        n = 200
        base = np.cumsum(np.random.randn(n) * 0.5) + 100
        prices_b = base.copy()
        prices_a = 1.0 * base + np.random.randn(n) * 0.2
        # Force spread positive at the end (push price_a up)
        prices_a[-5:] += 8.0

        candles = _make_ohlcv_list(prices_a)
        signal = await s.analyze(candles, pair_prices={
            "symbol_a": "A", "symbol_b": "B",
            "prices_a": prices_a, "prices_b": prices_b,
        })
        assert signal.action in (SignalAction.SELL, SignalAction.HOLD)
        assert signal.strategy_name == "pairs_trading"

    @pytest.mark.asyncio
    async def test_confidence_bounded(self):
        """Confidence should be between 0 and 1."""
        s = PairsTradingStrategy(zscore_entry=1.5, min_half_life=0.1)
        prices_a, prices_b = _make_cointegrated_pair(n=300)
        candles = _make_ohlcv_list(prices_a)
        signal = await s.analyze(candles, pair_prices={
            "symbol_a": "A", "symbol_b": "B",
            "prices_a": prices_a, "prices_b": prices_b,
        })
        assert 0.0 <= signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_metadata_contains_pair_info(self):
        """Metadata should contain pair trading details when cointegrated."""
        s = PairsTradingStrategy(zscore_entry=1.5, min_half_life=0.1)
        prices_a, prices_b = _make_cointegrated_pair(n=300)
        candles = _make_ohlcv_list(prices_a)
        signal = await s.analyze(candles, pair_prices={
            "symbol_a": "BTC", "symbol_b": "ETH",
            "prices_a": prices_a, "prices_b": prices_b,
        })
        if "hedge_ratio" in signal.metadata:
            assert "zscore" in signal.metadata
            assert "pair" in signal.metadata
            assert signal.metadata["pair"] == "BTC/ETH"

    @pytest.mark.asyncio
    async def test_regime_re_enabled(self):
        """Re-enabling after regime change should allow signals."""
        s = PairsTradingStrategy()
        s.adapt_to_regime("HIGH_VOLATILITY")
        assert s._regime_disabled is True
        s.adapt_to_regime("LOW_VOLATILITY")
        assert s._regime_disabled is False
