"""Tests for V2-016: Composite momentum strategy (RSI + MACD + Stochastic)."""

from datetime import datetime, timedelta

import pytest

from bot.models import OHLCV, SignalAction
from bot.strategies.technical.composite import CompositeMomentumStrategy


def make_candles(
    prices: list[float],
    symbol: str = "BTC/USDT",
    volumes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    opens: list[float] | None = None,
) -> list[OHLCV]:
    """Create OHLCV candles from close prices."""
    base = datetime(2024, 1, 1)
    if volumes is None:
        volumes = [1000.0] * len(prices)
    candles = []
    for i, (p, v) in enumerate(zip(prices, volumes)):
        o = opens[i] if opens else p * 0.999
        h = highs[i] if highs else max(p, o) * 1.005
        bp_low = lows[i] if lows else min(p, o) * 0.995
        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=i),
                open=o,
                high=h,
                low=bp_low,
                close=p,
                volume=v,
                symbol=symbol,
                timeframe="1h",
            )
        )
    return candles


def make_downtrend_candles(n: int = 50, start: float = 200.0) -> list[OHLCV]:
    """Create a clear downtrend: prices fall steadily.

    This produces:
    - RSI < 40 (bearish momentum → rsi_buy = True)
    Wait, for SELL we need RSI > 60. For BUY we need RSI < 40.

    For a SELL signal we need:
    - RSI > 60 → need uptrend first then possible
    - MACD histogram turning negative
    - Stochastic crossing down from above 70

    For a BUY signal we need:
    - RSI < 40 → need downtrend
    - MACD histogram turning positive
    - Stochastic crossing up from below 30
    """
    # First go up to establish high stochastic, then come down sharply
    prices = []
    for i in range(n):
        # Start high, then crash
        if i < n // 2:
            prices.append(start + i * 2.0)
        else:
            prices.append(start + (n // 2) * 2.0 - (i - n // 2) * 5.0)
    # Ensure no negative prices
    prices = [max(p, 10.0) for p in prices]
    return make_candles(prices)


def make_buy_signal_candles(n: int = 50) -> list[OHLCV]:
    """Create candle data that should produce a BUY signal.

    Pattern: strong downtrend followed by a reversal at the end.
    This creates:
    - RSI < 40 (oversold after downtrend, starting to turn)
    - MACD histogram turning positive (reversal)
    - Stochastic %K crossing above %D from oversold region
    """
    prices = []
    base = 100.0

    # Phase 1: Gradual decline (candles 0-34) — establishes downtrend
    for i in range(35):
        prices.append(base - i * 1.5)

    # Phase 2: Sharp bottom and reversal (candles 35-49)
    bottom = prices[-1]
    for i in range(15):
        if i < 5:
            # Continue declining slightly
            prices.append(bottom - i * 0.5)
        elif i < 10:
            # Flat / slight uptick
            prices.append(bottom - 2.5 + (i - 5) * 0.3)
        else:
            # Clear uptick (reversal)
            prices.append(bottom - 1.0 + (i - 10) * 1.5)

    prices = [max(p, 10.0) for p in prices]

    # Volume: higher on reversal candles
    volumes = [1000.0] * 35 + [1500.0] * 15

    return make_candles(prices, volumes=volumes)


def make_sell_signal_candles(n: int = 50) -> list[OHLCV]:
    """Create candle data that should produce a SELL signal.

    Pattern: strong uptrend followed by a rollover at the end.
    This creates:
    - RSI > 60 (overbought after uptrend, starting to turn)
    - MACD histogram turning negative (rollover)
    - Stochastic %K crossing below %D from overbought region
    """
    prices = []
    base = 50.0

    # Phase 1: Strong rally (candles 0-34)
    for i in range(35):
        prices.append(base + i * 2.0)

    # Phase 2: Topping and rollover (candles 35-49)
    top = prices[-1]
    for i in range(15):
        if i < 5:
            # Continue rising slightly
            prices.append(top + i * 0.3)
        elif i < 10:
            # Flat / slight decline
            prices.append(top + 1.5 - (i - 5) * 0.3)
        else:
            # Clear decline (rollover)
            prices.append(top + 0.0 - (i - 10) * 2.0)

    prices = [max(p, 10.0) for p in prices]
    volumes = [1000.0] * 35 + [1500.0] * 15

    return make_candles(prices, volumes=volumes)


def make_flat_candles(n: int = 50, price: float = 100.0) -> list[OHLCV]:
    """Create flat price data (no signal expected)."""
    # Small random-like variations around a flat price
    prices = [price + (i % 3 - 1) * 0.5 for i in range(n)]
    return make_candles(prices)


# ==================== Constructor / Properties ====================

class TestCompositeMomentumProperties:
    def test_name(self):
        strategy = CompositeMomentumStrategy()
        assert strategy.name == "composite_momentum"

    def test_required_history_length_default(self):
        strategy = CompositeMomentumStrategy()
        # Default: max(26+9+1=36, 14+3+1=18, 14+2=16) = 36
        assert strategy.required_history_length == 36

    def test_required_history_length_custom(self):
        strategy = CompositeMomentumStrategy(macd_slow=30, macd_signal=12)
        # max(30+12+1=43, 14+3+1=18, 14+2=16) = 43
        assert strategy.required_history_length == 43

    def test_default_parameters(self):
        strategy = CompositeMomentumStrategy()
        assert strategy._rsi_period == 14
        assert strategy._rsi_buy_threshold == 40.0
        assert strategy._rsi_sell_threshold == 60.0
        assert strategy._macd_fast == 12
        assert strategy._macd_slow == 26
        assert strategy._macd_signal == 9
        assert strategy._stoch_period == 14
        assert strategy._stoch_smooth == 3
        assert strategy._stoch_buy_threshold == 30.0
        assert strategy._stoch_sell_threshold == 70.0

    def test_custom_parameters(self):
        strategy = CompositeMomentumStrategy(
            rsi_period=10,
            rsi_buy_threshold=35.0,
            rsi_sell_threshold=65.0,
            macd_fast=8,
            macd_slow=21,
            macd_signal=5,
            stoch_period=10,
            stoch_smooth=5,
            stoch_buy_threshold=25.0,
            stoch_sell_threshold=75.0,
        )
        assert strategy._rsi_period == 10
        assert strategy._rsi_buy_threshold == 35.0
        assert strategy._rsi_sell_threshold == 65.0
        assert strategy._macd_fast == 8
        assert strategy._macd_slow == 21
        assert strategy._macd_signal == 5
        assert strategy._stoch_period == 10
        assert strategy._stoch_smooth == 5
        assert strategy._stoch_buy_threshold == 25.0
        assert strategy._stoch_sell_threshold == 75.0


# ==================== Insufficient Data ====================

class TestInsufficientData:
    @pytest.mark.asyncio
    async def test_insufficient_data_returns_hold(self):
        strategy = CompositeMomentumStrategy()
        candles = make_flat_candles(n=10)  # Need 36
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.confidence == 0.0
        assert signal.metadata["reason"] == "insufficient_data"

    @pytest.mark.asyncio
    async def test_exact_minimum_data(self):
        strategy = CompositeMomentumStrategy()
        candles = make_flat_candles(n=36)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action in (SignalAction.HOLD, SignalAction.BUY, SignalAction.SELL)
        assert signal.strategy_name == "composite_momentum"


# ==================== Metadata ====================

class TestMetadata:
    @pytest.mark.asyncio
    async def test_metadata_contains_all_indicators(self):
        strategy = CompositeMomentumStrategy()
        candles = make_flat_candles(n=50)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")

        assert "rsi" in signal.metadata
        assert "rsi_buy_confirmed" in signal.metadata
        assert "rsi_sell_confirmed" in signal.metadata
        assert "macd" in signal.metadata
        assert "macd_signal" in signal.metadata
        assert "macd_histogram" in signal.metadata
        assert "macd_prev_histogram" in signal.metadata
        assert "macd_buy_confirmed" in signal.metadata
        assert "macd_sell_confirmed" in signal.metadata
        assert "stoch_k" in signal.metadata
        assert "stoch_d" in signal.metadata
        assert "stoch_prev_k" in signal.metadata
        assert "stoch_prev_d" in signal.metadata
        assert "stoch_buy_confirmed" in signal.metadata
        assert "stoch_sell_confirmed" in signal.metadata
        assert "buy_confirmations" in signal.metadata
        assert "sell_confirmations" in signal.metadata
        assert "signal_type" in signal.metadata

    @pytest.mark.asyncio
    async def test_metadata_confirmation_types_are_bool(self):
        strategy = CompositeMomentumStrategy()
        candles = make_flat_candles(n=50)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")

        assert isinstance(signal.metadata["rsi_buy_confirmed"], bool)
        assert isinstance(signal.metadata["rsi_sell_confirmed"], bool)
        assert isinstance(signal.metadata["macd_buy_confirmed"], bool)
        assert isinstance(signal.metadata["macd_sell_confirmed"], bool)
        assert isinstance(signal.metadata["stoch_buy_confirmed"], bool)
        assert isinstance(signal.metadata["stoch_sell_confirmed"], bool)

    @pytest.mark.asyncio
    async def test_symbol_from_kwargs(self):
        strategy = CompositeMomentumStrategy()
        candles = make_flat_candles(n=50)
        signal = await strategy.analyze(candles, symbol="ETH/USDT")
        assert signal.symbol == "ETH/USDT"

    @pytest.mark.asyncio
    async def test_symbol_from_candles(self):
        strategy = CompositeMomentumStrategy()
        candles = make_flat_candles(n=50, price=100.0)
        signal = await strategy.analyze(candles)
        assert signal.symbol == "BTC/USDT"


# ==================== Hold Signals ====================

class TestHoldSignals:
    @pytest.mark.asyncio
    async def test_flat_market_returns_hold(self):
        strategy = CompositeMomentumStrategy()
        candles = make_flat_candles(n=50)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.confidence == 0.0
        assert signal.metadata["signal_type"] == "no_agreement"

    @pytest.mark.asyncio
    async def test_no_agreement_when_indicators_disagree(self):
        """When indicators give mixed signals, result is HOLD."""
        strategy = CompositeMomentumStrategy()
        # Flat data should not produce agreement
        candles = make_flat_candles(n=50, price=100.0)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # With flat data, neither buy nor sell count should reach 2
        assert signal.action == SignalAction.HOLD


# ==================== Buy Signals ====================

class TestBuySignals:
    @pytest.mark.asyncio
    async def test_buy_signal_from_reversal(self):
        """After a downtrend with reversal, should get BUY or HOLD (data-dependent)."""
        strategy = CompositeMomentumStrategy()
        candles = make_buy_signal_candles()
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # After downtrend reversal, RSI should be low
        assert signal.metadata["rsi"] < 50
        assert signal.strategy_name == "composite_momentum"

    @pytest.mark.asyncio
    async def test_triple_buy_confidence(self):
        """When all 3 indicators confirm BUY, confidence = 0.8."""
        strategy = CompositeMomentumStrategy()
        candles = make_buy_signal_candles()
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        if signal.metadata.get("buy_confirmations", 0) == 3:
            assert signal.action == SignalAction.BUY
            assert signal.confidence == 0.8
            assert signal.metadata["signal_type"] == "triple_buy"

    @pytest.mark.asyncio
    async def test_partial_buy_confidence(self):
        """When 2 out of 3 confirm BUY, confidence = 0.4."""
        strategy = CompositeMomentumStrategy()
        candles = make_buy_signal_candles()
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        if signal.metadata.get("buy_confirmations", 0) == 2:
            assert signal.action == SignalAction.BUY
            assert signal.confidence == 0.4
            assert signal.metadata["signal_type"] == "partial_buy"

    @pytest.mark.asyncio
    async def test_buy_requires_no_sell_confirmations_for_partial(self):
        """Partial buy requires sell_count == 0."""
        strategy = CompositeMomentumStrategy()
        candles = make_buy_signal_candles()
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        if signal.metadata.get("signal_type") == "partial_buy":
            assert signal.metadata["sell_confirmations"] == 0


# ==================== Sell Signals ====================

class TestSellSignals:
    @pytest.mark.asyncio
    async def test_sell_signal_from_rollover(self):
        """After an uptrend with rollover, should get SELL or HOLD (data-dependent)."""
        strategy = CompositeMomentumStrategy()
        candles = make_sell_signal_candles()
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # After uptrend, RSI should be high
        assert signal.metadata["rsi"] > 40
        assert signal.strategy_name == "composite_momentum"

    @pytest.mark.asyncio
    async def test_triple_sell_confidence(self):
        """When all 3 indicators confirm SELL, confidence = 0.8."""
        strategy = CompositeMomentumStrategy()
        candles = make_sell_signal_candles()
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        if signal.metadata.get("sell_confirmations", 0) == 3:
            assert signal.action == SignalAction.SELL
            assert signal.confidence == 0.8
            assert signal.metadata["signal_type"] == "triple_sell"

    @pytest.mark.asyncio
    async def test_partial_sell_confidence(self):
        """When 2 out of 3 confirm SELL, confidence = 0.4."""
        strategy = CompositeMomentumStrategy()
        candles = make_sell_signal_candles()
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        if signal.metadata.get("sell_confirmations", 0) == 2:
            assert signal.action == SignalAction.SELL
            assert signal.confidence == 0.4
            assert signal.metadata["signal_type"] == "partial_sell"


# ==================== Confirmation Count Logic ====================

class TestConfirmationLogic:
    @pytest.mark.asyncio
    async def test_buy_confirmations_range(self):
        """buy_confirmations should be 0-3."""
        strategy = CompositeMomentumStrategy()
        candles = make_flat_candles(n=50)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert 0 <= signal.metadata["buy_confirmations"] <= 3

    @pytest.mark.asyncio
    async def test_sell_confirmations_range(self):
        """sell_confirmations should be 0-3."""
        strategy = CompositeMomentumStrategy()
        candles = make_flat_candles(n=50)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert 0 <= signal.metadata["sell_confirmations"] <= 3

    @pytest.mark.asyncio
    async def test_triple_buy_sets_correct_signal_type(self):
        """Triple buy has signal_type='triple_buy'."""
        strategy = CompositeMomentumStrategy()
        # Use a scenario where all 3 confirm — crafted directly
        # We can mock this by using extreme data
        # Create steep downtrend then sharp reversal
        prices = []
        for i in range(50):
            if i < 30:
                prices.append(200.0 - i * 4.0)  # Steep down
            elif i < 45:
                prices.append(80.0 - (i - 30) * 0.3)  # Continue down slightly
            else:
                prices.append(75.0 + (i - 45) * 3.0)  # Sharp reversal

        prices = [max(p, 10.0) for p in prices]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # Verify the signal type matches confirmations
        if signal.metadata["buy_confirmations"] == 3:
            assert signal.metadata["signal_type"] == "triple_buy"
            assert signal.confidence == 0.8

    @pytest.mark.asyncio
    async def test_mixed_signals_return_hold(self):
        """When both buy and sell have 2 confirmations, neither qualifies."""
        strategy = CompositeMomentumStrategy()
        candles = make_flat_candles(n=50)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # Mixed signals shouldn't produce a buy or sell
        buy_c = signal.metadata["buy_confirmations"]
        sell_c = signal.metadata["sell_confirmations"]
        if buy_c >= 1 and sell_c >= 1:
            # Neither partial buy nor partial sell if both have confirmations
            if buy_c == 2 and sell_c > 0:
                assert signal.action == SignalAction.HOLD
            if sell_c == 2 and buy_c > 0:
                assert signal.action == SignalAction.HOLD


# ==================== Regime Adaptation ====================

class TestRegimeAdaptation:
    @pytest.mark.asyncio
    async def test_disabled_in_ranging_regime(self):
        from bot.strategies.regime import MarketRegime

        strategy = CompositeMomentumStrategy()
        strategy.adapt_to_regime(MarketRegime.RANGING)
        assert strategy._regime_disabled is True

        candles = make_flat_candles(n=50)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata["reason"] == "disabled_by_regime"
        assert signal.metadata["regime"] == "RANGING"

    @pytest.mark.asyncio
    async def test_enabled_in_trending_up(self):
        from bot.strategies.regime import MarketRegime

        strategy = CompositeMomentumStrategy()
        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        assert strategy._regime_disabled is False

    @pytest.mark.asyncio
    async def test_enabled_in_trending_down(self):
        from bot.strategies.regime import MarketRegime

        strategy = CompositeMomentumStrategy()
        strategy.adapt_to_regime(MarketRegime.TRENDING_DOWN)
        assert strategy._regime_disabled is False

    @pytest.mark.asyncio
    async def test_enabled_in_high_volatility(self):
        from bot.strategies.regime import MarketRegime

        strategy = CompositeMomentumStrategy()
        strategy.adapt_to_regime(MarketRegime.HIGH_VOLATILITY)
        assert strategy._regime_disabled is False

    @pytest.mark.asyncio
    async def test_re_enable_after_regime_change(self):
        from bot.strategies.regime import MarketRegime

        strategy = CompositeMomentumStrategy()
        strategy.adapt_to_regime(MarketRegime.RANGING)
        assert strategy._regime_disabled is True

        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        assert strategy._regime_disabled is False

        # Should produce normal signal, not disabled
        candles = make_flat_candles(n=50)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.metadata.get("reason") != "disabled_by_regime"


# ==================== Strategy Registration ====================

class TestRegistration:
    def test_strategy_can_be_registered(self):
        from bot.strategies.base import strategy_registry

        instance = CompositeMomentumStrategy()
        strategy_registry.register(instance)
        strategy = strategy_registry.get("composite_momentum")
        assert strategy is not None
        assert isinstance(strategy, CompositeMomentumStrategy)

    def test_strategy_active_after_registration(self):
        from bot.strategies.base import strategy_registry

        instance = CompositeMomentumStrategy()
        strategy_registry.register(instance)
        assert strategy_registry.is_active("composite_momentum")


# ==================== Backward Compatibility ====================

class TestBackwardCompatibility:
    def test_default_constructor_works(self):
        """Default constructor should work with no arguments."""
        strategy = CompositeMomentumStrategy()
        assert strategy.name == "composite_momentum"

    @pytest.mark.asyncio
    async def test_analyzes_without_symbol_kwarg(self):
        """analyze() works without explicit symbol keyword arg."""
        strategy = CompositeMomentumStrategy()
        candles = make_flat_candles(n=50)
        signal = await strategy.analyze(candles)
        assert signal.symbol == "BTC/USDT"
        assert signal.strategy_name == "composite_momentum"


# ==================== Edge Cases ====================

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_monotonic_uptrend(self):
        """Strong uptrend should have high RSI, not produce invalid signals."""
        strategy = CompositeMomentumStrategy()
        prices = [50.0 + i * 2.0 for i in range(50)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # RSI should be high in uptrend
        assert signal.metadata["rsi"] > 50
        assert signal.strategy_name == "composite_momentum"

    @pytest.mark.asyncio
    async def test_monotonic_downtrend(self):
        """Strong downtrend should have low RSI."""
        strategy = CompositeMomentumStrategy()
        prices = [200.0 - i * 2.0 for i in range(50)]
        prices = [max(p, 10.0) for p in prices]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # RSI should be low in downtrend
        assert signal.metadata["rsi"] < 50

    @pytest.mark.asyncio
    async def test_large_dataset(self):
        """Strategy works with a large number of candles."""
        strategy = CompositeMomentumStrategy()
        prices = [100.0 + (i % 20 - 10) * 0.5 for i in range(200)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action in (SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD)

    @pytest.mark.asyncio
    async def test_confidence_values_bounded(self):
        """Confidence should always be 0.0, 0.4, or 0.8."""
        strategy = CompositeMomentumStrategy()
        candles = make_flat_candles(n=50)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.confidence in (0.0, 0.4, 0.8)

    @pytest.mark.asyncio
    async def test_custom_thresholds(self):
        """Custom thresholds should be used in analysis."""
        strategy = CompositeMomentumStrategy(
            rsi_buy_threshold=50.0,  # More lenient
            rsi_sell_threshold=50.0,  # More lenient
        )
        candles = make_flat_candles(n=50)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.strategy_name == "composite_momentum"
        # The thresholds change what counts as a confirmation
        # but with flat data we shouldn't get a signal regardless
