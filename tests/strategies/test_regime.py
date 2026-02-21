"""Tests for MarketRegimeDetector and strategy regime adaptation."""

from datetime import datetime, timedelta

import pytest

from bot.models import OHLCV, SignalAction
from bot.strategies.regime import MarketRegime, MarketRegimeDetector
from bot.strategies.technical.bollinger import BollingerStrategy
from bot.strategies.technical.ma_crossover import MACrossoverStrategy
from bot.strategies.technical.rsi import RSIStrategy


def make_candles(
    prices: list[float],
    symbol: str = "BTC/USDT",
    volatility: float = 0.01,
) -> list[OHLCV]:
    """Create OHLCV candles from close prices with realistic OHLCV values."""
    base = datetime(2024, 1, 1)
    candles = []
    for i, p in enumerate(prices):
        high = p * (1 + volatility)
        low = p * (1 - volatility)
        open_price = p * (1 - volatility * 0.5)
        # Ensure constraints: high >= max(open, close), low <= min(open, close)
        high = max(high, open_price, p)
        low = min(low, open_price, p)
        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=i),
                open=open_price,
                high=high,
                low=low,
                close=p,
                volume=1000.0,
                symbol=symbol,
                timeframe="1h",
            )
        )
    return candles


def make_trending_up_candles(n: int = 50) -> list[OHLCV]:
    """Create candles with a strong uptrend (ADX > 25, +DI > -DI)."""
    prices = [100.0 + i * 2.0 for i in range(n)]
    return make_candles(prices, volatility=0.005)


def make_trending_down_candles(n: int = 50) -> list[OHLCV]:
    """Create candles with a strong downtrend (ADX > 25, -DI > +DI)."""
    prices = [200.0 - i * 2.0 for i in range(n)]
    return make_candles(prices, volatility=0.005)


def make_ranging_candles(n: int = 50) -> list[OHLCV]:
    """Create candles with ranging/sideways price action (low ADX, tight BB)."""
    # Oscillate around 100 with very small amplitude
    prices = [100.0 + (0.5 if i % 2 == 0 else -0.5) for i in range(n)]
    return make_candles(prices, volatility=0.002)


def make_volatile_candles(n: int = 50) -> list[OHLCV]:
    """Create candles with extreme volatility (ATR > 2x average)."""
    base = datetime(2024, 1, 1)
    candles = []
    # Start with calm period, then huge swings (all prices stay positive)
    for i in range(n):
        if i < n // 2:
            # Calm period — very tight range
            p = 100.0 + (0.5 if i % 2 == 0 else -0.5)
            vol = 0.003
        else:
            # Volatile period — large oscillations but always positive
            offset = i - n // 2
            p = 100.0 + offset * 8 * (1 if (offset % 2 == 0) else -1)
            p = max(p, 20.0)  # Floor at 20 to stay positive
            vol = 0.15  # 15% intrabar volatility
        high = p * (1 + vol)
        low = p * (1 - vol)
        open_price = p * (1 - vol * 0.3)
        high = max(high, open_price, p)
        low = min(low, open_price, p)
        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=i),
                open=open_price,
                high=high,
                low=low,
                close=p,
                volume=1000.0,
                symbol="BTC/USDT",
                timeframe="1h",
            )
        )
    return candles


class TestMarketRegime:
    def test_enum_values(self):
        assert MarketRegime.TRENDING_UP == "TRENDING_UP"
        assert MarketRegime.TRENDING_DOWN == "TRENDING_DOWN"
        assert MarketRegime.RANGING == "RANGING"
        assert MarketRegime.HIGH_VOLATILITY == "HIGH_VOLATILITY"

    def test_enum_is_string(self):
        assert isinstance(MarketRegime.TRENDING_UP, str)


class TestMarketRegimeDetector:
    @pytest.fixture
    def detector(self):
        return MarketRegimeDetector()

    def test_default_parameters(self, detector):
        assert detector._adx_period == 14
        assert detector._atr_period == 14
        assert detector._bb_period == 20
        assert detector._adx_trending_threshold == 25.0
        assert detector._adx_ranging_threshold == 20.0
        assert detector._atr_volatility_multiplier == 2.0

    def test_required_history_length(self, detector):
        assert detector.required_history_length == 25  # max(14,14,20) + 5

    def test_custom_parameters(self):
        detector = MarketRegimeDetector(
            adx_period=20,
            atr_period=20,
            bb_period=30,
            adx_trending_threshold=30.0,
            adx_ranging_threshold=15.0,
            atr_volatility_multiplier=3.0,
        )
        assert detector._adx_period == 20
        assert detector._bb_period == 30
        assert detector.required_history_length == 35  # max(20,20,30) + 5

    def test_insufficient_data(self, detector):
        candles = make_candles([100.0] * 5)
        result = detector.detect(candles)
        assert result == MarketRegime.RANGING

    def test_insufficient_data_details(self, detector):
        candles = make_candles([100.0] * 5)
        details = detector.detect_with_details(candles)
        assert details["regime"] == MarketRegime.RANGING
        assert details["sufficient_data"] is False

    def test_detect_trending_up(self, detector):
        candles = make_trending_up_candles(60)
        regime = detector.detect(candles)
        assert regime == MarketRegime.TRENDING_UP

    def test_detect_trending_down(self, detector):
        candles = make_trending_down_candles(60)
        regime = detector.detect(candles)
        assert regime == MarketRegime.TRENDING_DOWN

    def test_detect_ranging(self, detector):
        candles = make_ranging_candles(60)
        regime = detector.detect(candles)
        assert regime == MarketRegime.RANGING

    def test_detect_high_volatility(self, detector):
        candles = make_volatile_candles(60)
        regime = detector.detect(candles)
        assert regime == MarketRegime.HIGH_VOLATILITY

    def test_detect_with_details_has_all_fields(self, detector):
        candles = make_trending_up_candles(60)
        details = detector.detect_with_details(candles)
        assert "regime" in details
        assert "adx" in details
        assert "plus_di" in details
        assert "minus_di" in details
        assert "atr_ratio" in details
        assert "bb_width" in details
        assert details["sufficient_data"] is True

    def test_trending_up_has_positive_di(self, detector):
        candles = make_trending_up_candles(60)
        details = detector.detect_with_details(candles)
        assert details["plus_di"] > details["minus_di"]

    def test_trending_down_has_negative_di(self, detector):
        candles = make_trending_down_candles(60)
        details = detector.detect_with_details(candles)
        assert details["minus_di"] > details["plus_di"]

    def test_ranging_has_low_adx(self, detector):
        candles = make_ranging_candles(60)
        details = detector.detect_with_details(candles)
        assert details["adx"] < 25.0

    def test_volatile_has_high_atr_ratio(self, detector):
        candles = make_volatile_candles(60)
        details = detector.detect_with_details(candles)
        assert details["atr_ratio"] > 2.0


class TestMACrossoverRegimeAdaptation:
    @pytest.fixture
    def strategy(self):
        return MACrossoverStrategy(short_period=20, long_period=50)

    def test_default_not_disabled(self, strategy):
        assert strategy._regime_disabled is False

    def test_adapt_trending_up_uses_shorter_periods(self, strategy):
        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        assert strategy._short_period == 10
        assert strategy._long_period == 30
        assert strategy._regime_disabled is False

    def test_adapt_trending_down_uses_shorter_periods(self, strategy):
        strategy.adapt_to_regime(MarketRegime.TRENDING_DOWN)
        assert strategy._short_period == 10
        assert strategy._long_period == 30
        assert strategy._regime_disabled is False

    def test_adapt_ranging_disables(self, strategy):
        strategy.adapt_to_regime(MarketRegime.RANGING)
        assert strategy._regime_disabled is True

    def test_adapt_high_volatility_restores_defaults(self, strategy):
        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        strategy.adapt_to_regime(MarketRegime.HIGH_VOLATILITY)
        assert strategy._short_period == 20
        assert strategy._long_period == 50
        assert strategy._regime_disabled is False

    def test_adapt_ranging_then_trending_reenables(self, strategy):
        strategy.adapt_to_regime(MarketRegime.RANGING)
        assert strategy._regime_disabled is True
        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        assert strategy._regime_disabled is False

    @pytest.mark.asyncio
    async def test_disabled_returns_hold(self, strategy):
        strategy.adapt_to_regime(MarketRegime.RANGING)
        prices = [100 + i for i in range(60)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "disabled_by_regime"

    @pytest.mark.asyncio
    async def test_trending_uses_adapted_periods(self, strategy):
        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        # Strategy should now use periods 10/30
        assert strategy._short_period == 10
        assert strategy._long_period == 30
        # required_history_length should reflect adapted periods
        assert strategy.required_history_length == 31  # long_period + 1


class TestRSIRegimeAdaptation:
    @pytest.fixture
    def strategy(self):
        return RSIStrategy(period=14, overbought=70.0, oversold=30.0)

    def test_adapt_ranging_tighter_bounds(self, strategy):
        strategy.adapt_to_regime(MarketRegime.RANGING)
        assert strategy._overbought == 65.0
        assert strategy._oversold == 35.0

    def test_adapt_trending_standard_bounds(self, strategy):
        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        assert strategy._overbought == 70.0
        assert strategy._oversold == 30.0

    def test_adapt_trending_down_standard_bounds(self, strategy):
        strategy.adapt_to_regime(MarketRegime.TRENDING_DOWN)
        assert strategy._overbought == 70.0
        assert strategy._oversold == 30.0

    def test_adapt_high_volatility_restores_originals(self, strategy):
        strategy.adapt_to_regime(MarketRegime.RANGING)
        assert strategy._overbought == 65.0
        strategy.adapt_to_regime(MarketRegime.HIGH_VOLATILITY)
        assert strategy._overbought == 70.0
        assert strategy._oversold == 30.0

    def test_custom_thresholds_preserved(self):
        strategy = RSIStrategy(period=14, overbought=80.0, oversold=20.0)
        strategy.adapt_to_regime(MarketRegime.RANGING)
        assert strategy._overbought == 65.0
        strategy.adapt_to_regime(MarketRegime.HIGH_VOLATILITY)
        # Should restore to original custom values
        assert strategy._overbought == 80.0
        assert strategy._oversold == 20.0


class TestBollingerRegimeAdaptation:
    @pytest.fixture
    def strategy(self):
        return BollingerStrategy(period=20, std_dev=2.0)

    def test_default_not_disabled(self, strategy):
        assert strategy._regime_disabled is False

    def test_adapt_trending_up_disables(self, strategy):
        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        assert strategy._regime_disabled is True

    def test_adapt_trending_down_disables(self, strategy):
        strategy.adapt_to_regime(MarketRegime.TRENDING_DOWN)
        assert strategy._regime_disabled is True

    def test_adapt_ranging_enables(self, strategy):
        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        strategy.adapt_to_regime(MarketRegime.RANGING)
        assert strategy._regime_disabled is False

    def test_adapt_high_volatility_enables(self, strategy):
        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        strategy.adapt_to_regime(MarketRegime.HIGH_VOLATILITY)
        assert strategy._regime_disabled is False

    @pytest.mark.asyncio
    async def test_disabled_returns_hold(self, strategy):
        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        candles = make_candles([100.0] * 25)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "disabled_by_regime"

    @pytest.mark.asyncio
    async def test_enabled_analyzes_normally(self, strategy):
        strategy.adapt_to_regime(MarketRegime.RANGING)
        # Price drops below lower band — should give BUY signal
        candles = make_candles([100.0] * 22 + [60.0], volatility=0.01)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # Should not be disabled_by_regime
        assert signal.metadata.get("reason") != "disabled_by_regime"


class TestBaseStrategyAdaptToRegime:
    """Test that BaseStrategy.adapt_to_regime is a no-op by default."""

    def test_base_strategy_adapt_is_noop(self):
        """Strategies without adapt_to_regime override should not crash."""
        from bot.strategies.technical.macd import MACDStrategy

        strategy = MACDStrategy()
        # Should not raise
        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        strategy.adapt_to_regime(MarketRegime.RANGING)
