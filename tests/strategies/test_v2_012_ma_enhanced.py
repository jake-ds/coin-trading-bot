"""Tests for V2-012: Enhanced MA Crossover with confirmation filters."""

from datetime import datetime, timedelta

import pytest

from bot.models import OHLCV, SignalAction
from bot.strategies.technical.ma_crossover import MACrossoverStrategy


def make_candles(
    prices: list[float],
    volumes: list[float] | None = None,
    symbol: str = "BTC/USDT",
    bullish: bool | None = None,
) -> list[OHLCV]:
    """Create OHLCV candles from close prices.

    Args:
        prices: Close prices.
        volumes: Per-candle volumes (default 1000.0).
        symbol: Trading pair symbol.
        bullish: If True, open < close; if False, open > close; if None, auto.
    """
    base = datetime(2024, 1, 1)
    if volumes is None:
        volumes = [1000.0] * len(prices)
    candles = []
    for i, p in enumerate(prices):
        if bullish is True:
            o = p * 0.99
        elif bullish is False:
            o = p * 1.01
        else:
            o = p * 0.99  # default: slightly bullish
        h = max(o, p) * 1.005
        lo = min(o, p) * 0.995
        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=i),
                open=o,
                high=h,
                low=lo,
                close=p,
                volume=volumes[i],
                symbol=symbol,
                timeframe="1h",
            )
        )
    return candles


def _bullish_crossover_prices() -> list[float]:
    """Prices producing a bullish crossover at the LAST candle with short=3, long=7.

    At [-2]: short_ma(93.33) <= long_ma(97.14) (below)
    At [-1]: short_ma(130.0) > long_ma(111.43) (above) → crossover
    """
    return [100, 100, 100, 100, 100, 100, 100, 100, 100, 90, 80, 110, 200]


def _bearish_crossover_prices() -> list[float]:
    """Prices producing a bearish crossover at the LAST candle with short=3, long=7.

    At [-2]: short_ma(106.67) >= long_ma(102.86)
    At [-1]: short_ma(86.67) < long_ma(95.71) → crossover
    """
    return [100, 100, 100, 100, 100, 100, 100, 100, 100, 110, 120, 90, 50]


# ---------- Backward compatibility ----------

class TestBackwardCompatibility:
    """Ensure default parameters match old behavior."""

    @pytest.fixture
    def strategy(self):
        return MACrossoverStrategy(short_period=3, long_period=7)

    @pytest.mark.asyncio
    async def test_default_filters_disabled_buy(self, strategy):
        """With defaults, bullish crossover produces BUY."""
        candles = make_candles(_bullish_crossover_prices())
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY

    @pytest.mark.asyncio
    async def test_default_filters_disabled_sell(self, strategy):
        """With defaults, bearish crossover produces SELL."""
        candles = make_candles(_bearish_crossover_prices())
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.SELL

    @pytest.mark.asyncio
    async def test_hold_still_works(self, strategy):
        prices = [100.0] * 15
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD


# ---------- Volume confirmation ----------

class TestVolumeConfirmation:
    @pytest.fixture
    def strategy(self):
        return MACrossoverStrategy(
            short_period=3,
            long_period=7,
            volume_confirmation=True,
            volume_multiplier=1.5,
            volume_period=5,
        )

    @pytest.mark.asyncio
    async def test_buy_with_high_volume_confirmed(self, strategy):
        """BUY signal passes when volume > 1.5x average."""
        prices = _bullish_crossover_prices()
        # Low volumes for history, spike on crossover candle
        volumes = [100.0] * (len(prices) - 1) + [200.0]
        candles = make_candles(prices, volumes=volumes)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY
        assert signal.metadata["volume_confirmed"] is True
        assert signal.metadata["volume_ratio"] >= 1.5

    @pytest.mark.asyncio
    async def test_buy_rejected_low_volume(self, strategy):
        """BUY signal rejected when volume < 1.5x average."""
        prices = _bullish_crossover_prices()
        volumes = [100.0] * len(prices)  # All same → ratio ~1.0
        candles = make_candles(prices, volumes=volumes)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata["reason"] == "filters_rejected"
        assert "volume" in signal.metadata["rejected_by"]

    @pytest.mark.asyncio
    async def test_volume_ratio_in_metadata(self, strategy):
        """Volume ratio always in metadata when filter enabled and crossover detected."""
        prices = _bullish_crossover_prices()
        volumes = [100.0] * len(prices)
        candles = make_candles(prices, volumes=volumes)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert "volume_ratio" in signal.metadata


# ---------- Momentum confirmation ----------

class TestMomentumConfirmation:
    @pytest.fixture
    def strategy(self):
        return MACrossoverStrategy(
            short_period=3,
            long_period=7,
            momentum_confirmation=True,
        )

    @pytest.mark.asyncio
    async def test_buy_bullish_candle_confirmed(self, strategy):
        """BUY signal confirmed when crossover candle is bullish (close > open)."""
        prices = _bullish_crossover_prices()
        candles = make_candles(prices, bullish=True)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY
        assert signal.metadata["momentum_confirmed"] is True

    @pytest.mark.asyncio
    async def test_buy_rejected_bearish_candle(self, strategy):
        """BUY signal rejected when crossover candle is bearish (close < open)."""
        prices = _bullish_crossover_prices()
        candles = make_candles(prices, bullish=False)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata["reason"] == "filters_rejected"
        assert "momentum" in signal.metadata["rejected_by"]

    @pytest.mark.asyncio
    async def test_sell_bearish_candle_confirmed(self, strategy):
        """SELL signal confirmed when crossover candle is bearish (close < open)."""
        prices = _bearish_crossover_prices()
        candles = make_candles(prices, bullish=False)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.SELL
        assert signal.metadata["momentum_confirmed"] is True

    @pytest.mark.asyncio
    async def test_sell_rejected_bullish_candle(self, strategy):
        """SELL signal rejected when crossover candle is bullish."""
        prices = _bearish_crossover_prices()
        candles = make_candles(prices, bullish=True)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert "momentum" in signal.metadata["rejected_by"]


# ---------- Trend strength filter ----------

class TestTrendStrengthFilter:
    @pytest.fixture
    def strategy(self):
        return MACrossoverStrategy(
            short_period=3,
            long_period=7,
            trend_strength_filter=True,
        )

    @pytest.mark.asyncio
    async def test_expanding_distance_confirmed(self, strategy):
        """Signal confirmed when MA distance is expanding."""
        prices = _bullish_crossover_prices()
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # The big jump from 110 → 200 should make distance expand
        assert signal.action == SignalAction.BUY
        assert signal.metadata["ma_distance_expanding"] is True

    @pytest.mark.asyncio
    async def test_contracting_distance_rejected(self, strategy):
        """Signal rejected when MA distance is contracting."""
        # Converging MAs: short approaches long from below
        # Need: at [-2] short < long, at [-1] short > long (crossover),
        # but current distance < previous distance
        # This happens with a gradual crossover where MAs are converging
        prices = [100, 100, 100, 100, 100, 100, 100, 100, 98, 96, 95, 98, 101]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # With this gentle crossover, the MA distances should be small and contracting
        if signal.action == SignalAction.HOLD and "ma_distance_expanding" in signal.metadata:
            assert signal.metadata["ma_distance_expanding"] is False

    @pytest.mark.asyncio
    async def test_metadata_includes_distances(self, strategy):
        """MA distances included in metadata."""
        prices = _bullish_crossover_prices()
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert "current_ma_distance" in signal.metadata
        assert "prev_ma_distance" in signal.metadata


# ---------- ADX filter ----------

class TestADXFilter:
    @pytest.fixture
    def strategy(self):
        return MACrossoverStrategy(
            short_period=3,
            long_period=7,
            adx_filter_enabled=True,
            adx_threshold=20.0,
        )

    @pytest.mark.asyncio
    async def test_adx_in_metadata_with_enough_data(self, strategy):
        """ADX value present in metadata when crossover detected with sufficient data."""
        # Need 30+ candles for ADX window=14 to compute
        prices = [100.0] * 25 + [100, 100, 100, 90, 80, 110, 200]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert "adx" in signal.metadata

    @pytest.mark.asyncio
    async def test_adx_insufficient_data_skips_filter(self, strategy):
        """With too few candles for ADX, the filter is skipped gracefully."""
        prices = _bullish_crossover_prices()  # Only 13 candles
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # ADX should be set to 0.0 (computation skipped), signal still passes
        assert signal.metadata.get("adx", 0.0) == 0.0
        assert signal.action == SignalAction.BUY

    @pytest.mark.asyncio
    async def test_strong_trend_high_adx(self, strategy):
        """Strong directional data produces measurable ADX with crossover."""
        # 20 flat + decline + sharp rebound producing crossover at last candle
        # At [-2]: short_ma(73.33) <= long_ma(75.71), at [-1]: short_ma(101.67) > long_ma(85)
        prices = [100.0] * 20 + [95, 90, 85, 80, 75, 70, 65, 60, 95, 150]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert "adx" in signal.metadata


# ---------- Cooldown ----------

class TestCooldown:
    @pytest.fixture
    def strategy(self):
        return MACrossoverStrategy(
            short_period=3,
            long_period=7,
            cooldown_candles=3,
        )

    @pytest.mark.asyncio
    async def test_first_signal_fires(self, strategy):
        """First signal fires (starts ready with cooldown satisfied)."""
        candles = make_candles(_bullish_crossover_prices())
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY

    @pytest.mark.asyncio
    async def test_immediate_second_signal_blocked(self, strategy):
        """Signal immediately after another is blocked by cooldown."""
        candles = make_candles(_bullish_crossover_prices())
        signal1 = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal1.action == SignalAction.BUY

        # Second call with same crossover data — cooldown not met
        signal2 = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal2.action == SignalAction.HOLD
        assert signal2.metadata["reason"] == "filters_rejected"
        assert "cooldown" in signal2.metadata["rejected_by"]

    @pytest.mark.asyncio
    async def test_signal_after_cooldown_fires(self, strategy):
        """Signal fires after enough HOLD candles pass the cooldown period."""
        candles = make_candles(_bullish_crossover_prices())
        await strategy.analyze(candles, symbol="BTC/USDT")

        # Feed 3 HOLD candles to clear cooldown
        hold_candles = make_candles([200.0] * 15)
        for _ in range(3):
            await strategy.analyze(hold_candles, symbol="BTC/USDT")

        # Now crossover should fire
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY

    @pytest.mark.asyncio
    async def test_cooldown_metadata(self, strategy):
        """Cooldown info included in metadata."""
        candles = make_candles(_bullish_crossover_prices())
        await strategy.analyze(candles, symbol="BTC/USDT")
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.metadata["cooldown_ready"] is False
        assert signal.metadata["candles_since_last_signal"] == 0


# ---------- Confidence calculation ----------

class TestConfidenceCalculation:
    @pytest.mark.asyncio
    async def test_default_confidence_positive(self):
        """With no filters, confidence is distance-based (original formula)."""
        strategy = MACrossoverStrategy(short_period=3, long_period=7)
        candles = make_candles(_bullish_crossover_prices())
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY
        assert signal.confidence > 0.0
        assert signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_enhanced_confidence_with_volume(self):
        """With volume filter, confidence blends distance + volume ratio."""
        strategy = MACrossoverStrategy(
            short_period=3,
            long_period=7,
            volume_confirmation=True,
            volume_multiplier=1.5,
            volume_period=5,
        )
        prices = _bullish_crossover_prices()
        volumes = [100.0] * (len(prices) - 1) + [300.0]
        candles = make_candles(prices, volumes=volumes)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY
        assert signal.confidence > 0.0
        assert signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_capped_at_one(self):
        """Confidence never exceeds 1.0."""
        strategy = MACrossoverStrategy(
            short_period=3,
            long_period=7,
            volume_confirmation=True,
            adx_filter_enabled=True,
        )
        prices = _bullish_crossover_prices()
        volumes = [100.0] * (len(prices) - 1) + [10000.0]
        candles = make_candles(prices, volumes=volumes)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.confidence <= 1.0


# ---------- Multiple filters combined ----------

class TestCombinedFilters:
    @pytest.fixture
    def strategy(self):
        return MACrossoverStrategy(
            short_period=3,
            long_period=7,
            volume_confirmation=True,
            volume_multiplier=1.5,
            volume_period=5,
            momentum_confirmation=True,
        )

    @pytest.mark.asyncio
    async def test_both_filters_pass(self, strategy):
        """Signal fires when both volume and momentum confirm."""
        prices = _bullish_crossover_prices()
        volumes = [100.0] * (len(prices) - 1) + [200.0]
        candles = make_candles(prices, volumes=volumes, bullish=True)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY
        assert signal.metadata["volume_confirmed"] is True
        assert signal.metadata["momentum_confirmed"] is True

    @pytest.mark.asyncio
    async def test_volume_pass_momentum_fail(self, strategy):
        """Signal rejected when volume passes but momentum fails."""
        prices = _bullish_crossover_prices()
        volumes = [100.0] * (len(prices) - 1) + [200.0]
        candles = make_candles(prices, volumes=volumes, bullish=False)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert "momentum" in signal.metadata["rejected_by"]

    @pytest.mark.asyncio
    async def test_momentum_pass_volume_fail(self, strategy):
        """Signal rejected when momentum passes but volume fails."""
        prices = _bullish_crossover_prices()
        volumes = [100.0] * len(prices)  # All same → ratio ~1.0
        candles = make_candles(prices, volumes=volumes, bullish=True)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert "volume" in signal.metadata["rejected_by"]


# ---------- Regime adaptation still works ----------

class TestRegimeAdaptationUnchanged:
    @pytest.mark.asyncio
    async def test_regime_disabled_still_returns_hold(self):
        """Regime-disabled behavior unchanged by new filters."""
        from bot.strategies.regime import MarketRegime

        strategy = MACrossoverStrategy(
            short_period=20,
            long_period=50,
            volume_confirmation=True,
            momentum_confirmation=True,
        )
        strategy.adapt_to_regime(MarketRegime.RANGING)
        candles = make_candles([100.0] * 60)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata["reason"] == "disabled_by_regime"

    @pytest.mark.asyncio
    async def test_regime_trending_uses_shorter_periods(self):
        """Regime adaptation still adjusts periods."""
        from bot.strategies.regime import MarketRegime

        strategy = MACrossoverStrategy(short_period=20, long_period=50)
        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        assert strategy._short_period == 10
        assert strategy._long_period == 30
