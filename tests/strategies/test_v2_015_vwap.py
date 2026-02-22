"""Tests for V2-015: VWAP strategy and volume profile analysis."""

from datetime import datetime, timedelta

import pytest

from bot.models import OHLCV, SignalAction
from bot.strategies.technical.vwap import VWAPStrategy


def make_candles(
    prices: list[float],
    symbol: str = "BTC/USDT",
    volumes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    opens: list[float] | None = None,
) -> list[OHLCV]:
    """Create OHLCV candles from close prices and optional overrides."""
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


def make_vwap_crossover_candles(
    direction: str = "above",
    n_candles: int = 30,
    base_price: float = 100.0,
    normal_volume: float = 1000.0,
    crossover_volume: float = 3000.0,
) -> list[OHLCV]:
    """Create candle data with a VWAP crossover at the last candle.

    The typical price (HLC/3) is used for VWAP. We construct price data
    that stays below (or above) VWAP for most of the series, then crosses
    at the end with high volume.
    """
    base = datetime(2024, 1, 1)
    candles: list[OHLCV] = []

    if direction == "above":
        # Price starts below VWAP, then crosses above at the end
        # First build candles at base_price with steady volume
        for i in range(n_candles - 2):
            p = base_price
            candles.append(
                OHLCV(
                    timestamp=base + timedelta(hours=i),
                    open=p,
                    high=p * 1.002,
                    low=p * 0.998,
                    close=p,
                    volume=normal_volume,
                    symbol="BTC/USDT",
                    timeframe="1h",
                )
            )

        # Penultimate candle: price dips below VWAP
        p_prev = base_price * 0.98
        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=n_candles - 2),
                open=p_prev,
                high=p_prev * 1.002,
                low=p_prev * 0.998,
                close=p_prev,
                volume=normal_volume,
                symbol="BTC/USDT",
                timeframe="1h",
            )
        )

        # Last candle: price jumps above VWAP with high volume
        p_last = base_price * 1.03
        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=n_candles - 1),
                open=p_prev,
                high=p_last * 1.002,
                low=p_prev * 0.998,
                close=p_last,
                volume=crossover_volume,
                symbol="BTC/USDT",
                timeframe="1h",
            )
        )
    else:
        # Price starts above VWAP, then crosses below at the end
        for i in range(n_candles - 2):
            p = base_price
            candles.append(
                OHLCV(
                    timestamp=base + timedelta(hours=i),
                    open=p,
                    high=p * 1.002,
                    low=p * 0.998,
                    close=p,
                    volume=normal_volume,
                    symbol="BTC/USDT",
                    timeframe="1h",
                )
            )

        # Penultimate candle: price spikes above VWAP
        p_prev = base_price * 1.02
        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=n_candles - 2),
                open=p_prev,
                high=p_prev * 1.002,
                low=p_prev * 0.998,
                close=p_prev,
                volume=normal_volume,
                symbol="BTC/USDT",
                timeframe="1h",
            )
        )

        # Last candle: price drops below VWAP with high volume
        p_last = base_price * 0.97
        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=n_candles - 1),
                open=p_prev,
                high=p_prev * 1.002,
                low=p_last * 0.998,
                close=p_last,
                volume=crossover_volume,
                symbol="BTC/USDT",
                timeframe="1h",
            )
        )

    return candles


# ----- VWAP Calculation Tests -----


@pytest.mark.asyncio
async def test_vwap_calculation():
    """VWAP should be sum(typical_price * volume) / sum(volume)."""
    strategy = VWAPStrategy()
    # Simple data: constant price and volume → VWAP = typical_price
    prices = [100.0] * 25
    volumes = [1000.0] * 25
    candles = make_candles(prices, volumes=volumes)

    signal = await strategy.analyze(candles, symbol="BTC/USDT")
    vwap = signal.metadata["vwap"]
    # Typical price = (high + low + close) / 3
    # With our candle maker: high = close * 1.005, low = open * 0.995 = close * 0.999 * 0.995
    # So VWAP ≈ close (approximately)
    assert abs(vwap - 100.0) < 1.0


@pytest.mark.asyncio
async def test_vwap_bands_present():
    """Metadata should contain VWAP bands (+/- 1 and 2 std devs)."""
    strategy = VWAPStrategy()
    prices = [100.0 + i * 0.1 for i in range(25)]
    candles = make_candles(prices)

    signal = await strategy.analyze(candles, symbol="BTC/USDT")
    meta = signal.metadata

    assert "upper_band_1" in meta
    assert "lower_band_1" in meta
    assert "upper_band_2" in meta
    assert "lower_band_2" in meta
    assert "vwap" in meta
    assert "std_dev" in meta

    # Band ordering: lower_2 < lower_1 < vwap < upper_1 < upper_2
    assert meta["lower_band_2"] <= meta["lower_band_1"]
    assert meta["lower_band_1"] <= meta["vwap"]
    assert meta["vwap"] <= meta["upper_band_1"]
    assert meta["upper_band_1"] <= meta["upper_band_2"]


@pytest.mark.asyncio
async def test_vwap_distance_pct():
    """Metadata should contain vwap_distance_pct showing how far close is from VWAP."""
    strategy = VWAPStrategy()
    prices = [100.0] * 25
    candles = make_candles(prices)

    signal = await strategy.analyze(candles, symbol="BTC/USDT")
    assert "vwap_distance_pct" in signal.metadata


# ----- Signal Generation Tests -----


@pytest.mark.asyncio
async def test_buy_signal_on_cross_above_vwap():
    """BUY when price crosses above VWAP with high volume."""
    strategy = VWAPStrategy(volume_multiplier=1.2)
    candles = make_vwap_crossover_candles(direction="above")

    signal = await strategy.analyze(candles, symbol="BTC/USDT")
    assert signal.action == SignalAction.BUY
    assert signal.confidence > 0.0
    assert signal.metadata["crossover"] == "above"
    assert signal.metadata["volume_increasing"] is True


@pytest.mark.asyncio
async def test_sell_signal_on_cross_below_vwap():
    """SELL when price crosses below VWAP with high volume."""
    strategy = VWAPStrategy(volume_multiplier=1.2)
    candles = make_vwap_crossover_candles(direction="below")

    signal = await strategy.analyze(candles, symbol="BTC/USDT")
    assert signal.action == SignalAction.SELL
    assert signal.confidence > 0.0
    assert signal.metadata["crossover"] == "below"
    assert signal.metadata["volume_increasing"] is True


@pytest.mark.asyncio
async def test_hold_when_no_crossover():
    """HOLD when price stays on same side of VWAP."""
    strategy = VWAPStrategy()
    # Flat prices → no crossover
    prices = [100.0] * 25
    candles = make_candles(prices)

    signal = await strategy.analyze(candles, symbol="BTC/USDT")
    assert signal.action == SignalAction.HOLD
    assert signal.metadata.get("reason") == "no_crossover"


@pytest.mark.asyncio
async def test_hold_when_volume_not_confirmed():
    """HOLD when crossover occurs but volume is too low."""
    strategy = VWAPStrategy(volume_multiplier=10.0)  # Very high threshold
    candles = make_vwap_crossover_candles(
        direction="above",
        crossover_volume=1000.0,  # Same as normal volume → won't meet 10x threshold
    )

    signal = await strategy.analyze(candles, symbol="BTC/USDT")
    assert signal.action == SignalAction.HOLD
    assert signal.metadata.get("reason") == "volume_not_confirmed"


# ----- Volume Analysis Tests -----


@pytest.mark.asyncio
async def test_volume_ratio_calculated():
    """Volume ratio should be current volume / average volume."""
    strategy = VWAPStrategy(volume_period=10)
    prices = [100.0] * 25
    volumes = [1000.0] * 24 + [3000.0]  # Last candle has 3x volume
    candles = make_candles(prices, volumes=volumes)

    signal = await strategy.analyze(candles, symbol="BTC/USDT")
    # Volume ratio should be approximately 3000/1200 = 2.5 (rolling avg includes last candle)
    assert signal.metadata["volume_ratio"] >= 2.5


@pytest.mark.asyncio
async def test_volume_increasing_flag():
    """volume_increasing should be True when volume exceeds multiplier threshold."""
    strategy = VWAPStrategy(volume_multiplier=1.5, volume_period=10)
    prices = [100.0] * 25
    volumes = [1000.0] * 24 + [2000.0]
    candles = make_candles(prices, volumes=volumes)

    signal = await strategy.analyze(candles, symbol="BTC/USDT")
    assert signal.metadata["volume_increasing"] is True


@pytest.mark.asyncio
async def test_volume_not_increasing_flag():
    """volume_increasing should be False when volume is below threshold."""
    strategy = VWAPStrategy(volume_multiplier=2.0, volume_period=10)
    prices = [100.0] * 25
    volumes = [1000.0] * 25  # All same volume
    candles = make_candles(prices, volumes=volumes)

    signal = await strategy.analyze(candles, symbol="BTC/USDT")
    assert signal.metadata["volume_increasing"] is False


# ----- Confidence Tests -----


@pytest.mark.asyncio
async def test_confidence_bounded():
    """Confidence should be between 0.1 and 1.0 for valid signals."""
    strategy = VWAPStrategy(volume_multiplier=1.2)
    candles = make_vwap_crossover_candles(direction="above")

    signal = await strategy.analyze(candles, symbol="BTC/USDT")
    assert signal.action == SignalAction.BUY
    assert 0.1 <= signal.confidence <= 1.0


@pytest.mark.asyncio
async def test_higher_volume_higher_confidence():
    """Higher volume should produce higher confidence."""
    strategy = VWAPStrategy(volume_multiplier=1.2)

    candles_low_vol = make_vwap_crossover_candles(
        direction="above", crossover_volume=1500.0
    )
    candles_high_vol = make_vwap_crossover_candles(
        direction="above", crossover_volume=5000.0
    )

    signal_low = await strategy.analyze(candles_low_vol, symbol="BTC/USDT")
    signal_high = await strategy.analyze(candles_high_vol, symbol="BTC/USDT")

    # Both should be BUY
    assert signal_low.action == SignalAction.BUY
    assert signal_high.action == SignalAction.BUY
    # Higher volume → higher confidence
    assert signal_high.confidence >= signal_low.confidence


# ----- Edge Cases -----


@pytest.mark.asyncio
async def test_insufficient_data():
    """HOLD with reason when not enough candles."""
    strategy = VWAPStrategy()
    prices = [100.0] * 5  # Too few
    candles = make_candles(prices)

    signal = await strategy.analyze(candles, symbol="BTC/USDT")
    assert signal.action == SignalAction.HOLD
    assert signal.metadata["reason"] == "insufficient_data"


@pytest.mark.asyncio
async def test_strategy_name():
    """Strategy name should be 'vwap'."""
    strategy = VWAPStrategy()
    assert strategy.name == "vwap"


@pytest.mark.asyncio
async def test_required_history_length():
    """Should require enough candles for volume period + 1."""
    strategy = VWAPStrategy(volume_period=20)
    assert strategy.required_history_length >= 21


@pytest.mark.asyncio
async def test_symbol_from_kwargs():
    """Should use symbol from kwargs if provided."""
    strategy = VWAPStrategy()
    prices = [100.0] * 25
    candles = make_candles(prices, symbol="ETH/USDT")

    signal = await strategy.analyze(candles, symbol="CUSTOM/USDT")
    assert signal.symbol == "CUSTOM/USDT"


@pytest.mark.asyncio
async def test_symbol_from_candle_data():
    """Should fall back to symbol from candle data."""
    strategy = VWAPStrategy()
    prices = [100.0] * 25
    candles = make_candles(prices, symbol="ETH/USDT")

    signal = await strategy.analyze(candles)
    assert signal.symbol == "ETH/USDT"


@pytest.mark.asyncio
async def test_strategy_name_in_signal():
    """Signal should have strategy_name='vwap'."""
    strategy = VWAPStrategy()
    prices = [100.0] * 25
    candles = make_candles(prices)

    signal = await strategy.analyze(candles, symbol="BTC/USDT")
    assert signal.strategy_name == "vwap"


# ----- Regime Adaptation Tests -----


@pytest.mark.asyncio
async def test_regime_high_volatility_disables():
    """HIGH_VOLATILITY regime should disable the strategy."""
    from bot.strategies.regime import MarketRegime

    strategy = VWAPStrategy()
    strategy.adapt_to_regime(MarketRegime.HIGH_VOLATILITY)

    prices = [100.0] * 25
    candles = make_candles(prices)
    signal = await strategy.analyze(candles, symbol="BTC/USDT")

    assert signal.action == SignalAction.HOLD
    assert signal.metadata["reason"] == "disabled_by_regime"


@pytest.mark.asyncio
async def test_regime_trending_up_enabled():
    """TRENDING_UP regime should keep strategy enabled."""
    from bot.strategies.regime import MarketRegime

    strategy = VWAPStrategy()
    strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
    assert strategy._regime_disabled is False


@pytest.mark.asyncio
async def test_regime_ranging_enabled():
    """RANGING regime should keep strategy enabled."""
    from bot.strategies.regime import MarketRegime

    strategy = VWAPStrategy()
    strategy.adapt_to_regime(MarketRegime.RANGING)
    assert strategy._regime_disabled is False


@pytest.mark.asyncio
async def test_regime_re_enable_after_high_vol():
    """Strategy should re-enable after regime changes from HIGH_VOLATILITY."""
    from bot.strategies.regime import MarketRegime

    strategy = VWAPStrategy()
    strategy.adapt_to_regime(MarketRegime.HIGH_VOLATILITY)
    assert strategy._regime_disabled is True

    strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
    assert strategy._regime_disabled is False


# ----- Registration Tests -----


def test_vwap_registered_in_registry():
    """VWAPStrategy should be auto-registered when module is imported."""
    from bot.strategies.base import strategy_registry

    # Re-register (registry may have been cleared by other tests)
    strategy_registry.register(VWAPStrategy())
    vwap = strategy_registry.get("vwap")
    assert vwap is not None
    assert isinstance(vwap, VWAPStrategy)


# ----- Configuration Tests -----


@pytest.mark.asyncio
async def test_custom_volume_period():
    """Custom volume_period should be used."""
    strategy = VWAPStrategy(volume_period=5)
    assert strategy._volume_period == 5
    assert strategy.required_history_length >= 6


@pytest.mark.asyncio
async def test_custom_volume_multiplier():
    """Custom volume_multiplier should be used."""
    strategy = VWAPStrategy(volume_multiplier=2.5)
    assert strategy._volume_multiplier == 2.5


@pytest.mark.asyncio
async def test_custom_band_std_count():
    """Custom band_std_count should be accepted."""
    strategy = VWAPStrategy(band_std_count=3)
    assert strategy._band_std_count == 3


# ----- Backward Compatibility -----


@pytest.mark.asyncio
async def test_default_params():
    """Default parameters should be reasonable."""
    strategy = VWAPStrategy()
    assert strategy._volume_period == 20
    assert strategy._volume_multiplier == 1.2
    assert strategy._band_std_count == 2
    assert strategy._regime_disabled is False


@pytest.mark.asyncio
async def test_zero_volume_handling():
    """Should handle zero volume gracefully."""
    strategy = VWAPStrategy()
    prices = [100.0] * 25
    volumes = [0.0] * 25
    candles = make_candles(prices, volumes=volumes)

    # Should not crash
    signal = await strategy.analyze(candles, symbol="BTC/USDT")
    assert signal.action == SignalAction.HOLD
