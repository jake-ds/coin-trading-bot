"""Tests for V2-014: Bollinger Band squeeze breakout strategy."""

from datetime import datetime, timedelta

import pytest

from bot.models import OHLCV, SignalAction
from bot.strategies.technical.bollinger import BollingerStrategy


def make_candles(
    prices: list[float],
    symbol: str = "BTC/USDT",
    volumes: list[float] | None = None,
) -> list[OHLCV]:
    """Create OHLCV candles from close prices and optional volumes."""
    base = datetime(2024, 1, 1)
    if volumes is None:
        volumes = [1000.0] * len(prices)
    return [
        OHLCV(
            timestamp=base + timedelta(hours=i),
            open=p * 0.99,
            high=p * 1.01,
            low=p * 0.98,
            close=p,
            volume=v,
            symbol=symbol,
            timeframe="1h",
        )
        for i, (p, v) in enumerate(zip(prices, volumes))
    ]


def make_squeeze_candles(
    stable_count: int = 50,
    stable_price: float = 100.0,
    squeeze_count: int = 8,
    squeeze_price: float = 100.0,
    breakout_price: float | None = None,
    breakout_volume: float = 5000.0,
    normal_volume: float = 1000.0,
    squeeze_variation: float = 0.001,
) -> list[OHLCV]:
    """Create candle data that produces a Bollinger squeeze followed by optional breakout.

    - stable_count candles at varying prices (normal volatility)
    - squeeze_count candles at very tight prices (squeeze / low volatility)
    - Optional breakout candle at breakout_price with high volume
    """
    base = datetime(2024, 1, 1)
    candles: list[OHLCV] = []

    # Phase 1: Normal volatility (wide bands)
    for i in range(stable_count):
        # Create meaningful price variation so BB width is significant
        variation = 3.0 * (1 if i % 2 == 0 else -1) + (i % 5) * 0.5
        p = stable_price + variation
        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=i),
                open=p * 0.99,
                high=p * 1.02,
                low=p * 0.97,
                close=p,
                volume=normal_volume,
                symbol="BTC/USDT",
                timeframe="1h",
            )
        )

    # Phase 2: Squeeze (very tight prices → BB width contracts)
    for j in range(squeeze_count):
        idx = stable_count + j
        tiny_var = squeeze_variation * (1 if j % 2 == 0 else -1)
        p = squeeze_price + tiny_var
        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=idx),
                open=p * 0.999,
                high=p * 1.001,
                low=p * 0.999,
                close=p,
                volume=normal_volume,
                symbol="BTC/USDT",
                timeframe="1h",
            )
        )

    # Phase 3: Optional breakout candle
    if breakout_price is not None:
        idx = stable_count + squeeze_count
        # Breakout candle needs proper OHLCV constraints
        if breakout_price > squeeze_price:
            # Upward breakout
            bp_open = squeeze_price
            bp_high = breakout_price * 1.01
            bp_low = squeeze_price * 0.99
            bp_close = breakout_price
        else:
            # Downward breakout
            bp_open = squeeze_price
            bp_high = squeeze_price * 1.01
            bp_low = breakout_price * 0.99
            bp_close = breakout_price

        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=idx),
                open=bp_open,
                high=bp_high,
                low=bp_low,
                close=bp_close,
                volume=breakout_volume,
                symbol="BTC/USDT",
                timeframe="1h",
            )
        )

    return candles


class TestBollingerBackwardCompat:
    """Ensure existing mean_reversion behavior is preserved."""

    @pytest.fixture
    def strategy(self):
        return BollingerStrategy(period=10, std_dev=2.0)

    def test_default_mode_is_mean_reversion(self, strategy):
        assert strategy._mode == "mean_reversion"

    def test_name(self, strategy):
        assert strategy.name == "bollinger"

    def test_required_history_length_mean_reversion(self, strategy):
        assert strategy.required_history_length == 11

    @pytest.mark.asyncio
    async def test_price_below_lower_band_buy(self, strategy):
        prices = [100.0] * 12 + [60.0]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY
        assert signal.metadata["mode"] == "mean_reversion"

    @pytest.mark.asyncio
    async def test_price_above_upper_band_sell(self, strategy):
        prices = [100.0] * 12 + [140.0]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.SELL
        assert signal.metadata["mode"] == "mean_reversion"

    @pytest.mark.asyncio
    async def test_price_within_bands_hold(self, strategy):
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


class TestBollingerSqueezeDetection:
    """Test squeeze detection logic."""

    @pytest.fixture
    def strategy(self):
        return BollingerStrategy(
            period=20,
            std_dev=2.0,
            mode="squeeze",
            squeeze_candles_required=5,
            breakout_volume_multiplier=1.5,
            cooldown_candles=10,
        )

    def test_squeeze_mode_name(self, strategy):
        assert strategy.name == "bollinger"
        assert strategy._mode == "squeeze"

    def test_required_history_length_squeeze(self, strategy):
        # 2 * period + squeeze_candles_required = 2 * 20 + 5 = 45
        assert strategy.required_history_length == 45

    @pytest.mark.asyncio
    async def test_insufficient_data(self, strategy):
        prices = [100.0] * 10
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "insufficient_data"

    @pytest.mark.asyncio
    async def test_no_squeeze_returns_hold(self, strategy):
        """Volatile data without squeeze should return HOLD."""
        # Wide price swings — no squeeze
        prices = [100.0 + (i % 10) * 5.0 for i in range(50)]
        candles = make_candles(prices)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "no_squeeze"
        assert signal.metadata["mode"] == "squeeze"

    @pytest.mark.asyncio
    async def test_squeeze_detected(self, strategy):
        """Squeeze is detected when BB width contracts below average."""
        candles = make_squeeze_candles(
            stable_count=50, squeeze_count=8, breakout_price=None
        )
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.metadata["in_squeeze"] is True
        assert signal.metadata["squeeze_count"] >= 5
        # No breakout → HOLD
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "squeeze_no_breakout"

    @pytest.mark.asyncio
    async def test_squeeze_metadata(self, strategy):
        """Squeeze mode includes squeeze-specific metadata."""
        candles = make_squeeze_candles(stable_count=50, squeeze_count=8)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert "squeeze_count" in signal.metadata
        assert "in_squeeze" in signal.metadata
        assert "avg_band_width" in signal.metadata
        assert "volume_ratio" in signal.metadata
        assert "volume_confirmed" in signal.metadata


class TestBollingerSqueezeBreakout:
    """Test breakout detection after squeeze."""

    @pytest.fixture
    def strategy(self):
        return BollingerStrategy(
            period=20,
            std_dev=2.0,
            mode="squeeze",
            squeeze_candles_required=5,
            breakout_volume_multiplier=1.5,
            cooldown_candles=10,
        )

    @pytest.mark.asyncio
    async def test_upward_breakout_with_volume(self, strategy):
        """Breakout above upper band with volume → BUY."""
        candles = make_squeeze_candles(
            stable_count=50,
            squeeze_count=8,
            breakout_price=120.0,
            breakout_volume=5000.0,
            normal_volume=1000.0,
        )
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY
        assert signal.confidence > 0.5
        assert signal.metadata["breakout_direction"] == "up"
        assert signal.metadata["volume_confirmed"] is True

    @pytest.mark.asyncio
    async def test_downward_breakout_with_volume(self, strategy):
        """Breakout below lower band with volume → SELL."""
        candles = make_squeeze_candles(
            stable_count=50,
            squeeze_count=8,
            breakout_price=80.0,
            breakout_volume=5000.0,
            normal_volume=1000.0,
        )
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.SELL
        assert signal.confidence > 0.5
        assert signal.metadata["breakout_direction"] == "down"
        assert signal.metadata["volume_confirmed"] is True

    @pytest.mark.asyncio
    async def test_breakout_without_volume_rejected(self, strategy):
        """Breakout without volume confirmation → HOLD."""
        candles = make_squeeze_candles(
            stable_count=50,
            squeeze_count=8,
            breakout_price=120.0,
            breakout_volume=1000.0,  # Same as normal — no volume spike
            normal_volume=1000.0,
        )
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "breakout_no_volume"
        assert signal.metadata["breakout_direction"] == "up"

    @pytest.mark.asyncio
    async def test_downward_breakout_without_volume_rejected(self, strategy):
        """Downward breakout without volume confirmation → HOLD."""
        candles = make_squeeze_candles(
            stable_count=50,
            squeeze_count=8,
            breakout_price=80.0,
            breakout_volume=1000.0,
            normal_volume=1000.0,
        )
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "breakout_no_volume"
        assert signal.metadata["breakout_direction"] == "down"


class TestBollingerSqueezeDuration:
    """Test squeeze duration requirement."""

    @pytest.mark.asyncio
    async def test_short_squeeze_rejected(self):
        """Squeeze lasting fewer than required candles → no breakout signal."""
        strategy = BollingerStrategy(
            period=20,
            std_dev=2.0,
            mode="squeeze",
            squeeze_candles_required=5,
            breakout_volume_multiplier=1.5,
            cooldown_candles=0,
        )
        # Only 3 squeeze candles — below the 5-candle requirement
        candles = make_squeeze_candles(
            stable_count=50,
            squeeze_count=3,
            breakout_price=120.0,
            breakout_volume=5000.0,
            normal_volume=1000.0,
        )
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        # Squeeze not established → HOLD (no_squeeze or breakout ignored)
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_exact_minimum_squeeze_accepted(self):
        """Squeeze lasting exactly the required candles → breakout accepted."""
        strategy = BollingerStrategy(
            period=20,
            std_dev=2.0,
            mode="squeeze",
            squeeze_candles_required=5,
            breakout_volume_multiplier=1.5,
            cooldown_candles=0,
        )
        candles = make_squeeze_candles(
            stable_count=50,
            squeeze_count=6,  # Need 6: 5 before + breakout at the end
            breakout_price=120.0,
            breakout_volume=5000.0,
            normal_volume=1000.0,
        )
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY


class TestBollingerCooldown:
    """Test cooldown mechanism."""

    @pytest.mark.asyncio
    async def test_cooldown_after_signal(self):
        """After a breakout signal, cooldown prevents next signal."""
        strategy = BollingerStrategy(
            period=20,
            std_dev=2.0,
            mode="squeeze",
            squeeze_candles_required=5,
            breakout_volume_multiplier=1.5,
            cooldown_candles=10,
        )
        # First call: breakout → BUY
        candles = make_squeeze_candles(
            stable_count=50,
            squeeze_count=8,
            breakout_price=120.0,
            breakout_volume=5000.0,
            normal_volume=1000.0,
        )
        signal1 = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal1.action == SignalAction.BUY

        # Second call immediately: should be in cooldown
        signal2 = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal2.action == SignalAction.HOLD
        assert signal2.metadata.get("reason") == "cooldown"

    @pytest.mark.asyncio
    async def test_cooldown_expires(self):
        """After cooldown period, signals are allowed again."""
        strategy = BollingerStrategy(
            period=20,
            std_dev=2.0,
            mode="squeeze",
            squeeze_candles_required=5,
            breakout_volume_multiplier=1.5,
            cooldown_candles=3,
        )
        # First call: breakout
        candles = make_squeeze_candles(
            stable_count=50,
            squeeze_count=8,
            breakout_price=120.0,
            breakout_volume=5000.0,
            normal_volume=1000.0,
        )
        signal1 = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal1.action == SignalAction.BUY

        # Calls 2, 3, 4: in cooldown (candles_since = 1, 2, 3)
        for _ in range(3):
            signal = await strategy.analyze(candles, symbol="BTC/USDT")
            assert signal.action == SignalAction.HOLD
            assert signal.metadata.get("reason") == "cooldown"

        # Call 5: cooldown expired (candles_since = 4 > 3)
        signal5 = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal5.action != SignalAction.HOLD or signal5.metadata.get("reason") != "cooldown"

    @pytest.mark.asyncio
    async def test_cooldown_zero_disables(self):
        """cooldown_candles=0 means no cooldown."""
        strategy = BollingerStrategy(
            period=20,
            std_dev=2.0,
            mode="squeeze",
            squeeze_candles_required=5,
            breakout_volume_multiplier=1.5,
            cooldown_candles=0,
        )
        candles = make_squeeze_candles(
            stable_count=50,
            squeeze_count=8,
            breakout_price=120.0,
            breakout_volume=5000.0,
            normal_volume=1000.0,
        )
        signal1 = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal1.action == SignalAction.BUY

        # Second call immediately: no cooldown
        signal2 = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal2.action == SignalAction.BUY


class TestBollingerSqueezeConfidence:
    """Test confidence calculation for squeeze breakout."""

    @pytest.mark.asyncio
    async def test_confidence_scales_with_volume(self):
        """Higher volume → higher confidence."""
        strategy_low = BollingerStrategy(
            period=20, std_dev=2.0, mode="squeeze",
            squeeze_candles_required=5, breakout_volume_multiplier=1.5,
            cooldown_candles=0,
        )
        strategy_high = BollingerStrategy(
            period=20, std_dev=2.0, mode="squeeze",
            squeeze_candles_required=5, breakout_volume_multiplier=1.5,
            cooldown_candles=0,
        )

        candles_low_vol = make_squeeze_candles(
            stable_count=50, squeeze_count=8,
            breakout_price=120.0, breakout_volume=2000.0, normal_volume=1000.0,
        )
        candles_high_vol = make_squeeze_candles(
            stable_count=50, squeeze_count=8,
            breakout_price=120.0, breakout_volume=10000.0, normal_volume=1000.0,
        )

        sig_low = await strategy_low.analyze(candles_low_vol, symbol="BTC/USDT")
        sig_high = await strategy_high.analyze(candles_high_vol, symbol="BTC/USDT")

        assert sig_low.action == SignalAction.BUY
        assert sig_high.action == SignalAction.BUY
        assert sig_high.confidence >= sig_low.confidence

    @pytest.mark.asyncio
    async def test_confidence_capped_at_one(self):
        """Confidence never exceeds 1.0."""
        strategy = BollingerStrategy(
            period=20, std_dev=2.0, mode="squeeze",
            squeeze_candles_required=5, breakout_volume_multiplier=1.5,
            cooldown_candles=0,
        )
        candles = make_squeeze_candles(
            stable_count=50, squeeze_count=8,
            breakout_price=150.0, breakout_volume=50000.0, normal_volume=1000.0,
        )
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.confidence <= 1.0


class TestBollingerSqueezeRegime:
    """Test regime adaptation with squeeze mode."""

    @pytest.mark.asyncio
    async def test_regime_disabled_in_squeeze_mode(self):
        """Regime disable works in squeeze mode too."""
        from bot.strategies.regime import MarketRegime

        strategy = BollingerStrategy(
            period=20, std_dev=2.0, mode="squeeze",
            squeeze_candles_required=5, breakout_volume_multiplier=1.5,
            cooldown_candles=0,
        )
        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)

        candles = make_squeeze_candles(
            stable_count=50, squeeze_count=8,
            breakout_price=120.0, breakout_volume=5000.0,
        )
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata.get("reason") == "disabled_by_regime"

    @pytest.mark.asyncio
    async def test_regime_enabled_in_ranging(self):
        """Squeeze mode works in ranging regime."""
        from bot.strategies.regime import MarketRegime

        strategy = BollingerStrategy(
            period=20, std_dev=2.0, mode="squeeze",
            squeeze_candles_required=5, breakout_volume_multiplier=1.5,
            cooldown_candles=0,
        )
        strategy.adapt_to_regime(MarketRegime.RANGING)

        candles = make_squeeze_candles(
            stable_count=50, squeeze_count=8,
            breakout_price=120.0, breakout_volume=5000.0,
        )
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY


class TestBollingerModeConfig:
    """Test mode configuration."""

    def test_mode_parameter(self):
        s1 = BollingerStrategy(mode="mean_reversion")
        assert s1._mode == "mean_reversion"

        s2 = BollingerStrategy(mode="squeeze")
        assert s2._mode == "squeeze"

    def test_squeeze_params_configurable(self):
        s = BollingerStrategy(
            mode="squeeze",
            squeeze_candles_required=10,
            breakout_volume_multiplier=2.0,
            cooldown_candles=15,
        )
        assert s._squeeze_candles_required == 10
        assert s._breakout_volume_multiplier == 2.0
        assert s._cooldown_candles == 15

    def test_default_constructor_uses_mean_reversion(self):
        """Default BollingerStrategy() uses mean_reversion for backward compat."""
        strategy = BollingerStrategy()
        assert strategy._mode == "mean_reversion"
