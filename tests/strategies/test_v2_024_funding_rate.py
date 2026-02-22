"""Tests for FundingRateStrategy (V2-024)."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from bot.data.funding import FundingRateMonitor
from bot.models import OHLCV, SignalAction
from bot.strategies.technical.funding_rate import FundingRateStrategy


def make_candles(n: int = 5, symbol: str = "BTC/USDT:USDT") -> list[OHLCV]:
    """Create minimal OHLCV candles for strategy calls."""
    candles = []
    base = datetime(2026, 2, 22, 0, 0, 0)
    for i in range(n):
        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=i),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50000.0,
                volume=1000.0,
                symbol=symbol,
                timeframe="1h",
            )
        )
    return candles


@pytest.fixture
def strategy():
    """Create a FundingRateStrategy with default params."""
    return FundingRateStrategy()


@pytest.fixture
def custom_strategy():
    """Create a FundingRateStrategy with custom thresholds."""
    return FundingRateStrategy(
        extreme_positive_rate=0.001,
        extreme_negative_rate=-0.001,
        confidence_scale_factor=5.0,
        min_confidence=0.2,
        max_confidence=0.8,
    )


@pytest.fixture
def mock_monitor():
    """Create a mock FundingRateMonitor."""
    exchange = MagicMock()
    exchange.name = "binance"
    monitor = FundingRateMonitor(exchange=exchange, store=None)
    return monitor


class TestFundingRateStrategyProperties:
    """Test strategy properties."""

    def test_name(self, strategy):
        assert strategy.name == "funding_rate"

    def test_required_history_length(self, strategy):
        assert strategy.required_history_length == 1


class TestFundingRateSignals:
    """Test signal generation based on funding rates."""

    @pytest.mark.asyncio
    async def test_extreme_positive_funding_generates_sell(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.001
        )
        assert signal.action == SignalAction.SELL
        assert signal.confidence > 0
        assert signal.metadata["signal_type"] == "extreme_positive_funding"
        assert signal.metadata["funding_rate"] == 0.001

    @pytest.mark.asyncio
    async def test_extreme_negative_funding_generates_buy(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=-0.001
        )
        assert signal.action == SignalAction.BUY
        assert signal.confidence > 0
        assert signal.metadata["signal_type"] == "extreme_negative_funding"
        assert signal.metadata["funding_rate"] == -0.001

    @pytest.mark.asyncio
    async def test_normal_funding_generates_hold(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.0001
        )
        assert signal.action == SignalAction.HOLD
        assert signal.confidence == 0.0
        assert signal.metadata["reason"] == "funding_rate_normal"

    @pytest.mark.asyncio
    async def test_no_funding_data_generates_hold(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(candles, symbol="BTC/USDT:USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata["reason"] == "no_funding_data"

    @pytest.mark.asyncio
    async def test_zero_funding_rate_is_hold(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.0
        )
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_exactly_at_positive_threshold_is_hold(self, strategy):
        candles = make_candles()
        # Exactly at threshold — not extreme
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.0005
        )
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_exactly_at_negative_threshold_is_hold(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=-0.0003
        )
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_slightly_above_positive_threshold(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.00051
        )
        assert signal.action == SignalAction.SELL

    @pytest.mark.asyncio
    async def test_slightly_below_negative_threshold(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=-0.00031
        )
        assert signal.action == SignalAction.BUY


class TestConfidenceScaling:
    """Test confidence calculation."""

    @pytest.mark.asyncio
    async def test_higher_deviation_higher_confidence(self, strategy):
        candles = make_candles()

        # Small deviation
        sig1 = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.0006
        )
        # Large deviation
        sig2 = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.005
        )

        assert sig2.confidence > sig1.confidence

    @pytest.mark.asyncio
    async def test_confidence_capped_at_max(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.1
        )
        assert signal.confidence <= strategy._max_confidence

    @pytest.mark.asyncio
    async def test_confidence_at_least_min(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.00051
        )
        assert signal.confidence >= strategy._min_confidence

    @pytest.mark.asyncio
    async def test_custom_thresholds(self, custom_strategy):
        candles = make_candles()
        # Below custom positive threshold (0.001) → HOLD
        signal = await custom_strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.0005
        )
        assert signal.action == SignalAction.HOLD

        # Above custom positive threshold → SELL
        signal = await custom_strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.002
        )
        assert signal.action == SignalAction.SELL
        assert signal.confidence <= 0.8  # Custom max


class TestFundingDataDict:
    """Test funding_data dict input."""

    @pytest.mark.asyncio
    async def test_funding_data_dict(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles,
            symbol="BTC/USDT:USDT",
            funding_data={
                "funding_rate": 0.001,
                "mark_price": 50000.0,
                "spot_price": 49950.0,
                "spread_pct": 0.1,
            },
        )
        assert signal.action == SignalAction.SELL
        assert signal.metadata["mark_price"] == 50000.0
        assert signal.metadata["spot_price"] == 49950.0
        assert signal.metadata["spread_pct"] == 0.1

    @pytest.mark.asyncio
    async def test_funding_rate_kwarg_takes_precedence(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles,
            symbol="BTC/USDT:USDT",
            funding_rate=0.001,
            funding_data={
                "funding_rate": 0.0001,  # Would be HOLD
                "mark_price": 50000.0,
                "spot_price": 49950.0,
            },
        )
        # funding_rate kwarg (0.001 = SELL) takes precedence
        assert signal.action == SignalAction.SELL


class TestMonitorIntegration:
    """Test integration with FundingRateMonitor."""

    @pytest.mark.asyncio
    async def test_reads_from_monitor(self, strategy, mock_monitor):
        strategy.set_funding_monitor(mock_monitor)
        mock_monitor.update_rate("BTC/USDT:USDT", 0.001)

        candles = make_candles()
        signal = await strategy.analyze(candles, symbol="BTC/USDT:USDT")
        assert signal.action == SignalAction.SELL

    @pytest.mark.asyncio
    async def test_monitor_no_data_returns_hold(self, strategy, mock_monitor):
        strategy.set_funding_monitor(mock_monitor)
        # No rate data for this symbol

        candles = make_candles()
        signal = await strategy.analyze(candles, symbol="ETH/USDT:USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.metadata["reason"] == "no_funding_data"

    @pytest.mark.asyncio
    async def test_kwarg_overrides_monitor(self, strategy, mock_monitor):
        strategy.set_funding_monitor(mock_monitor)
        mock_monitor.update_rate("BTC/USDT:USDT", 0.0001)  # Normal → HOLD

        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.001  # Extreme → SELL
        )
        assert signal.action == SignalAction.SELL


class TestRateTrend:
    """Test rate trend analysis."""

    @pytest.mark.asyncio
    async def test_rising_trend_boosts_sell_confidence(self, strategy, mock_monitor):
        strategy.set_funding_monitor(mock_monitor)

        # Create rising rate history
        for i in range(5):
            ts = datetime(2026, 2, 22) + timedelta(hours=8 * i)
            mock_monitor.update_rate(
                "BTC/USDT:USDT", 0.0003 + 0.0001 * i, ts
            )

        candles = make_candles()
        # Extreme positive funding with rising trend
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.001
        )
        assert signal.action == SignalAction.SELL
        assert signal.metadata["rate_trend"] == "rising"

    @pytest.mark.asyncio
    async def test_falling_trend_boosts_buy_confidence(self, strategy, mock_monitor):
        strategy.set_funding_monitor(mock_monitor)

        # Create falling rate history
        for i in range(5):
            ts = datetime(2026, 2, 22) + timedelta(hours=8 * i)
            mock_monitor.update_rate(
                "BTC/USDT:USDT", -0.0001 - 0.0001 * i, ts
            )

        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=-0.001
        )
        assert signal.action == SignalAction.BUY
        assert signal.metadata["rate_trend"] == "falling"

    @pytest.mark.asyncio
    async def test_no_trend_without_monitor(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.001
        )
        assert signal.metadata["rate_trend"] is None

    @pytest.mark.asyncio
    async def test_trend_disabled(self):
        strategy = FundingRateStrategy(use_rate_trend=False)
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.001
        )
        assert signal.metadata["rate_trend"] is None


class TestSpreadSignal:
    """Test perpetual-spot spread signal enhancement."""

    @pytest.mark.asyncio
    async def test_positive_spread_enhances_sell(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles,
            symbol="BTC/USDT:USDT",
            funding_rate=0.001,
            funding_data={
                "funding_rate": 0.001,
                "mark_price": 50500.0,
                "spot_price": 50000.0,
                "spread_pct": 1.0,
            },
        )
        assert signal.action == SignalAction.SELL
        assert signal.metadata["spread_signal"] == "perp_premium"

    @pytest.mark.asyncio
    async def test_negative_spread_enhances_buy(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles,
            symbol="BTC/USDT:USDT",
            funding_rate=-0.001,
            funding_data={
                "funding_rate": -0.001,
                "mark_price": 49500.0,
                "spot_price": 50000.0,
                "spread_pct": -1.0,
            },
        )
        assert signal.action == SignalAction.BUY
        assert signal.metadata["spread_signal"] == "perp_discount"

    @pytest.mark.asyncio
    async def test_no_spread_signal_when_below_threshold(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles,
            symbol="BTC/USDT:USDT",
            funding_rate=0.001,
            funding_data={
                "funding_rate": 0.001,
                "mark_price": 50010.0,
                "spot_price": 50000.0,
                "spread_pct": 0.02,
            },
        )
        assert signal.action == SignalAction.SELL
        assert "spread_signal" not in signal.metadata


class TestRegimeAdaptation:
    """Test market regime adaptation."""

    @pytest.mark.asyncio
    async def test_disabled_in_high_volatility(self, strategy):
        from bot.strategies.regime import MarketRegime

        strategy.adapt_to_regime(MarketRegime.HIGH_VOLATILITY)
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.01
        )
        assert signal.action == SignalAction.HOLD
        assert signal.metadata["reason"] == "disabled_by_regime"

    @pytest.mark.asyncio
    async def test_enabled_in_trending(self, strategy):
        from bot.strategies.regime import MarketRegime

        strategy.adapt_to_regime(MarketRegime.TRENDING_UP)
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.001
        )
        assert signal.action == SignalAction.SELL

    @pytest.mark.asyncio
    async def test_enabled_in_ranging(self, strategy):
        from bot.strategies.regime import MarketRegime

        strategy.adapt_to_regime(MarketRegime.RANGING)
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=-0.001
        )
        assert signal.action == SignalAction.BUY

    @pytest.mark.asyncio
    async def test_re_enabled_after_high_volatility(self, strategy):
        from bot.strategies.regime import MarketRegime

        strategy.adapt_to_regime(MarketRegime.HIGH_VOLATILITY)
        strategy.adapt_to_regime(MarketRegime.TRENDING_DOWN)

        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.001
        )
        assert signal.action == SignalAction.SELL


class TestSymbolExtraction:
    """Test symbol extraction from various sources."""

    @pytest.mark.asyncio
    async def test_symbol_from_kwargs(self, strategy):
        candles = make_candles(symbol="ETH/USDT:USDT")
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.0001
        )
        assert signal.symbol == "BTC/USDT:USDT"

    @pytest.mark.asyncio
    async def test_symbol_from_candles(self, strategy):
        candles = make_candles(symbol="ETH/USDT:USDT")
        signal = await strategy.analyze(candles, funding_rate=0.0001)
        assert signal.symbol == "ETH/USDT:USDT"

    @pytest.mark.asyncio
    async def test_symbol_unknown_with_empty_candles(self, strategy):
        signal = await strategy.analyze([], funding_rate=0.0001)
        assert signal.symbol == "UNKNOWN"


class TestStrategyRegistration:
    """Test that the strategy can be registered with the registry."""

    def test_strategy_can_register(self):
        from bot.strategies.base import strategy_registry

        # Registry may be cleared by other tests, so register explicitly
        strat = FundingRateStrategy()
        strategy_registry.register(strat)
        assert strategy_registry.get("funding_rate") is not None
        assert strategy_registry.get("funding_rate").name == "funding_rate"


class TestMetadata:
    """Test metadata completeness."""

    @pytest.mark.asyncio
    async def test_sell_signal_metadata(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles,
            symbol="BTC/USDT:USDT",
            funding_rate=0.001,
            funding_data={
                "funding_rate": 0.001,
                "mark_price": 50000.0,
                "spot_price": 49950.0,
                "spread_pct": 0.1,
            },
        )
        assert "funding_rate" in signal.metadata
        assert "extreme_positive_threshold" in signal.metadata
        assert "extreme_negative_threshold" in signal.metadata
        assert "mark_price" in signal.metadata
        assert "spot_price" in signal.metadata
        assert "spread_pct" in signal.metadata
        assert "signal_type" in signal.metadata
        assert "deviation" in signal.metadata
        assert "rate_trend" in signal.metadata

    @pytest.mark.asyncio
    async def test_hold_signal_metadata(self, strategy):
        candles = make_candles()
        signal = await strategy.analyze(
            candles, symbol="BTC/USDT:USDT", funding_rate=0.0001
        )
        assert signal.metadata["reason"] == "funding_rate_normal"
        assert "funding_rate" in signal.metadata
        assert "extreme_positive_threshold" in signal.metadata
        assert "extreme_negative_threshold" in signal.metadata


class TestConfigSettings:
    """Test config settings integration."""

    def test_funding_rate_settings_exist(self):
        from bot.config import Settings

        settings = Settings(
            funding_extreme_positive_rate=0.001,
            funding_extreme_negative_rate=-0.001,
            funding_confidence_scale=5.0,
            funding_spread_threshold_pct=1.0,
            funding_rate_history_limit=100,
        )
        assert settings.funding_extreme_positive_rate == 0.001
        assert settings.funding_extreme_negative_rate == -0.001
        assert settings.funding_confidence_scale == 5.0
        assert settings.funding_spread_threshold_pct == 1.0
        assert settings.funding_rate_history_limit == 100

    def test_funding_rate_default_settings(self):
        from bot.config import Settings

        settings = Settings()
        assert settings.funding_extreme_positive_rate == 0.0005
        assert settings.funding_extreme_negative_rate == -0.0003
        assert settings.funding_confidence_scale == 10.0
        assert settings.funding_spread_threshold_pct == 0.5
        assert settings.funding_rate_history_limit == 50
