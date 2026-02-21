"""Tests for V2-010: Multi-timeframe trend filter."""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.config import Settings, TradingMode
from bot.main import TradingBot
from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.ensemble import SignalEnsemble
from bot.strategies.trend_filter import TrendDirection, TrendFilter


def make_candle(
    close: float,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: float = 1000.0,
    timestamp: datetime | None = None,
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
) -> OHLCV:
    """Create an OHLCV candle for testing."""
    if high is None:
        high = close * 1.01
    if low is None:
        low = close * 0.99
    if open_ is None:
        open_ = close
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    return OHLCV(
        timestamp=timestamp,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        symbol=symbol,
        timeframe=timeframe,
    )


def make_trending_up_candles(
    n: int = 50, start_price: float = 100.0, step: float = 2.0
) -> list[OHLCV]:
    """Create candles showing a clear uptrend (bullish candles).

    Each candle: open near bottom, close near top.
    Constraints: high >= max(open, close), low <= min(open, close).
    """
    candles = []
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        close = start_price + i * step
        open_ = close - step * 0.5
        high = close + step * 0.3
        low = open_ - step * 0.2
        candles.append(
            make_candle(
                close=close,
                high=high,
                low=low,
                open_=open_,
                timestamp=base_time + timedelta(hours=4 * i),
            )
        )
    return candles


def make_trending_down_candles(
    n: int = 50, start_price: float = 200.0, step: float = 2.0
) -> list[OHLCV]:
    """Create candles showing a clear downtrend (bearish candles).

    Each candle: open near top, close near bottom.
    Constraints: high >= max(open, close), low <= min(open, close).
    """
    candles = []
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        close = start_price - i * step
        open_ = close + step * 0.5
        high = open_ + step * 0.3
        low = close - step * 0.2
        candles.append(
            make_candle(
                close=close,
                high=high,
                low=low,
                open_=open_,
                timestamp=base_time + timedelta(hours=4 * i),
            )
        )
    return candles


def make_ranging_candles(
    n: int = 50, center: float = 100.0, amplitude: float = 1.0
) -> list[OHLCV]:
    """Create candles showing a ranging/sideways market.

    All candles oscillate around center with very small bodies.
    """
    candles = []
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        offset = amplitude * math.sin(i * 0.5)
        close = center + offset
        open_ = center + offset * 0.5
        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(open_, close) + amplitude * 0.3
        low = min(open_, close) - amplitude * 0.3
        candles.append(
            make_candle(
                close=close,
                high=high,
                low=low,
                open_=open_,
                timestamp=base_time + timedelta(hours=4 * i),
            )
        )
    return candles


def make_signal(
    strategy_name: str = "test",
    symbol: str = "BTC/USDT",
    action: SignalAction = SignalAction.HOLD,
    confidence: float = 0.5,
) -> TradingSignal:
    return TradingSignal(
        strategy_name=strategy_name,
        symbol=symbol,
        action=action,
        confidence=confidence,
    )


def make_settings(**kwargs):
    """Create test settings with safe defaults."""
    defaults = {
        "trading_mode": TradingMode.PAPER,
        "database_url": "sqlite+aiosqlite:///:memory:",
        "binance_api_key": "",
        "upbit_api_key": "",
        "symbols": ["BTC/USDT"],
        "loop_interval_seconds": 1,
        "signal_min_agreement": 1,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


class TestTrendFilter:
    """Test TrendFilter trend detection."""

    def test_bullish_trend_detected(self):
        """Uptrending candles should be detected as BULLISH."""
        tf = TrendFilter()
        candles = make_trending_up_candles(n=50)
        result = tf.get_trend("BTC/USDT", candles)
        assert result == TrendDirection.BULLISH

    def test_bearish_trend_detected(self):
        """Downtrending candles should be detected as BEARISH."""
        tf = TrendFilter()
        candles = make_trending_down_candles(n=50)
        result = tf.get_trend("BTC/USDT", candles)
        assert result == TrendDirection.BEARISH

    def test_neutral_trend_in_ranging_market(self):
        """Sideways/ranging candles should be detected as NEUTRAL."""
        tf = TrendFilter()
        candles = make_ranging_candles(n=50, amplitude=0.5)
        result = tf.get_trend("BTC/USDT", candles)
        assert result == TrendDirection.NEUTRAL

    def test_insufficient_data_returns_neutral(self):
        """Not enough candles should return NEUTRAL."""
        tf = TrendFilter()
        candles = make_trending_up_candles(n=5)
        result = tf.get_trend("BTC/USDT", candles)
        assert result == TrendDirection.NEUTRAL

    def test_empty_candles_returns_neutral(self):
        """Empty candle list should return NEUTRAL."""
        tf = TrendFilter()
        result = tf.get_trend("BTC/USDT", [])
        assert result == TrendDirection.NEUTRAL

    def test_required_history_length(self):
        """required_history_length should be at least sma_period + 5."""
        tf = TrendFilter(sma_period=20, adx_period=14)
        assert tf.required_history_length >= 25

    def test_custom_parameters(self):
        """Custom SMA/ADX parameters should work."""
        tf = TrendFilter(sma_period=10, adx_period=7, adx_trending_threshold=20.0)
        candles = make_trending_up_candles(n=30)
        result = tf.get_trend("BTC/USDT", candles)
        # With lower threshold, trend detection should be easier
        assert result in (TrendDirection.BULLISH, TrendDirection.NEUTRAL)

    def test_get_trend_details_returns_values(self):
        """get_trend_details should return dict with direction, adx, sma_slope."""
        tf = TrendFilter()
        candles = make_trending_up_candles(n=50)
        details = tf.get_trend_details("BTC/USDT", candles)

        assert "direction" in details
        assert "adx" in details
        assert "sma_slope" in details
        assert "sufficient_data" in details
        assert details["sufficient_data"] is True
        assert isinstance(details["adx"], float)
        assert isinstance(details["sma_slope"], float)

    def test_get_trend_details_insufficient_data(self):
        """get_trend_details with insufficient data returns neutral with flag."""
        tf = TrendFilter()
        candles = make_trending_up_candles(n=5)
        details = tf.get_trend_details("BTC/USDT", candles)

        assert details["direction"] == TrendDirection.NEUTRAL
        assert details["sufficient_data"] is False


class TestEnsembleTrendFiltering:
    """Test ensemble vote() with trend_direction parameter."""

    def test_buy_rejected_in_bearish_trend(self):
        """BUY signals should be rejected when trend is BEARISH."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.8),
            make_signal(strategy_name="s2", action=SignalAction.BUY, confidence=0.7),
        ]
        result = ensemble.vote(
            signals, "BTC/USDT", trend_direction=TrendDirection.BEARISH
        )
        assert result.action == SignalAction.HOLD
        assert result.metadata["reason"] == "trend_filter_rejected"
        assert result.metadata["rejected_action"] == "BUY"
        assert result.metadata["trend"] == "BEARISH"

    def test_buy_allowed_in_bullish_trend(self):
        """BUY signals should be allowed when trend is BULLISH."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.8),
            make_signal(strategy_name="s2", action=SignalAction.BUY, confidence=0.7),
        ]
        result = ensemble.vote(
            signals, "BTC/USDT", trend_direction=TrendDirection.BULLISH
        )
        assert result.action == SignalAction.BUY
        assert result.metadata.get("trend") == "BULLISH"

    def test_buy_allowed_in_neutral_trend(self):
        """BUY signals should be allowed when trend is NEUTRAL."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.8),
            make_signal(strategy_name="s2", action=SignalAction.BUY, confidence=0.7),
        ]
        result = ensemble.vote(
            signals, "BTC/USDT", trend_direction=TrendDirection.NEUTRAL
        )
        assert result.action == SignalAction.BUY

    def test_sell_rejected_in_bullish_trend(self):
        """SELL signals should be rejected when trend is BULLISH."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.SELL, confidence=0.8),
            make_signal(strategy_name="s2", action=SignalAction.SELL, confidence=0.7),
        ]
        result = ensemble.vote(
            signals, "BTC/USDT", trend_direction=TrendDirection.BULLISH
        )
        assert result.action == SignalAction.HOLD
        assert result.metadata["reason"] == "trend_filter_rejected"
        assert result.metadata["rejected_action"] == "SELL"

    def test_sell_allowed_in_bearish_trend(self):
        """SELL signals should be allowed when trend is BEARISH."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.SELL, confidence=0.8),
            make_signal(strategy_name="s2", action=SignalAction.SELL, confidence=0.7),
        ]
        result = ensemble.vote(
            signals, "BTC/USDT", trend_direction=TrendDirection.BEARISH
        )
        assert result.action == SignalAction.SELL

    def test_sell_allowed_in_neutral_trend(self):
        """SELL signals should be allowed when trend is NEUTRAL."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.SELL, confidence=0.8),
            make_signal(strategy_name="s2", action=SignalAction.SELL, confidence=0.7),
        ]
        result = ensemble.vote(
            signals, "BTC/USDT", trend_direction=TrendDirection.NEUTRAL
        )
        assert result.action == SignalAction.SELL

    def test_no_trend_direction_passes_all(self):
        """Without trend_direction, all signals pass through (backward compat)."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.8),
            make_signal(strategy_name="s2", action=SignalAction.BUY, confidence=0.7),
        ]
        result = ensemble.vote(signals, "BTC/USDT")
        assert result.action == SignalAction.BUY
        assert "trend" not in result.metadata

    def test_conflict_still_detected_with_trend(self):
        """Conflict detection still works when trend is provided."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.8),
            make_signal(strategy_name="s2", action=SignalAction.SELL, confidence=0.7),
        ]
        result = ensemble.vote(
            signals, "BTC/USDT", trend_direction=TrendDirection.BULLISH
        )
        assert result.action == SignalAction.HOLD
        assert result.metadata["reason"] == "conflict"

    def test_hold_signals_not_filtered_by_trend(self):
        """HOLD signals should not be affected by trend direction."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.HOLD),
            make_signal(strategy_name="s2", action=SignalAction.HOLD),
        ]
        result = ensemble.vote(
            signals, "BTC/USDT", trend_direction=TrendDirection.BEARISH
        )
        assert result.action == SignalAction.HOLD
        assert result.metadata["reason"] == "insufficient_agreement"


class TestConfigIntegration:
    """Test config settings for multi-timeframe."""

    def test_default_timeframes(self):
        """Default timeframes should be ['15m', '1h', '4h', '1d']."""
        settings = Settings(database_url="sqlite+aiosqlite:///:memory:")
        assert settings.timeframes == ["15m", "1h", "4h", "1d"]

    def test_custom_timeframes(self):
        """Custom timeframes should be accepted."""
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            timeframes=["5m", "15m", "1h"],
        )
        assert settings.timeframes == ["5m", "15m", "1h"]

    def test_default_trend_timeframe(self):
        """Default trend_timeframe should be '4h'."""
        settings = Settings(database_url="sqlite+aiosqlite:///:memory:")
        assert settings.trend_timeframe == "4h"

    def test_custom_trend_timeframe(self):
        """Custom trend_timeframe should be accepted."""
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            trend_timeframe="1d",
        )
        assert settings.trend_timeframe == "1d"


class TestTrendFilterMainIntegration:
    """Test TrendFilter wiring in TradingBot."""

    @pytest.mark.asyncio
    async def test_trend_filter_created_on_initialize(self):
        """TrendFilter should be created during bot initialization."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        assert bot._trend_filter is not None
        assert isinstance(bot._trend_filter, TrendFilter)

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_trading_cycle_fetches_trend_candles(self):
        """_trading_cycle should fetch higher-timeframe candles for trend."""
        settings = make_settings(trend_timeframe="4h")
        bot = TradingBot(settings=settings)
        await bot.initialize()
        bot._risk_manager.update_portfolio_value(10000.0)

        mock_candle = MagicMock()
        mock_candle.close = 50000.0
        mock_candle.high = 50500.0
        mock_candle.low = 49500.0
        mock_candle.open = 50000.0
        mock_candle.volume = 1000.0

        # Track calls to get_candles to verify trend timeframe was fetched
        call_args_list = []

        async def tracking_get_candles(*args, **kwargs):
            call_args_list.append((args, kwargs))
            return [mock_candle] * 200

        bot._store.get_candles = tracking_get_candles

        mock_strategy = MagicMock()
        mock_strategy.name = "test"
        mock_strategy.required_history_length = 1
        mock_strategy.analyze = AsyncMock(
            return_value=make_signal(action=SignalAction.HOLD)
        )

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]
            await bot._trading_cycle()

        # Should have called get_candles at least with trend_timeframe="4h"
        timeframe_calls = [
            c for c in call_args_list if c[1].get("timeframe") == "4h"
        ]
        assert len(timeframe_calls) > 0

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_buy_blocked_by_bearish_trend_in_cycle(self):
        """BUY should be rejected when higher-timeframe trend is BEARISH."""
        settings = make_settings(signal_min_agreement=1)
        bot = TradingBot(settings=settings)
        await bot.initialize()
        bot._risk_manager.update_portfolio_value(10000.0)

        mock_candle = MagicMock()
        mock_candle.close = 50000.0

        # Mock trend filter to return BEARISH
        bot._trend_filter.get_trend = MagicMock(return_value=TrendDirection.BEARISH)

        async def mock_get_candles(*args, **kwargs):
            return [mock_candle] * 200

        bot._store.get_candles = mock_get_candles

        mock_strategy = MagicMock()
        mock_strategy.name = "test"
        mock_strategy.required_history_length = 1
        mock_strategy.analyze = AsyncMock(
            return_value=make_signal(
                strategy_name="test", action=SignalAction.BUY, confidence=0.8
            )
        )

        mock_engine = AsyncMock()
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]
            await bot._trading_cycle()

        # BUY should be blocked — no trade executed
        mock_engine.execute_signal.assert_not_called()

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_collector_uses_configured_timeframes(self):
        """DataCollector should be initialized with configured timeframes."""
        settings = make_settings(timeframes=["5m", "15m", "1h", "4h"])
        bot = TradingBot(settings=settings)
        await bot.initialize()

        assert bot._collector._timeframes == ["5m", "15m", "1h", "4h"]

        await bot.shutdown()
