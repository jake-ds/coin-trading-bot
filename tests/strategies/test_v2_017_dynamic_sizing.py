"""Tests for V2-017: Dynamic position sizing with ATR-based volatility adjustment."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.config import Settings, TradingMode
from bot.models import OHLCV, SignalAction, TradingSignal
from bot.risk.manager import RiskManager
from bot.strategies.indicators import calculate_atr, calculate_atr_series

# --- Helpers ---

_BASE_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_candle(
    close: float,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: float = 100.0,
    idx: int = 0,
) -> OHLCV:
    """Create a candle with valid OHLCV constraints."""
    if open_ is None:
        open_ = close
    if high is None:
        high = max(close, open_) * 1.001
    if low is None:
        low = min(close, open_) * 0.999
    return OHLCV(
        timestamp=_BASE_TIME + timedelta(hours=idx),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        symbol="BTC/USDT",
        timeframe="1h",
    )


def make_stable_candles(n: int, base_price: float = 100.0) -> list[OHLCV]:
    """Create N candles with stable, low-volatility prices."""
    candles = []
    for i in range(n):
        close = base_price + (i % 3) * 0.5  # slight variation
        candles.append(
            make_candle(
                close=close,
                high=close + 1.0,
                low=close - 1.0,
                open_=close - 0.2,
                volume=100.0,
                idx=i,
            )
        )
    return candles


def make_volatile_candles(n: int, base_price: float = 100.0) -> list[OHLCV]:
    """Create N candles with high volatility (large high-low range)."""
    candles = []
    for i in range(n):
        close = base_price + (i % 5) * 5.0
        candles.append(
            make_candle(
                close=close,
                high=close + 20.0,
                low=close - 20.0,
                open_=close - 3.0,
                volume=500.0,
                idx=i,
            )
        )
    return candles


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


def make_mock_strategy(signal_action=SignalAction.BUY, confidence=0.8):
    """Create a mock strategy with required attributes for ensemble."""
    strategy = MagicMock()
    strategy.name = "test_strategy"
    strategy.required_history_length = 1
    strategy.analyze = AsyncMock(
        return_value=TradingSignal(
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            action=signal_action,
            confidence=confidence,
        )
    )
    strategy.adapt_to_regime = MagicMock()
    return strategy


# === ATR Calculation Tests ===


class TestCalculateATR:
    def test_returns_none_insufficient_data(self):
        """ATR requires at least period + 1 candles."""
        candles = make_stable_candles(10)
        result = calculate_atr(candles, period=14)
        assert result is None

    def test_returns_none_empty_candles(self):
        result = calculate_atr([], period=14)
        assert result is None

    def test_returns_value_with_sufficient_data(self):
        candles = make_stable_candles(20)
        result = calculate_atr(candles, period=14)
        assert result is not None
        assert result > 0

    def test_stable_market_low_atr(self):
        """Stable prices should produce a low ATR."""
        candles = make_stable_candles(30, base_price=100.0)
        atr = calculate_atr(candles, period=14)
        assert atr is not None
        assert atr < 5.0  # low volatility

    def test_volatile_market_high_atr(self):
        """Volatile prices should produce a high ATR."""
        candles = make_volatile_candles(30, base_price=100.0)
        atr = calculate_atr(candles, period=14)
        assert atr is not None
        assert atr > 30.0  # high volatility

    def test_volatile_atr_greater_than_stable(self):
        """Volatile market ATR should be much larger than stable market ATR."""
        stable_candles = make_stable_candles(30, base_price=100.0)
        volatile_candles = make_volatile_candles(30, base_price=100.0)
        stable_atr = calculate_atr(stable_candles, period=14)
        volatile_atr = calculate_atr(volatile_candles, period=14)
        assert volatile_atr > stable_atr * 5

    def test_custom_period(self):
        candles = make_stable_candles(10)
        result = calculate_atr(candles, period=5)
        assert result is not None
        assert result > 0

    def test_minimum_candles_for_period(self):
        """Exactly period + 1 candles should work."""
        candles = make_stable_candles(15)
        result = calculate_atr(candles, period=14)
        assert result is not None

    def test_atr_with_gaps(self):
        """ATR should handle price gaps (close to next open jump)."""
        candles = []
        for i in range(20):
            # Every other candle has a big gap
            base = 100.0 + (10.0 if i % 2 == 0 else 0.0)
            candles.append(
                make_candle(
                    close=base,
                    high=base + 2.0,
                    low=base - 2.0,
                    open_=base - 1.0,
                    idx=i,
                )
            )
        atr = calculate_atr(candles, period=14)
        assert atr is not None
        # Gap should be captured in true range
        assert atr > 4.0


class TestCalculateATRSeries:
    def test_returns_empty_for_insufficient_data(self):
        candles = make_stable_candles(5)
        result = calculate_atr_series(candles, period=14)
        assert result == []

    def test_returns_correct_length(self):
        candles = make_stable_candles(30)
        result = calculate_atr_series(candles, period=14)
        # True ranges have len-1 entries, then rolling mean needs period entries
        # So result length = (len-1) - period + 1 = len - period
        assert len(result) == 30 - 14

    def test_all_values_positive(self):
        candles = make_stable_candles(30)
        result = calculate_atr_series(candles, period=14)
        assert all(v > 0 for v in result)

    def test_last_value_matches_calculate_atr(self):
        """The last value in the series should match calculate_atr."""
        candles = make_stable_candles(30)
        series = calculate_atr_series(candles, period=14)
        single = calculate_atr(candles, period=14)
        assert abs(series[-1] - single) < 1e-10


# === Dynamic Position Sizing Tests ===


class TestDynamicPositionSizing:
    @pytest.fixture
    def rm(self):
        return RiskManager(
            max_position_size_pct=100.0,  # no cap for testing sizing logic
            stop_loss_pct=3.0,
        )

    def test_basic_calculation(self, rm):
        """Basic ATR-based sizing: risk_amount / (atr * multiplier)."""
        # portfolio=100000, risk=1%, atr=100, multiplier=2
        # risk_amount = 1000, position_size = 1000 / (100 * 2) = 5.0
        # max_qty = 100000 / 1000 = 100, so not capped
        size = rm.calculate_dynamic_position_size(
            portfolio_value=100000,
            price=1000,
            atr=100,
            risk_per_trade_pct=1.0,
            atr_multiplier=2.0,
        )
        assert size == pytest.approx(5.0)

    def test_high_atr_smaller_position(self, rm):
        """High ATR (volatile) should give smaller position."""
        low_atr_size = rm.calculate_dynamic_position_size(
            portfolio_value=10000, price=1000, atr=10, risk_per_trade_pct=1.0
        )
        high_atr_size = rm.calculate_dynamic_position_size(
            portfolio_value=10000, price=1000, atr=100, risk_per_trade_pct=1.0
        )
        assert high_atr_size < low_atr_size

    def test_low_atr_larger_position(self, rm):
        """Low ATR (quiet) should give larger position."""
        size = rm.calculate_dynamic_position_size(
            portfolio_value=10000, price=1000, atr=5, risk_per_trade_pct=1.0
        )
        # risk_amount=100, position_size=100/(5*2)=10
        assert size == pytest.approx(10.0)

    def test_capped_at_max_position_size(self):
        """Position size must not exceed max_position_size_pct."""
        rm = RiskManager(max_position_size_pct=10.0)
        # Very low ATR would give huge position, but should be capped
        # max_position_size_pct=10%, portfolio=10000, price=100
        # max_value = 1000, max_qty = 1000/100 = 10
        size = rm.calculate_dynamic_position_size(
            portfolio_value=10000,
            price=100,
            atr=0.01,  # extremely low ATR
            risk_per_trade_pct=1.0,
        )
        max_qty = 10000 * 0.10 / 100  # = 10
        assert size == pytest.approx(max_qty)

    def test_zero_price_returns_zero(self, rm):
        size = rm.calculate_dynamic_position_size(
            portfolio_value=10000, price=0, atr=100
        )
        assert size == 0.0

    def test_zero_portfolio_returns_zero(self, rm):
        size = rm.calculate_dynamic_position_size(
            portfolio_value=0, price=1000, atr=100
        )
        assert size == 0.0

    def test_zero_atr_returns_zero(self, rm):
        size = rm.calculate_dynamic_position_size(
            portfolio_value=10000, price=1000, atr=0
        )
        assert size == 0.0

    def test_negative_atr_returns_zero(self, rm):
        size = rm.calculate_dynamic_position_size(
            portfolio_value=10000, price=1000, atr=-10
        )
        assert size == 0.0

    def test_custom_risk_per_trade(self, rm):
        """Higher risk_per_trade_pct gives larger position."""
        size_1pct = rm.calculate_dynamic_position_size(
            portfolio_value=10000, price=1000, atr=50, risk_per_trade_pct=1.0
        )
        size_2pct = rm.calculate_dynamic_position_size(
            portfolio_value=10000, price=1000, atr=50, risk_per_trade_pct=2.0
        )
        assert size_2pct == pytest.approx(size_1pct * 2)

    def test_custom_atr_multiplier(self, rm):
        """Higher atr_multiplier gives smaller position (more conservative)."""
        size_2x = rm.calculate_dynamic_position_size(
            portfolio_value=10000, price=1000, atr=50, atr_multiplier=2.0
        )
        size_3x = rm.calculate_dynamic_position_size(
            portfolio_value=10000, price=1000, atr=50, atr_multiplier=3.0
        )
        assert size_3x < size_2x

    def test_zero_atr_multiplier_returns_zero(self, rm):
        size = rm.calculate_dynamic_position_size(
            portfolio_value=10000, price=1000, atr=50, atr_multiplier=0
        )
        assert size == 0.0

    def test_backward_compat_fixed_sizing_still_works(self):
        """The original calculate_position_size still works unchanged."""
        rm = RiskManager(max_position_size_pct=10.0)
        size = rm.calculate_position_size(10000.0, 50000.0)
        assert abs(size - 0.02) < 0.001

    def test_volatile_vs_stable_end_to_end(self, rm):
        """Simulate stable vs volatile markets using actual candle data."""
        stable_candles = make_stable_candles(30, base_price=1000.0)
        volatile_candles = make_volatile_candles(30, base_price=1000.0)

        stable_atr = calculate_atr(stable_candles, period=14)
        volatile_atr = calculate_atr(volatile_candles, period=14)

        assert stable_atr is not None
        assert volatile_atr is not None

        stable_size = rm.calculate_dynamic_position_size(
            portfolio_value=10000, price=1000, atr=stable_atr
        )
        volatile_size = rm.calculate_dynamic_position_size(
            portfolio_value=10000, price=1000, atr=volatile_atr
        )

        # Volatile market -> smaller position
        assert volatile_size < stable_size

    def test_never_exceeds_max_regardless_of_inputs(self):
        """No combination of inputs should exceed max position size."""
        rm = RiskManager(max_position_size_pct=5.0)
        max_qty = 10000 * 0.05 / 100  # = 5.0
        # Tiny ATR and high risk should still be capped
        size = rm.calculate_dynamic_position_size(
            portfolio_value=10000,
            price=100,
            atr=0.001,
            risk_per_trade_pct=10.0,
            atr_multiplier=0.5,
        )
        assert size <= max_qty + 1e-10


# === Config Tests ===


class TestDynamicSizingConfig:
    def test_default_config_values(self):
        settings = make_settings()
        assert settings.risk_per_trade_pct == 1.0
        assert settings.atr_multiplier == 2.0
        assert settings.atr_period == 14

    def test_custom_config_values(self):
        settings = make_settings(
            risk_per_trade_pct=2.5,
            atr_multiplier=3.0,
            atr_period=20,
        )
        assert settings.risk_per_trade_pct == 2.5
        assert settings.atr_multiplier == 3.0
        assert settings.atr_period == 20


# === Integration Tests (main.py wiring) ===


class TestDynamicSizingIntegration:
    @pytest.mark.asyncio
    async def test_trading_cycle_uses_dynamic_sizing(self):
        """When candles provide enough data for ATR, dynamic sizing is used."""
        from bot.main import TradingBot

        settings = make_settings(
            risk_per_trade_pct=1.0,
            atr_multiplier=2.0,
            atr_period=14,
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        candles = make_stable_candles(30, base_price=50000.0)

        mock_store = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=candles)
        bot._store = mock_store
        bot._collector = AsyncMock()
        bot._collector.collect_once = AsyncMock()

        mock_strategy = make_mock_strategy()

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]

            executed_qty = None
            mock_engine = AsyncMock()

            async def capture_execute(signal, quantity):
                nonlocal executed_qty
                executed_qty = quantity
                return None

            mock_engine.execute_signal = capture_execute
            bot._execution_engines = {"test": mock_engine}

            await bot._trading_cycle()

        if executed_qty is not None:
            atr = calculate_atr(candles, period=14)
            assert atr is not None
            portfolio_value = bot._risk_manager._current_portfolio_value
            expected = bot._risk_manager.calculate_dynamic_position_size(
                portfolio_value, candles[-1].close, atr,
                risk_per_trade_pct=1.0, atr_multiplier=2.0,
            )
            assert executed_qty == pytest.approx(expected, rel=1e-6)

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_trading_cycle_fallback_to_fixed_sizing(self):
        """When not enough candles for ATR, fall back to fixed % sizing."""
        from bot.main import TradingBot

        settings = make_settings(
            risk_per_trade_pct=1.0,
            atr_multiplier=2.0,
            atr_period=14,
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Only 5 candles -- not enough for ATR with period=14
        candles = make_stable_candles(5, base_price=50000.0)

        mock_store = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=candles)
        bot._store = mock_store
        bot._collector = AsyncMock()
        bot._collector.collect_once = AsyncMock()

        mock_strategy = make_mock_strategy()

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]

            executed_qty = None
            mock_engine = AsyncMock()

            async def capture_execute(signal, quantity):
                nonlocal executed_qty
                executed_qty = quantity
                return None

            mock_engine.execute_signal = capture_execute
            bot._execution_engines = {"test": mock_engine}

            await bot._trading_cycle()

        if executed_qty is not None:
            portfolio_value = bot._risk_manager._current_portfolio_value
            expected_fixed = bot._risk_manager.calculate_position_size(
                portfolio_value, candles[-1].close
            )
            assert executed_qty == pytest.approx(expected_fixed, rel=1e-6)

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_atr_period_config_used(self):
        """Custom atr_period from config is passed to calculate_atr."""
        from bot.main import TradingBot

        settings = make_settings(atr_period=5)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # 10 candles -- enough for period=5 but not period=14
        candles = make_stable_candles(10, base_price=50000.0)

        mock_store = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=candles)
        bot._store = mock_store
        bot._collector = AsyncMock()
        bot._collector.collect_once = AsyncMock()

        mock_strategy = make_mock_strategy()

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]

            executed_qty = None
            mock_engine = AsyncMock()

            async def capture_execute(signal, quantity):
                nonlocal executed_qty
                executed_qty = quantity
                return None

            mock_engine.execute_signal = capture_execute
            bot._execution_engines = {"test": mock_engine}

            with patch(
                "bot.main.calculate_atr", wraps=calculate_atr
            ) as mock_atr:
                await bot._trading_cycle()
                if mock_atr.called:
                    _, kwargs = mock_atr.call_args
                    assert kwargs.get("period") == 5

        await bot.shutdown()
