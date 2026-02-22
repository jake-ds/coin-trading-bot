"""Tests for RiskManager pipeline integration: position tracking, PnL recording, daily reset."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.config import Settings, TradingMode
from bot.main import TradingBot
from bot.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    SignalAction,
    TradingSignal,
)
from bot.risk.manager import RiskManager


def make_settings(**kwargs):
    """Create test settings with safe defaults."""
    defaults = {
        "trading_mode": TradingMode.PAPER,
        "database_url": "sqlite+aiosqlite:///:memory:",
        "binance_api_key": "",
        "upbit_api_key": "",
        "symbols": ["BTC/USDT"],
        "loop_interval_seconds": 1,
        "paper_initial_balance": 10000.0,
        "paper_fee_pct": 0.1,
        "signal_min_agreement": 1,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


def make_signal(
    action: SignalAction = SignalAction.BUY,
    symbol: str = "BTC/USDT",
    confidence: float = 0.8,
) -> TradingSignal:
    return TradingSignal(
        strategy_name="test",
        symbol=symbol,
        action=action,
        confidence=confidence,
    )


def make_order(
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    qty: float = 0.01,
    price: float = 50000.0,
) -> Order:
    from datetime import timezone

    now = datetime.now(timezone.utc)
    return Order(
        id=f"paper-{side.value.lower()}-001",
        exchange="binance",
        symbol=symbol,
        side=side,
        type=OrderType.MARKET,
        price=0,
        quantity=qty,
        status=OrderStatus.FILLED,
        created_at=now,
        filled_at=now,
        filled_price=price,
        filled_quantity=qty,
    )


class TestRiskManagerGetPosition:
    """Tests for the get_position method."""

    def test_get_position_returns_none_when_empty(self):
        rm = RiskManager()
        assert rm.get_position("BTC/USDT") is None

    def test_get_position_returns_position_data(self):
        rm = RiskManager()
        rm.add_position("BTC/USDT", 1.0, 50000.0)
        pos = rm.get_position("BTC/USDT")
        assert pos is not None
        assert pos["quantity"] == 1.0
        assert pos["entry_price"] == 50000.0

    def test_get_position_returns_none_after_removal(self):
        rm = RiskManager()
        rm.add_position("BTC/USDT", 1.0, 50000.0)
        rm.remove_position("BTC/USDT")
        assert rm.get_position("BTC/USDT") is None


class TestCheckAndResetDaily:
    """Tests for the check_and_reset_daily method."""

    def test_first_call_resets(self):
        rm = RiskManager()
        # No date set yet
        assert rm._daily_pnl_reset_date is None
        result = rm.check_and_reset_daily()
        assert result is True
        assert rm._daily_pnl_reset_date == datetime.utcnow().date()

    def test_same_day_no_reset(self):
        rm = RiskManager()
        rm.check_and_reset_daily()  # set today
        result = rm.check_and_reset_daily()
        assert result is False

    def test_new_day_resets(self):
        rm = RiskManager()
        rm._daily_pnl_reset_date = (datetime.utcnow() - timedelta(days=1)).date()
        rm._daily_pnl = -100.0
        result = rm.check_and_reset_daily()
        assert result is True
        assert rm._daily_pnl == 0.0

    def test_new_day_clears_daily_loss_halt(self):
        rm = RiskManager(daily_loss_limit_pct=5.0)
        rm.update_portfolio_value(10000.0)
        rm.record_trade_pnl(-500.0)  # 5% loss triggers halt
        rm.validate_signal(make_signal())
        assert rm.is_halted
        assert rm.halt_reason == "daily_loss_limit"

        # Simulate next day
        rm._daily_pnl_reset_date = (datetime.utcnow() - timedelta(days=1)).date()
        rm.check_and_reset_daily()

        assert not rm.is_halted
        # Can trade again
        signal = rm.validate_signal(make_signal())
        assert signal.action == SignalAction.BUY


class TestRiskPipelineIntegration:
    """Tests for RiskManager wired into trading cycle."""

    @pytest.mark.asyncio
    async def test_buy_registers_position_in_risk_manager(self):
        """After a BUY order is executed, risk manager should track the position."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Mock store to return candles
        mock_candle = MagicMock()
        mock_candle.close = 50000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle] * 200)

        # Mock a strategy that returns a BUY signal
        mock_strategy = MagicMock()
        mock_strategy.name = "test_strategy"
        mock_strategy.required_history_length = 1
        mock_signal = TradingSignal(
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            confidence=0.8,
        )
        mock_strategy.analyze = AsyncMock(return_value=mock_signal)

        buy_order = make_order(side=OrderSide.BUY, qty=0.02, price=50000.0)
        mock_engine = AsyncMock()
        mock_engine.execute_signal = AsyncMock(return_value=buy_order)
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]
            await bot._trading_cycle()

        # Risk manager should now track BTC/USDT position
        pos = bot._risk_manager.get_position("BTC/USDT")
        assert pos is not None
        assert pos["quantity"] == 0.02
        assert pos["entry_price"] == 50000.0

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_duplicate_buy_blocked_after_position_opened(self):
        """After a BUY fills, a second BUY for the same symbol should be rejected."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        mock_candle = MagicMock()
        mock_candle.close = 50000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle] * 200)

        mock_strategy = MagicMock()
        mock_strategy.name = "test_strategy"
        mock_strategy.required_history_length = 1
        buy_signal = TradingSignal(
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            confidence=0.8,
        )
        mock_strategy.analyze = AsyncMock(return_value=buy_signal)

        buy_order = make_order(side=OrderSide.BUY, qty=0.02, price=50000.0)
        mock_engine = AsyncMock()
        mock_engine.execute_signal = AsyncMock(return_value=buy_order)
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]

            # First cycle: BUY fills
            await bot._trading_cycle()
            assert bot._risk_manager.get_position("BTC/USDT") is not None

            # Second cycle: same BUY signal, should be blocked
            mock_engine.execute_signal.reset_mock()
            await bot._trading_cycle()

            # Engine should NOT have been called the second time (signal rejected)
            mock_engine.execute_signal.assert_not_called()

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_sell_records_pnl_and_removes_position(self):
        """After a SELL, PnL should be recorded and position removed."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Pre-load a position in risk manager
        bot._risk_manager.update_portfolio_value(10000.0)
        bot._risk_manager.add_position("BTC/USDT", 0.02, 50000.0)

        mock_candle = MagicMock()
        mock_candle.close = 51000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle] * 200)

        mock_strategy = MagicMock()
        mock_strategy.name = "test_strategy"
        mock_strategy.required_history_length = 1
        sell_signal = TradingSignal(
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            action=SignalAction.SELL,
            confidence=0.8,
        )
        mock_strategy.analyze = AsyncMock(return_value=sell_signal)

        sell_order = make_order(side=OrderSide.SELL, qty=0.02, price=51000.0)
        mock_engine = AsyncMock()
        mock_engine.execute_signal = AsyncMock(return_value=sell_order)
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]
            await bot._trading_cycle()

        # Position should be removed
        assert bot._risk_manager.get_position("BTC/USDT") is None

        # PnL should be recorded: (51000 - 50000) * 0.02 = 20.0
        assert bot._risk_manager._daily_pnl == 20.0

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_daily_loss_limit_blocks_all_signals(self):
        """After daily loss limit is hit, all trading signals should be rejected."""
        settings = make_settings(daily_loss_limit_pct=5.0)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Manually set up a loss scenario
        bot._risk_manager.update_portfolio_value(10000.0)
        bot._risk_manager.record_trade_pnl(-500.0)  # 5% loss

        mock_candle = MagicMock()
        mock_candle.close = 50000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle] * 200)

        mock_strategy = MagicMock()
        mock_strategy.name = "test_strategy"
        mock_strategy.required_history_length = 1
        buy_signal = TradingSignal(
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            confidence=0.8,
        )
        mock_strategy.analyze = AsyncMock(return_value=buy_signal)

        mock_engine = AsyncMock()
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]
            await bot._trading_cycle()

        # Engine should not have been called
        mock_engine.execute_signal.assert_not_called()
        assert bot._risk_manager.is_halted

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_daily_reset_resumes_trading_next_day(self):
        """After daily loss halt, trading should resume the next day."""
        settings = make_settings(daily_loss_limit_pct=5.0)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        bot._risk_manager.update_portfolio_value(10000.0)
        bot._risk_manager.record_trade_pnl(-500.0)  # triggers halt

        # Trigger halt
        bot._risk_manager.validate_signal(make_signal())
        assert bot._risk_manager.is_halted

        # Simulate next day
        bot._risk_manager._daily_pnl_reset_date = (
            datetime.utcnow() - timedelta(days=1)
        ).date()

        mock_candle = MagicMock()
        mock_candle.close = 50000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle] * 200)

        mock_strategy = MagicMock()
        mock_strategy.name = "test_strategy"
        mock_strategy.required_history_length = 1
        buy_signal = TradingSignal(
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            confidence=0.8,
        )
        mock_strategy.analyze = AsyncMock(return_value=buy_signal)

        buy_order = make_order(side=OrderSide.BUY, qty=0.02, price=50000.0)
        mock_engine = AsyncMock()
        mock_engine.execute_signal = AsyncMock(return_value=buy_order)
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]
            # This cycle should trigger daily reset and allow trading
            await bot._trading_cycle()

        # Trading should have resumed
        assert not bot._risk_manager.is_halted
        mock_engine.execute_signal.assert_called_once()

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_portfolio_value_updated_from_paper_portfolio(self):
        """Risk manager portfolio value should be updated from PaperPortfolio each cycle."""
        settings = make_settings(paper_initial_balance=20000.0)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Paper portfolio should be created with 20000
        assert bot._paper_portfolio is not None
        assert bot._paper_portfolio.cash == 20000.0

        mock_candle = MagicMock()
        mock_candle.close = 50000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle] * 200)

        # No strategies - just run cycle to update portfolio value
        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = []
            await bot._trading_cycle()

        # Risk manager should have portfolio value from paper portfolio
        assert bot._risk_manager._current_portfolio_value == 20000.0

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_sell_pnl_negative_loss(self):
        """Selling at a loss should record negative PnL."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        bot._risk_manager.update_portfolio_value(10000.0)
        bot._risk_manager.add_position("BTC/USDT", 0.02, 50000.0)

        mock_candle = MagicMock()
        mock_candle.close = 48000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle] * 200)

        mock_strategy = MagicMock()
        mock_strategy.name = "test_strategy"
        mock_strategy.required_history_length = 1
        sell_signal = TradingSignal(
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            action=SignalAction.SELL,
            confidence=0.8,
        )
        mock_strategy.analyze = AsyncMock(return_value=sell_signal)

        sell_order = make_order(side=OrderSide.SELL, qty=0.02, price=48000.0)
        mock_engine = AsyncMock()
        mock_engine.execute_signal = AsyncMock(return_value=sell_order)
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]
            await bot._trading_cycle()

        # PnL should be negative: (48000 - 50000) * 0.02 = -40.0
        assert bot._risk_manager._daily_pnl == -40.0

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_sell_without_prior_position_no_pnl_crash(self):
        """Selling when no position is tracked in risk manager should not crash."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        bot._risk_manager.update_portfolio_value(10000.0)
        # No position added to risk manager

        mock_candle = MagicMock()
        mock_candle.close = 50000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle] * 200)

        mock_strategy = MagicMock()
        mock_strategy.name = "test_strategy"
        mock_strategy.required_history_length = 1
        sell_signal = TradingSignal(
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            action=SignalAction.SELL,
            confidence=0.8,
        )
        mock_strategy.analyze = AsyncMock(return_value=sell_signal)

        sell_order = make_order(side=OrderSide.SELL, qty=0.02, price=50000.0)
        mock_engine = AsyncMock()
        mock_engine.execute_signal = AsyncMock(return_value=sell_order)
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]
            # Should not crash even without prior position
            await bot._trading_cycle()

        # No PnL should be recorded (no position to calculate from)
        assert bot._risk_manager._daily_pnl == 0.0
        # Position should still be removed (no-op but no crash)
        assert bot._risk_manager.get_position("BTC/USDT") is None

        await bot.shutdown()
