"""Tests for V2-008: PositionManager wiring into TradingBot orchestrator."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.config import Settings, TradingMode
from bot.execution.position_manager import PositionManager
from bot.main import TradingBot
from bot.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    SignalAction,
    TradingSignal,
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


class TestPositionManagerInitialization:
    @pytest.mark.asyncio
    async def test_position_manager_created_on_initialize(self):
        """PositionManager should be created during bot initialization."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        assert bot._position_manager is not None
        assert isinstance(bot._position_manager, PositionManager)

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_position_manager_uses_config_settings(self):
        """PositionManager should use settings from config."""
        settings = make_settings(
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
            trailing_stop_enabled=True,
            trailing_stop_pct=3.0,
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        pm = bot._position_manager
        assert pm._stop_loss_pct == 5.0
        assert pm._take_profit_pct == 10.0
        assert pm._trailing_stop_enabled is True
        assert pm._trailing_stop_pct == 3.0

        await bot.shutdown()


class TestPositionManagerExitChecks:
    @pytest.mark.asyncio
    async def test_stop_loss_triggers_sell_in_cycle(self):
        """When price drops below stop-loss, a SELL order should be executed."""
        settings = make_settings(stop_loss_pct=3.0, take_profit_pct=5.0)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Add a managed position
        bot._position_manager.add_position(
            "BTC/USDT", entry_price=50000.0, quantity=1.0
        )
        # Also add to risk manager so PnL tracking works
        bot._risk_manager.add_position("BTC/USDT", 1.0, 50000.0)
        bot._risk_manager.update_portfolio_value(50000.0)

        # Mock store to return a candle below stop-loss (48500 = 50000 * 0.97)
        mock_candle = MagicMock()
        mock_candle.close = 47000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle])

        # Mock execution engine
        mock_order = Order(
            id="exit-001",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            price=0,
            quantity=1.0,
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            filled_price=47000.0,
            filled_quantity=1.0,
        )
        mock_engine = AsyncMock()
        mock_engine.execute_signal = AsyncMock(return_value=mock_order)
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = []
            await bot._trading_cycle()

        # Verify sell was executed
        mock_engine.execute_signal.assert_called_once()
        call_args = mock_engine.execute_signal.call_args
        signal = call_args[0][0]
        assert signal.action == SignalAction.SELL
        assert signal.strategy_name == "position_manager"
        assert call_args[1]["quantity"] == 1.0

        # Position should be removed from both managers
        assert bot._position_manager.get_position("BTC/USDT") is None
        assert bot._risk_manager.get_position("BTC/USDT") is None

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_take_profit_1_partial_sell(self):
        """TP1 should trigger a partial sell (50%) and keep remaining position."""
        settings = make_settings(stop_loss_pct=3.0, take_profit_pct=5.0)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        bot._position_manager.add_position(
            "BTC/USDT", entry_price=50000.0, quantity=1.0
        )
        bot._risk_manager.add_position("BTC/USDT", 1.0, 50000.0)
        bot._risk_manager.update_portfolio_value(50000.0)

        # Price at TP1 level (50000 * 1.03 = 51500)
        mock_candle = MagicMock()
        mock_candle.close = 51500.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle])

        mock_order = Order(
            id="tp1-001",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            price=0,
            quantity=0.5,
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            filled_price=51500.0,
            filled_quantity=0.5,
        )
        mock_engine = AsyncMock()
        mock_engine.execute_signal = AsyncMock(return_value=mock_order)
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = []
            await bot._trading_cycle()

        # Partial sell executed
        mock_engine.execute_signal.assert_called_once()
        call_args = mock_engine.execute_signal.call_args
        assert call_args[1]["quantity"] == 0.5

        # Position still managed with remaining quantity
        managed_pos = bot._position_manager.get_position("BTC/USDT")
        assert managed_pos is not None
        assert managed_pos.quantity == 0.5
        assert managed_pos.tp1_hit is True

        # RiskManager position updated with remaining
        rm_pos = bot._risk_manager.get_position("BTC/USDT")
        assert rm_pos is not None
        assert rm_pos["quantity"] == 0.5

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_take_profit_2_after_tp1(self):
        """TP2 should trigger after TP1 and sell remaining position."""
        settings = make_settings(stop_loss_pct=3.0, take_profit_pct=5.0)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        bot._position_manager.add_position(
            "BTC/USDT", entry_price=50000.0, quantity=1.0
        )
        bot._risk_manager.add_position("BTC/USDT", 1.0, 50000.0)
        bot._risk_manager.update_portfolio_value(50000.0)

        mock_engine = AsyncMock()
        bot._execution_engines = {"binance": mock_engine}

        # First cycle: TP1 triggers
        mock_candle_tp1 = MagicMock()
        mock_candle_tp1.close = 51500.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle_tp1])

        mock_order_tp1 = Order(
            id="tp1-001",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            price=0,
            quantity=0.5,
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            filled_price=51500.0,
            filled_quantity=0.5,
        )
        mock_engine.execute_signal = AsyncMock(return_value=mock_order_tp1)

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = []
            await bot._trading_cycle()

        # Second cycle: TP2 triggers
        mock_candle_tp2 = MagicMock()
        mock_candle_tp2.close = 52500.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle_tp2])

        mock_order_tp2 = Order(
            id="tp2-001",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            price=0,
            quantity=0.5,
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            filled_price=52500.0,
            filled_quantity=0.5,
        )
        mock_engine.execute_signal = AsyncMock(return_value=mock_order_tp2)

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = []
            await bot._trading_cycle()

        # Position fully closed
        assert bot._position_manager.get_position("BTC/USDT") is None
        assert bot._risk_manager.get_position("BTC/USDT") is None

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_exit_checked_before_strategies(self):
        """Exit checks should run BEFORE strategy analysis."""
        settings = make_settings(stop_loss_pct=3.0, take_profit_pct=5.0)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Add a position that will trigger stop-loss
        bot._position_manager.add_position(
            "BTC/USDT", entry_price=50000.0, quantity=1.0
        )
        bot._risk_manager.add_position("BTC/USDT", 1.0, 50000.0)
        bot._risk_manager.update_portfolio_value(50000.0)

        mock_candle = MagicMock()
        mock_candle.close = 47000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle] * 200)

        mock_order = Order(
            id="exit-001",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            price=0,
            quantity=1.0,
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            filled_price=47000.0,
            filled_quantity=1.0,
        )
        mock_engine = AsyncMock()
        mock_engine.execute_signal = AsyncMock(return_value=mock_order)
        bot._execution_engines = {"binance": mock_engine}

        # Track call order
        call_order = []

        original_check_exits = bot._position_manager.check_exits

        def tracked_check_exits(*args, **kwargs):
            call_order.append("check_exits")
            return original_check_exits(*args, **kwargs)

        bot._position_manager.check_exits = tracked_check_exits

        hold_signal = TradingSignal(
            strategy_name="test",
            symbol="BTC/USDT",
            action=SignalAction.HOLD,
            confidence=0.0,
        )

        mock_strategy = MagicMock()
        mock_strategy.name = "test"
        mock_strategy.required_history_length = 1

        async def tracked_analyze(*args, **kwargs):
            call_order.append("strategy_analyze")
            return hold_signal

        mock_strategy.analyze = AsyncMock(side_effect=tracked_analyze)

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]
            await bot._trading_cycle()

        # check_exits was called before strategy_analyze
        assert call_order.index("check_exits") < call_order.index(
            "strategy_analyze"
        )

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_buy_registers_position_in_manager(self):
        """After BUY execution, position should be registered in PositionManager."""
        settings = make_settings(stop_loss_pct=3.0, take_profit_pct=5.0)
        bot = TradingBot(settings=settings)
        await bot.initialize()
        bot._risk_manager.update_portfolio_value(10000.0)

        mock_candle = MagicMock()
        mock_candle.close = 50000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle] * 200)

        mock_strategy = MagicMock()
        mock_strategy.name = "test_strategy"
        mock_strategy.required_history_length = 1
        mock_strategy.analyze = AsyncMock(
            return_value=TradingSignal(
                strategy_name="test_strategy",
                symbol="BTC/USDT",
                action=SignalAction.BUY,
                confidence=0.8,
            )
        )

        mock_order = Order(
            id="buy-001",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            price=0,
            quantity=0.02,
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            filled_price=50000.0,
            filled_quantity=0.02,
        )
        mock_engine = AsyncMock()
        mock_engine.execute_signal = AsyncMock(return_value=mock_order)
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]
            await bot._trading_cycle()

        # Position registered in PositionManager
        managed = bot._position_manager.get_position("BTC/USDT")
        assert managed is not None
        assert managed.entry_price == 50000.0
        assert managed.quantity == 0.02

        # Also registered in RiskManager
        rm_pos = bot._risk_manager.get_position("BTC/USDT")
        assert rm_pos is not None

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_strategy_sell_removes_from_position_manager(self):
        """When a strategy triggers a SELL, position should be removed from PositionManager."""
        settings = make_settings(stop_loss_pct=3.0, take_profit_pct=5.0)
        bot = TradingBot(settings=settings)
        await bot.initialize()
        bot._risk_manager.update_portfolio_value(10000.0)

        # Pre-register position
        bot._position_manager.add_position(
            "BTC/USDT", entry_price=50000.0, quantity=0.02
        )
        bot._risk_manager.add_position("BTC/USDT", 0.02, 50000.0)

        mock_candle = MagicMock()
        mock_candle.close = 51000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle] * 200)

        mock_strategy = MagicMock()
        mock_strategy.name = "test_strategy"
        mock_strategy.required_history_length = 1
        mock_strategy.analyze = AsyncMock(
            return_value=TradingSignal(
                strategy_name="test_strategy",
                symbol="BTC/USDT",
                action=SignalAction.SELL,
                confidence=0.8,
            )
        )

        mock_order = Order(
            id="sell-001",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            price=0,
            quantity=0.02,
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            filled_price=51000.0,
            filled_quantity=0.02,
        )
        mock_engine = AsyncMock()
        mock_engine.execute_signal = AsyncMock(return_value=mock_order)
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]
            await bot._trading_cycle()

        # Position removed from both managers
        assert bot._position_manager.get_position("BTC/USDT") is None
        assert bot._risk_manager.get_position("BTC/USDT") is None

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_pnl_recorded_on_stop_loss_exit(self):
        """PnL should be recorded when stop-loss triggers."""
        settings = make_settings(stop_loss_pct=3.0, take_profit_pct=5.0)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        bot._position_manager.add_position(
            "BTC/USDT", entry_price=50000.0, quantity=1.0
        )
        bot._risk_manager.add_position("BTC/USDT", 1.0, 50000.0)
        bot._risk_manager.update_portfolio_value(50000.0)

        # Price drops to trigger stop-loss
        mock_candle = MagicMock()
        mock_candle.close = 47000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle])

        # Stop-loss price is 48500, filled at 47000
        mock_order = Order(
            id="sl-001",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            price=0,
            quantity=1.0,
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            filled_price=47000.0,
            filled_quantity=1.0,
        )
        mock_engine = AsyncMock()
        mock_engine.execute_signal = AsyncMock(return_value=mock_order)
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = []
            await bot._trading_cycle()

        # PnL should be negative: (47000 - 50000) * 1.0 = -3000
        assert bot._risk_manager._daily_pnl == -3000.0

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_no_exit_when_price_in_range(self):
        """No exit should trigger when price is between stop-loss and take-profit."""
        settings = make_settings(stop_loss_pct=3.0, take_profit_pct=5.0)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        bot._position_manager.add_position(
            "BTC/USDT", entry_price=50000.0, quantity=1.0
        )

        mock_candle = MagicMock()
        mock_candle.close = 50500.0  # Between SL (48500) and TP1 (51500)
        bot._store.get_candles = AsyncMock(return_value=[mock_candle])

        mock_engine = AsyncMock()
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = []
            await bot._trading_cycle()

        # No sell executed
        mock_engine.execute_signal.assert_not_called()

        # Position still managed
        assert bot._position_manager.get_position("BTC/USDT") is not None

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_exit_signal_metadata_includes_exit_type(self):
        """Exit sell signal should include exit_type in metadata."""
        settings = make_settings(stop_loss_pct=3.0, take_profit_pct=5.0)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        bot._position_manager.add_position(
            "BTC/USDT", entry_price=50000.0, quantity=1.0
        )
        bot._risk_manager.add_position("BTC/USDT", 1.0, 50000.0)
        bot._risk_manager.update_portfolio_value(50000.0)

        mock_candle = MagicMock()
        mock_candle.close = 47000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle])

        mock_order = Order(
            id="sl-001",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            price=0,
            quantity=1.0,
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            filled_price=47000.0,
            filled_quantity=1.0,
        )
        mock_engine = AsyncMock()
        mock_engine.execute_signal = AsyncMock(return_value=mock_order)
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = []
            await bot._trading_cycle()

        # Check the signal passed to execute_signal
        call_args = mock_engine.execute_signal.call_args
        signal = call_args[0][0]
        assert signal.metadata["exit_type"] == "stop_loss"
        assert signal.strategy_name == "position_manager"

        await bot.shutdown()


class TestConfigSettings:
    def test_default_take_profit_pct(self):
        settings = make_settings()
        assert settings.take_profit_pct == 5.0

    def test_default_trailing_stop_enabled(self):
        settings = make_settings()
        assert settings.trailing_stop_enabled is False

    def test_default_trailing_stop_pct(self):
        settings = make_settings()
        assert settings.trailing_stop_pct == 2.0

    def test_custom_take_profit_pct(self):
        settings = make_settings(take_profit_pct=10.0)
        assert settings.take_profit_pct == 10.0

    def test_custom_trailing_stop(self):
        settings = make_settings(
            trailing_stop_enabled=True, trailing_stop_pct=3.0
        )
        assert settings.trailing_stop_enabled is True
        assert settings.trailing_stop_pct == 3.0
