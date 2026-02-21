"""Tests for TradingBot orchestrator."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from bot.config import Settings, TradingMode
from bot.main import TradingBot


def make_settings(**kwargs):
    """Create test settings with safe defaults."""
    defaults = {
        "trading_mode": TradingMode.PAPER,
        "database_url": "sqlite+aiosqlite:///:memory:",
        "binance_api_key": "",
        "upbit_api_key": "",
        "symbols": ["BTC/USDT"],
        "loop_interval_seconds": 1,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


class TestTradingBot:
    def test_init_default(self):
        settings = make_settings()
        bot = TradingBot(settings=settings)
        assert bot._running is False
        assert bot._settings.trading_mode == TradingMode.PAPER

    @pytest.mark.asyncio
    async def test_initialize_no_exchanges(self):
        """Bot should initialize even without exchange credentials."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        assert bot._store is not None
        assert bot._risk_manager is not None
        assert len(bot._exchanges) == 0

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown(self):
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()
        await bot.shutdown()

        assert bot._running is False

    def test_stop(self):
        settings = make_settings()
        bot = TradingBot(settings=settings)
        bot._running = True
        bot.stop()
        assert bot._running is False

    @pytest.mark.asyncio
    async def test_trading_cycle_no_exchanges(self):
        """Trading cycle should handle no exchanges gracefully."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Should not crash
        await bot._trading_cycle()

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_paper_mode_creates_paper_engines(self):
        settings = make_settings()
        bot = TradingBot(settings=settings)
        # No exchange keys so no engines created
        await bot.initialize()
        assert len(bot._execution_engines) == 0
        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_run_trading_loop_stops(self):
        """Trading loop should stop when stop() is called."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Schedule stop after a short delay
        async def delayed_stop():
            await asyncio.sleep(0.05)
            bot.stop()

        asyncio.create_task(delayed_stop())
        await bot.run_trading_loop()

        assert bot._running is False
        await bot.shutdown()


class TestTelegramWiring:
    def test_telegram_not_created_when_not_configured(self):
        """Telegram notifier should not be created when credentials are missing."""
        settings = make_settings(telegram_bot_token="", telegram_chat_id="")
        bot = TradingBot(settings=settings)
        bot._init_telegram()
        assert bot._telegram is None

    def test_telegram_created_when_configured(self):
        """Telegram notifier should be created when both token and chat_id are set."""
        settings = make_settings(
            telegram_bot_token="test-token", telegram_chat_id="12345"
        )
        bot = TradingBot(settings=settings)
        bot._init_telegram()
        assert bot._telegram is not None
        assert bot._telegram._bot_token == "test-token"
        assert bot._telegram._chat_id == "12345"

    def test_telegram_skipped_if_only_token(self):
        """Telegram notifier should not be created if only token is set (no chat_id)."""
        settings = make_settings(telegram_bot_token="test-token", telegram_chat_id="")
        bot = TradingBot(settings=settings)
        bot._init_telegram()
        assert bot._telegram is None

    def test_telegram_skipped_if_only_chat_id(self):
        """Telegram notifier should not be created if only chat_id is set (no token)."""
        settings = make_settings(telegram_bot_token="", telegram_chat_id="12345")
        bot = TradingBot(settings=settings)
        bot._init_telegram()
        assert bot._telegram is None

    @pytest.mark.asyncio
    async def test_telegram_shutdown_notification(self):
        """Bot sends Telegram notification on shutdown when configured."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Inject mock telegram
        mock_telegram = AsyncMock()
        mock_telegram.send_message = AsyncMock(return_value=True)
        bot._telegram = mock_telegram

        await bot.shutdown()

        mock_telegram.send_message.assert_called_with("Bot shutting down.")

    @pytest.mark.asyncio
    async def test_telegram_error_notification_on_cycle_error(self):
        """Bot sends Telegram error notification with full traceback when trading cycle raises."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Inject mock telegram
        mock_telegram = AsyncMock()
        mock_telegram.notify_error = AsyncMock(return_value=True)
        bot._telegram = mock_telegram

        # Make collector raise an error
        bot._collector = AsyncMock()
        bot._collector.collect_once = AsyncMock(
            side_effect=RuntimeError("test error")
        )

        bot._running = True

        async def stop_after_one():
            await asyncio.sleep(0.05)
            bot.stop()

        asyncio.create_task(stop_after_one())
        await bot.run_trading_loop()

        # Telegram receives full traceback (not just error message)
        mock_telegram.notify_error.assert_called()
        error_msg = mock_telegram.notify_error.call_args[0][0]
        assert "test error" in error_msg
        assert "Traceback" in error_msg

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_no_telegram_error_when_not_configured(self):
        """No crash when Telegram is not configured and an error occurs."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()
        assert bot._telegram is None

        # Make collector raise an error
        bot._collector = AsyncMock()
        bot._collector.collect_once = AsyncMock(
            side_effect=RuntimeError("test error")
        )

        bot._running = True

        async def stop_after_one():
            await asyncio.sleep(0.05)
            bot.stop()

        asyncio.create_task(stop_after_one())
        # Should not crash
        await bot.run_trading_loop()

        await bot.shutdown()


class TestResilientExchangeWiring:
    @pytest.mark.asyncio
    async def test_exchanges_wrapped_in_resilient(self):
        """Exchange adapters should be wrapped in ResilientExchange."""
        from bot.execution.resilient import ResilientExchange

        settings = make_settings(
            binance_api_key="test-key", binance_secret_key="test-secret"
        )
        bot = TradingBot(settings=settings)

        mock_adapter = AsyncMock()
        type(mock_adapter).name = PropertyMock(return_value="binance")

        with patch("bot.main.ExchangeFactory.create", return_value=mock_adapter):
            bot._init_exchanges()

        assert len(bot._exchanges) == 1
        assert isinstance(bot._exchanges[0], ResilientExchange)
        assert bot._exchanges[0].name == "binance"

    @pytest.mark.asyncio
    async def test_multiple_exchanges_wrapped(self):
        """Both Binance and Upbit adapters should be wrapped in ResilientExchange."""
        from bot.execution.resilient import ResilientExchange

        settings = make_settings(
            binance_api_key="bkey",
            binance_secret_key="bsecret",
            upbit_api_key="ukey",
            upbit_secret_key="usecret",
        )
        bot = TradingBot(settings=settings)

        mock_binance = AsyncMock()
        type(mock_binance).name = PropertyMock(return_value="binance")
        mock_upbit = AsyncMock()
        type(mock_upbit).name = PropertyMock(return_value="upbit")

        def create_side_effect(name, **kwargs):
            if name == "binance":
                return mock_binance
            return mock_upbit

        with patch(
            "bot.main.ExchangeFactory.create", side_effect=create_side_effect
        ):
            bot._init_exchanges()

        assert len(bot._exchanges) == 2
        assert all(isinstance(e, ResilientExchange) for e in bot._exchanges)

    @pytest.mark.asyncio
    async def test_resilient_exchange_used_in_execution_engine(self):
        """ExecutionEngine should receive a ResilientExchange, not a raw adapter."""
        from bot.execution.resilient import ResilientExchange

        settings = make_settings(
            binance_api_key="test-key", binance_secret_key="test-secret"
        )
        bot = TradingBot(settings=settings)

        mock_adapter = AsyncMock()
        type(mock_adapter).name = PropertyMock(return_value="binance")

        with patch("bot.main.ExchangeFactory.create", return_value=mock_adapter):
            bot._exchanges = []
            bot._init_exchanges()
            bot._store = AsyncMock()

            is_paper = bot._settings.trading_mode == TradingMode.PAPER
            from bot.execution.engine import ExecutionEngine

            for exchange in bot._exchanges:
                bot._execution_engines[exchange.name] = ExecutionEngine(
                    exchange=exchange,
                    store=bot._store,
                    paper_trading=is_paper,
                )

        assert "binance" in bot._execution_engines
        engine = bot._execution_engines["binance"]
        assert isinstance(engine._exchange, ResilientExchange)


class TestDashboardWiring:
    @pytest.mark.asyncio
    async def test_dashboard_state_updated_after_cycle(self):
        """Dashboard state should be updated after each trading cycle."""
        from bot.dashboard import app as dashboard_module

        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Reset dashboard state
        dashboard_module.update_state(
            status="stopped", trades=[], metrics={}, portfolio={}
        )

        await bot._trading_cycle()

        state = dashboard_module.get_state()
        assert state["status"] == "running"

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_dashboard_trade_appended_on_execution(self):
        """Executed trades should appear in dashboard state."""
        from bot.dashboard import app as dashboard_module
        from bot.models import (
            Order,
            OrderSide,
            OrderStatus,
            OrderType,
            SignalAction,
            TradingSignal,
        )

        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Reset dashboard state
        dashboard_module.update_state(
            status="stopped", trades=[], metrics={}, portfolio={}
        )

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

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = [mock_strategy]

            mock_order = Order(
                id="test-001",
                exchange="binance",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                price=0,
                quantity=0.01,
                status=OrderStatus.FILLED,
                created_at=datetime.now(timezone.utc),
                filled_price=50000.0,
                filled_quantity=0.01,
            )
            mock_engine = AsyncMock()
            mock_engine.execute_signal = AsyncMock(return_value=mock_order)
            bot._execution_engines = {"binance": mock_engine}

            await bot._trading_cycle()

        state = dashboard_module.get_state()
        assert len(state["trades"]) == 1
        assert state["trades"][0]["symbol"] == "BTC/USDT"
        assert state["trades"][0]["side"] == "BUY"

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_dashboard_status_stopped_on_shutdown(self):
        """Dashboard status should be 'stopped' after shutdown."""
        from bot.dashboard import app as dashboard_module

        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        await bot.shutdown()

        state = dashboard_module.get_state()
        assert state["status"] == "stopped"
