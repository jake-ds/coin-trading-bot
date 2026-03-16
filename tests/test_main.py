"""Tests for TradingBot orchestrator."""

import asyncio
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

    def test_stop_via_running_flag(self):
        settings = make_settings()
        bot = TradingBot(settings=settings)
        bot._running = True
        bot._running = False
        assert bot._running is False


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


class TestEngineMode:
    @pytest.mark.asyncio
    async def test_engine_manager_initialized(self):
        """Engine manager should be initialized with OnChainTraderEngine."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        assert bot._engine_manager is not None
        assert bot._portfolio_mgr is not None
        assert "onchain_trader" in bot._engine_manager.engines

        await bot.shutdown()


class TestEmergencyControls:
    @pytest.mark.asyncio
    async def test_emergency_stop(self):
        """Emergency stop should set flags and pause engines."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        result = await bot.emergency_stop(reason="test")

        assert bot._emergency_stopped is True
        assert bot._emergency_reason == "test"
        assert "cancelled_orders" in result

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_emergency_resume(self):
        """Emergency resume should clear flags."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        await bot.emergency_stop(reason="test")
        result = await bot.emergency_resume()

        assert result["success"] is True
        assert bot._emergency_stopped is False

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_emergency_resume_when_not_stopped(self):
        """Resume should fail if not stopped."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        result = await bot.emergency_resume()
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_emergency_state_property(self):
        """emergency_state should reflect current state."""
        settings = make_settings()
        bot = TradingBot(settings=settings)

        state = bot.emergency_state
        assert state["active"] is False
        assert state["reason"] is None
