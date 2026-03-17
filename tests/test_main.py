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


class TestCalculatePortfolio:
    """Tests for TradingBot._calculate_portfolio() — TPV calculation."""

    def _make_bot(self) -> TradingBot:
        settings = make_settings()
        bot = TradingBot(settings=settings)
        bot._engine_manager = MagicMock()
        bot._engine_manager.engines = {}
        return bot

    def _make_ccxt_exchange(self, balance_total: dict, tickers: dict | None = None):
        """Create a mock exchange adapter wrapping a mock ccxt instance."""
        ccxt_mock = AsyncMock()
        ccxt_mock.fetch_balance = AsyncMock(return_value={"total": balance_total})
        if tickers is not None:
            ccxt_mock.fetch_tickers = AsyncMock(return_value=tickers)
            ccxt_mock.fetch_ticker = AsyncMock(
                side_effect=lambda pair: tickers.get(pair, {})
            )
        else:
            ccxt_mock.fetch_tickers = AsyncMock(return_value={})

        # Wrap in adapter-like object that _get_ccxt can unwrap
        wrapper = MagicMock()
        wrapper._exchange = ccxt_mock
        return wrapper

    @pytest.mark.asyncio
    async def test_stablecoin_only(self):
        """Exchange has only USDT — TPV equals cash."""
        bot = self._make_bot()
        bot._exchanges = [
            self._make_ccxt_exchange({"USDT": 1000.0})
        ]

        result = await bot._calculate_portfolio()

        assert result["cash_value"] == 1000.0
        assert result["position_value"] == 0.0
        assert result["total_value"] == 1000.0

    @pytest.mark.asyncio
    async def test_live_with_crypto_holdings(self):
        """Exchange has USDT=850 + BTC=0.0015 + ETH=0.05.
        TPV should include crypto valued via tickers."""
        tickers = {
            "BTC/USDT": {"last": 100000.0},
            "ETH/USDT": {"last": 3000.0},
        }
        bot = self._make_bot()
        bot._exchanges = [
            self._make_ccxt_exchange(
                {"USDT": 850.0, "BTC": 0.0015, "ETH": 0.05},
                tickers=tickers,
            )
        ]

        result = await bot._calculate_portfolio()

        assert result["cash_value"] == 850.0
        # BTC: 0.0015 * 100000 = 150, ETH: 0.05 * 3000 = 150
        assert result["position_value"] == 300.0
        assert result["total_value"] == 1150.0

    @pytest.mark.asyncio
    async def test_paper_mode_engine_positions_fallback(self):
        """Paper mode: exchange has USDT only, engine tracks positions."""
        bot = self._make_bot()
        bot._exchanges = [
            self._make_ccxt_exchange({"USDT": 900.0})
        ]

        # Simulate engine with a tracked position
        mock_engine = MagicMock()
        mock_engine.positions = {
            "BTC/USDT": {
                "quantity": 0.001,
                "current_price": 100000.0,
                "entry_price": 95000.0,
                "unrealized_pnl": 5.0,
            }
        }
        bot._engine_manager.engines = {"onchain_trader": mock_engine}

        result = await bot._calculate_portfolio()

        assert result["cash_value"] == 900.0
        # 0.001 * 100000 = 100
        assert result["position_value"] == 100.0
        assert result["total_value"] == 1000.0
        assert result["unrealized_pnl"] == 5.0

    @pytest.mark.asyncio
    async def test_engine_positions_uses_entry_price_fallback(self):
        """Engine position without current_price falls back to entry_price."""
        bot = self._make_bot()
        bot._exchanges = [
            self._make_ccxt_exchange({"USDT": 850.0})
        ]

        mock_engine = MagicMock()
        mock_engine.positions = {
            "BTC/USDT": {
                "quantity": 0.001,
                "entry_price": 100000.0,
                # no current_price set
            }
        }
        bot._engine_manager.engines = {"onchain_trader": mock_engine}

        result = await bot._calculate_portfolio()

        assert result["cash_value"] == 850.0
        assert result["position_value"] == 100.0
        assert result["total_value"] == 950.0

    @pytest.mark.asyncio
    async def test_fetch_tickers_failure_falls_back_to_individual(self):
        """If fetch_tickers fails, should try individual fetch_ticker calls."""
        bot = self._make_bot()

        # Create ccxt mock where fetch_tickers fails but fetch_ticker works
        ccxt_mock = AsyncMock()
        ccxt_mock.fetch_balance = AsyncMock(
            return_value={"total": {"USDT": 850.0, "BTC": 0.001}}
        )
        ccxt_mock.fetch_tickers = AsyncMock(side_effect=Exception("API error"))
        ccxt_mock.fetch_ticker = AsyncMock(
            return_value={"last": 100000.0}
        )

        wrapper = MagicMock()
        wrapper._exchange = ccxt_mock
        bot._exchanges = [wrapper]

        result = await bot._calculate_portfolio()

        assert result["cash_value"] == 850.0
        assert result["position_value"] == 100.0  # 0.001 * 100000
        assert result["total_value"] == 950.0

    @pytest.mark.asyncio
    async def test_no_exchange_available(self):
        """No exchange connected — portfolio should be all zeros."""
        bot = self._make_bot()
        bot._exchanges = []

        result = await bot._calculate_portfolio()

        assert result["total_value"] == 0.0
        assert result["cash_value"] == 0.0
        assert result["position_value"] == 0.0

    @pytest.mark.asyncio
    async def test_multiple_stablecoins(self):
        """Multiple stablecoins should all count as cash."""
        bot = self._make_bot()
        bot._exchanges = [
            self._make_ccxt_exchange(
                {"USDT": 500.0, "USDC": 300.0, "BUSD": 200.0}
            )
        ]

        result = await bot._calculate_portfolio()

        assert result["cash_value"] == 1000.0
        assert result["total_value"] == 1000.0

    @pytest.mark.asyncio
    async def test_ticker_preferred_over_engine_positions(self):
        """When exchange has crypto and tickers work, engine positions
        should NOT be used (would double-count)."""
        tickers = {"BTC/USDT": {"last": 100000.0}}
        bot = self._make_bot()
        bot._exchanges = [
            self._make_ccxt_exchange(
                {"USDT": 850.0, "BTC": 0.0015},
                tickers=tickers,
            )
        ]

        # Engine also tracks this position — should NOT be double counted
        mock_engine = MagicMock()
        mock_engine.positions = {
            "BTC/USDT": {
                "quantity": 0.0015,
                "current_price": 100000.0,
                "entry_price": 95000.0,
                "unrealized_pnl": 7.5,
            }
        }
        bot._engine_manager.engines = {"onchain_trader": mock_engine}

        result = await bot._calculate_portfolio()

        assert result["cash_value"] == 850.0
        # Should be 150 from ticker, NOT 300 from double counting
        assert result["position_value"] == 150.0
        assert result["total_value"] == 1000.0
        # unrealized_pnl still comes from engine positions
        assert result["unrealized_pnl"] == 7.5
