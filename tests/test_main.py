"""Tests for TradingBot orchestrator."""

import asyncio

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
