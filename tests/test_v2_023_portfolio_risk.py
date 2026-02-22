"""Integration tests for V2-023: Portfolio-level risk management wiring in main.py."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.config import Settings, TradingMode
from bot.dashboard.app import update_state
from bot.main import TradingBot
from bot.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    SignalAction,
    TradingSignal,
)
from bot.risk.portfolio_risk import PortfolioRiskManager


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


@pytest.fixture(autouse=True)
def reset_dashboard_state():
    """Reset bot state before each test."""
    update_state(
        status="stopped",
        started_at=None,
        trades=[],
        metrics={},
        portfolio={"balances": {}, "positions": [], "total_value": 0.0},
        cycle_metrics={
            "cycle_count": 0,
            "average_cycle_duration": 0.0,
            "last_cycle_time": None,
        },
        strategy_stats={},
    )


class TestPortfolioRiskWiring:
    @pytest.mark.asyncio
    async def test_portfolio_risk_created_in_initialize(self):
        """PortfolioRiskManager should be created during bot initialization."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        assert bot._portfolio_risk is not None
        assert isinstance(bot._portfolio_risk, PortfolioRiskManager)

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_portfolio_risk_config_params(self):
        """PortfolioRiskManager should use config settings."""
        settings = make_settings(
            max_total_exposure_pct=50.0,
            max_correlation=0.7,
            correlation_window=20,
            max_positions_per_sector=2,
            max_portfolio_heat=0.10,
            sector_map={"BTC/USDT": "large_cap"},
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        prm = bot._portfolio_risk
        assert prm._max_total_exposure_pct == 50.0
        assert prm._max_correlation == 0.7
        assert prm._correlation_window == 20
        assert prm._max_positions_per_sector == 2
        assert prm._max_portfolio_heat == 0.10
        assert prm._sector_map == {"BTC/USDT": "large_cap"}

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_portfolio_value_synced_from_paper(self):
        """Portfolio risk manager should have its value synced from paper portfolio."""
        settings = make_settings(paper_initial_balance=20000.0)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Replace store with mock to run a cycle
        mock_store = MagicMock()
        mock_store.get_candles = AsyncMock(return_value=[])
        mock_store.close = AsyncMock()
        bot._store = mock_store
        bot._collector = None

        await bot._trading_cycle()

        assert bot._portfolio_risk.portfolio_value == pytest.approx(20000.0)

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_exposure_limit_blocks_buy(self):
        """BUY should be rejected when portfolio exposure limit would be exceeded."""
        settings = make_settings(
            max_total_exposure_pct=10.0,
            paper_initial_balance=10000.0,
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Pre-fill portfolio risk with a large position
        bot._portfolio_risk.update_portfolio_value(10000.0)
        bot._portfolio_risk.add_position("ETH/USDT", 5000.0)
        # Already at 50% exposure, but limit is 10%

        # Create a mock store with candles
        mock_candle = MagicMock()
        mock_candle.close = 50000.0
        mock_candle.high = 50500.0
        mock_candle.low = 49500.0
        mock_candle.open = 50000.0
        mock_candle.volume = 100.0

        mock_store = MagicMock()
        mock_store.get_candles = AsyncMock(return_value=[mock_candle] * 50)
        mock_store.close = AsyncMock()
        bot._store = mock_store
        bot._collector = None

        # Mock ensemble to return a BUY signal
        buy_signal = TradingSignal(
            strategy_name="test",
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            confidence=0.8,
        )
        bot._signal_ensemble.vote = MagicMock(return_value=buy_signal)
        bot._signal_ensemble.collect_signals = AsyncMock(return_value=[buy_signal])

        # Mock execution engine
        mock_engine = MagicMock()
        mock_engine.execute_signal = AsyncMock(return_value=None)
        bot._execution_engines = {"test": mock_engine}

        await bot._trading_cycle()

        # Execution should NOT have been called because portfolio risk blocked it
        mock_engine.execute_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_position_tracked_after_buy(self):
        """After BUY execution, position should be tracked in portfolio risk manager."""
        settings = make_settings(
            paper_initial_balance=10000.0,
            max_total_exposure_pct=90.0,
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Create mock candles
        mock_candle = MagicMock()
        mock_candle.close = 50000.0
        mock_candle.high = 50500.0
        mock_candle.low = 49500.0
        mock_candle.open = 50000.0
        mock_candle.volume = 100.0

        mock_store = MagicMock()
        mock_store.get_candles = AsyncMock(return_value=[mock_candle] * 50)
        mock_store.close = AsyncMock()
        bot._store = mock_store
        bot._collector = None

        # Mock ensemble to return BUY
        buy_signal = TradingSignal(
            strategy_name="test",
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            confidence=0.8,
        )
        bot._signal_ensemble.vote = MagicMock(return_value=buy_signal)
        bot._signal_ensemble.collect_signals = AsyncMock(return_value=[buy_signal])

        # Mock engine to return filled order
        now = datetime.now(timezone.utc)
        filled_order = Order(
            id="test-order-1",
            exchange="test",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.01,
            price=0,
            status=OrderStatus.FILLED,
            filled_price=50000.0,
            created_at=now,
        )
        mock_engine = MagicMock()
        mock_engine.execute_signal = AsyncMock(return_value=filled_order)
        bot._execution_engines = {"test": mock_engine}

        await bot._trading_cycle()

        # Position should be tracked in portfolio risk
        assert "BTC/USDT" in bot._portfolio_risk.positions

    @pytest.mark.asyncio
    async def test_position_removed_after_sell(self):
        """After SELL execution, position should be removed from portfolio risk manager."""
        settings = make_settings(
            paper_initial_balance=10000.0,
            max_total_exposure_pct=90.0,
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Pre-fill with existing position
        bot._portfolio_risk.update_portfolio_value(10000.0)
        bot._portfolio_risk.add_position("BTC/USDT", 5000.0)
        bot._risk_manager.add_position("BTC/USDT", 0.1, 50000.0)

        # Create mock candles
        mock_candle = MagicMock()
        mock_candle.close = 51000.0
        mock_candle.high = 51500.0
        mock_candle.low = 50500.0
        mock_candle.open = 51000.0
        mock_candle.volume = 100.0

        mock_store = MagicMock()
        mock_store.get_candles = AsyncMock(return_value=[mock_candle] * 50)
        mock_store.close = AsyncMock()
        bot._store = mock_store
        bot._collector = None

        # Mock ensemble to return SELL
        sell_signal = TradingSignal(
            strategy_name="test",
            symbol="BTC/USDT",
            action=SignalAction.SELL,
            confidence=0.8,
        )
        bot._signal_ensemble.vote = MagicMock(return_value=sell_signal)
        bot._signal_ensemble.collect_signals = AsyncMock(return_value=[sell_signal])

        # Mock engine to return filled sell order
        now = datetime.now(timezone.utc)
        filled_order = Order(
            id="test-order-2",
            exchange="test",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=0.1,
            price=0,
            status=OrderStatus.FILLED,
            filled_price=51000.0,
            created_at=now,
        )
        mock_engine = MagicMock()
        mock_engine.execute_signal = AsyncMock(return_value=filled_order)
        bot._execution_engines = {"test": mock_engine}

        await bot._trading_cycle()

        # Position should be removed from portfolio risk
        assert "BTC/USDT" not in bot._portfolio_risk.positions


class TestPortfolioRiskConfigDefaults:
    def test_default_config_values(self):
        """Config should have reasonable defaults for portfolio risk settings."""
        settings = make_settings()
        assert settings.max_total_exposure_pct == 60.0
        assert settings.max_correlation == 0.8
        assert settings.correlation_window == 30
        assert settings.max_positions_per_sector == 3
        assert settings.max_portfolio_heat == 0.15
        assert settings.sector_map == {}
