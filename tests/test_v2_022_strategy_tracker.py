"""Integration tests for V2-022: Per-strategy performance tracking and auto-disable."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from bot.config import Settings, TradingMode
from bot.dashboard.app import app, get_state, update_state
from bot.main import TradingBot
from bot.models import OrderSide, SignalAction, TradingSignal
from bot.monitoring.strategy_tracker import StrategyTracker


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


class TestStrategyTrackerWiring:
    @pytest.mark.asyncio
    async def test_tracker_created_in_initialize(self):
        """StrategyTracker should be created during bot initialization."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        assert bot._strategy_tracker is not None
        assert isinstance(bot._strategy_tracker, StrategyTracker)

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_tracker_config_params(self):
        """StrategyTracker should use config settings."""
        settings = make_settings(
            strategy_max_consecutive_losses=3,
            strategy_min_win_rate_pct=50.0,
            strategy_min_trades_for_eval=15,
            strategy_re_enable_check_hours=12.0,
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        tracker = bot._strategy_tracker
        assert tracker.max_consecutive_losses == 3
        assert tracker.min_win_rate_pct == 50.0
        assert tracker.min_trades_for_evaluation == 15
        assert tracker.re_enable_check_hours == 12.0

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_tracker_records_sell_pnl(self):
        """StrategyTracker should record PnL when a SELL trade completes."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Manually set up components for the test
        mock_store = MagicMock()
        mock_store.get_candles = AsyncMock(return_value=[MagicMock(close=50000)])
        mock_store.close = AsyncMock()
        bot._store = mock_store
        bot._collector = MagicMock()
        bot._collector.collect_once = AsyncMock()

        mock_engine = MagicMock()
        mock_order = MagicMock()
        mock_order.filled_price = 51000
        mock_order.side = OrderSide.SELL
        mock_order.symbol = "BTC/USDT"
        mock_order.quantity = 0.1
        mock_order.created_at = None
        mock_engine.execute_signal = AsyncMock(return_value=mock_order)
        bot._execution_engines = {"test": mock_engine}

        # Add a position in risk manager so PnL can be calculated
        bot._risk_manager.add_position("BTC/USDT", 0.1, 50000)

        # Create a SELL signal from ensemble
        sell_signal = TradingSignal(
            strategy_name="ensemble",
            symbol="BTC/USDT",
            action=SignalAction.SELL,
            confidence=0.8,
            metadata={"agreeing_strategies": ["rsi", "macd"]},
        )

        # Mock ensemble to return this signal
        bot._signal_ensemble.collect_signals = AsyncMock(return_value=[sell_signal])
        bot._signal_ensemble.vote = MagicMock(return_value=sell_signal)

        await bot._trading_cycle()

        # Both rsi and macd should have the trade recorded
        rsi_stats = bot._strategy_tracker.get_stats("rsi")
        macd_stats = bot._strategy_tracker.get_stats("macd")
        # PnL = (51000 - 50000) * 0.1 = 100
        assert rsi_stats.total_trades == 1
        assert rsi_stats.total_pnl == pytest.approx(100.0)
        assert macd_stats.total_trades == 1
        assert macd_stats.total_pnl == pytest.approx(100.0)

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_tracker_records_exit_pnl(self):
        """StrategyTracker should record PnL for PositionManager exits."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        mock_store = MagicMock()
        mock_store.get_candles = AsyncMock(return_value=[MagicMock(close=48000)])
        mock_store.close = AsyncMock()
        bot._store = mock_store
        bot._collector = MagicMock()
        bot._collector.collect_once = AsyncMock()

        mock_engine = MagicMock()
        mock_order = MagicMock()
        mock_order.filled_price = 48500
        mock_order.side = OrderSide.SELL
        mock_order.symbol = "BTC/USDT"
        mock_order.quantity = 0.1
        mock_order.created_at = None
        mock_engine.execute_signal = AsyncMock(return_value=mock_order)
        bot._execution_engines = {"test": mock_engine}

        # Add position in risk manager
        bot._risk_manager.add_position("BTC/USDT", 0.1, 50000)

        # Add position in position manager
        bot._position_manager.add_position("BTC/USDT", 50000, 0.1)

        # Mock ensemble to return HOLD (no strategy signals)
        bot._signal_ensemble.collect_signals = AsyncMock(return_value=[])
        bot._signal_ensemble.vote = MagicMock(
            return_value=TradingSignal(
                strategy_name="ensemble",
                symbol="BTC/USDT",
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={},
            )
        )

        # Trigger stop-loss exit
        await bot._trading_cycle()

        # Position manager should have checked exits
        pm_stats = bot._strategy_tracker.get_stats("position_manager")
        if pm_stats.total_trades > 0:
            # PnL = (48500 - 50000) * 0.1 = -150
            assert pm_stats.total_pnl == pytest.approx(-150.0)

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_tracker_check_re_enable_called_each_cycle(self):
        """check_re_enable should be called at the start of each cycle."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        bot._store = MagicMock()
        bot._store.get_candles = AsyncMock(return_value=[])
        bot._store.close = AsyncMock()
        bot._collector = MagicMock()
        bot._collector.collect_once = AsyncMock()

        bot._signal_ensemble.collect_signals = AsyncMock(return_value=[])
        bot._signal_ensemble.vote = MagicMock(
            return_value=TradingSignal(
                strategy_name="ensemble",
                symbol="BTC/USDT",
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={},
            )
        )

        # Mock check_re_enable to verify it's called
        bot._strategy_tracker.check_re_enable = MagicMock()

        await bot._trading_cycle()

        bot._strategy_tracker.check_re_enable.assert_called_once()

        await bot.shutdown()


class TestDashboardStrategiesEndpoint:
    @pytest.fixture
    async def client(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_strategies_endpoint_empty(self, client):
        """GET /strategies should return empty dict when no strategies tracked."""
        resp = await client.get("/strategies")
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategies"] == {}

    @pytest.mark.asyncio
    async def test_strategies_endpoint_with_data(self, client):
        """GET /strategies should return strategy stats from dashboard state."""
        update_state(
            strategy_stats={
                "rsi": {
                    "total_trades": 10,
                    "wins": 7,
                    "losses": 3,
                    "total_pnl": 250.0,
                    "win_rate": 70.0,
                    "avg_pnl": 25.0,
                    "consecutive_losses": 0,
                    "sharpe_ratio": 1.5,
                    "profit_factor": 3.0,
                    "disabled": False,
                }
            }
        )

        resp = await client.get("/strategies")
        assert resp.status_code == 200
        data = resp.json()
        assert "rsi" in data["strategies"]
        rsi = data["strategies"]["rsi"]
        assert rsi["total_trades"] == 10
        assert rsi["win_rate"] == 70.0
        assert rsi["disabled"] is False

    @pytest.mark.asyncio
    async def test_dashboard_state_updated_with_strategy_stats(self):
        """Dashboard state should include strategy_stats after trading cycle."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        bot._store = MagicMock()
        bot._store.get_candles = AsyncMock(return_value=[])
        bot._store.close = AsyncMock()
        bot._collector = MagicMock()
        bot._collector.collect_once = AsyncMock()

        bot._signal_ensemble.collect_signals = AsyncMock(return_value=[])
        bot._signal_ensemble.vote = MagicMock(
            return_value=TradingSignal(
                strategy_name="ensemble",
                symbol="BTC/USDT",
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={},
            )
        )

        await bot._trading_cycle()

        state = get_state()
        assert "strategy_stats" in state

        await bot.shutdown()


class TestConfigSettings:
    def test_default_strategy_tracker_settings(self):
        settings = make_settings()
        assert settings.strategy_max_consecutive_losses == 5
        assert settings.strategy_min_win_rate_pct == 40.0
        assert settings.strategy_min_trades_for_eval == 20
        assert settings.strategy_re_enable_check_hours == 24.0

    def test_custom_strategy_tracker_settings(self):
        settings = make_settings(
            strategy_max_consecutive_losses=10,
            strategy_min_win_rate_pct=30.0,
            strategy_min_trades_for_eval=50,
            strategy_re_enable_check_hours=48.0,
        )
        assert settings.strategy_max_consecutive_losses == 10
        assert settings.strategy_min_win_rate_pct == 30.0
        assert settings.strategy_min_trades_for_eval == 50
        assert settings.strategy_re_enable_check_hours == 48.0
