"""Integration tests for the full trading pipeline."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, PropertyMock

import pytest

from bot.config import Settings, TradingMode
from bot.data.store import DataStore
from bot.execution.engine import ExecutionEngine
from bot.main import TradingBot
from bot.models import OHLCV, OrderSide, OrderStatus, SignalAction, TradingSignal
from bot.risk.manager import RiskManager
from bot.strategies.base import BaseStrategy


def make_ohlcv(price: float, i: int = 0) -> OHLCV:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return OHLCV(
        timestamp=base + timedelta(hours=i),
        open=price,
        high=price * 1.02,
        low=price * 0.98,
        close=price,
        volume=1000.0,
    )


class SimpleBuyStrategy(BaseStrategy):
    """Strategy that generates BUY when price > 100, SELL when < 95."""

    @property
    def name(self) -> str:
        return "simple_test"

    @property
    def required_history_length(self) -> int:
        return 1

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        last_price = ohlcv_data[-1].close
        if last_price > 100:
            action = SignalAction.BUY
        elif last_price < 95:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD
        return TradingSignal(
            strategy_name=self.name,
            symbol=kwargs.get("symbol", "BTC/USDT"),
            action=action,
            confidence=0.8,
        )


class TestFullPipeline:
    """E2E test: data collection -> strategy signal -> risk check -> order execution."""

    @pytest.mark.asyncio
    async def test_data_to_signal_to_risk_to_execution(self):
        """Test the full pipeline in paper trading mode."""
        # Setup components
        store = DataStore(database_url="sqlite+aiosqlite:///:memory:")
        await store.initialize()

        mock_exchange = AsyncMock()
        type(mock_exchange).name = PropertyMock(return_value="mock")
        mock_exchange.get_ticker = AsyncMock(
            return_value={"last": 105.0, "bid": 104.9, "ask": 105.1}
        )

        risk_manager = RiskManager(
            max_position_size_pct=10.0,
            stop_loss_pct=5.0,
            daily_loss_limit_pct=5.0,
            max_drawdown_pct=15.0,
            max_concurrent_positions=5,
        )

        engine = ExecutionEngine(
            exchange=mock_exchange,
            store=store,
            paper_trading=True,
        )

        strategy = SimpleBuyStrategy()

        # Create candle data (price > 100, should trigger BUY)
        candles = [make_ohlcv(105.0, i) for i in range(5)]

        # Run strategy
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY

        # Risk check
        validated_signal = risk_manager.validate_signal(signal)
        assert validated_signal.action == SignalAction.BUY

        # Execute
        qty = risk_manager.calculate_position_size(10000.0, 105.0)
        assert qty > 0

        order = await engine.execute_signal(validated_signal, quantity=qty)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.side == OrderSide.BUY

        # Verify trade stored
        trades = await store.get_trades()
        assert len(trades) == 1

        await store.close()

    @pytest.mark.asyncio
    async def test_risk_blocks_excessive_position(self):
        """Risk manager should block signals when limits are exceeded."""
        risk_manager = RiskManager(
            max_concurrent_positions=1,
        )

        signal = TradingSignal(
            strategy_name="test",
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            confidence=0.8,
        )

        # First buy should pass
        validated = risk_manager.validate_signal(signal)
        assert validated.action == SignalAction.BUY
        risk_manager.add_position("BTC/USDT", 0.1, 50000.0)

        # Second buy on different symbol should be blocked
        signal2 = TradingSignal(
            strategy_name="test",
            symbol="ETH/USDT",
            action=SignalAction.BUY,
            confidence=0.8,
        )
        validated2 = risk_manager.validate_signal(signal2)
        assert validated2.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_pipeline_with_sell_signal(self):
        """Test sell signal flow through the pipeline."""
        store = DataStore(database_url="sqlite+aiosqlite:///:memory:")
        await store.initialize()

        mock_exchange = AsyncMock()
        type(mock_exchange).name = PropertyMock(return_value="mock")
        mock_exchange.get_ticker = AsyncMock(
            return_value={"last": 90.0, "bid": 89.9, "ask": 90.1}
        )

        engine = ExecutionEngine(
            exchange=mock_exchange,
            store=store,
            paper_trading=True,
        )

        strategy = SimpleBuyStrategy()
        candles = [make_ohlcv(90.0, i) for i in range(5)]

        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.SELL

        order = await engine.execute_signal(signal, quantity=0.1)
        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.status == OrderStatus.FILLED

        await store.close()


class TestGracefulShutdown:
    @pytest.mark.asyncio
    async def test_bot_shutdown_during_trading(self):
        """Bot should shut down gracefully during active trading."""
        settings = Settings(
            trading_mode=TradingMode.PAPER,
            database_url="sqlite+aiosqlite:///:memory:",
            binance_api_key="",
            upbit_api_key="",
            symbols=["BTC/USDT"],
            loop_interval_seconds=1,
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Schedule shutdown after a brief delay
        async def delayed_shutdown():
            await asyncio.sleep(0.05)
            bot.stop()

        asyncio.create_task(delayed_shutdown())
        await bot.run_trading_loop()

        assert bot._running is False
        await bot.shutdown()


class TestConfigValidation:
    def test_paper_mode_default(self):
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
        )
        assert settings.trading_mode == TradingMode.PAPER

    def test_invalid_percentage_rejected(self):
        with pytest.raises(Exception):
            Settings(
                database_url="sqlite+aiosqlite:///:memory:",
                max_position_size_pct=200.0,
            )

    def test_invalid_log_level_rejected(self):
        with pytest.raises(Exception):
            Settings(
                database_url="sqlite+aiosqlite:///:memory:",
                log_level="INVALID",
            )
