"""Tests for execution engine."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from bot.data.store import DataStore
from bot.execution.engine import ExecutionEngine
from bot.models import OrderSide, OrderStatus, OrderType, SignalAction, TradingSignal


@pytest.fixture
def mock_exchange():
    exchange = AsyncMock()
    type(exchange).name = PropertyMock(return_value="mock")
    exchange.get_ticker = AsyncMock(return_value={"last": 50000.0, "bid": 49990, "ask": 50010})
    return exchange


@pytest.fixture
async def store():
    ds = DataStore(database_url="sqlite+aiosqlite:///:memory:")
    await ds.initialize()
    yield ds
    await ds.close()


def make_signal(action=SignalAction.BUY, symbol="BTC/USDT"):
    return TradingSignal(
        strategy_name="test", symbol=symbol, action=action, confidence=0.8,
    )


class TestExecutionEngine:
    @pytest.mark.asyncio
    async def test_paper_trade_buy(self, mock_exchange, store):
        engine = ExecutionEngine(mock_exchange, store, paper_trading=True)
        signal = make_signal(SignalAction.BUY)
        order = await engine.execute_signal(signal, quantity=0.1)

        assert order is not None
        assert order.id.startswith("paper-")
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 50000.0
        assert order.filled_quantity == 0.1

    @pytest.mark.asyncio
    async def test_paper_trade_sell(self, mock_exchange, store):
        engine = ExecutionEngine(mock_exchange, store, paper_trading=True)
        signal = make_signal(SignalAction.SELL)
        order = await engine.execute_signal(signal, quantity=0.5)

        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_hold_signal_skipped(self, mock_exchange, store):
        engine = ExecutionEngine(mock_exchange, store, paper_trading=True)
        signal = make_signal(SignalAction.HOLD)
        order = await engine.execute_signal(signal, quantity=0.1)
        assert order is None

    @pytest.mark.asyncio
    async def test_paper_trade_saves_to_store(self, mock_exchange, store):
        engine = ExecutionEngine(mock_exchange, store, paper_trading=True)
        signal = make_signal(SignalAction.BUY)
        await engine.execute_signal(signal, quantity=0.1)

        trades = await store.get_trades()
        assert len(trades) == 1
        assert trades[0]["side"] == "BUY"

    @pytest.mark.asyncio
    async def test_live_execute_success(self, mock_exchange, store):
        mock_exchange.create_order = AsyncMock(return_value=MagicMock(
            id="live-001",
            exchange="mock",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            price=0,
            quantity=0.1,
            status=OrderStatus.FILLED,
            created_at=None,
            filled_at=None,
            filled_price=50000.0,
            filled_quantity=0.1,
            fee=0.05,
        ))
        # Need real Order for store.save_trade
        from bot.models import Order
        mock_order = Order(
            id="live-001", exchange="mock", symbol="BTC/USDT",
            side=OrderSide.BUY, type=OrderType.MARKET, price=0,
            quantity=0.1, status=OrderStatus.FILLED,
        )
        mock_exchange.create_order = AsyncMock(return_value=mock_order)

        engine = ExecutionEngine(mock_exchange, store, paper_trading=False)
        signal = make_signal(SignalAction.BUY)
        order = await engine.execute_signal(signal, quantity=0.1)

        assert order is not None
        assert order.id == "live-001"

    @pytest.mark.asyncio
    async def test_live_execute_retry(self, mock_exchange, store):
        from bot.models import Order
        mock_order = Order(
            id="retry-001", exchange="mock", symbol="BTC/USDT",
            side=OrderSide.BUY, type=OrderType.MARKET, price=0,
            quantity=0.1, status=OrderStatus.FILLED,
        )
        mock_exchange.create_order = AsyncMock(
            side_effect=[ConnectionError("timeout"), mock_order]
        )

        engine = ExecutionEngine(
            mock_exchange, store, paper_trading=False,
            max_retries=3, retry_delay=0.01,
        )
        signal = make_signal(SignalAction.BUY)
        order = await engine.execute_signal(signal, quantity=0.1)

        assert order is not None
        assert mock_exchange.create_order.call_count == 2

    @pytest.mark.asyncio
    async def test_live_execute_max_retries_exceeded(self, mock_exchange, store):
        mock_exchange.create_order = AsyncMock(side_effect=ConnectionError("fail"))

        engine = ExecutionEngine(
            mock_exchange, store, paper_trading=False,
            max_retries=2, retry_delay=0.01,
        )
        signal = make_signal(SignalAction.BUY)
        order = await engine.execute_signal(signal, quantity=0.1)

        assert order is None

    @pytest.mark.asyncio
    async def test_cancel_order(self, mock_exchange, store):
        mock_exchange.cancel_order = AsyncMock(return_value=True)
        engine = ExecutionEngine(mock_exchange, store, paper_trading=False)
        result = await engine.cancel_order("order-1", "BTC/USDT")
        assert result is True
