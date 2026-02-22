"""Tests for DataStore with in-memory SQLite."""

from datetime import datetime, timedelta

import pytest

from bot.data.store import DataStore
from bot.models import OHLCV, Order, OrderSide, OrderStatus, OrderType


@pytest.fixture
async def store():
    """Create an in-memory DataStore for testing."""
    ds = DataStore(database_url="sqlite+aiosqlite:///:memory:")
    await ds.initialize()
    yield ds
    await ds.close()


class TestDataStoreCandles:
    @pytest.mark.asyncio
    async def test_save_and_get_candles(self, store):
        candles = [
            OHLCV(
                timestamp=datetime(2024, 1, 1, i),
                open=100.0 + i,
                high=110.0 + i,
                low=95.0 + i,
                close=105.0 + i,
                volume=1000.0,
                symbol="BTC/USDT",
                timeframe="1h",
            )
            for i in range(5)
        ]
        await store.save_candles(candles)

        result = await store.get_candles("BTC/USDT", "1h")
        assert len(result) == 5
        assert result[0].timestamp == datetime(2024, 1, 1, 0)
        assert result[4].timestamp == datetime(2024, 1, 1, 4)

    @pytest.mark.asyncio
    async def test_get_candles_by_date_range(self, store):
        candles = [
            OHLCV(
                timestamp=datetime(2024, 1, d, 0),
                open=100.0,
                high=110.0,
                low=95.0,
                close=105.0,
                volume=1000.0,
                symbol="BTC/USDT",
                timeframe="1d",
            )
            for d in range(1, 11)
        ]
        await store.save_candles(candles)

        result = await store.get_candles(
            "BTC/USDT",
            "1d",
            start=datetime(2024, 1, 3),
            end=datetime(2024, 1, 7),
        )
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_get_candles_limit(self, store):
        candles = [
            OHLCV(
                timestamp=datetime(2024, 1, 1, i),
                open=100.0,
                high=110.0,
                low=95.0,
                close=105.0,
                volume=1000.0,
                symbol="ETH/USDT",
                timeframe="1h",
            )
            for i in range(10)
        ]
        await store.save_candles(candles)

        result = await store.get_candles("ETH/USDT", "1h", limit=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_candles_empty(self, store):
        result = await store.get_candles("NONEXISTENT/PAIR", "1h")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_candles_different_symbols(self, store):
        btc = [
            OHLCV(
                timestamp=datetime(2024, 1, 1),
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=100.0,
                symbol="BTC/USDT",
                timeframe="1h",
            )
        ]
        eth = [
            OHLCV(
                timestamp=datetime(2024, 1, 1),
                open=3000.0,
                high=3100.0,
                low=2900.0,
                close=3050.0,
                volume=500.0,
                symbol="ETH/USDT",
                timeframe="1h",
            )
        ]
        await store.save_candles(btc)
        await store.save_candles(eth)

        btc_result = await store.get_candles("BTC/USDT")
        eth_result = await store.get_candles("ETH/USDT")
        assert len(btc_result) == 1
        assert len(eth_result) == 1
        assert btc_result[0].open == 50000.0
        assert eth_result[0].open == 3000.0


class TestDataStoreTrades:
    @pytest.mark.asyncio
    async def test_save_and_get_trade(self, store):
        order = Order(
            id="trade-001",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            price=0,
            quantity=0.1,
            status=OrderStatus.FILLED,
            filled_price=50000.0,
            filled_quantity=0.1,
            fee=0.05,
        )
        await store.save_trade(order)

        trades = await store.get_trades()
        assert len(trades) == 1
        assert trades[0]["order_id"] == "trade-001"
        assert trades[0]["symbol"] == "BTC/USDT"
        assert trades[0]["side"] == "BUY"
        assert trades[0]["price"] == 50000.0

    @pytest.mark.asyncio
    async def test_get_trades_by_symbol(self, store):
        for i, symbol in enumerate(["BTC/USDT", "ETH/USDT", "BTC/USDT"]):
            order = Order(
                id=f"trade-{i}",
                exchange="binance",
                symbol=symbol,
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                price=100.0,
                quantity=1.0,
                status=OrderStatus.FILLED,
            )
            await store.save_trade(order)

        btc_trades = await store.get_trades(symbol="BTC/USDT")
        assert len(btc_trades) == 2

    @pytest.mark.asyncio
    async def test_get_trades_by_date_range(self, store):
        base_time = datetime(2024, 1, 5)
        for i in range(5):
            order = Order(
                id=f"trade-{i}",
                exchange="binance",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                price=0,
                quantity=1.0,
                status=OrderStatus.FILLED,
                created_at=base_time + timedelta(days=i),
            )
            await store.save_trade(order)

        trades = await store.get_trades(
            start=datetime(2024, 1, 6),
            end=datetime(2024, 1, 8),
        )
        assert len(trades) == 3

    @pytest.mark.asyncio
    async def test_get_trades_empty(self, store):
        trades = await store.get_trades()
        assert trades == []


class TestDataStorePortfolio:
    @pytest.mark.asyncio
    async def test_save_and_get_portfolio_snapshot(self, store):
        await store.save_portfolio_snapshot(
            total_value=50000.0,
            unrealized_pnl=500.0,
            balances={"USDT": 10000.0, "BTC": 0.5},
            positions=[{"symbol": "BTC/USDT", "quantity": 0.5}],
        )

        snapshot = await store.get_latest_portfolio_snapshot()
        assert snapshot is not None
        assert snapshot["total_value"] == 50000.0
        assert snapshot["unrealized_pnl"] == 500.0
        assert snapshot["balances"]["USDT"] == 10000.0
        assert len(snapshot["positions"]) == 1

    @pytest.mark.asyncio
    async def test_get_latest_snapshot_returns_most_recent(self, store):
        await store.save_portfolio_snapshot(total_value=10000.0)
        await store.save_portfolio_snapshot(total_value=20000.0)
        await store.save_portfolio_snapshot(total_value=30000.0)

        snapshot = await store.get_latest_portfolio_snapshot()
        assert snapshot["total_value"] == 30000.0

    @pytest.mark.asyncio
    async def test_get_latest_snapshot_empty(self, store):
        snapshot = await store.get_latest_portfolio_snapshot()
        assert snapshot is None


class TestDataStoreLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self):
        ds = DataStore(database_url="sqlite+aiosqlite:///:memory:")
        await ds.initialize()
        # Should be able to query without errors
        result = await ds.get_candles("BTC/USDT")
        assert result == []
        await ds.close()
