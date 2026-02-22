"""Tests for DataCollector with mocked exchange."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, PropertyMock

import pytest

from bot.data.collector import DataCollector
from bot.data.store import DataStore
from bot.models import OHLCV


def make_candles(count: int, symbol: str = "BTC/USDT", base_hour: int = 0) -> list[OHLCV]:
    """Create sample OHLCV candles for testing."""
    return [
        OHLCV(
            timestamp=datetime(2024, 1, 1, base_hour + i, tzinfo=timezone.utc),
            open=100.0 + i,
            high=110.0 + i,
            low=95.0 + i,
            close=105.0 + i,
            volume=1000.0,
            symbol=symbol,
            timeframe="1h",
        )
        for i in range(count)
    ]


@pytest.fixture
def mock_exchange():
    """Create a mock exchange adapter."""
    exchange = AsyncMock()
    type(exchange).name = PropertyMock(return_value="mock_exchange")
    return exchange


@pytest.fixture
async def store():
    """Create an in-memory DataStore."""
    ds = DataStore(database_url="sqlite+aiosqlite:///:memory:")
    await ds.initialize()
    yield ds
    await ds.close()


class TestDataCollector:
    @pytest.mark.asyncio
    async def test_collect_once(self, mock_exchange, store):
        candles = make_candles(5)
        mock_exchange.get_ohlcv = AsyncMock(return_value=candles)

        collector = DataCollector(
            exchanges=[mock_exchange],
            store=store,
            symbols=["BTC/USDT"],
            timeframes=["1h"],
        )
        total = await collector.collect_once()
        assert total == 5

        stored = await store.get_candles("BTC/USDT", "1h")
        assert len(stored) == 5

    @pytest.mark.asyncio
    async def test_collect_multiple_symbols(self, mock_exchange, store):
        btc_candles = make_candles(3, symbol="BTC/USDT")
        eth_candles = make_candles(3, symbol="ETH/USDT")

        async def mock_ohlcv(symbol, **kwargs):
            if symbol == "BTC/USDT":
                return btc_candles
            return eth_candles

        mock_exchange.get_ohlcv = AsyncMock(side_effect=mock_ohlcv)

        collector = DataCollector(
            exchanges=[mock_exchange],
            store=store,
            symbols=["BTC/USDT", "ETH/USDT"],
            timeframes=["1h"],
        )
        total = await collector.collect_once()
        assert total == 6

    @pytest.mark.asyncio
    async def test_collect_multiple_timeframes(self, mock_exchange, store):
        candles = make_candles(2)
        mock_exchange.get_ohlcv = AsyncMock(return_value=candles)

        collector = DataCollector(
            exchanges=[mock_exchange],
            store=store,
            symbols=["BTC/USDT"],
            timeframes=["1h", "4h"],
        )
        total = await collector.collect_once()
        assert total == 4  # 2 candles × 2 timeframes

    @pytest.mark.asyncio
    async def test_collect_handles_exchange_error(self, mock_exchange, store):
        mock_exchange.get_ohlcv = AsyncMock(side_effect=ConnectionError("timeout"))

        collector = DataCollector(
            exchanges=[mock_exchange],
            store=store,
            symbols=["BTC/USDT"],
        )
        total = await collector.collect_once()
        assert total == 0

    @pytest.mark.asyncio
    async def test_collect_empty_candles(self, mock_exchange, store):
        mock_exchange.get_ohlcv = AsyncMock(return_value=[])

        collector = DataCollector(
            exchanges=[mock_exchange],
            store=store,
            symbols=["BTC/USDT"],
        )
        total = await collector.collect_once()
        assert total == 0

    @pytest.mark.asyncio
    async def test_backfill_no_existing_data(self, mock_exchange, store):
        candles = make_candles(10)
        mock_exchange.get_ohlcv = AsyncMock(return_value=candles)

        collector = DataCollector(
            exchanges=[mock_exchange],
            store=store,
            symbols=["BTC/USDT"],
        )
        count = await collector.backfill(mock_exchange, "BTC/USDT", "1h", limit=10)
        assert count == 10

    @pytest.mark.asyncio
    async def test_backfill_with_existing_data(self, mock_exchange, store):
        # First save some existing candles
        existing = make_candles(5)
        await store.save_candles(existing)

        # Now simulate fetching including old + new
        all_candles = make_candles(8)
        mock_exchange.get_ohlcv = AsyncMock(return_value=all_candles)

        collector = DataCollector(
            exchanges=[mock_exchange],
            store=store,
            symbols=["BTC/USDT"],
        )
        count = await collector.backfill(mock_exchange, "BTC/USDT", "1h")
        assert count == 3  # Only 3 new candles (hours 5, 6, 7)

    @pytest.mark.asyncio
    async def test_backfill_error_handling(self, mock_exchange, store):
        mock_exchange.get_ohlcv = AsyncMock(side_effect=ConnectionError("fail"))

        collector = DataCollector(
            exchanges=[mock_exchange],
            store=store,
            symbols=["BTC/USDT"],
        )
        count = await collector.backfill(mock_exchange, "BTC/USDT")
        assert count == 0

    @pytest.mark.asyncio
    async def test_default_timeframe(self, mock_exchange, store):
        collector = DataCollector(
            exchanges=[mock_exchange],
            store=store,
            symbols=["BTC/USDT"],
        )
        assert collector._timeframes == ["1h"]

    @pytest.mark.asyncio
    async def test_stop(self, mock_exchange, store):
        collector = DataCollector(
            exchanges=[mock_exchange],
            store=store,
            symbols=["BTC/USDT"],
        )
        assert collector._running is False
        collector._running = True
        collector.stop()
        assert collector._running is False
