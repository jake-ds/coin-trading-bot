"""Tests for BinanceFuturesAdapter — all exchange calls are mocked."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.exchanges.binance_futures import BinanceFuturesAdapter
from bot.models import OrderSide, OrderStatus, OrderType


@pytest.fixture
def adapter():
    """Create adapter with mocked ccxt exchange."""
    with patch("bot.exchanges.binance_futures.ccxt.binance") as mock_cls:
        mock_exchange = MagicMock()
        mock_cls.return_value = mock_exchange
        ad = BinanceFuturesAdapter(
            api_key="test", secret_key="test", testnet=True
        )
        yield ad


class TestBinanceFuturesInit:
    def test_name(self, adapter):
        assert adapter.name == "binance_futures"


class TestGetTicker:
    @pytest.mark.asyncio
    async def test_get_ticker(self, adapter):
        adapter._exchange.fetch_ticker = AsyncMock(
            return_value={
                "bid": 50000.0,
                "ask": 50010.0,
                "last": 50005.0,
                "baseVolume": 1234.5,
            }
        )
        ticker = await adapter.get_ticker("BTC/USDT")
        assert ticker["bid"] == 50000.0
        assert ticker["ask"] == 50010.0
        assert ticker["last"] == 50005.0
        assert ticker["volume"] == 1234.5


class TestGetOHLCV:
    @pytest.mark.asyncio
    async def test_get_ohlcv(self, adapter):
        ts = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        adapter._exchange.fetch_ohlcv = AsyncMock(
            return_value=[
                [ts, 50000, 51000, 49000, 50500, 100.0],
            ]
        )
        candles = await adapter.get_ohlcv("BTC/USDT", "1h", limit=1)
        assert len(candles) == 1
        assert candles[0].symbol == "BTC/USDT"
        assert candles[0].close == 50500.0


class TestGetBalance:
    @pytest.mark.asyncio
    async def test_get_balance(self, adapter):
        adapter._exchange.fetch_balance = AsyncMock(
            return_value={"free": {"USDT": 5000.0, "BTC": 0.1}}
        )
        balance = await adapter.get_balance()
        assert balance["USDT"] == 5000.0
        assert balance["BTC"] == 0.1


class TestCreateOrder:
    @pytest.mark.asyncio
    async def test_create_market_order(self, adapter):
        adapter._exchange.create_order = AsyncMock(
            return_value={
                "id": "12345",
                "symbol": "BTC/USDT",
                "side": "buy",
                "type": "market",
                "amount": 0.01,
                "price": None,
                "status": "closed",
                "average": 50000.0,
                "filled": 0.01,
                "fee": {"cost": 0.5},
                "lastTradeTimestamp": 1704067200000,
            }
        )
        order = await adapter.create_order(
            "BTC/USDT", OrderSide.BUY, OrderType.MARKET, 0.01
        )
        assert order.id == "12345"
        assert order.exchange == "binance_futures"
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 50000.0


class TestCancelOrder:
    @pytest.mark.asyncio
    async def test_cancel_success(self, adapter):
        adapter._exchange.cancel_order = AsyncMock(return_value={})
        assert await adapter.cancel_order("123", "BTC/USDT") is True


class TestFuturesSpecific:
    @pytest.mark.asyncio
    async def test_set_leverage(self, adapter):
        adapter._exchange.set_leverage = AsyncMock(return_value={})
        await adapter.set_leverage("BTC/USDT", 5)
        adapter._exchange.set_leverage.assert_called_once_with(5, "BTC/USDT")

    @pytest.mark.asyncio
    async def test_get_funding_rate(self, adapter):
        adapter._exchange.fetch_funding_rate = AsyncMock(
            return_value={
                "fundingRate": 0.0003,
                "markPrice": 50000.0,
                "indexPrice": 49990.0,
                "fundingTimestamp": 1704067200000,
            }
        )
        rate = await adapter.get_funding_rate("BTC/USDT")
        assert rate["funding_rate"] == 0.0003
        assert rate["mark_price"] == 50000.0
        assert rate["index_price"] == 49990.0

    @pytest.mark.asyncio
    async def test_get_positions(self, adapter):
        adapter._exchange.fetch_positions = AsyncMock(
            return_value=[
                {
                    "symbol": "BTC/USDT",
                    "side": "long",
                    "contracts": 0.1,
                    "entryPrice": 50000.0,
                    "markPrice": 51000.0,
                    "unrealizedPnl": 100.0,
                    "leverage": 5,
                    "marginMode": "cross",
                },
                {
                    "symbol": "ETH/USDT",
                    "side": "short",
                    "contracts": 0,  # No position
                    "entryPrice": 0,
                    "markPrice": 0,
                    "unrealizedPnl": 0,
                    "leverage": 1,
                    "marginMode": "cross",
                },
            ]
        )
        positions = await adapter.get_positions()
        assert len(positions) == 1  # Only non-zero
        assert positions[0]["symbol"] == "BTC/USDT"
        assert positions[0]["contracts"] == 0.1
        assert positions[0]["side"] == "long"

    @pytest.mark.asyncio
    async def test_set_margin_mode(self, adapter):
        adapter._exchange.set_margin_mode = AsyncMock(return_value={})
        await adapter.set_margin_mode("BTC/USDT", "isolated")
        adapter._exchange.set_margin_mode.assert_called_once_with(
            "isolated", "BTC/USDT"
        )

    @pytest.mark.asyncio
    async def test_close(self, adapter):
        adapter._exchange.close = AsyncMock()
        await adapter.close()
        adapter._exchange.close.assert_called_once()
