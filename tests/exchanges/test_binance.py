"""Tests for Binance exchange adapter with mocked ccxt."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.exchanges.binance import BinanceAdapter
from bot.models import OrderSide, OrderStatus, OrderType


@pytest.fixture
def adapter():
    """Create a BinanceAdapter with mocked ccxt exchange."""
    with patch("bot.exchanges.binance.ccxt.binance") as mock_cls:
        mock_exchange = MagicMock()
        mock_cls.return_value = mock_exchange
        a = BinanceAdapter(api_key="test", secret_key="test", testnet=True)
        yield a, mock_exchange


class TestBinanceAdapter:
    def test_name(self, adapter):
        a, _ = adapter
        assert a.name == "binance"

    @pytest.mark.asyncio
    async def test_get_ticker(self, adapter):
        a, mock_ex = adapter
        mock_ex.fetch_ticker = AsyncMock(return_value={
            "bid": 50000.0,
            "ask": 50100.0,
            "last": 50050.0,
            "baseVolume": 1234.56,
        })
        ticker = await a.get_ticker("BTC/USDT")
        assert ticker["bid"] == 50000.0
        assert ticker["ask"] == 50100.0
        assert ticker["last"] == 50050.0
        assert ticker["volume"] == 1234.56
        mock_ex.fetch_ticker.assert_called_once_with("BTC/USDT")

    @pytest.mark.asyncio
    async def test_get_ticker_invalid_symbol(self, adapter):
        import ccxt.async_support as ccxt_async

        a, mock_ex = adapter
        mock_ex.fetch_ticker = AsyncMock(side_effect=ccxt_async.BadSymbol("bad"))
        with pytest.raises(ValueError, match="Invalid symbol"):
            await a.get_ticker("INVALID/PAIR")

    @pytest.mark.asyncio
    async def test_get_ticker_network_error(self, adapter):
        import ccxt.async_support as ccxt_async

        a, mock_ex = adapter
        mock_ex.fetch_ticker = AsyncMock(side_effect=ccxt_async.NetworkError("timeout"))
        with pytest.raises(ConnectionError):
            await a.get_ticker("BTC/USDT")

    @pytest.mark.asyncio
    async def test_get_ohlcv(self, adapter):
        a, mock_ex = adapter
        ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=[
            [ts, 100.0, 110.0, 95.0, 105.0, 1000.0],
            [ts + 3600000, 105.0, 115.0, 100.0, 110.0, 1200.0],
        ])
        candles = await a.get_ohlcv("BTC/USDT", "1h", 2)
        assert len(candles) == 2
        assert candles[0].open == 100.0
        assert candles[0].high == 110.0
        assert candles[0].symbol == "BTC/USDT"
        assert candles[1].close == 110.0

    @pytest.mark.asyncio
    async def test_get_balance(self, adapter):
        a, mock_ex = adapter
        mock_ex.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 10000.0, "BTC": 0.5, "ETH": 0.0},
        })
        balances = await a.get_balance()
        assert balances["USDT"] == 10000.0
        assert balances["BTC"] == 0.5
        assert "ETH" not in balances  # Zero balances excluded

    @pytest.mark.asyncio
    async def test_create_market_order(self, adapter):
        a, mock_ex = adapter
        mock_ex.create_order = AsyncMock(return_value={
            "id": "12345",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "market",
            "amount": 0.1,
            "price": None,
            "status": "closed",
            "average": 50000.0,
            "filled": 0.1,
            "fee": {"cost": 0.05, "currency": "USDT"},
        })
        order = await a.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )
        assert order.id == "12345"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.MARKET
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 50000.0

    @pytest.mark.asyncio
    async def test_create_limit_order(self, adapter):
        a, mock_ex = adapter
        mock_ex.create_order = AsyncMock(return_value={
            "id": "12346",
            "symbol": "ETH/USDT",
            "side": "sell",
            "type": "limit",
            "amount": 5.0,
            "price": 3000.0,
            "status": "open",
            "average": None,
            "filled": 0,
        })
        order = await a.create_order(
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=5.0,
            price=3000.0,
        )
        assert order.id == "12346"
        assert order.side == OrderSide.SELL
        assert order.type == OrderType.LIMIT
        assert order.price == 3000.0
        assert order.status == OrderStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, adapter):
        a, mock_ex = adapter
        mock_ex.cancel_order = AsyncMock(return_value={})
        result = await a.cancel_order("12345", "BTC/USDT")
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, adapter):
        import ccxt.async_support as ccxt_async

        a, mock_ex = adapter
        mock_ex.cancel_order = AsyncMock(side_effect=ccxt_async.OrderNotFound("not found"))
        result = await a.cancel_order("nonexistent", "BTC/USDT")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_order_status(self, adapter):
        a, mock_ex = adapter
        mock_ex.fetch_order = AsyncMock(return_value={
            "id": "12345",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "amount": 1.0,
            "price": 50000.0,
            "status": "closed",
            "average": 49950.0,
            "filled": 1.0,
            "lastTradeTimestamp": 1704067200000,
            "fee": {"cost": 0.5},
        })
        order = await a.get_order_status("12345", "BTC/USDT")
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 49950.0

    @pytest.mark.asyncio
    async def test_get_order_book(self, adapter):
        a, mock_ex = adapter
        mock_ex.fetch_order_book = AsyncMock(return_value={
            "bids": [[50000.0, 1.0], [49999.0, 2.0]],
            "asks": [[50001.0, 0.5], [50002.0, 1.5]],
        })
        book = await a.get_order_book("BTC/USDT", limit=5)
        assert len(book["bids"]) == 2
        assert len(book["asks"]) == 2
        assert book["bids"][0] == [50000.0, 1.0]

    @pytest.mark.asyncio
    async def test_close(self, adapter):
        a, mock_ex = adapter
        mock_ex.close = AsyncMock()
        await a.close()
        mock_ex.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_insufficient_funds(self, adapter):
        import ccxt.async_support as ccxt_async

        a, mock_ex = adapter
        mock_ex.create_order = AsyncMock(
            side_effect=ccxt_async.InsufficientFunds("not enough")
        )
        with pytest.raises(ValueError, match="Insufficient funds"):
            await a.create_order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
            )
