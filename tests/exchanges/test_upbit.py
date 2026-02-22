"""Tests for Upbit exchange adapter with mocked ccxt."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.exchanges.upbit import UpbitAdapter
from bot.models import OrderSide, OrderStatus, OrderType


@pytest.fixture
def adapter():
    """Create an UpbitAdapter with mocked ccxt exchange."""
    with patch("bot.exchanges.upbit.ccxt.upbit") as mock_cls:
        mock_exchange = MagicMock()
        mock_cls.return_value = mock_exchange
        a = UpbitAdapter(api_key="test", secret_key="test")
        yield a, mock_exchange


class TestUpbitAdapter:
    def test_name(self, adapter):
        a, _ = adapter
        assert a.name == "upbit"

    @pytest.mark.asyncio
    async def test_get_ticker_krw_pair(self, adapter):
        a, mock_ex = adapter
        mock_ex.fetch_ticker = AsyncMock(return_value={
            "bid": 95000000.0,
            "ask": 95100000.0,
            "last": 95050000.0,
            "baseVolume": 500.0,
        })
        ticker = await a.get_ticker("BTC/KRW")
        assert ticker["bid"] == 95000000.0
        assert ticker["last"] == 95050000.0
        mock_ex.fetch_ticker.assert_called_once_with("BTC/KRW")

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
            await a.get_ticker("BTC/KRW")

    @pytest.mark.asyncio
    async def test_get_ohlcv(self, adapter):
        a, mock_ex = adapter
        ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=[
            [ts, 95000000.0, 96000000.0, 94000000.0, 95500000.0, 300.0],
        ])
        candles = await a.get_ohlcv("BTC/KRW", "1h", 1)
        assert len(candles) == 1
        assert candles[0].open == 95000000.0
        assert candles[0].symbol == "BTC/KRW"

    @pytest.mark.asyncio
    async def test_get_balance(self, adapter):
        a, mock_ex = adapter
        mock_ex.fetch_balance = AsyncMock(return_value={
            "free": {"KRW": 5000000.0, "BTC": 0.1, "ETH": 0.0},
        })
        balances = await a.get_balance()
        assert balances["KRW"] == 5000000.0
        assert balances["BTC"] == 0.1
        assert "ETH" not in balances

    @pytest.mark.asyncio
    async def test_create_market_order(self, adapter):
        a, mock_ex = adapter
        mock_ex.create_order = AsyncMock(return_value={
            "id": "upbit-001",
            "symbol": "BTC/KRW",
            "side": "buy",
            "type": "market",
            "amount": 0.01,
            "price": None,
            "status": "closed",
            "average": 95000000.0,
            "filled": 0.01,
            "fee": {"cost": 500.0, "currency": "KRW"},
        })
        order = await a.create_order(
            symbol="BTC/KRW",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
        )
        assert order.id == "upbit-001"
        assert order.exchange == "upbit"
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, adapter):
        a, mock_ex = adapter
        mock_ex.cancel_order = AsyncMock(return_value={})
        result = await a.cancel_order("upbit-001", "BTC/KRW")
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, adapter):
        import ccxt.async_support as ccxt_async

        a, mock_ex = adapter
        mock_ex.cancel_order = AsyncMock(side_effect=ccxt_async.OrderNotFound("not found"))
        result = await a.cancel_order("nonexistent", "BTC/KRW")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_order_status(self, adapter):
        a, mock_ex = adapter
        mock_ex.fetch_order = AsyncMock(return_value={
            "id": "upbit-001",
            "symbol": "BTC/KRW",
            "side": "buy",
            "type": "limit",
            "amount": 0.05,
            "price": 94000000.0,
            "status": "open",
            "average": None,
            "filled": 0,
        })
        order = await a.get_order_status("upbit-001", "BTC/KRW")
        assert order.status == OrderStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_get_order_book(self, adapter):
        a, mock_ex = adapter
        mock_ex.fetch_order_book = AsyncMock(return_value={
            "bids": [[95000000.0, 0.5], [94999000.0, 1.0]],
            "asks": [[95001000.0, 0.3], [95002000.0, 0.8]],
        })
        book = await a.get_order_book("BTC/KRW", limit=5)
        assert len(book["bids"]) == 2
        assert book["bids"][0][0] == 95000000.0

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
            side_effect=ccxt_async.InsufficientFunds("not enough KRW")
        )
        with pytest.raises(ValueError, match="Insufficient funds"):
            await a.create_order(
                symbol="BTC/KRW",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
            )
