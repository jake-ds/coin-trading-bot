"""Tests for Binance Futures exchange adapter with mocked ccxt."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.exchanges.binance_futures import BinanceFuturesAdapter
from bot.models import OrderSide, OrderStatus, OrderType


@pytest.fixture
def adapter():
    """Create a BinanceFuturesAdapter with mocked ccxt exchange."""
    with patch("bot.exchanges.binance_futures.ccxt.binance") as mock_cls:
        mock_exchange = MagicMock()
        mock_cls.return_value = mock_exchange
        a = BinanceFuturesAdapter(api_key="test", secret_key="test", testnet=True)
        yield a, mock_exchange


class TestBinanceFuturesAdapter:
    def test_name(self, adapter):
        a, _ = adapter
        assert a.name == "binance_futures"

    def test_to_futures_symbol(self):
        assert BinanceFuturesAdapter.to_futures_symbol("BTC/USDT") == "BTC/USDT:USDT"
        assert BinanceFuturesAdapter.to_futures_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"

    def test_to_spot_symbol(self):
        assert BinanceFuturesAdapter.to_spot_symbol("BTC/USDT:USDT") == "BTC/USDT"
        assert BinanceFuturesAdapter.to_spot_symbol("BTC/USDT") == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_get_ticker(self, adapter):
        a, mock_ex = adapter
        mock_ex.fetch_ticker = AsyncMock(return_value={
            "bid": 84900.0,
            "ask": 85100.0,
            "last": 85000.0,
            "baseVolume": 50000.0,
        })
        ticker = await a.get_ticker("BTC/USDT")
        assert ticker["last"] == 85000.0
        mock_ex.fetch_ticker.assert_called_once_with("BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_get_ticker_invalid_symbol(self, adapter):
        import ccxt.async_support as ccxt_async

        a, mock_ex = adapter
        mock_ex.fetch_ticker = AsyncMock(
            side_effect=ccxt_async.BadSymbol("bad")
        )
        with pytest.raises(ValueError, match="Invalid futures symbol"):
            await a.get_ticker("INVALID/PAIR")

    @pytest.mark.asyncio
    async def test_set_leverage(self, adapter):
        a, mock_ex = adapter
        mock_ex.set_leverage = AsyncMock()
        await a.set_leverage("BTC/USDT", 3)
        mock_ex.set_leverage.assert_called_once_with(3, "BTC/USDT:USDT")
        assert a._leverage_set["BTC/USDT"] == 3

    @pytest.mark.asyncio
    async def test_set_leverage_already_set(self, adapter):
        import ccxt.async_support as ccxt_async

        a, mock_ex = adapter
        mock_ex.set_leverage = AsyncMock(
            side_effect=ccxt_async.ExchangeError("No need to change leverage")
        )
        await a.set_leverage("BTC/USDT", 2)
        assert a._leverage_set["BTC/USDT"] == 2

    @pytest.mark.asyncio
    async def test_set_margin_mode(self, adapter):
        a, mock_ex = adapter
        mock_ex.set_margin_mode = AsyncMock()
        await a.set_margin_mode("BTC/USDT", "isolated")
        mock_ex.set_margin_mode.assert_called_once_with("isolated", "BTC/USDT:USDT")
        assert a._margin_mode_set["BTC/USDT"] == "isolated"

    @pytest.mark.asyncio
    async def test_set_margin_mode_already_set(self, adapter):
        import ccxt.async_support as ccxt_async

        a, mock_ex = adapter
        mock_ex.set_margin_mode = AsyncMock(
            side_effect=ccxt_async.ExchangeError("No need to change margin type")
        )
        await a.set_margin_mode("ETH/USDT", "isolated")
        assert a._margin_mode_set["ETH/USDT"] == "isolated"

    @pytest.mark.asyncio
    async def test_ensure_leverage_and_margin(self, adapter):
        a, mock_ex = adapter
        mock_ex.set_margin_mode = AsyncMock()
        mock_ex.set_leverage = AsyncMock()
        await a.ensure_leverage_and_margin("BTC/USDT", 3, "isolated")
        mock_ex.set_margin_mode.assert_called_once()
        mock_ex.set_leverage.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_leverage_idempotent(self, adapter):
        a, mock_ex = adapter
        mock_ex.set_margin_mode = AsyncMock()
        mock_ex.set_leverage = AsyncMock()
        a._margin_mode_set["BTC/USDT"] = "isolated"
        a._leverage_set["BTC/USDT"] = 2
        await a.ensure_leverage_and_margin("BTC/USDT", 2, "isolated")
        # Already set — should NOT call again
        mock_ex.set_margin_mode.assert_not_called()
        mock_ex.set_leverage.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_order_sell_short(self, adapter):
        a, mock_ex = adapter
        mock_ex.create_order = AsyncMock(return_value={
            "id": "order-123",
            "symbol": "BTC/USDT:USDT",
            "side": "sell",
            "type": "market",
            "status": "closed",
            "amount": 0.001,
            "price": 85000.0,
            "average": 84950.0,
            "filled": 0.001,
            "fee": {"cost": 0.04, "currency": "USDT"},
        })
        order = await a.create_order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.001,
        )
        assert order.side == OrderSide.SELL
        assert order.exchange == "binance_futures"
        assert order.filled_price == 84950.0
        mock_ex.create_order.assert_called_once_with(
            symbol="BTC/USDT:USDT",
            type="market",
            side="sell",
            amount=0.001,
            price=None,
            params={},
        )

    @pytest.mark.asyncio
    async def test_create_order_buy_reduce_only(self, adapter):
        a, mock_ex = adapter
        mock_ex.create_order = AsyncMock(return_value={
            "id": "order-456",
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "type": "market",
            "status": "closed",
            "amount": 0.001,
            "price": 83000.0,
            "average": 83000.0,
            "filled": 0.001,
            "fee": {"cost": 0.04, "currency": "USDT"},
        })
        order = await a.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            reduce_only=True,
        )
        assert order.side == OrderSide.BUY
        mock_ex.create_order.assert_called_once_with(
            symbol="BTC/USDT:USDT",
            type="market",
            side="buy",
            amount=0.001,
            price=None,
            params={"reduceOnly": True},
        )

    @pytest.mark.asyncio
    async def test_create_order_insufficient_funds(self, adapter):
        import ccxt.async_support as ccxt_async

        a, mock_ex = adapter
        mock_ex.create_order = AsyncMock(
            side_effect=ccxt_async.InsufficientFunds("not enough margin")
        )
        with pytest.raises(ValueError, match="Insufficient futures margin"):
            await a.create_order(
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=0.001,
            )

    @pytest.mark.asyncio
    async def test_get_balance(self, adapter):
        a, mock_ex = adapter
        mock_ex.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 500.0, "BNB": 0.0},
        })
        balance = await a.get_balance()
        assert balance == {"USDT": 500.0}

    @pytest.mark.asyncio
    async def test_fetch_futures_positions(self, adapter):
        a, mock_ex = adapter
        mock_ex.fetch_positions = AsyncMock(return_value=[
            {
                "symbol": "BTC/USDT:USDT",
                "side": "short",
                "contracts": 0.002,
                "notional": 170.0,
                "unrealizedPnl": 5.0,
                "entryPrice": 85000.0,
                "markPrice": 82500.0,
                "leverage": 2,
                "marginMode": "isolated",
                "liquidationPrice": 95000.0,
            },
            {
                "symbol": "ETH/USDT:USDT",
                "side": "short",
                "contracts": 0,  # No position
                "notional": 0,
                "unrealizedPnl": 0,
                "entryPrice": 0,
                "markPrice": 0,
                "leverage": 1,
                "marginMode": "cross",
                "liquidationPrice": 0,
            },
        ])
        positions = await a.fetch_futures_positions()
        assert len(positions) == 1  # Only non-zero contracts
        assert positions[0]["symbol"] == "BTC/USDT"
        assert positions[0]["contracts"] == 0.002
        assert positions[0]["entry_price"] == 85000.0

    @pytest.mark.asyncio
    async def test_fetch_funding_rate(self, adapter):
        a, mock_ex = adapter
        mock_ex.fetch_funding_rate = AsyncMock(return_value={
            "fundingRate": 0.0003,
            "fundingDatetime": "2026-03-17T08:00:00Z",
        })
        rate = await a.fetch_funding_rate("BTC/USDT")
        assert rate["funding_rate"] == 0.0003

    @pytest.mark.asyncio
    async def test_close(self, adapter):
        a, mock_ex = adapter
        mock_ex.close = AsyncMock()
        await a.close()
        mock_ex.close.assert_called_once()
