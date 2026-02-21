"""Tests for resilient exchange wrapper."""

from unittest.mock import AsyncMock, PropertyMock

import pytest

from bot.execution.circuit_breaker import CircuitState
from bot.execution.resilient import ResilientExchange


@pytest.fixture
def mock_exchange():
    exchange = AsyncMock()
    type(exchange).name = PropertyMock(return_value="mock_exchange")
    exchange.get_ticker = AsyncMock(return_value={"last": 50000.0})
    exchange.close = AsyncMock()
    return exchange


class TestResilientExchange:
    def test_name(self, mock_exchange):
        re = ResilientExchange(mock_exchange)
        assert re.name == "mock_exchange"

    def test_initially_available(self, mock_exchange):
        re = ResilientExchange(mock_exchange)
        assert re.is_available

    @pytest.mark.asyncio
    async def test_successful_call(self, mock_exchange):
        re = ResilientExchange(mock_exchange)
        result = await re.get_ticker("BTC/USDT")
        assert result["last"] == 50000.0
        mock_exchange.get_ticker.assert_called_once_with("BTC/USDT")

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self, mock_exchange):
        mock_exchange.get_ticker = AsyncMock(
            side_effect=[ConnectionError("timeout"), {"last": 50000.0}]
        )
        re = ResilientExchange(mock_exchange, retry_delay=0.01)
        result = await re.get_ticker("BTC/USDT")
        assert result["last"] == 50000.0
        assert mock_exchange.get_ticker.call_count == 2

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, mock_exchange):
        mock_exchange.get_ticker = AsyncMock(side_effect=ConnectionError("fail"))
        re = ResilientExchange(
            mock_exchange, failure_threshold=3, max_retries=1, retry_delay=0.01
        )

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await re.get_ticker("BTC/USDT")

        assert re.circuit_breaker.state == CircuitState.OPEN
        assert not re.is_available

    @pytest.mark.asyncio
    async def test_circuit_open_rejects_calls(self, mock_exchange):
        mock_exchange.get_ticker = AsyncMock(side_effect=ConnectionError("fail"))
        re = ResilientExchange(
            mock_exchange, failure_threshold=1, max_retries=1, retry_delay=0.01
        )

        with pytest.raises(ConnectionError):
            await re.get_ticker("BTC/USDT")

        # Circuit is now open, next call should be rejected
        with pytest.raises(ConnectionError, match="Circuit breaker open"):
            await re.get_ticker("BTC/USDT")

    @pytest.mark.asyncio
    async def test_value_error_propagates(self, mock_exchange):
        mock_exchange.get_ticker = AsyncMock(side_effect=ValueError("bad symbol"))
        re = ResilientExchange(mock_exchange)

        with pytest.raises(ValueError, match="bad symbol"):
            await re.get_ticker("INVALID")

    @pytest.mark.asyncio
    async def test_get_balance(self, mock_exchange):
        mock_exchange.get_balance = AsyncMock(return_value={"USDT": 10000.0})
        re = ResilientExchange(mock_exchange)
        result = await re.get_balance()
        assert result == {"USDT": 10000.0}
        mock_exchange.get_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_order(self, mock_exchange):
        mock_exchange.cancel_order = AsyncMock(return_value=True)
        re = ResilientExchange(mock_exchange)
        result = await re.cancel_order("order-123", "BTC/USDT")
        assert result is True
        mock_exchange.cancel_order.assert_called_once_with("order-123", "BTC/USDT")

    @pytest.mark.asyncio
    async def test_get_order_status(self, mock_exchange):
        mock_order = {"id": "order-123", "status": "filled"}
        mock_exchange.get_order_status = AsyncMock(return_value=mock_order)
        re = ResilientExchange(mock_exchange)
        result = await re.get_order_status("order-123", "BTC/USDT")
        assert result["status"] == "filled"
        mock_exchange.get_order_status.assert_called_once_with("order-123", "BTC/USDT")

    @pytest.mark.asyncio
    async def test_get_order_book(self, mock_exchange):
        book = {"bids": [[50000, 1.0]], "asks": [[50001, 0.5]]}
        mock_exchange.get_order_book = AsyncMock(return_value=book)
        re = ResilientExchange(mock_exchange)
        result = await re.get_order_book("BTC/USDT")
        assert result["bids"][0][0] == 50000
        mock_exchange.get_order_book.assert_called_once_with("BTC/USDT", 20)

    @pytest.mark.asyncio
    async def test_get_order_book_custom_limit(self, mock_exchange):
        book = {"bids": [], "asks": []}
        mock_exchange.get_order_book = AsyncMock(return_value=book)
        re = ResilientExchange(mock_exchange)
        await re.get_order_book("BTC/USDT", limit=10)
        mock_exchange.get_order_book.assert_called_once_with("BTC/USDT", 10)

    @pytest.mark.asyncio
    async def test_close(self, mock_exchange):
        re = ResilientExchange(mock_exchange)
        await re.close()
        mock_exchange.close.assert_called_once()
