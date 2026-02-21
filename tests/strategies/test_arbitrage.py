"""Tests for arbitrage strategy."""

from datetime import datetime
from unittest.mock import AsyncMock, PropertyMock

import pytest

from bot.models import OHLCV, SignalAction
from bot.strategies.arbitrage.arbitrage_strategy import ArbitrageStrategy


def make_mock_exchange(name: str, bid: float, ask: float):
    exchange = AsyncMock()
    type(exchange).name = PropertyMock(return_value=name)
    exchange.get_ticker = AsyncMock(return_value={
        "bid": bid,
        "ask": ask,
        "last": (bid + ask) / 2,
        "volume": 100.0,
    })
    return exchange


def make_candle():
    return OHLCV(
        timestamp=datetime(2024, 1, 1),
        open=50000.0, high=51000.0, low=49000.0, close=50500.0,
        volume=100.0, symbol="BTC/USDT",
    )


class TestArbitrageStrategy:
    @pytest.fixture
    def strategy(self):
        return ArbitrageStrategy(min_spread_pct=0.5, fee_pct=0.1)

    def test_name(self, strategy):
        assert strategy.name == "arbitrage"

    @pytest.mark.asyncio
    async def test_insufficient_exchanges(self, strategy):
        exchange = make_mock_exchange("binance", 50000, 50100)
        signal = await strategy.analyze(
            [make_candle()], symbol="BTC/USDT", exchanges=[exchange]
        )
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_profitable_arbitrage(self, strategy):
        # Binance: bid=50500, ask=50100 — sell here
        # Upbit: bid=50000, ask=49800 — buy here
        binance = make_mock_exchange("binance", bid=50500, ask=50100)
        upbit = make_mock_exchange("upbit", bid=50000, ask=49800)

        signal = await strategy.analyze(
            [make_candle()], symbol="BTC/USDT", exchanges=[binance, upbit]
        )
        # Spread = (50500 - 49800) / 49800 * 100 = 1.41% minus 0.2% fees = 1.21%
        assert signal.action == SignalAction.BUY
        assert signal.metadata["buy_exchange"] == "upbit"
        assert signal.metadata["sell_exchange"] == "binance"
        assert signal.metadata["net_spread_pct"] > 0.5

    @pytest.mark.asyncio
    async def test_no_profitable_arbitrage(self, strategy):
        # Same prices on both exchanges
        binance = make_mock_exchange("binance", bid=50000, ask=50100)
        upbit = make_mock_exchange("upbit", bid=50000, ask=50100)

        signal = await strategy.analyze(
            [make_candle()], symbol="BTC/USDT", exchanges=[binance, upbit]
        )
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_exchange_error_handled(self, strategy):
        binance = make_mock_exchange("binance", bid=50000, ask=50100)
        upbit = AsyncMock()
        type(upbit).name = PropertyMock(return_value="upbit")
        upbit.get_ticker = AsyncMock(side_effect=ConnectionError("timeout"))

        signal = await strategy.analyze(
            [make_candle()], symbol="BTC/USDT", exchanges=[binance, upbit]
        )
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_accounts_for_fees(self, strategy):
        # Spread = 0.3% which is less than min_spread + fees
        binance = make_mock_exchange("binance", bid=50150, ask=50100)
        upbit = make_mock_exchange("upbit", bid=50000, ask=50000)

        signal = await strategy.analyze(
            [make_candle()], symbol="BTC/USDT", exchanges=[binance, upbit]
        )
        assert signal.action == SignalAction.HOLD
