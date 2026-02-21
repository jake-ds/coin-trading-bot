"""Tests for DCA strategy."""

from datetime import datetime, timedelta

import pytest

from bot.models import OHLCV, SignalAction
from bot.strategies.dca.dca_strategy import DCAStrategy


def make_candles(count: int, base_price: float = 100.0) -> list[OHLCV]:
    base = datetime(2024, 1, 1)
    return [
        OHLCV(
            timestamp=base + timedelta(hours=i),
            open=base_price,
            high=base_price * 1.01,
            low=base_price * 0.99,
            close=base_price,
            volume=1000.0,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        for i in range(count)
    ]


class TestDCAStrategy:
    @pytest.fixture
    def strategy(self):
        return DCAStrategy(
            interval="daily",
            buy_amount=100.0,
            use_rsi_enhancement=False,
        )

    def test_name(self, strategy):
        assert strategy.name == "dca"

    def test_required_history(self, strategy):
        assert strategy.required_history_length == 1

    @pytest.mark.asyncio
    async def test_first_buy(self, strategy):
        candles = make_candles(5)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY
        assert signal.metadata["buy_amount"] == 100.0
        assert signal.metadata["total_invested"] == 100.0

    @pytest.mark.asyncio
    async def test_second_buy_too_soon(self, strategy):
        candles = make_candles(5)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")  # first buy
        # Confirm the buy to update state
        strategy.confirm_buy(
            candles[-1].timestamp,
            signal.metadata["buy_amount"],
            signal.metadata["quantity"],
        )

        # Same time — should hold
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_second_buy_after_interval(self, strategy):
        candles = make_candles(5)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")  # first buy
        strategy.confirm_buy(
            candles[-1].timestamp,
            signal.metadata["buy_amount"],
            signal.metadata["quantity"],
        )

        # Create candles 25 hours later
        base = datetime(2024, 1, 2, 1)
        later_candles = [
            OHLCV(
                timestamp=base + timedelta(hours=i),
                open=100.0, high=101.0, low=99.0, close=100.0,
                volume=1000.0, symbol="BTC/USDT", timeframe="1h",
            )
            for i in range(5)
        ]
        signal = await strategy.analyze(later_candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY
        assert signal.metadata["total_invested"] == 200.0

    @pytest.mark.asyncio
    async def test_tracks_average_price(self, strategy):
        candles = make_candles(5, base_price=50000.0)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.metadata["average_price"] > 0

    @pytest.mark.asyncio
    async def test_empty_data(self, strategy):
        signal = await strategy.analyze([], symbol="BTC/USDT")
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_rsi_enhanced_dca(self):
        # Create declining prices that produce low RSI
        strategy = DCAStrategy(
            interval="daily",
            buy_amount=100.0,
            use_rsi_enhancement=True,
            rsi_period=7,
            rsi_oversold=30.0,
            rsi_bonus_multiplier=2.0,
        )
        base = datetime(2024, 1, 1)
        candles = [
            OHLCV(
                timestamp=base + timedelta(hours=i),
                open=100.0 - i * 3,
                high=(100.0 - i * 3) * 1.01,
                low=(100.0 - i * 3) * 0.99,
                close=100.0 - i * 3,
                volume=1000.0,
                symbol="BTC/USDT",
                timeframe="1h",
            )
            for i in range(15)
        ]
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal.action == SignalAction.BUY
        # With RSI enhancement and oversold, buy amount should be doubled
        assert signal.metadata["buy_amount"] == 200.0
