"""Tests for V2-006 bug fixes: arbitrage same-exchange, DCA idempotency, backtest PnL formula."""

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, PropertyMock

import pytest

from bot.backtest.engine import BacktestEngine
from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.arbitrage.arbitrage_strategy import ArbitrageStrategy
from bot.strategies.base import BaseStrategy
from bot.strategies.dca.dca_strategy import DCAStrategy

# ── Helpers ──────────────────────────────────────────────────────────────


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


def make_candles(
    prices: list[float],
    start_time: datetime | None = None,
    symbol: str = "BTC/USDT",
) -> list[OHLCV]:
    base = start_time or datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        OHLCV(
            timestamp=base + timedelta(hours=i),
            open=price,
            high=price * 1.01,
            low=price * 0.99,
            close=price,
            volume=1000.0,
            symbol=symbol,
        )
        for i, price in enumerate(prices)
    ]


def make_dca_candles(count: int, base_price: float = 100.0) -> list[OHLCV]:
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


class AlternateBuySellStrategy(BaseStrategy):
    """Test strategy that alternates between BUY and SELL."""

    def __init__(self):
        self._count = 0

    @property
    def name(self) -> str:
        return "alternate"

    @property
    def required_history_length(self) -> int:
        return 1

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        self._count += 1
        action = SignalAction.BUY if self._count % 2 == 1 else SignalAction.SELL
        return TradingSignal(
            strategy_name="alternate",
            symbol=kwargs.get("symbol", "BTC/USDT"),
            action=action,
            confidence=0.8,
        )


# ── Arbitrage: same-exchange bug fix ─────────────────────────────────────


class TestArbitrageSameExchange:
    @pytest.mark.asyncio
    async def test_same_exchange_returns_hold(self):
        """When best bid and best ask are from the same exchange, return HOLD."""
        strategy = ArbitrageStrategy(min_spread_pct=0.5, fee_pct=0.1)

        # Both exchanges have identical prices, so best bid & ask come from same one
        # Exchange A has better bid AND better ask
        exchange_a = make_mock_exchange("exchange_a", bid=50500, ask=49800)
        exchange_b = make_mock_exchange("exchange_b", bid=50000, ask=50100)

        candle = OHLCV(
            timestamp=datetime(2024, 1, 1),
            open=50000.0, high=51000.0, low=49000.0, close=50500.0,
            volume=100.0, symbol="BTC/USDT",
        )
        signal = await strategy.analyze(
            [candle], symbol="BTC/USDT", exchanges=[exchange_a, exchange_b]
        )
        assert signal.action == SignalAction.HOLD
        assert signal.metadata["reason"] == "same_exchange"

    @pytest.mark.asyncio
    async def test_different_exchanges_still_works(self):
        """Profitable arbitrage across different exchanges still returns BUY."""
        strategy = ArbitrageStrategy(min_spread_pct=0.5, fee_pct=0.1)

        # Best bid on binance (50500), best ask on upbit (49800)
        binance = make_mock_exchange("binance", bid=50500, ask=50100)
        upbit = make_mock_exchange("upbit", bid=50000, ask=49800)

        candle = OHLCV(
            timestamp=datetime(2024, 1, 1),
            open=50000.0, high=51000.0, low=49000.0, close=50500.0,
            volume=100.0, symbol="BTC/USDT",
        )
        signal = await strategy.analyze(
            [candle], symbol="BTC/USDT", exchanges=[binance, upbit]
        )
        assert signal.action == SignalAction.BUY
        assert signal.metadata["buy_exchange"] == "upbit"
        assert signal.metadata["sell_exchange"] == "binance"


# ── DCA: idempotency bug fix ────────────────────────────────────────────


class TestDCAIdempotency:
    @pytest.fixture
    def strategy(self):
        return DCAStrategy(
            interval="daily",
            buy_amount=100.0,
            use_rsi_enhancement=False,
        )

    @pytest.mark.asyncio
    async def test_analyze_idempotent(self, strategy):
        """analyze() called twice without confirm_buy returns same result."""
        candles = make_dca_candles(5)
        signal1 = await strategy.analyze(candles, symbol="BTC/USDT")
        signal2 = await strategy.analyze(candles, symbol="BTC/USDT")

        assert signal1.action == signal2.action == SignalAction.BUY
        assert signal1.metadata["total_invested"] == signal2.metadata["total_invested"]
        assert signal1.metadata["buy_amount"] == signal2.metadata["buy_amount"]
        assert signal1.metadata["total_quantity"] == signal2.metadata["total_quantity"]

    @pytest.mark.asyncio
    async def test_analyze_no_side_effects(self, strategy):
        """analyze() does not mutate internal state."""
        candles = make_dca_candles(5)

        # Call analyze multiple times
        await strategy.analyze(candles, symbol="BTC/USDT")
        await strategy.analyze(candles, symbol="BTC/USDT")
        await strategy.analyze(candles, symbol="BTC/USDT")

        # Internal state should not have changed
        assert strategy._total_invested == 0.0
        assert strategy._total_quantity == 0.0
        assert strategy._last_buy_time is None

    @pytest.mark.asyncio
    async def test_confirm_buy_updates_state(self, strategy):
        """confirm_buy() properly updates internal tracking state."""
        candles = make_dca_candles(5)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")

        strategy.confirm_buy(
            candles[-1].timestamp,
            signal.metadata["buy_amount"],
            signal.metadata["quantity"],
        )

        assert strategy._total_invested == 100.0
        assert strategy._total_quantity == signal.metadata["quantity"]
        assert strategy._last_buy_time == candles[-1].timestamp

    @pytest.mark.asyncio
    async def test_confirm_buy_prevents_immediate_rebuy(self, strategy):
        """After confirm_buy, analyze() returns HOLD until interval elapses."""
        candles = make_dca_candles(5)
        signal = await strategy.analyze(candles, symbol="BTC/USDT")
        strategy.confirm_buy(
            candles[-1].timestamp,
            signal.metadata["buy_amount"],
            signal.metadata["quantity"],
        )

        # Same candles — should hold
        signal2 = await strategy.analyze(candles, symbol="BTC/USDT")
        assert signal2.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_projected_totals_accumulate(self, strategy):
        """After confirm_buy, next analyze shows accumulated totals."""
        candles = make_dca_candles(5)
        signal1 = await strategy.analyze(candles, symbol="BTC/USDT")
        strategy.confirm_buy(
            candles[-1].timestamp,
            signal1.metadata["buy_amount"],
            signal1.metadata["quantity"],
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
        signal2 = await strategy.analyze(later_candles, symbol="BTC/USDT")
        assert signal2.action == SignalAction.BUY
        assert signal2.metadata["total_invested"] == 200.0


# ── Backtest: PnL formula bug fix ───────────────────────────────────────


class TestBacktestPnLFormula:
    @pytest.mark.asyncio
    async def test_pnl_accounts_for_both_fees(self):
        """PnL correctly deducts both buy fee and sell fee."""
        # With fees, PnL should be negative if price doesn't move
        fee_pct = 1.0  # 1% fee to make effect visible
        engine = BacktestEngine(
            initial_capital=10000.0,
            fee_pct=fee_pct,
            slippage_pct=0.0,
        )
        # Buy at 100, sell at 100 => should lose money due to fees
        prices = [100.0, 100.0, 100.0]
        data = make_candles(prices)
        result = await engine.run(AlternateBuySellStrategy(), data)

        assert len(result.trades) == 2
        sell_trade = result.trades[1]

        # PnL should be negative (lost money on fees)
        assert sell_trade.pnl < 0

        # Verify exact PnL:
        # Buy: entry_cost = 10000, fee = 100, available = 9900, qty = 99.0
        # Sell: proceeds = 99.0 * 100 = 9900, fee = 99, cash = 9801
        # PnL = 9801 - 10000 = -199
        assert abs(sell_trade.pnl - (-199.0)) < 0.01

    @pytest.mark.asyncio
    async def test_pnl_with_profit_and_fees(self):
        """PnL accounts for fees even when trade is profitable."""
        engine = BacktestEngine(
            initial_capital=10000.0,
            fee_pct=0.1,  # 0.1% fee
            slippage_pct=0.0,
        )
        # Buy at 100, sell at 120 => profitable even after fees
        prices = [100.0, 100.0, 120.0]
        data = make_candles(prices)
        result = await engine.run(AlternateBuySellStrategy(), data)

        sell_trade = result.trades[1]

        # Buy: entry_cost = 10000, fee = 10, available = 9990, qty = 99.9
        # Sell: proceeds = 99.9 * 120 = 11988, fee = 11.988, cash = 11976.012
        # PnL = 11976.012 - 10000 = 1976.012
        assert sell_trade.pnl > 0
        assert abs(sell_trade.pnl - 1976.012) < 0.01

    @pytest.mark.asyncio
    async def test_pnl_zero_fees_unchanged(self):
        """With zero fees, PnL formula gives same result as before."""
        engine = BacktestEngine(
            initial_capital=10000.0,
            fee_pct=0.0,
            slippage_pct=0.0,
        )
        prices = [100.0, 100.0, 120.0]
        data = make_candles(prices)
        result = await engine.run(AlternateBuySellStrategy(), data)

        sell_trade = result.trades[1]
        # Buy at 100: qty = 100, cash = 0
        # Sell at 120: proceeds = 12000, cash = 12000
        # PnL = 12000 - 10000 = 2000
        assert abs(sell_trade.pnl - 2000.0) < 0.01
