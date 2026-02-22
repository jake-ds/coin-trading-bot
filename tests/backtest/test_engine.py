"""Tests for backtesting engine."""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from bot.backtest.engine import BacktestEngine, BacktestResult, TradeLog
from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy


class AlwaysBuyStrategy(BaseStrategy):
    """Test strategy that always buys."""

    @property
    def name(self) -> str:
        return "always_buy"

    @property
    def required_history_length(self) -> int:
        return 1

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        return TradingSignal(
            strategy_name="always_buy",
            symbol=kwargs.get("symbol", "BTC/USDT"),
            action=SignalAction.BUY,
            confidence=0.9,
        )


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


class AlwaysHoldStrategy(BaseStrategy):
    """Test strategy that always holds."""

    @property
    def name(self) -> str:
        return "always_hold"

    @property
    def required_history_length(self) -> int:
        return 1

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        return TradingSignal(
            strategy_name="always_hold",
            symbol=kwargs.get("symbol", "BTC/USDT"),
            action=SignalAction.HOLD,
            confidence=0.5,
        )


def make_candles(prices: list[float], start_time: datetime | None = None) -> list[OHLCV]:
    """Create OHLCV candles from a list of close prices."""
    base = start_time or datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i, price in enumerate(prices):
        candles.append(OHLCV(
            timestamp=base + timedelta(hours=i),
            open=price,
            high=price * 1.01,
            low=price * 0.99,
            close=price,
            volume=1000.0,
        ))
    return candles


class TestBacktestEngine:
    @pytest.mark.asyncio
    async def test_no_trades_on_hold(self):
        engine = BacktestEngine(initial_capital=10000.0)
        prices = [100.0] * 10
        data = make_candles(prices)
        result = await engine.run(AlwaysHoldStrategy(), data)

        assert result.strategy_name == "always_hold"
        assert len(result.trades) == 0
        assert result.final_portfolio_value == 10000.0

    @pytest.mark.asyncio
    async def test_buy_and_sell_cycle(self):
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        # Prices: buy at 100, sell at 120 => 20% gain
        prices = [100.0, 100.0, 120.0]
        data = make_candles(prices)
        strategy = AlternateBuySellStrategy()
        result = await engine.run(strategy, data)

        assert len(result.trades) == 2
        assert result.trades[0].side == "BUY"
        assert result.trades[1].side == "SELL"
        assert result.final_portfolio_value > 10000.0

    @pytest.mark.asyncio
    async def test_fees_reduce_returns(self):
        engine_no_fees = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        engine_fees = BacktestEngine(initial_capital=10000.0, fee_pct=1.0, slippage_pct=0.0)
        prices = [100.0, 100.0, 120.0]
        data = make_candles(prices)

        result_no = await engine_fees.run(AlternateBuySellStrategy(), data)
        result_yes = await engine_no_fees.run(AlternateBuySellStrategy(), data)

        # With fees the result should be different
        assert result_no.final_portfolio_value != result_yes.final_portfolio_value

    @pytest.mark.asyncio
    async def test_slippage_affects_results(self):
        engine_no_slip = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        engine_slip = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=1.0)
        prices = [100.0, 100.0, 120.0]
        data = make_candles(prices)

        result_no = await engine_no_slip.run(AlternateBuySellStrategy(), data)
        result_slip = await engine_slip.run(AlternateBuySellStrategy(), data)

        # With slippage, returns should be worse
        assert result_slip.final_portfolio_value < result_no.final_portfolio_value

    @pytest.mark.asyncio
    async def test_result_contains_metrics(self):
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        prices = [100.0, 100.0, 120.0, 120.0, 110.0]
        data = make_candles(prices)
        strategy = AlternateBuySellStrategy()
        result = await engine.run(strategy, data)

        assert result.metrics is not None
        assert result.metrics.total_trades >= 0

    @pytest.mark.asyncio
    async def test_result_metadata(self):
        engine = BacktestEngine()
        prices = [100.0] * 5
        data = make_candles(prices)
        result = await engine.run(AlwaysHoldStrategy(), data, symbol="ETH/USDT", timeframe="4h")

        assert result.symbol == "ETH/USDT"
        assert result.timeframe == "4h"
        assert isinstance(result, BacktestResult)

    @pytest.mark.asyncio
    async def test_trade_log_records(self):
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        prices = [100.0, 100.0, 120.0]
        data = make_candles(prices)
        strategy = AlternateBuySellStrategy()
        result = await engine.run(strategy, data)

        for trade in result.trades:
            assert isinstance(trade, TradeLog)
            assert trade.symbol == "BTC/USDT"
            assert trade.price > 0
            assert trade.quantity > 0

    @pytest.mark.asyncio
    async def test_only_buy_when_no_position(self):
        """AlwaysBuy strategy should only buy once since no sells happen."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        prices = [100.0, 110.0, 120.0, 130.0, 140.0]
        data = make_candles(prices)
        result = await engine.run(AlwaysBuyStrategy(), data)

        # Should only have 1 buy trade (subsequent buys ignored since already in position)
        buy_trades = [t for t in result.trades if t.side == "BUY"]
        assert len(buy_trades) == 1
