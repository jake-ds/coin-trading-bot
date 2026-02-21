"""Backtesting engine for replaying historical data through strategies."""

from dataclasses import dataclass, field
from datetime import datetime

import structlog

from bot.models import OHLCV, SignalAction
from bot.monitoring.metrics import MetricsCollector, PerformanceMetrics
from bot.strategies.base import BaseStrategy

logger = structlog.get_logger()


@dataclass
class TradeLog:
    """Record of a single backtested trade."""

    timestamp: datetime
    symbol: str
    side: str
    price: float
    quantity: float
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    strategy_name: str
    symbol: str
    timeframe: str
    metrics: PerformanceMetrics
    trades: list[TradeLog] = field(default_factory=list)
    final_portfolio_value: float = 0.0


class BacktestEngine:
    """Replays historical OHLCV data through a strategy."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_pct: float = 0.1,
        slippage_pct: float = 0.05,
    ):
        self._initial_capital = initial_capital
        self._fee_pct = fee_pct
        self._slippage_pct = slippage_pct

    async def run(
        self,
        strategy: BaseStrategy,
        data: list[OHLCV],
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
    ) -> BacktestResult:
        """Run a backtest with the given strategy and data."""
        cash = self._initial_capital
        position_qty = 0.0
        entry_cost = 0.0  # Total cash spent on buy (including fee)
        trades: list[TradeLog] = []
        metrics = MetricsCollector(initial_capital=self._initial_capital)

        min_history = strategy.required_history_length

        for i in range(min_history, len(data)):
            window = data[max(0, i - min_history):i + 1]
            current_price = data[i].close
            current_time = data[i].timestamp

            signal = await strategy.analyze(window, symbol=symbol)

            if signal.action == SignalAction.BUY and position_qty == 0:
                # Apply slippage
                buy_price = current_price * (1 + self._slippage_pct / 100)
                # Calculate quantity (use all cash minus fees)
                entry_cost = cash  # Total cash spent (including buy fee)
                fee = cash * (self._fee_pct / 100)
                available = cash - fee
                position_qty = available / buy_price if buy_price > 0 else 0
                cash = 0

                trades.append(TradeLog(
                    timestamp=current_time,
                    symbol=symbol,
                    side="BUY",
                    price=buy_price,
                    quantity=position_qty,
                ))

            elif signal.action == SignalAction.SELL and position_qty > 0:
                sell_price = current_price * (1 - self._slippage_pct / 100)
                proceeds = position_qty * sell_price
                fee = proceeds * (self._fee_pct / 100)
                cash = proceeds - fee

                pnl = cash - entry_cost
                metrics.record_trade(pnl)

                trades.append(TradeLog(
                    timestamp=current_time,
                    symbol=symbol,
                    side="SELL",
                    price=sell_price,
                    quantity=position_qty,
                    pnl=pnl,
                ))

                position_qty = 0

            # Record portfolio value
            portfolio_value = cash + (position_qty * current_price)
            metrics.record_portfolio_value(portfolio_value)

        # Final value
        final_price = data[-1].close if data else 0
        final_value = cash + (position_qty * final_price)

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=timeframe,
            metrics=metrics.calculate(),
            trades=trades,
            final_portfolio_value=round(final_value, 2),
        )
