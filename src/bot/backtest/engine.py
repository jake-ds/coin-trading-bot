"""Backtesting engine for replaying historical data through strategies."""

from __future__ import annotations

import csv
import io
import json
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
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    strategy_name: str
    symbol: str
    timeframe: str
    metrics: PerformanceMetrics
    trades: list[TradeLog] = field(default_factory=list)
    final_portfolio_value: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    drawdown_curve: list[float] = field(default_factory=list)
    monthly_returns: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        """Export result as JSON string."""
        data = {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "final_portfolio_value": self.final_portfolio_value,
            "metrics": {
                "total_return_pct": self.metrics.total_return_pct,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "win_rate": self.metrics.win_rate,
                "max_drawdown_pct": self.metrics.max_drawdown_pct,
                "total_trades": self.metrics.total_trades,
                "winning_trades": self.metrics.winning_trades,
                "losing_trades": self.metrics.losing_trades,
            },
            "trades": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "symbol": t.symbol,
                    "side": t.side,
                    "price": t.price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "exit_reason": t.exit_reason,
                }
                for t in self.trades
            ],
            "equity_curve": self.equity_curve,
            "drawdown_curve": self.drawdown_curve,
            "monthly_returns": self.monthly_returns,
        }
        return json.dumps(data, indent=2)

    def to_csv(self) -> str:
        """Export trade log as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "timestamp", "symbol", "side", "price", "quantity", "pnl", "exit_reason",
        ])
        for t in self.trades:
            writer.writerow([
                t.timestamp.isoformat(),
                t.symbol,
                t.side,
                t.price,
                t.quantity,
                t.pnl,
                t.exit_reason,
            ])
        return output.getvalue()


class BacktestEngine:
    """Replays historical OHLCV data through a strategy.

    Supports:
    - Stop-loss and take-profit enforcement on every candle
    - Trailing stop that adjusts upward as price rises
    - Dynamic slippage based on order size relative to volume
    - Walk-forward optimization mode
    - Equity curve, drawdown curve, and monthly returns
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_pct: float = 0.1,
        slippage_pct: float = 0.05,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
        trailing_stop_pct: float = 0.0,
        dynamic_slippage: bool = False,
        base_slippage_pct: float = 0.05,
        avg_daily_volume: float = 0.0,
    ):
        self._initial_capital = initial_capital
        self._fee_pct = fee_pct
        self._slippage_pct = slippage_pct
        self._stop_loss_pct = stop_loss_pct
        self._take_profit_pct = take_profit_pct
        self._trailing_stop_pct = trailing_stop_pct
        self._dynamic_slippage = dynamic_slippage
        self._base_slippage_pct = base_slippage_pct
        self._avg_daily_volume = avg_daily_volume

    def _calculate_slippage(self, order_value: float) -> float:
        """Calculate slippage percentage.

        If dynamic_slippage is enabled, slippage scales with order size
        relative to average daily volume.
        """
        if self._dynamic_slippage and self._avg_daily_volume > 0:
            volume_ratio = order_value / self._avg_daily_volume
            return self._base_slippage_pct * (1 + volume_ratio)
        return self._slippage_pct

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
        entry_cost = 0.0
        entry_price = 0.0
        highest_price_since_entry = 0.0
        stop_loss_price = 0.0
        take_profit_price = 0.0
        trades: list[TradeLog] = []
        metrics = MetricsCollector(initial_capital=self._initial_capital)
        equity_curve: list[float] = [self._initial_capital]
        monthly_values: dict[str, list[float]] = {}

        min_history = strategy.required_history_length

        for i in range(min_history, len(data)):
            window = data[max(0, i - min_history):i + 1]
            candle = data[i]
            current_price = candle.close
            current_time = candle.timestamp

            # --- Check exit conditions BEFORE strategy signal ---
            if position_qty > 0:
                exit_result = self._check_exit_conditions(
                    candle=candle,
                    entry_price=entry_price,
                    position_qty=position_qty,
                    entry_cost=entry_cost,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    highest_price_since_entry=highest_price_since_entry,
                    symbol=symbol,
                )
                if exit_result is not None:
                    sell_price, exit_reason, new_highest = exit_result
                    order_value = position_qty * sell_price
                    slippage = self._calculate_slippage(order_value)
                    # For SL/TP exits, apply slippage to the exit price
                    if exit_reason == "stop_loss" or exit_reason == "trailing_stop":
                        sell_price = sell_price * (1 - slippage / 100)
                    elif exit_reason == "take_profit":
                        sell_price = sell_price * (1 - slippage / 100)
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
                        exit_reason=exit_reason,
                    ))
                    position_qty = 0
                    entry_cost = 0.0
                    entry_price = 0.0
                    highest_price_since_entry = 0.0
                    stop_loss_price = 0.0
                    take_profit_price = 0.0
                else:
                    # Update highest price for trailing stop
                    if candle.high > highest_price_since_entry:
                        highest_price_since_entry = candle.high
                    # Update trailing stop level
                    if self._trailing_stop_pct > 0:
                        trailing_price = highest_price_since_entry * (
                            1 - self._trailing_stop_pct / 100
                        )
                        if trailing_price > stop_loss_price:
                            stop_loss_price = trailing_price

            # --- Strategy signal processing ---
            signal = await strategy.analyze(window, symbol=symbol)

            if signal.action == SignalAction.BUY and position_qty == 0:
                order_value = cash
                slippage = self._calculate_slippage(order_value)
                buy_price = current_price * (1 + slippage / 100)
                entry_cost = cash
                fee = cash * (self._fee_pct / 100)
                available = cash - fee
                position_qty = available / buy_price if buy_price > 0 else 0
                cash = 0
                entry_price = buy_price
                highest_price_since_entry = entry_price

                # Set exit levels
                if self._stop_loss_pct > 0:
                    stop_loss_price = entry_price * (1 - self._stop_loss_pct / 100)
                if self._take_profit_pct > 0:
                    take_profit_price = entry_price * (1 + self._take_profit_pct / 100)
                if self._trailing_stop_pct > 0 and self._stop_loss_pct == 0:
                    # If only trailing stop specified without fixed SL, use trailing as initial
                    stop_loss_price = entry_price * (1 - self._trailing_stop_pct / 100)

                trades.append(TradeLog(
                    timestamp=current_time,
                    symbol=symbol,
                    side="BUY",
                    price=buy_price,
                    quantity=position_qty,
                ))

            elif signal.action == SignalAction.SELL and position_qty > 0:
                order_value = position_qty * current_price
                slippage = self._calculate_slippage(order_value)
                sell_price = current_price * (1 - slippage / 100)
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
                    exit_reason="strategy",
                ))
                position_qty = 0
                entry_cost = 0.0
                entry_price = 0.0
                highest_price_since_entry = 0.0
                stop_loss_price = 0.0
                take_profit_price = 0.0

            # Record portfolio value
            portfolio_value = cash + (position_qty * current_price)
            metrics.record_portfolio_value(portfolio_value)
            equity_curve.append(portfolio_value)

            # Track monthly values
            month_key = current_time.strftime("%Y-%m")
            if month_key not in monthly_values:
                monthly_values[month_key] = []
            monthly_values[month_key].append(portfolio_value)

        # Final value
        final_price = data[-1].close if data else 0
        final_value = cash + (position_qty * final_price)

        # Calculate monthly returns
        monthly_returns = self._calculate_monthly_returns(
            monthly_values, self._initial_capital,
        )

        # Calculate drawdown curve
        drawdown_curve = self._calculate_drawdown_curve(equity_curve)

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=timeframe,
            metrics=metrics.calculate(),
            trades=trades,
            final_portfolio_value=round(final_value, 2),
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            monthly_returns=monthly_returns,
        )

    def _check_exit_conditions(
        self,
        candle: OHLCV,
        entry_price: float,
        position_qty: float,
        entry_cost: float,
        stop_loss_price: float,
        take_profit_price: float,
        highest_price_since_entry: float,
        symbol: str = "",
    ) -> tuple[float, str, float] | None:
        """Check if exit conditions are met on this candle.

        Uses candle low for stop-loss check and candle high for take-profit check.
        This is more realistic than using close price.

        Returns:
            Tuple of (exit_price, exit_reason, new_highest) or None.
        """
        # Update highest price with this candle's high
        new_highest = max(highest_price_since_entry, candle.high)

        # Check trailing stop update
        effective_sl = stop_loss_price
        if self._trailing_stop_pct > 0 and new_highest > entry_price:
            trailing_price = new_highest * (1 - self._trailing_stop_pct / 100)
            if trailing_price > effective_sl:
                effective_sl = trailing_price

        # Take-profit: check if candle high reached take-profit (checked first —
        # price likely reached TP at the high before SL at the low)
        if take_profit_price > 0 and candle.high >= take_profit_price:
            return (take_profit_price, "take_profit", new_highest)

        # Stop-loss: check if candle low breached stop-loss
        if effective_sl > 0 and candle.low <= effective_sl:
            exit_reason = "trailing_stop" if (
                self._trailing_stop_pct > 0
                and effective_sl > stop_loss_price
            ) else "stop_loss"
            return (effective_sl, exit_reason, new_highest)

        return None

    @staticmethod
    def _calculate_monthly_returns(
        monthly_values: dict[str, list[float]],
        initial_capital: float,
    ) -> dict[str, float]:
        """Calculate monthly returns from monthly portfolio values."""
        monthly_returns: dict[str, float] = {}
        prev_end_value = initial_capital
        for month_key in sorted(monthly_values.keys()):
            values = monthly_values[month_key]
            end_value = values[-1]
            if prev_end_value > 0:
                ret = ((end_value - prev_end_value) / prev_end_value) * 100
                monthly_returns[month_key] = round(ret, 2)
            else:
                monthly_returns[month_key] = 0.0
            prev_end_value = end_value
        return monthly_returns

    @staticmethod
    def _calculate_drawdown_curve(equity_curve: list[float]) -> list[float]:
        """Calculate drawdown percentage at each point in the equity curve."""
        if not equity_curve:
            return []
        drawdowns: list[float] = []
        peak = equity_curve[0]
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak * 100) if peak > 0 else 0.0
            drawdowns.append(round(dd, 4))
        return drawdowns

    async def walk_forward(
        self,
        strategy_factory: callable,
        data: list[OHLCV],
        train_ratio: float = 0.7,
        n_windows: int = 3,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
    ) -> BacktestResult:
        """Run walk-forward backtesting.

        Splits data into train/test windows, runs strategy on each test window,
        and combines results.

        Args:
            strategy_factory: Callable that returns a fresh strategy instance.
            data: Full OHLCV dataset.
            train_ratio: Fraction of each window used for training (0-1).
            n_windows: Number of walk-forward windows.
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            Combined BacktestResult from all test windows.
        """
        if len(data) < 10:
            strategy = strategy_factory()
            return await self.run(strategy, data, symbol, timeframe)

        window_size = len(data) // n_windows
        if window_size < 2:
            strategy = strategy_factory()
            return await self.run(strategy, data, symbol, timeframe)

        all_trades: list[TradeLog] = []
        all_equity: list[float] = [self._initial_capital]
        combined_metrics = MetricsCollector(initial_capital=self._initial_capital)
        cash = self._initial_capital
        monthly_values: dict[str, list[float]] = {}

        for w in range(n_windows):
            start = w * window_size
            end = min(start + window_size, len(data))
            if w == n_windows - 1:
                end = len(data)

            window_data = data[start:end]
            train_end = int(len(window_data) * train_ratio)

            if train_end < 2 or len(window_data) - train_end < 2:
                continue

            train_data = window_data[:train_end]
            test_data = window_data[train_end:]

            # Create fresh strategy for this window
            strategy = strategy_factory()

            # Train phase: strategy sees training data (for ML strategies)
            if hasattr(strategy, "train") and callable(strategy.train):
                await strategy.train(train_data, symbol=symbol)

            # Test phase: run on test data with current capital
            test_engine = BacktestEngine(
                initial_capital=cash,
                fee_pct=self._fee_pct,
                slippage_pct=self._slippage_pct,
                stop_loss_pct=self._stop_loss_pct,
                take_profit_pct=self._take_profit_pct,
                trailing_stop_pct=self._trailing_stop_pct,
                dynamic_slippage=self._dynamic_slippage,
                base_slippage_pct=self._base_slippage_pct,
                avg_daily_volume=self._avg_daily_volume,
            )
            result = await test_engine.run(strategy, test_data, symbol, timeframe)

            # Accumulate results
            all_trades.extend(result.trades)
            cash = result.final_portfolio_value

            # Add equity values (skip initial since it duplicates previous end)
            if result.equity_curve:
                all_equity.extend(result.equity_curve[1:])

            # Record trades in combined metrics
            for trade in result.trades:
                if trade.side == "SELL":
                    combined_metrics.record_trade(trade.pnl)

            # Track monthly
            for t in result.trades:
                month_key = t.timestamp.strftime("%Y-%m")
                if month_key not in monthly_values:
                    monthly_values[month_key] = []

        # Record final portfolio value
        combined_metrics.record_portfolio_value(cash)

        # Build monthly returns from equity curve timestamps
        monthly_returns: dict[str, float] = {}
        if monthly_values:
            monthly_returns = self._calculate_monthly_returns(
                {k: [cash] for k, v in monthly_values.items()},
                self._initial_capital,
            )

        drawdown_curve = self._calculate_drawdown_curve(all_equity)

        strategy_name = strategy_factory().name if callable(strategy_factory) else "unknown"

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            metrics=combined_metrics.calculate(),
            trades=all_trades,
            final_portfolio_value=round(cash, 2),
            equity_curve=all_equity,
            drawdown_curve=drawdown_curve,
            monthly_returns=monthly_returns,
        )

    async def compare(
        self,
        strategies: list[BaseStrategy],
        data: list[OHLCV],
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
    ) -> list[BacktestResult]:
        """Run multiple strategies on the same data and return results sorted by Sharpe ratio.

        Args:
            strategies: List of strategies to compare.
            data: OHLCV data to test against.
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            List of BacktestResult sorted by Sharpe ratio (descending).
        """
        results: list[BacktestResult] = []
        for strategy in strategies:
            result = await self.run(strategy, data, symbol, timeframe)
            results.append(result)

        # Sort by Sharpe ratio descending
        results.sort(key=lambda r: r.metrics.sharpe_ratio, reverse=True)
        return results
