"""Simple backtest runner for research experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BacktestResult:
    """Result of a simple backtest simulation."""

    returns: list[float]
    sharpe: float
    max_drawdown: float
    total_return: float
    num_trades: int


class SimpleBacktestRunner:
    """Runs basic backtests over historical price data for research experiments."""

    def run(
        self,
        prices: list[float],
        strategy_fn: Callable[[list[float], int], float],
    ) -> BacktestResult:
        """Run a backtest.

        Args:
            prices: Historical price series.
            strategy_fn: Function(prices, index) -> position (-1, 0, or 1).
                         Called at each bar with all prices up to that point.

        Returns:
            BacktestResult with returns, Sharpe, drawdown, etc.
        """
        if len(prices) < 2:
            return BacktestResult(
                returns=[], sharpe=0.0, max_drawdown=0.0,
                total_return=0.0, num_trades=0,
            )

        returns: list[float] = []
        positions: list[float] = []
        num_trades = 0
        prev_pos = 0.0

        for i in range(1, len(prices)):
            pos = strategy_fn(prices, i - 1)
            positions.append(pos)

            ret = pos * (prices[i] - prices[i - 1]) / prices[i - 1]
            returns.append(ret)

            if pos != prev_pos:
                num_trades += 1
            prev_pos = pos

        arr = np.array(returns)
        total_return = float(np.sum(arr))

        if len(arr) > 1 and np.std(arr) > 0:
            sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(365 * 24))
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = np.cumsum(arr)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        return BacktestResult(
            returns=returns,
            sharpe=sharpe,
            max_drawdown=max_dd,
            total_return=total_return,
            num_trades=num_trades,
        )
