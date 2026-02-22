"""Performance metrics collector."""

import math
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Trading performance metrics."""

    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0


class MetricsCollector:
    """Calculates and stores trading performance metrics."""

    def __init__(self, initial_capital: float = 10000.0, risk_free_rate: float = 0.0):
        self._initial_capital = initial_capital
        self._risk_free_rate = risk_free_rate
        self._portfolio_values: list[float] = [initial_capital]
        self._trade_returns: list[float] = []
        self._peak_value: float = initial_capital

    def record_portfolio_value(self, value: float) -> None:
        """Record a portfolio value snapshot."""
        self._portfolio_values.append(value)
        if value > self._peak_value:
            self._peak_value = value

    def record_trade(self, pnl: float) -> None:
        """Record the P&L of a completed trade."""
        self._trade_returns.append(pnl)

    def calculate(self) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        current_value = self._portfolio_values[-1]
        total_return = ((current_value - self._initial_capital) / self._initial_capital) * 100

        # Win/loss stats
        wins = [r for r in self._trade_returns if r > 0]
        losses = [r for r in self._trade_returns if r <= 0]
        total = len(self._trade_returns)
        win_rate = (len(wins) / total * 100) if total > 0 else 0.0

        # Max drawdown
        max_dd = 0.0
        peak = self._portfolio_values[0]
        for v in self._portfolio_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe ratio (simplified: using trade returns)
        sharpe = 0.0
        if len(self._trade_returns) > 1:
            mean_return = sum(self._trade_returns) / len(self._trade_returns)
            variance = sum((r - mean_return) ** 2 for r in self._trade_returns) / (
                len(self._trade_returns) - 1
            )
            std_dev = math.sqrt(variance) if variance > 0 else 0
            sharpe = (mean_return - self._risk_free_rate) / std_dev if std_dev > 0 else 0

        return PerformanceMetrics(
            total_return_pct=round(total_return, 2),
            sharpe_ratio=round(sharpe, 4),
            win_rate=round(win_rate, 2),
            max_drawdown_pct=round(max_dd, 2),
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
        )
