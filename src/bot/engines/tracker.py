"""In-memory performance tracker for trading engines.

Records every trade and cycle result, computing rolling metrics such as
win rate, Sharpe ratio, max drawdown, and profit factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class TradeRecord:
    """Record of a single completed trade."""

    engine_name: str
    symbol: str
    side: str  # "buy" / "sell" / "long_short" etc.
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float  # gross PnL
    cost: float
    net_pnl: float  # pnl - cost
    entry_time: str  # ISO format
    exit_time: str  # ISO format
    hold_time_seconds: float = 0.0


@dataclass
class EngineMetrics:
    """Aggregated performance metrics for a single engine."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_profit_per_trade: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    avg_hold_time_min: float = 0.0
    cost_ratio: float = 0.0  # total_cost / total_gross_pnl
    best_trade: float = 0.0
    worst_trade: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 4),
            "avg_profit_per_trade": round(self.avg_profit_per_trade, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_hold_time_min": round(self.avg_hold_time_min, 2),
            "cost_ratio": round(self.cost_ratio, 4),
            "best_trade": round(self.best_trade, 4),
            "worst_trade": round(self.worst_trade, 4),
            "total_cost": round(self.total_cost, 4),
        }


class EngineTracker:
    """In-memory performance tracker across all engines.

    Stores trade records and cycle results.  Computes rolling metrics
    over configurable time windows.  Data resets on restart.
    """

    def __init__(self) -> None:
        self._trades: dict[str, list[TradeRecord]] = {}
        self._cycles: dict[str, list[dict]] = {}
        self._pnl_history: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_trade(self, engine_name: str, trade: TradeRecord) -> None:
        """Store a completed trade record."""
        self._trades.setdefault(engine_name, []).append(trade)
        # Update cumulative PnL history
        history = self._pnl_history.setdefault(engine_name, [])
        cum_pnl = history[-1]["cumulative_pnl"] if history else 0.0
        cum_pnl += trade.net_pnl
        history.append({
            "timestamp": trade.exit_time,
            "pnl": trade.net_pnl,
            "cumulative_pnl": round(cum_pnl, 4),
            "symbol": trade.symbol,
        })

    def record_cycle(
        self, engine_name: str, result: Any  # EngineCycleResult
    ) -> None:
        """Store a cycle result summary."""
        entry = {
            "cycle_num": result.cycle_num,
            "timestamp": result.timestamp,
            "pnl_update": result.pnl_update,
            "actions_count": len(result.actions_taken),
            "duration_ms": result.duration_ms,
        }
        self._cycles.setdefault(engine_name, []).append(entry)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(
        self, engine_name: str, window_hours: float = 24
    ) -> EngineMetrics:
        """Compute metrics from trades within a time window."""
        trades = self._filter_trades(engine_name, window_hours)
        return self._compute_metrics(trades)

    def get_all_metrics(
        self, window_hours: float = 24
    ) -> dict[str, EngineMetrics]:
        """Compute metrics for every engine with recorded trades."""
        result: dict[str, EngineMetrics] = {}
        all_engines = set(self._trades.keys()) | set(self._cycles.keys())
        for name in all_engines:
            result[name] = self.get_metrics(name, window_hours)
        return result

    def get_pnl_history(self, engine_name: str) -> list[dict]:
        """Return cumulative PnL time-series for charting."""
        return list(self._pnl_history.get(engine_name, []))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _filter_trades(
        self, engine_name: str, window_hours: float
    ) -> list[TradeRecord]:
        """Return trades within the time window."""
        all_trades = self._trades.get(engine_name, [])
        if not all_trades or window_hours <= 0:
            return all_trades

        now = datetime.now(timezone.utc)
        cutoff_seconds = window_hours * 3600
        filtered: list[TradeRecord] = []
        for t in all_trades:
            try:
                exit_dt = datetime.fromisoformat(t.exit_time)
                if exit_dt.tzinfo is None:
                    exit_dt = exit_dt.replace(tzinfo=timezone.utc)
                age = (now - exit_dt).total_seconds()
                if age <= cutoff_seconds:
                    filtered.append(t)
            except (ValueError, TypeError):
                # Include trades with unparseable timestamps
                filtered.append(t)
        return filtered

    @staticmethod
    def _compute_metrics(trades: list[TradeRecord]) -> EngineMetrics:
        """Compute aggregated metrics from a list of trades."""
        m = EngineMetrics()

        if not trades:
            return m

        m.total_trades = len(trades)
        net_pnls = [t.net_pnl for t in trades]
        total_wins = 0.0
        total_losses = 0.0

        for t in trades:
            if t.net_pnl > 0:
                m.winning_trades += 1
                total_wins += t.net_pnl
            elif t.net_pnl < 0:
                m.losing_trades += 1
                total_losses += abs(t.net_pnl)

        m.total_pnl = sum(net_pnls)
        m.total_cost = sum(t.cost for t in trades)
        m.avg_profit_per_trade = m.total_pnl / m.total_trades
        m.win_rate = m.winning_trades / m.total_trades if m.total_trades > 0 else 0.0

        # Profit factor = total_wins / total_losses (0 if no losses)
        m.profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # Best / worst trade
        m.best_trade = max(net_pnls)
        m.worst_trade = min(net_pnls)

        # Average hold time
        hold_times = [t.hold_time_seconds for t in trades if t.hold_time_seconds > 0]
        m.avg_hold_time_min = (sum(hold_times) / len(hold_times) / 60.0) if hold_times else 0.0

        # Cost ratio = total_cost / total_gross_pnl (0 if no gross pnl)
        total_gross = sum(t.pnl for t in trades)
        m.cost_ratio = m.total_cost / abs(total_gross) if total_gross != 0 else 0.0

        # Sharpe ratio (annualized, assuming hourly returns)
        # Need at least 2 trades
        if len(net_pnls) >= 2:
            mean_return = sum(net_pnls) / len(net_pnls)
            variance = sum((p - mean_return) ** 2 for p in net_pnls) / (len(net_pnls) - 1)
            std_return = math.sqrt(variance) if variance > 0 else 0.0
            if std_return > 0:
                m.sharpe_ratio = (mean_return / std_return) * math.sqrt(365 * 24)
            else:
                m.sharpe_ratio = 0.0

        # Max drawdown (percentage from peak equity)
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in net_pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            if peak > 0:
                dd = (peak - cumulative) / peak
                if dd > max_dd:
                    max_dd = dd
        m.max_drawdown = max_dd

        return m
