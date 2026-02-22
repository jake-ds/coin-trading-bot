"""Paper trading validation framework.

Runs paper trading for a configurable duration and generates a go/no-go report
based on performance criteria.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ValidationCriteria:
    """Configurable go/no-go criteria for validation."""

    min_win_rate_pct: float = 45.0
    min_sharpe_ratio: float = 0.5
    max_drawdown_pct: float = 15.0
    min_trades: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_win_rate_pct": self.min_win_rate_pct,
            "min_sharpe_ratio": self.min_sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "min_trades": self.min_trades,
        }


@dataclass
class CriterionResult:
    """Result of evaluating a single criterion."""

    name: str
    passed: bool
    actual: float
    threshold: float
    comparison: str  # ">=", "<=", ">"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "actual": round(self.actual, 4),
            "threshold": self.threshold,
            "comparison": self.comparison,
        }


@dataclass
class ValidationReport:
    """Report generated after a paper trading validation run."""

    start_time: datetime
    end_time: datetime
    duration_seconds: float
    initial_balance: float
    final_balance: float
    total_trades: int
    wins: int
    losses: int
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    strategy_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    criteria_results: list[CriterionResult] = field(default_factory=list)
    recommendation: str = ""  # "GO" or "NO-GO"
    trade_pnls: list[float] = field(default_factory=list)

    @property
    def is_go(self) -> bool:
        return self.recommendation == "GO"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": round(self.duration_seconds, 2),
            "initial_balance": self.initial_balance,
            "final_balance": round(self.final_balance, 4),
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_return_pct": round(self.total_return_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "win_rate_pct": round(self.win_rate_pct, 2),
            "strategy_breakdown": self.strategy_breakdown,
            "recommendation": self.recommendation,
            "criteria_results": [c.to_dict() for c in self.criteria_results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def format_summary(self) -> str:
        """Format a human-readable summary of the validation report."""
        lines = []
        lines.append("=" * 60)
        lines.append("PAPER TRADING VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Duration: {self.duration_seconds:.0f}s")
        lines.append(
            f"Period: {self.start_time.strftime('%Y-%m-%d %H:%M')} "
            f"-> {self.end_time.strftime('%Y-%m-%d %H:%M')}"
        )
        lines.append("")
        lines.append("--- Performance ---")
        lines.append(f"Initial Balance: ${self.initial_balance:,.2f}")
        lines.append(f"Final Balance:   ${self.final_balance:,.2f}")
        lines.append(f"Total Return:    {self.total_return_pct:+.2f}%")
        lines.append(f"Max Drawdown:    {self.max_drawdown_pct:.2f}%")
        lines.append(f"Sharpe Ratio:    {self.sharpe_ratio:.4f}")
        lines.append(f"Total Trades:    {self.total_trades}")
        lines.append(f"Win Rate:        {self.win_rate_pct:.1f}%")
        lines.append(f"Wins/Losses:     {self.wins}/{self.losses}")
        lines.append("")

        if self.strategy_breakdown:
            lines.append("--- Strategy Breakdown ---")
            for name, stats in self.strategy_breakdown.items():
                pnl = stats.get("total_pnl", 0)
                wr = stats.get("win_rate", 0)
                trades = stats.get("total_trades", 0)
                lines.append(f"  {name}: PnL={pnl:+.4f} WR={wr:.1f}% Trades={trades}")
            lines.append("")

        lines.append("--- Criteria Evaluation ---")
        for cr in self.criteria_results:
            status = "PASS" if cr.passed else "FAIL"
            lines.append(
                f"  [{status}] {cr.name}: "
                f"{cr.actual:.4f} {cr.comparison} {cr.threshold}"
            )
        lines.append("")
        lines.append("=" * 60)
        if self.is_go:
            lines.append("  RECOMMENDATION:  GO  ")
        else:
            lines.append("  RECOMMENDATION:  NO-GO  ")
            failed = [c for c in self.criteria_results if not c.passed]
            if failed:
                lines.append("")
                lines.append("  Failed criteria:")
                for c in failed:
                    diff = c.actual - c.threshold
                    lines.append(
                        f"    - {c.name}: {c.actual:.4f} vs {c.threshold} "
                        f"(off by {diff:+.4f})"
                    )
        lines.append("=" * 60)
        return "\n".join(lines)


def evaluate_criteria(
    total_trades: int,
    win_rate_pct: float,
    sharpe_ratio: float,
    max_drawdown_pct: float,
    criteria: ValidationCriteria,
) -> tuple[str, list[CriterionResult]]:
    """Evaluate validation criteria and return recommendation + results.

    Returns:
        Tuple of (recommendation: "GO" or "NO-GO", list of CriterionResult).
    """
    results = []

    # Min trades
    results.append(CriterionResult(
        name="min_trades",
        passed=total_trades >= criteria.min_trades,
        actual=float(total_trades),
        threshold=float(criteria.min_trades),
        comparison=">=",
    ))

    # Win rate
    results.append(CriterionResult(
        name="win_rate_pct",
        passed=win_rate_pct >= criteria.min_win_rate_pct,
        actual=win_rate_pct,
        threshold=criteria.min_win_rate_pct,
        comparison=">=",
    ))

    # Sharpe ratio
    results.append(CriterionResult(
        name="sharpe_ratio",
        passed=sharpe_ratio >= criteria.min_sharpe_ratio,
        actual=sharpe_ratio,
        threshold=criteria.min_sharpe_ratio,
        comparison=">=",
    ))

    # Max drawdown (lower is better, so actual must be <= threshold)
    results.append(CriterionResult(
        name="max_drawdown_pct",
        passed=max_drawdown_pct <= criteria.max_drawdown_pct,
        actual=max_drawdown_pct,
        threshold=criteria.max_drawdown_pct,
        comparison="<=",
    ))

    all_passed = all(r.passed for r in results)
    recommendation = "GO" if all_passed else "NO-GO"
    return recommendation, results


def calculate_sharpe_from_pnls(trade_pnls: list[float]) -> float:
    """Calculate simplified Sharpe ratio from a list of trade PnLs."""
    if len(trade_pnls) < 2:
        return 0.0
    mean_return = sum(trade_pnls) / len(trade_pnls)
    variance = sum((r - mean_return) ** 2 for r in trade_pnls) / (len(trade_pnls) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0
    if std_dev == 0:
        return 0.0
    return mean_return / std_dev


def calculate_max_drawdown(equity_curve: list[float]) -> float:
    """Calculate maximum drawdown percentage from an equity curve.

    Args:
        equity_curve: List of portfolio values over time.

    Returns:
        Maximum drawdown as a percentage (e.g., 15.0 for 15%).
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

    return max_dd


def build_validation_report(
    start_time: datetime,
    end_time: datetime,
    initial_balance: float,
    final_balance: float,
    trade_pnls: list[float],
    equity_curve: list[float],
    strategy_breakdown: dict[str, dict[str, Any]],
    criteria: ValidationCriteria,
) -> ValidationReport:
    """Build a complete validation report from trading results.

    Args:
        start_time: When validation started.
        end_time: When validation ended.
        initial_balance: Starting portfolio value.
        final_balance: Ending portfolio value.
        trade_pnls: List of realized PnLs per trade.
        equity_curve: List of portfolio values over time.
        strategy_breakdown: Per-strategy stats from StrategyTracker.
        criteria: Go/no-go criteria.

    Returns:
        Complete ValidationReport with recommendation.
    """
    duration = (end_time - start_time).total_seconds()
    total_trades = len(trade_pnls)
    wins = sum(1 for p in trade_pnls if p > 0)
    losses = sum(1 for p in trade_pnls if p <= 0)
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    total_return = (
        (final_balance - initial_balance) / initial_balance * 100.0
        if initial_balance > 0
        else 0.0
    )
    sharpe = calculate_sharpe_from_pnls(trade_pnls)
    max_dd = calculate_max_drawdown(equity_curve)

    recommendation, criteria_results = evaluate_criteria(
        total_trades=total_trades,
        win_rate_pct=win_rate,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        criteria=criteria,
    )

    return ValidationReport(
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration,
        initial_balance=initial_balance,
        final_balance=final_balance,
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        total_return_pct=total_return,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        win_rate_pct=win_rate,
        strategy_breakdown=strategy_breakdown,
        criteria_results=criteria_results,
        recommendation=recommendation,
        trade_pnls=trade_pnls,
    )


def save_report(report: ValidationReport, directory: str = "data") -> str:
    """Save validation report to a JSON file.

    Args:
        report: The validation report to save.
        directory: Directory to save the report in.

    Returns:
        The file path of the saved report.
    """
    os.makedirs(directory, exist_ok=True)
    timestamp = report.end_time.strftime("%Y%m%d_%H%M%S")
    filename = f"validation_report_{timestamp}.json"
    filepath = os.path.join(directory, filename)

    with open(filepath, "w") as f:
        f.write(report.to_json())

    return filepath


def parse_duration(duration_str: str) -> float:
    """Parse a duration string like '48h', '30m', '2d' into seconds.

    Supported suffixes: s (seconds), m (minutes), h (hours), d (days).

    Args:
        duration_str: Duration string to parse.

    Returns:
        Duration in seconds.

    Raises:
        ValueError: If the duration string is invalid.
    """
    duration_str = duration_str.strip().lower()
    if not duration_str:
        raise ValueError("Empty duration string")

    multipliers = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
    }

    suffix = duration_str[-1]
    if suffix in multipliers:
        try:
            value = float(duration_str[:-1])
        except ValueError:
            raise ValueError(f"Invalid duration value: {duration_str}")
        if value <= 0:
            raise ValueError(f"Duration must be positive: {duration_str}")
        return value * multipliers[suffix]
    else:
        # Try parsing as plain seconds
        try:
            value = float(duration_str)
        except ValueError:
            raise ValueError(
                f"Invalid duration format: {duration_str}. "
                f"Use format like '48h', '30m', '2d', or plain seconds."
            )
        if value <= 0:
            raise ValueError(f"Duration must be positive: {duration_str}")
        return value
