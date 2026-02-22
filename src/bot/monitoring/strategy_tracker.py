"""Per-strategy performance tracking and auto-disable logic."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from bot.strategies.base import StrategyRegistry
    from bot.strategies.regime import MarketRegime

logger = structlog.get_logger()


class StrategyStats:
    """Performance statistics for a single strategy."""

    def __init__(self) -> None:
        self.total_trades: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.total_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.trade_pnls: list[float] = []
        self.disabled: bool = False
        self.disabled_at: float | None = None
        self.disabled_regime: MarketRegime | None = None
        self.disabled_reason: str | None = None

    @property
    def win_rate(self) -> float:
        """Win rate as a percentage (0-100)."""
        if self.total_trades == 0:
            return 0.0
        return (self.wins / self.total_trades) * 100.0

    @property
    def avg_pnl(self) -> float:
        """Average PnL per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    @property
    def sharpe_ratio(self) -> float:
        """Simplified Sharpe ratio from trade PnLs."""
        if len(self.trade_pnls) < 2:
            return 0.0
        mean_return = sum(self.trade_pnls) / len(self.trade_pnls)
        variance = sum(
            (r - mean_return) ** 2 for r in self.trade_pnls
        ) / (len(self.trade_pnls) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        if std_dev == 0:
            return 0.0
        return mean_return / std_dev

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss. Returns 0 if no losses."""
        gross_profit = sum(p for p in self.trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in self.trade_pnls if p < 0))
        if gross_loss == 0:
            return 0.0
        return gross_profit / gross_loss

    def to_dict(self) -> dict[str, Any]:
        """Serialize stats to a dictionary."""
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": round(self.total_pnl, 4),
            "win_rate": round(self.win_rate, 2),
            "avg_pnl": round(self.avg_pnl, 4),
            "consecutive_losses": self.consecutive_losses,
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "profit_factor": round(self.profit_factor, 4),
            "disabled": self.disabled,
            "disabled_reason": self.disabled_reason,
            "pnl_history": [round(p, 4) for p in self.trade_pnls[-50:]],
        }


class StrategyTracker:
    """Tracks per-strategy performance and auto-disables underperforming strategies.

    Auto-disable rules:
    - If consecutive_losses >= max_consecutive_losses, disable the strategy.
    - If win_rate < min_win_rate_pct after min_trades_for_evaluation trades, disable.

    Re-enable rules:
    - Disabled strategies are re-evaluated every re_enable_check_hours hours.
    - If the market regime has changed since the strategy was disabled, re-enable it.
    """

    def __init__(
        self,
        max_consecutive_losses: int = 5,
        min_win_rate_pct: float = 40.0,
        min_trades_for_evaluation: int = 20,
        re_enable_check_hours: float = 24.0,
        registry: StrategyRegistry | None = None,
    ) -> None:
        self._max_consecutive_losses = max_consecutive_losses
        self._min_win_rate_pct = min_win_rate_pct
        self._min_trades_for_evaluation = min_trades_for_evaluation
        self._re_enable_check_hours = re_enable_check_hours
        self._registry = registry
        self._stats: dict[str, StrategyStats] = {}
        self._current_regime: MarketRegime | None = None

    @property
    def max_consecutive_losses(self) -> int:
        return self._max_consecutive_losses

    @property
    def min_win_rate_pct(self) -> float:
        return self._min_win_rate_pct

    @property
    def min_trades_for_evaluation(self) -> int:
        return self._min_trades_for_evaluation

    @property
    def re_enable_check_hours(self) -> float:
        return self._re_enable_check_hours

    def set_registry(self, registry: StrategyRegistry) -> None:
        """Set the strategy registry reference."""
        self._registry = registry

    def get_stats(self, strategy_name: str) -> StrategyStats:
        """Get or create stats for a strategy."""
        if strategy_name not in self._stats:
            self._stats[strategy_name] = StrategyStats()
        return self._stats[strategy_name]

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all tracked strategies as serializable dicts."""
        return {name: stats.to_dict() for name, stats in self._stats.items()}

    def record_trade(self, strategy_name: str, pnl: float) -> None:
        """Record a completed trade and check auto-disable conditions.

        Args:
            strategy_name: Name of the strategy that generated the signal.
            pnl: Realized profit/loss of the trade.
        """
        stats = self.get_stats(strategy_name)
        stats.total_trades += 1
        stats.total_pnl += pnl
        stats.trade_pnls.append(pnl)

        if pnl > 0:
            stats.wins += 1
            stats.consecutive_losses = 0
        else:
            stats.losses += 1
            stats.consecutive_losses += 1

        logger.info(
            "strategy_trade_recorded",
            strategy=strategy_name,
            pnl=round(pnl, 4),
            total_trades=stats.total_trades,
            win_rate=round(stats.win_rate, 2),
            consecutive_losses=stats.consecutive_losses,
        )

        # Check auto-disable conditions
        self._check_auto_disable(strategy_name, stats)

    def _check_auto_disable(
        self, strategy_name: str, stats: StrategyStats
    ) -> None:
        """Check if a strategy should be auto-disabled."""
        if stats.disabled:
            return

        reason = None

        # Check consecutive losses
        if stats.consecutive_losses >= self._max_consecutive_losses:
            reason = (
                f"consecutive_losses ({stats.consecutive_losses} "
                f">= {self._max_consecutive_losses})"
            )

        # Check win rate after minimum trades
        if (
            stats.total_trades >= self._min_trades_for_evaluation
            and stats.win_rate < self._min_win_rate_pct
        ):
            reason = (
                f"low_win_rate ({stats.win_rate:.1f}% "
                f"< {self._min_win_rate_pct}% after "
                f"{stats.total_trades} trades)"
            )

        if reason:
            stats.disabled = True
            stats.disabled_at = time.time()
            stats.disabled_regime = self._current_regime
            stats.disabled_reason = reason

            # Actually disable in the registry
            if self._registry:
                self._registry.disable(strategy_name)

            logger.warning(
                "strategy_auto_disabled",
                strategy=strategy_name,
                reason=reason,
                total_pnl=round(stats.total_pnl, 4),
                win_rate=round(stats.win_rate, 2),
            )

    def update_regime(self, regime: MarketRegime) -> None:
        """Update the current market regime and check re-enable conditions.

        Args:
            regime: The newly detected market regime.
        """
        old_regime = self._current_regime
        self._current_regime = regime

        if old_regime != regime:
            self._check_re_enable_on_regime_change()

    def check_re_enable(self) -> None:
        """Check if any disabled strategies should be re-enabled (time-based)."""
        now = time.time()
        for strategy_name, stats in self._stats.items():
            if not stats.disabled:
                continue
            if stats.disabled_at is None:
                continue

            elapsed_hours = (now - stats.disabled_at) / 3600.0
            if elapsed_hours >= self._re_enable_check_hours:
                # Time-based re-enable: check if regime changed
                if self._current_regime != stats.disabled_regime:
                    self._re_enable(strategy_name, stats, "regime_changed_after_timeout")
                else:
                    # Reset timer for another check period
                    stats.disabled_at = now
                    logger.debug(
                        "strategy_re_enable_deferred",
                        strategy=strategy_name,
                        reason="same_regime",
                    )

    def _check_re_enable_on_regime_change(self) -> None:
        """Re-enable strategies when market regime changes."""
        for strategy_name, stats in self._stats.items():
            if not stats.disabled:
                continue

            # Re-enable if the regime has changed from when it was disabled
            if (
                stats.disabled_regime is not None
                and self._current_regime != stats.disabled_regime
            ):
                self._re_enable(strategy_name, stats, "regime_changed")

    def _re_enable(
        self, strategy_name: str, stats: StrategyStats, reason: str
    ) -> None:
        """Re-enable a disabled strategy."""
        stats.disabled = False
        stats.disabled_at = None
        stats.disabled_regime = None
        stats.disabled_reason = None
        stats.consecutive_losses = 0

        if self._registry:
            self._registry.enable(strategy_name)

        logger.info(
            "strategy_re_enabled",
            strategy=strategy_name,
            reason=reason,
            current_regime=self._current_regime.value if self._current_regime else None,
        )
