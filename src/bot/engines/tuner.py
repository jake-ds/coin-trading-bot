"""Automatic strategy parameter adjustment based on performance metrics.

Evaluates engine performance and adjusts parameters within bounded ranges
to improve profitability.  All adjustments are small (max 10-20% per cycle)
to prevent wild swings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from bot.engines.base import DecisionStep

if TYPE_CHECKING:
    from bot.config import Settings
    from bot.engines.tracker import EngineMetrics

logger = structlog.get_logger(__name__)


@dataclass
class ParamChange:
    """Record of a single parameter adjustment."""

    engine_name: str
    param_name: str
    old_value: Any
    new_value: Any
    reason: str
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "param_name": self.param_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass
class TunerBounds:
    """Min/max bounds for a tunable parameter."""

    min_val: float
    max_val: float


# Engine-specific tunable parameters with bounds
TUNER_CONFIG: dict[str, dict[str, TunerBounds]] = {
    "funding_rate_arb": {
        "funding_arb_min_rate": TunerBounds(0.0001, 0.001),
        "funding_arb_max_spread_pct": TunerBounds(0.1, 1.0),
    },
    "grid_trading": {
        "grid_spacing_pct": TunerBounds(0.1, 2.0),
        "grid_levels": TunerBounds(5, 20),
    },
    "cross_exchange_arb": {
        "cross_arb_min_spread_pct": TunerBounds(0.05, 1.0),
    },
    "stat_arb": {
        "stat_arb_entry_zscore": TunerBounds(1.0, 3.0),
        "stat_arb_exit_zscore": TunerBounds(0.1, 1.0),
        "stat_arb_lookback": TunerBounds(50, 200),
        "stat_arb_min_correlation": TunerBounds(0.5, 0.95),
    },
}

# Max adjustment per cycle (as fraction of current value)
MAX_ADJUSTMENT_PCT = 0.15  # 15%


class ParameterTuner:
    """Evaluates engine performance and recommends parameter adjustments.

    Adjustment logic:
    - Sharpe < 0:  conservative (raise thresholds, reduce sizes)
    - 0 <= Sharpe < 0.5: minor adjustments toward conservative
    - 0.5 <= Sharpe < 1.0: hold steady (no changes)
    - Sharpe >= 1.0: can be slightly more aggressive (lower thresholds)
    """

    def __init__(self) -> None:
        self._history: dict[str, list[ParamChange]] = {}

    def evaluate_and_adjust(
        self,
        engine_name: str,
        metrics: EngineMetrics,
        current_params: dict[str, Any],
    ) -> list[ParamChange]:
        """Evaluate performance and return recommended parameter changes."""
        config = TUNER_CONFIG.get(engine_name)
        if not config:
            return []

        changes: list[ParamChange] = []
        sharpe = metrics.sharpe_ratio

        for param_name, bounds in config.items():
            current = current_params.get(param_name)
            if current is None:
                continue

            current_val = float(current)
            new_val = current_val

            if sharpe < 0:
                # Conservative: raise thresholds
                new_val = self._adjust_conservative(
                    param_name, current_val, bounds
                )
                reason = f"Sharpe {sharpe:.2f} < 0 — conservative adjustment"
            elif sharpe < 0.5:
                # Minor conservative adjustment
                new_val = self._adjust_slightly_conservative(
                    param_name, current_val, bounds
                )
                reason = f"Sharpe {sharpe:.2f} < 0.5 — minor conservative"
            elif sharpe >= 1.0:
                # Slightly more aggressive
                new_val = self._adjust_aggressive(
                    param_name, current_val, bounds
                )
                reason = f"Sharpe {sharpe:.2f} >= 1.0 — slight aggressive"
            else:
                # 0.5 <= sharpe < 1.0: no change
                continue

            # Clamp to bounds
            new_val = max(bounds.min_val, min(bounds.max_val, new_val))

            # Round integers
            if isinstance(current, int):
                new_val = round(new_val)

            if new_val != current_val:
                change = ParamChange(
                    engine_name=engine_name,
                    param_name=param_name,
                    old_value=current_val,
                    new_value=new_val,
                    reason=reason,
                )
                changes.append(change)

        # Store in history
        self._history.setdefault(engine_name, []).extend(changes)
        return changes

    def get_history(self, engine_name: str) -> list[ParamChange]:
        """Return adjustment history for an engine."""
        return list(self._history.get(engine_name, []))

    def get_decisions(self, changes: list[ParamChange]) -> list[DecisionStep]:
        """Convert parameter changes to DecisionSteps for dashboard."""
        decisions: list[DecisionStep] = []
        for c in changes:
            decisions.append(DecisionStep(
                label=f"파라미터 조정: {c.param_name}",
                observation=f"{c.old_value} → {c.new_value}",
                threshold="Sharpe 기반 자동 조정",
                result=c.reason,
                category="decide",
            ))
        return decisions

    def apply_changes(
        self, changes: list[ParamChange], settings: Settings
    ) -> list[str]:
        """Apply parameter changes via settings.reload()."""
        if not changes:
            return []

        updates = {c.param_name: c.new_value for c in changes}
        try:
            changed = settings.reload(updates)
            logger.info(
                "tuner_applied_changes",
                changes=[c.to_dict() for c in changes],
                applied=changed,
            )
            return changed
        except (ValueError, AttributeError) as e:
            logger.warning("tuner_apply_failed", error=str(e))
            return []

    # ------------------------------------------------------------------
    # Adjustment strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _adjust_conservative(
        param_name: str, current: float, bounds: TunerBounds
    ) -> float:
        """Raise thresholds / reduce aggressiveness (15% max)."""
        # For "threshold" params (min_rate, entry_zscore, min_spread, min_correlation)
        # raising means more conservative → increase value
        # For "size" params (grid_levels, lookback) → decrease
        if any(
            kw in param_name
            for kw in ("min_rate", "entry_zscore", "min_spread", "min_correlation")
        ):
            return current * (1 + MAX_ADJUSTMENT_PCT)
        elif any(kw in param_name for kw in ("levels", "lookback")):
            return current * (1 - MAX_ADJUSTMENT_PCT)
        elif "spacing" in param_name or "max_spread" in param_name:
            # Wider spacing = more conservative
            return current * (1 + MAX_ADJUSTMENT_PCT)
        elif "exit_zscore" in param_name:
            # Lower exit threshold = faster exit = more conservative
            return current * (1 - MAX_ADJUSTMENT_PCT)
        return current

    @staticmethod
    def _adjust_slightly_conservative(
        param_name: str, current: float, bounds: TunerBounds
    ) -> float:
        """Minor conservative adjustment (5% max)."""
        small_pct = MAX_ADJUSTMENT_PCT / 3
        if any(
            kw in param_name
            for kw in ("min_rate", "entry_zscore", "min_spread", "min_correlation")
        ):
            return current * (1 + small_pct)
        elif any(kw in param_name for kw in ("levels", "lookback")):
            return current * (1 - small_pct)
        elif "spacing" in param_name or "max_spread" in param_name:
            return current * (1 + small_pct)
        elif "exit_zscore" in param_name:
            return current * (1 - small_pct)
        return current

    @staticmethod
    def _adjust_aggressive(
        param_name: str, current: float, bounds: TunerBounds
    ) -> float:
        """Slightly more aggressive (10% max) — only when performing well."""
        agg_pct = MAX_ADJUSTMENT_PCT * 0.67
        if any(
            kw in param_name
            for kw in ("min_rate", "entry_zscore", "min_spread", "min_correlation")
        ):
            return current * (1 - agg_pct)
        elif any(kw in param_name for kw in ("levels", "lookback")):
            return current * (1 + agg_pct)
        elif "spacing" in param_name or "max_spread" in param_name:
            return current * (1 - agg_pct)
        elif "exit_zscore" in param_name:
            return current * (1 + agg_pct)
        return current
