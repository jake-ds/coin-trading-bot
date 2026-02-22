"""Portfolio rebalancer for periodic optimal allocation adjustment.

Calculates target allocations using Risk Parity or Markowitz optimization,
generates rebalancing signals when drift exceeds threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from bot.quant.portfolio import max_sharpe_portfolio, risk_parity_portfolio

logger = structlog.get_logger()


class PortfolioRebalancer:
    """Manages periodic portfolio rebalancing.

    Computes target weights, compares to current allocations,
    and generates rebalancing orders when drift exceeds threshold.
    """

    def __init__(
        self,
        method: str = "risk_parity",
        rebalance_threshold_pct: float = 5.0,
        min_rebalance_interval: int = 24,
        annualization: float = 365 * 24,
    ):
        """Initialize rebalancer.

        Args:
            method: Optimization method ('risk_parity' or 'max_sharpe').
            rebalance_threshold_pct: Min drift % to trigger rebalancing.
            min_rebalance_interval: Min hours between rebalances.
            annualization: Annualization factor for returns (default hourly crypto).
        """
        self._method = method
        self._rebalance_threshold_pct = rebalance_threshold_pct
        self._min_rebalance_interval = min_rebalance_interval
        self._annualization = annualization
        self._hours_since_rebalance = 0
        self._target_weights: dict[str, float] = {}

    def compute_target_weights(
        self,
        returns: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Compute optimal target weights from return histories.

        Args:
            returns: Dict mapping symbol -> return array.

        Returns:
            Dict mapping symbol -> target weight (0 to 1).
        """
        symbols = list(returns.keys())
        if len(symbols) < 2:
            return {s: 1.0 / max(len(symbols), 1) for s in symbols}

        # Align return lengths
        min_len = min(len(v) for v in returns.values())
        if min_len < 20:
            return {s: 1.0 / len(symbols) for s in symbols}

        # Stack returns into matrix
        ret_matrix = np.column_stack([returns[s][-min_len:] for s in symbols])

        if self._method == "max_sharpe":
            result = max_sharpe_portfolio(ret_matrix, annualization=self._annualization)
        else:
            result = risk_parity_portfolio(ret_matrix, annualization=self._annualization)

        weights = result["weights"]
        self._target_weights = {
            symbols[i]: round(float(weights[i]), 6) for i in range(len(symbols))
        }

        logger.info(
            "rebalancer_target_computed",
            method=self._method,
            weights=self._target_weights,
            expected_return=round(result["expected_return"], 4),
            volatility=round(result["volatility"], 4),
            sharpe=round(result["sharpe"], 4),
        )

        return self._target_weights

    def check_rebalance_needed(
        self,
        current_allocations: dict[str, float],
        portfolio_value: float,
    ) -> list[dict[str, Any]]:
        """Check if rebalancing is needed and return required trades.

        Args:
            current_allocations: Dict mapping symbol -> current value.
            portfolio_value: Total portfolio value.

        Returns:
            List of rebalancing orders with symbol, action, pct fields.
        """
        self._hours_since_rebalance += 1

        if not self._target_weights:
            return []

        if self._hours_since_rebalance < self._min_rebalance_interval:
            return []

        if portfolio_value <= 0:
            return []

        orders = []
        max_drift = 0.0

        for symbol, target_weight in self._target_weights.items():
            current_value = current_allocations.get(symbol, 0.0)
            current_pct = (current_value / portfolio_value) * 100
            target_pct = target_weight * 100
            drift_pct = target_pct - current_pct
            max_drift = max(max_drift, abs(drift_pct))

            if abs(drift_pct) > self._rebalance_threshold_pct:
                orders.append({
                    "symbol": symbol,
                    "action": "BUY" if drift_pct > 0 else "SELL",
                    "target_pct": round(target_pct, 2),
                    "current_pct": round(current_pct, 2),
                    "drift_pct": round(drift_pct, 2),
                    "target_value": round(portfolio_value * target_weight, 2),
                    "current_value": round(current_value, 2),
                })

        if orders:
            self._hours_since_rebalance = 0
            logger.info(
                "rebalance_triggered",
                max_drift_pct=round(max_drift, 2),
                n_trades=len(orders),
            )

        return orders

    @property
    def target_weights(self) -> dict[str, float]:
        return dict(self._target_weights)

    def reset_timer(self) -> None:
        """Reset the rebalance interval timer."""
        self._hours_since_rebalance = 0
