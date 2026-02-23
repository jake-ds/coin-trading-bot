"""Central capital allocation and global risk management for the multi-engine system."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)


class PortfolioManager:
    """Manages capital allocation across multiple trading engines.

    Each engine requests capital from the PortfolioManager and returns it
    when positions are closed.  The PM enforces per-engine allocation limits
    and a global drawdown circuit-breaker.
    """

    def __init__(
        self,
        total_capital: float,
        engine_allocations: dict[str, float] | None = None,
        max_drawdown_pct: float = 20.0,
    ):
        """
        Args:
            total_capital: Total capital available for all engines.
            engine_allocations: Fraction of total capital each engine may use.
                                E.g. {"funding_rate_arb": 0.30, "grid_trading": 0.25}.
                                Values should sum to <= 1.0.
            max_drawdown_pct: Global portfolio drawdown % that triggers a halt.
        """
        self._total_capital = total_capital
        self._engine_allocations = engine_allocations or {}
        self._max_drawdown_pct = max_drawdown_pct

        # Runtime state
        self._allocated: dict[str, float] = {}  # engine -> currently allocated
        self._engine_pnl: dict[str, float] = {}  # engine -> cumulative PnL
        self._peak_capital = total_capital
        self._drawdown_history: list[dict] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_capital(self) -> float:
        return self._total_capital

    @property
    def available_capital(self) -> float:
        return self._total_capital - sum(self._allocated.values())

    @property
    def total_allocated(self) -> float:
        return sum(self._allocated.values())

    @property
    def total_pnl(self) -> float:
        return sum(self._engine_pnl.values())

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def get_max_allocation(self, engine_name: str) -> float:
        """Return the maximum capital an engine is allowed to have."""
        fraction = self._engine_allocations.get(engine_name, 0.0)
        return self._total_capital * fraction

    def request_capital(self, engine_name: str, amount: float) -> float:
        """Request capital for an engine.

        Returns the actually allocated amount (may be less than requested
        if available capital is insufficient or the engine's limit is hit).
        """
        max_allowed = self.get_max_allocation(engine_name)
        currently_allocated = self._allocated.get(engine_name, 0.0)
        remaining_allowance = max(0.0, max_allowed - currently_allocated)
        available = self.available_capital

        actual = min(amount, remaining_allowance, available)
        if actual <= 0:
            return 0.0

        self._allocated[engine_name] = currently_allocated + actual
        logger.info(
            "capital_allocated",
            engine=engine_name,
            requested=amount,
            allocated=actual,
            total_allocated=self.total_allocated,
        )
        return actual

    def release_capital(self, engine_name: str, amount: float) -> None:
        """Return capital after an engine closes positions or shuts down."""
        currently = self._allocated.get(engine_name, 0.0)
        release = min(amount, currently)
        self._allocated[engine_name] = currently - release
        if self._allocated[engine_name] <= 0:
            self._allocated.pop(engine_name, None)
        logger.info(
            "capital_released",
            engine=engine_name,
            released=release,
            total_allocated=self.total_allocated,
        )

    # ------------------------------------------------------------------
    # PnL tracking
    # ------------------------------------------------------------------

    def report_pnl(self, engine_name: str, pnl_delta: float) -> None:
        """Report PnL change from an engine cycle."""
        current = self._engine_pnl.get(engine_name, 0.0)
        self._engine_pnl[engine_name] = current + pnl_delta
        # Update peak
        current_total = self._total_capital + self.total_pnl
        if current_total > self._peak_capital:
            self._peak_capital = current_total
        # Record drawdown history
        self._drawdown_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drawdown_pct": round(self.get_global_drawdown(), 2),
            "equity": round(current_total, 2),
        })
        if len(self._drawdown_history) > 1000:
            self._drawdown_history = self._drawdown_history[-1000:]

    def get_engine_pnl(self, engine_name: str) -> float:
        return self._engine_pnl.get(engine_name, 0.0)

    # ------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------

    def get_global_drawdown(self) -> float:
        """Current drawdown as a percentage from peak."""
        current_total = self._total_capital + self.total_pnl
        if self._peak_capital <= 0:
            return 0.0
        drawdown = (self._peak_capital - current_total) / self._peak_capital * 100
        return max(0.0, drawdown)

    def is_drawdown_breached(self) -> bool:
        """Check if global drawdown exceeds the configured limit."""
        return self.get_global_drawdown() >= self._max_drawdown_pct

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance_allocations(
        self, metrics: dict[str, object]
    ) -> dict[str, float]:
        """Rebalance engine allocations based on Sharpe-weighted performance.

        Args:
            metrics: dict mapping engine_name -> EngineMetrics (with sharpe_ratio attr).

        Returns:
            New allocation fractions after rebalancing.
        """
        if not self._engine_allocations:
            return {}

        # Compute Sharpe-weighted allocations
        weights: dict[str, float] = {}
        for name in self._engine_allocations:
            m = metrics.get(name)
            sharpe = getattr(m, "sharpe_ratio", 0.0) if m else 0.0
            weights[name] = max(sharpe, 0.1)

        # Normalize to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return dict(self._engine_allocations)

        normalized = {k: v / total_weight for k, v in weights.items()}

        # Project onto bounded simplex {x: sum=1, min<=x_i<=max}
        MIN_ALLOC = 0.10
        MAX_ALLOC = 0.40
        n = len(normalized)

        # Ensure feasibility: n * max >= 1.0
        if n * MAX_ALLOC < 1.0:
            effective_max = 1.0 - (n - 1) * MIN_ALLOC
        else:
            effective_max = MAX_ALLOC

        keys = list(normalized.keys())
        y = [normalized[k] for k in keys]

        # Binary search for Lagrange multiplier lambda such that
        # sum(clip(y_i - lambda, MIN, MAX)) = 1.0
        lo = min(y) - effective_max - 1.0
        hi = max(y) + 1.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            s = sum(
                max(MIN_ALLOC, min(effective_max, yi - mid)) for yi in y
            )
            if s > 1.0 + 1e-12:
                lo = mid
            elif s < 1.0 - 1e-12:
                hi = mid
            else:
                break

        lam = (lo + hi) / 2.0
        clipped = {
            keys[i]: max(MIN_ALLOC, min(effective_max, y[i] - lam))
            for i in range(n)
        }

        old = dict(self._engine_allocations)
        self._engine_allocations = clipped

        logger.info(
            "portfolio_rebalanced",
            old_allocations=old,
            new_allocations={k: round(v, 4) for k, v in clipped.items()},
        )
        return clipped

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_engine_allocation(self, engine_name: str) -> dict:
        """Return allocation details for a specific engine."""
        return {
            "engine": engine_name,
            "allocated": self._allocated.get(engine_name, 0.0),
            "max_allowed": self.get_max_allocation(engine_name),
            "pnl": self._engine_pnl.get(engine_name, 0.0),
        }

    def get_summary(self) -> dict:
        """Overall portfolio summary."""
        return {
            "total_capital": self._total_capital,
            "total_allocated": self.total_allocated,
            "available_capital": self.available_capital,
            "total_pnl": round(self.total_pnl, 2),
            "peak_capital": self._peak_capital,
            "global_drawdown_pct": round(self.get_global_drawdown(), 2),
            "drawdown_breached": self.is_drawdown_breached(),
            "engine_allocations": {
                name: {
                    "allocated": self._allocated.get(name, 0.0),
                    "max_fraction": frac,
                    "pnl": self._engine_pnl.get(name, 0.0),
                }
                for name, frac in self._engine_allocations.items()
            },
        }
