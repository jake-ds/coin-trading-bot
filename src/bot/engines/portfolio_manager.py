"""Central capital allocation and global risk management for the multi-engine system."""

from __future__ import annotations

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
