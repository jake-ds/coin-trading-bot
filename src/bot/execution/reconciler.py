"""Position reconciliation: sync local state with exchange."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import structlog

logger = structlog.get_logger()


class DiscrepancyType(str, Enum):
    """Type of position discrepancy."""

    LOCAL_ONLY = "local_only"
    EXCHANGE_ONLY = "exchange_only"
    QTY_MISMATCH = "qty_mismatch"


@dataclass
class PositionDiscrepancy:
    """A single discrepancy between local and exchange state."""

    symbol: str
    discrepancy_type: DiscrepancyType
    local_qty: float
    exchange_qty: float
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "type": self.discrepancy_type.value,
            "local_qty": self.local_qty,
            "exchange_qty": self.exchange_qty,
            "details": self.details,
        }


@dataclass
class ReconciliationResult:
    """Result of a position reconciliation check."""

    timestamp: str = ""
    exchange_name: str = ""
    matched: list[str] = field(default_factory=list)
    local_only: list[PositionDiscrepancy] = field(default_factory=list)
    exchange_only: list[PositionDiscrepancy] = field(default_factory=list)
    qty_mismatch: list[PositionDiscrepancy] = field(default_factory=list)
    error: str | None = None

    @property
    def has_discrepancies(self) -> bool:
        return bool(self.local_only or self.exchange_only or self.qty_mismatch)

    @property
    def total_discrepancies(self) -> int:
        return len(self.local_only) + len(self.exchange_only) + len(self.qty_mismatch)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "exchange_name": self.exchange_name,
            "matched": self.matched,
            "local_only": [d.to_dict() for d in self.local_only],
            "exchange_only": [d.to_dict() for d in self.exchange_only],
            "qty_mismatch": [d.to_dict() for d in self.qty_mismatch],
            "has_discrepancies": self.has_discrepancies,
            "total_discrepancies": self.total_discrepancies,
            "error": self.error,
        }


def _base_currency(symbol: str) -> str:
    """Extract the base currency from a trading pair symbol.

    E.g., "BTC/USDT" -> "BTC", "ETH/USDT" -> "ETH".
    """
    if "/" in symbol:
        return symbol.split("/")[0]
    return symbol


class PositionReconciler:
    """Compares local position state with actual exchange balances.

    For spot trading, positions are inferred from exchange balances
    via get_balance(). The reconciler compares the base currency
    quantity of each local position against the exchange balance.
    """

    def __init__(
        self,
        tolerance_pct: float = 1.0,
        auto_fix: bool = False,
    ):
        """Initialize the reconciler.

        Args:
            tolerance_pct: Percentage tolerance for qty mismatch (default 1%).
                           Quantities within this tolerance are considered matched.
            auto_fix: If True, update local state to match exchange on discrepancy.
        """
        self.tolerance_pct = tolerance_pct
        self.auto_fix = auto_fix
        self._last_result: ReconciliationResult | None = None
        self._last_run_time: float | None = None

    @property
    def last_result(self) -> ReconciliationResult | None:
        return self._last_result

    async def reconcile(
        self,
        exchange,
        local_positions: dict[str, dict],
    ) -> ReconciliationResult:
        """Compare local positions with exchange balances.

        Args:
            exchange: Exchange adapter (must have get_balance() and name).
            local_positions: Dict of symbol -> {"quantity": float, "entry_price": float}.

        Returns:
            ReconciliationResult with matched, local_only, exchange_only, qty_mismatch.
        """
        result = ReconciliationResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            exchange_name=getattr(exchange, "name", "unknown"),
        )

        # Fetch exchange balances
        try:
            exchange_balances = await exchange.get_balance()
        except Exception as e:
            logger.warning(
                "reconciliation_balance_fetch_failed",
                exchange=result.exchange_name,
                error=str(e),
            )
            result.error = f"Failed to fetch exchange balance: {e}"
            self._last_result = result
            self._last_run_time = time.monotonic()
            return result

        # Build a map of base_currency -> exchange_qty (only non-trivial balances)
        exchange_positions: dict[str, float] = {}
        for currency, qty in exchange_balances.items():
            if qty > 1e-8:  # ignore dust
                exchange_positions[currency] = qty

        # Track which exchange currencies we've matched
        matched_currencies: set[str] = set()

        # Compare each local position against exchange
        for symbol, pos_data in local_positions.items():
            local_qty = pos_data.get("quantity", 0.0)
            if local_qty <= 0:
                continue

            base = _base_currency(symbol)
            exchange_qty = exchange_positions.get(base, 0.0)

            if exchange_qty <= 1e-8:
                # Position exists locally but not on exchange
                result.local_only.append(PositionDiscrepancy(
                    symbol=symbol,
                    discrepancy_type=DiscrepancyType.LOCAL_ONLY,
                    local_qty=local_qty,
                    exchange_qty=0.0,
                    details=f"Position {symbol} exists locally ({local_qty}) but not on exchange",
                ))
            else:
                # Both exist — check quantity match within tolerance
                diff_pct = abs(local_qty - exchange_qty) / local_qty * 100 if local_qty > 0 else 0
                if diff_pct <= self.tolerance_pct:
                    result.matched.append(symbol)
                else:
                    result.qty_mismatch.append(PositionDiscrepancy(
                        symbol=symbol,
                        discrepancy_type=DiscrepancyType.QTY_MISMATCH,
                        local_qty=local_qty,
                        exchange_qty=exchange_qty,
                        details=(
                            f"Quantity mismatch for {symbol}: "
                            f"local={local_qty}, exchange={exchange_qty} "
                            f"(diff={diff_pct:.1f}%)"
                        ),
                    ))
                matched_currencies.add(base)

        # Check for exchange-only positions (on exchange but not local)
        # Build set of base currencies from local positions
        local_bases = {
            _base_currency(s)
            for s in local_positions
            if local_positions[s].get("quantity", 0) > 0
        }
        # Quote currencies to ignore (USDT, USD, BUSD, etc.)
        quote_currencies = {"USDT", "USD", "BUSD", "USDC", "KRW", "EUR", "GBP", "JPY"}

        for currency, qty in exchange_positions.items():
            if currency in quote_currencies:
                continue
            if currency in local_bases:
                continue
            if currency in matched_currencies:
                continue
            # Exchange has this currency but we don't track it locally
            result.exchange_only.append(PositionDiscrepancy(
                symbol=f"{currency}/USDT",  # Assume USDT pair for display
                discrepancy_type=DiscrepancyType.EXCHANGE_ONLY,
                local_qty=0.0,
                exchange_qty=qty,
                details=f"{currency} balance ({qty}) found on exchange but not tracked locally",
            ))

        # Log results
        if result.has_discrepancies:
            logger.warning(
                "reconciliation_discrepancies_found",
                exchange=result.exchange_name,
                total_discrepancies=result.total_discrepancies,
                local_only=len(result.local_only),
                exchange_only=len(result.exchange_only),
                qty_mismatch=len(result.qty_mismatch),
            )
        else:
            logger.info(
                "reconciliation_clean",
                exchange=result.exchange_name,
                matched=len(result.matched),
            )

        self._last_result = result
        self._last_run_time = time.monotonic()
        return result

    def format_alert_message(self, result: ReconciliationResult) -> str:
        """Format a reconciliation result as a human-readable alert message."""
        if not result.has_discrepancies:
            return (
                f"Reconciliation OK ({result.exchange_name}): "
                f"{len(result.matched)} positions matched."
            )

        lines = [
            f"RECONCILIATION ALERT ({result.exchange_name}): "
            f"{result.total_discrepancies} discrepancy(ies) found!"
        ]
        for d in result.local_only:
            lines.append(f"  LOCAL ONLY: {d.symbol} qty={d.local_qty}")
        for d in result.exchange_only:
            lines.append(f"  EXCHANGE ONLY: {d.symbol} qty={d.exchange_qty}")
        for d in result.qty_mismatch:
            lines.append(
                f"  QTY MISMATCH: {d.symbol} local={d.local_qty} exchange={d.exchange_qty}"
            )
        return "\n".join(lines)
