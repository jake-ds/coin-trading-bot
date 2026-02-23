"""Shared opportunity registry for dynamic symbol discovery.

The TokenScannerEngine publishes opportunities here; trading engines read them
each cycle to augment their static symbol lists.  TTL-based expiry ensures
stale opportunities are automatically cleaned up.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class OpportunityType(str, Enum):
    """Category of discovered trading opportunity."""

    FUNDING_RATE = "funding_rate"
    VOLATILITY = "volatility"
    CROSS_EXCHANGE_SPREAD = "cross_exchange_spread"
    CORRELATION = "correlation"


@dataclass
class Opportunity:
    """A single discovered trading opportunity."""

    symbol: str
    type: OpportunityType
    score: float  # 0-100, higher = more attractive
    metrics: dict[str, Any] = field(default_factory=dict)
    discovered_at: str = ""
    expires_at: str = ""
    source_exchange: str = ""

    def __post_init__(self) -> None:
        if not self.discovered_at:
            self.discovered_at = datetime.now(timezone.utc).isoformat()

    def is_expired(self) -> bool:
        """Check if this opportunity has expired."""
        if not self.expires_at:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at)
            return datetime.now(timezone.utc) >= exp
        except (ValueError, TypeError):
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "type": self.type.value,
            "score": round(self.score, 2),
            "metrics": self.metrics,
            "discovered_at": self.discovered_at,
            "expires_at": self.expires_at,
            "source_exchange": self.source_exchange,
        }


class OpportunityRegistry:
    """Thread-safe registry for trading opportunities.

    - Scanner publishes opportunities via ``publish()``.
    - Engines read via ``get_top()`` / ``get_symbols()`` / ``get_pairs()``.
    - Entries expire automatically based on TTL set at publish time.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # type -> list of Opportunity, atomically replaced on each publish
        self._store: dict[OpportunityType, list[Opportunity]] = {
            t: [] for t in OpportunityType
        }

    # ------------------------------------------------------------------
    # Write API (scanner calls these)
    # ------------------------------------------------------------------

    def publish(
        self,
        opportunity_type: OpportunityType,
        opportunities: list[Opportunity],
    ) -> None:
        """Atomically replace all opportunities of a given type."""
        with self._lock:
            self._store[opportunity_type] = list(opportunities)

    # ------------------------------------------------------------------
    # Read API (engines call these)
    # ------------------------------------------------------------------

    def get_top(
        self,
        opportunity_type: OpportunityType,
        n: int = 10,
        min_score: float = 0.0,
    ) -> list[Opportunity]:
        """Return top-N non-expired opportunities of the given type."""
        with self._lock:
            items = self._store.get(opportunity_type, [])
        # Filter expired and below min_score
        valid = [
            o for o in items
            if not o.is_expired() and o.score >= min_score
        ]
        valid.sort(key=lambda o: o.score, reverse=True)
        return valid[:n]

    def get_symbols(
        self,
        opportunity_type: OpportunityType,
        n: int = 10,
        min_score: float = 0.0,
    ) -> list[str]:
        """Return symbol strings only (convenience for engines)."""
        return [o.symbol for o in self.get_top(opportunity_type, n, min_score)]

    def get_pairs(
        self,
        n: int = 10,
        min_score: float = 0.0,
    ) -> list[list[str]]:
        """Return correlated symbol pairs for stat_arb engine.

        Pairs are stored in ``metrics["pair"]`` as ``[sym_a, sym_b]``.
        """
        opps = self.get_top(OpportunityType.CORRELATION, n, min_score)
        pairs: list[list[str]] = []
        for o in opps:
            pair = o.metrics.get("pair")
            if isinstance(pair, list) and len(pair) == 2:
                pairs.append(pair)
        return pairs

    def clear_expired(self) -> int:
        """Remove expired opportunities. Returns count removed."""
        removed = 0
        with self._lock:
            for otype in OpportunityType:
                before = len(self._store[otype])
                self._store[otype] = [
                    o for o in self._store[otype] if not o.is_expired()
                ]
                removed += before - len(self._store[otype])
        return removed

    # ------------------------------------------------------------------
    # Dashboard / monitoring
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of all opportunity types for the dashboard."""
        with self._lock:
            summary: dict[str, Any] = {}
            for otype in OpportunityType:
                items = self._store.get(otype, [])
                valid = [o for o in items if not o.is_expired()]
                summary[otype.value] = {
                    "count": len(valid),
                    "top_score": round(max((o.score for o in valid), default=0.0), 2),
                    "symbols": [o.symbol for o in sorted(
                        valid, key=lambda x: x.score, reverse=True
                    )[:5]],
                }
            return summary

    def get_all_opportunities(self) -> dict[str, list[dict[str, Any]]]:
        """Return all non-expired opportunities grouped by type."""
        with self._lock:
            result: dict[str, list[dict[str, Any]]] = {}
            for otype in OpportunityType:
                items = self._store.get(otype, [])
                valid = [o for o in items if not o.is_expired()]
                valid.sort(key=lambda o: o.score, reverse=True)
                result[otype.value] = [o.to_dict() for o in valid]
            return result
