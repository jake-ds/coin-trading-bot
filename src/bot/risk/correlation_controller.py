"""Cross-engine correlation risk controller.

Monitors symbol concentration across engines and prevents excessive
exposure to any single symbol when multiple engines trade it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from bot.risk.portfolio_risk import PortfolioRiskManager

logger = structlog.get_logger(__name__)


@dataclass
class EnginePosition:
    """A single engine's position in a symbol."""

    symbol: str
    side: str
    notional: float


class CorrelationRiskController:
    """Monitor and limit cross-engine symbol concentration.

    Tracks positions across all engines to detect when multiple engines
    hold overlapping symbols, which can create hidden concentration risk.
    """

    def __init__(
        self,
        portfolio_risk: PortfolioRiskManager | None = None,
        max_cross_engine_correlation: float = 0.85,
        max_symbol_concentration: float = 0.4,
    ):
        self._portfolio_risk = portfolio_risk
        self._max_cross_engine_correlation = max_cross_engine_correlation
        self._max_symbol_concentration = max_symbol_concentration

        # engine_name -> list of positions
        self._engine_positions: dict[str, list[dict]] = {}

    def update_positions(
        self, engine_positions: dict[str, list[dict]]
    ) -> None:
        """Update the full engine-position map.

        Args:
            engine_positions: Mapping of engine_name to list of position
                dicts, each with keys: symbol, side, notional.
        """
        self._engine_positions = dict(engine_positions)

    def calculate_cross_engine_correlation(self) -> dict:
        """Calculate position overlap between each pair of engines.

        Returns:
            Dict keyed by 'engineA|engineB' with overlap details.
        """
        engine_names = sorted(self._engine_positions.keys())
        result: dict[str, dict] = {}

        for i, eng_a in enumerate(engine_names):
            for eng_b in engine_names[i + 1:]:
                symbols_a = {
                    p["symbol"] for p in self._engine_positions.get(eng_a, [])
                }
                symbols_b = {
                    p["symbol"] for p in self._engine_positions.get(eng_b, [])
                }

                overlap = symbols_a & symbols_b
                union = symbols_a | symbols_b

                overlap_pct = len(overlap) / len(union) if union else 0.0

                # Concentration score: notional-weighted overlap
                notional_overlap = 0.0
                total_notional = 0.0
                for p in self._engine_positions.get(eng_a, []):
                    total_notional += abs(p.get("notional", 0))
                    if p["symbol"] in overlap:
                        notional_overlap += abs(p.get("notional", 0))
                for p in self._engine_positions.get(eng_b, []):
                    total_notional += abs(p.get("notional", 0))
                    if p["symbol"] in overlap:
                        notional_overlap += abs(p.get("notional", 0))

                concentration_score = (
                    notional_overlap / total_notional
                    if total_notional > 0
                    else 0.0
                )

                key = f"{eng_a}|{eng_b}"
                result[key] = {
                    "overlap_symbols": sorted(overlap),
                    "overlap_pct": round(overlap_pct, 4),
                    "concentration_score": round(concentration_score, 4),
                }

        return result

    def check_symbol_concentration(
        self, symbol: str
    ) -> tuple[bool, str]:
        """Check if total exposure to a symbol exceeds the limit.

        Args:
            symbol: The symbol to check.

        Returns:
            (allowed, reason) tuple. False if concentration too high.
        """
        total_notional = 0.0
        engines_with_symbol: list[str] = []

        for engine_name, positions in self._engine_positions.items():
            for p in positions:
                if p.get("symbol") == symbol:
                    total_notional += abs(p.get("notional", 0))
                    engines_with_symbol.append(engine_name)

        # Get portfolio value from PortfolioRiskManager
        portfolio_value = 0.0
        if self._portfolio_risk:
            portfolio_value = self._portfolio_risk.portfolio_value

        if portfolio_value <= 0:
            return True, ""

        concentration_pct = total_notional / portfolio_value

        if concentration_pct > self._max_symbol_concentration:
            reason = (
                f"symbol {symbol} concentration "
                f"{concentration_pct:.1%} exceeds limit "
                f"{self._max_symbol_concentration:.1%} "
                f"(engines: {', '.join(engines_with_symbol)})"
            )
            logger.warning(
                "cross_engine_concentration_limit",
                symbol=symbol,
                concentration_pct=round(concentration_pct, 4),
                limit=self._max_symbol_concentration,
                engines=engines_with_symbol,
            )
            return False, reason

        return True, ""

    def get_concentration_report(self) -> dict:
        """Build a comprehensive concentration report.

        Returns:
            Dict with per_symbol breakdown, cross_engine correlations,
            and any active alerts.
        """
        portfolio_value = 0.0
        if self._portfolio_risk:
            portfolio_value = self._portfolio_risk.portfolio_value

        # Per-symbol aggregation
        symbol_data: dict[str, dict] = {}
        for engine_name, positions in self._engine_positions.items():
            for p in positions:
                sym = p.get("symbol", "")
                if not sym:
                    continue
                if sym not in symbol_data:
                    symbol_data[sym] = {
                        "engines": [],
                        "total_notional": 0.0,
                        "pct_of_capital": 0.0,
                    }
                symbol_data[sym]["engines"].append(engine_name)
                symbol_data[sym]["total_notional"] += abs(
                    p.get("notional", 0)
                )

        # Compute pct_of_capital
        for sym_info in symbol_data.values():
            if portfolio_value > 0:
                sym_info["pct_of_capital"] = round(
                    sym_info["total_notional"] / portfolio_value, 4
                )
            sym_info["total_notional"] = round(
                sym_info["total_notional"], 2
            )

        # Cross-engine correlations
        cross_engine = self.calculate_cross_engine_correlation()

        # Alerts
        alerts: list[str] = []
        for sym, info in symbol_data.items():
            if info["pct_of_capital"] > self._max_symbol_concentration:
                alerts.append(
                    f"{sym}: {info['pct_of_capital']:.1%} concentration "
                    f"exceeds {self._max_symbol_concentration:.1%} limit"
                )
        for pair_key, pair_info in cross_engine.items():
            if (
                pair_info["concentration_score"]
                > self._max_cross_engine_correlation
            ):
                alerts.append(
                    f"{pair_key}: concentration_score "
                    f"{pair_info['concentration_score']:.2f} exceeds "
                    f"{self._max_cross_engine_correlation:.2f} limit"
                )

        return {
            "per_symbol": symbol_data,
            "cross_engine_correlations": cross_engine,
            "alerts": alerts,
        }
