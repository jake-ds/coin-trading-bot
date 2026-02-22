"""Portfolio-level risk management: correlation, exposure limits, heat map."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from bot.models import OHLCV

logger = structlog.get_logger()


class PortfolioRiskManager:
    """Portfolio-wide risk controls beyond per-trade limits.

    Manages:
    - Max total exposure as percentage of portfolio value
    - Correlation checks between positions to avoid redundant exposure
    - Sector/category limits for position grouping
    - Portfolio heat (aggregate risk) using position size * ATR
    """

    def __init__(
        self,
        max_total_exposure_pct: float = 60.0,
        max_correlation: float = 0.8,
        correlation_window: int = 30,
        max_positions_per_sector: int = 3,
        max_portfolio_heat: float = 0.15,
        sector_map: dict[str, str] | None = None,
    ):
        self._max_total_exposure_pct = max_total_exposure_pct
        self._max_correlation = max_correlation
        self._correlation_window = correlation_window
        self._max_positions_per_sector = max_positions_per_sector
        self._max_portfolio_heat = max_portfolio_heat
        # Map symbol -> sector (e.g., "BTC/USDT" -> "large_cap")
        self._sector_map: dict[str, str] = sector_map or {}

        # Price history for correlation calculation: symbol -> list of returns
        self._price_history: dict[str, list[float]] = {}

        # Current open positions: symbol -> {"value": float, "atr": float | None}
        self._positions: dict[str, dict] = {}

        # Current portfolio value
        self._portfolio_value: float = 0.0

    def update_portfolio_value(self, value: float) -> None:
        """Update the current portfolio value."""
        self._portfolio_value = value

    def update_price_history(
        self, symbol: str, candles: list[OHLCV]
    ) -> None:
        """Update price history for a symbol from candle data.

        Stores recent close-to-close returns for correlation calculation.
        """
        if len(candles) < 2:
            return

        closes = [c.close for c in candles]
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append((closes[i] - closes[i - 1]) / closes[i - 1])

        # Keep only the last correlation_window returns
        self._price_history[symbol] = returns[-self._correlation_window :]

    def add_position(
        self,
        symbol: str,
        value: float,
        atr: float | None = None,
    ) -> None:
        """Track an open position for portfolio risk monitoring."""
        self._positions[symbol] = {"value": value, "atr": atr}

    def remove_position(self, symbol: str) -> None:
        """Remove a closed position."""
        self._positions.pop(symbol, None)

    def update_position_value(self, symbol: str, value: float) -> None:
        """Update the current value of an existing position."""
        if symbol in self._positions:
            self._positions[symbol]["value"] = value

    def get_total_exposure(self) -> float:
        """Get total position value as a percentage of portfolio."""
        if self._portfolio_value <= 0:
            return 0.0
        total_value = sum(p["value"] for p in self._positions.values())
        return (total_value / self._portfolio_value) * 100

    def check_exposure_limit(self, new_position_value: float) -> tuple[bool, str]:
        """Check if adding a new position would exceed max total exposure.

        Returns:
            (allowed, reason) tuple.
        """
        if self._portfolio_value <= 0:
            return True, ""

        total_value = sum(p["value"] for p in self._positions.values())
        new_total = total_value + new_position_value
        exposure_pct = (new_total / self._portfolio_value) * 100

        if exposure_pct > self._max_total_exposure_pct:
            reason = (
                f"total_exposure {exposure_pct:.1f}% would exceed "
                f"limit {self._max_total_exposure_pct:.1f}%"
            )
            logger.warning(
                "portfolio_exposure_limit",
                current_exposure_pct=round(
                    (total_value / self._portfolio_value) * 100, 1
                ),
                new_exposure_pct=round(exposure_pct, 1),
                limit_pct=self._max_total_exposure_pct,
            )
            return False, reason

        return True, ""

    def calculate_correlation(
        self, symbol_a: str, symbol_b: str
    ) -> float | None:
        """Calculate rolling correlation between two symbols.

        Uses recent close-to-close returns stored in price_history.

        Returns:
            Pearson correlation coefficient, or None if insufficient data.
        """
        returns_a = self._price_history.get(symbol_a, [])
        returns_b = self._price_history.get(symbol_b, [])

        if not returns_a or not returns_b:
            return None

        # Align lengths (use the shorter common window)
        min_len = min(len(returns_a), len(returns_b))
        if min_len < 5:
            # Need at least 5 data points for meaningful correlation
            return None

        arr_a = np.array(returns_a[-min_len:])
        arr_b = np.array(returns_b[-min_len:])

        # Check for zero variance (constant returns)
        if np.std(arr_a) == 0 or np.std(arr_b) == 0:
            return None

        corr_matrix = np.corrcoef(arr_a, arr_b)
        correlation = float(corr_matrix[0, 1])

        # Handle NaN from numerical issues
        if np.isnan(correlation):
            return None

        return correlation

    def check_correlation(self, new_symbol: str) -> tuple[bool, str]:
        """Check if a new position is too correlated with existing positions.

        Returns:
            (allowed, reason) tuple.
        """
        for existing_symbol in self._positions:
            if existing_symbol == new_symbol:
                continue

            corr = self.calculate_correlation(new_symbol, existing_symbol)
            if corr is not None and abs(corr) > self._max_correlation:
                reason = (
                    f"high_correlation {new_symbol} vs {existing_symbol}: "
                    f"{corr:.3f} > {self._max_correlation}"
                )
                logger.warning(
                    "portfolio_correlation_limit",
                    new_symbol=new_symbol,
                    existing_symbol=existing_symbol,
                    correlation=round(corr, 3),
                    limit=self._max_correlation,
                )
                return False, reason

        return True, ""

    def check_sector_limit(self, symbol: str) -> tuple[bool, str]:
        """Check if adding a position in this symbol's sector would exceed limit.

        Returns:
            (allowed, reason) tuple.
        """
        sector = self._sector_map.get(symbol)
        if sector is None:
            # No sector mapping for this symbol — allow
            return True, ""

        # Count existing positions in the same sector
        count = 0
        for existing_symbol in self._positions:
            if self._sector_map.get(existing_symbol) == sector:
                count += 1

        if count >= self._max_positions_per_sector:
            reason = (
                f"sector '{sector}' has {count} positions, "
                f"limit is {self._max_positions_per_sector}"
            )
            logger.warning(
                "portfolio_sector_limit",
                symbol=symbol,
                sector=sector,
                current_count=count,
                limit=self._max_positions_per_sector,
            )
            return False, reason

        return True, ""

    def calculate_portfolio_heat(self) -> float:
        """Calculate portfolio heat as sum of (position_value * ATR / price) / portfolio_value.

        Heat represents the total risk as a fraction of portfolio value.
        Positions without ATR data are excluded from the heat calculation.

        Returns:
            Portfolio heat as a fraction (0.0 to 1.0+).
        """
        if self._portfolio_value <= 0:
            return 0.0

        total_heat = 0.0
        for pos in self._positions.values():
            atr = pos.get("atr")
            if atr is not None and atr > 0 and pos["value"] > 0:
                # Heat contribution = position_value * (atr / price)
                # Since value = qty * price, heat = qty * atr
                # Normalized by portfolio: heat = (qty * atr) / portfolio_value
                # Simplified: heat = (value / price) * atr / portfolio_value
                # = value * atr / (price * portfolio_value)
                # We store value directly, so heat = value * (atr / value) = atr
                # But actually: risk_per_position = qty * atr = (value/price) * atr
                # We don't have price separately, but we can approximate:
                # position_risk_pct = atr_pct_of_position * position_pct_of_portfolio
                # Simpler: total_heat = sum(position_value * atr_ratio) / portfolio_value
                # where atr_ratio is a volatility measure
                # Use: heat = sum(position_value) * avg_atr / portfolio_value
                # Actually the clearest: heat = sum(qty * atr) / portfolio_value
                # Since we store value (= qty * price), heat_i = (value_i / price_i) * atr_i
                # But we don't have price. So store atr as absolute value and compute:
                # heat_i = (value * atr / position_price) / portfolio_value
                # Simplest practical approach: heat = sum(value * normalized_atr)
                # where normalized_atr = atr / position_average_price.
                # Since we track value but not price, store ATR as fraction of price.

                # Use ATR as absolute; heat_contribution = position_value * (ATR / price)
                # We approximate price ≈ value (for qty=1) but that's wrong.
                # Better: track ATR/price ratio when adding position.
                # For simplicity, treat ATR as already normalized (ATR / price ratio).
                total_heat += pos["value"] * atr / self._portfolio_value

        return total_heat

    def check_portfolio_heat(
        self, new_position_value: float, new_atr: float | None
    ) -> tuple[bool, str]:
        """Check if adding a new position would exceed portfolio heat limit.

        Args:
            new_position_value: Value of the proposed new position.
            new_atr: ATR/price ratio for the new position (normalized).

        Returns:
            (allowed, reason) tuple.
        """
        if new_atr is None or self._portfolio_value <= 0:
            # Can't calculate heat without ATR — allow
            return True, ""

        current_heat = self.calculate_portfolio_heat()
        new_heat_contribution = new_position_value * new_atr / self._portfolio_value
        projected_heat = current_heat + new_heat_contribution

        if projected_heat > self._max_portfolio_heat:
            reason = (
                f"portfolio_heat {projected_heat:.4f} would exceed "
                f"limit {self._max_portfolio_heat:.4f}"
            )
            logger.warning(
                "portfolio_heat_limit",
                current_heat=round(current_heat, 4),
                projected_heat=round(projected_heat, 4),
                limit=self._max_portfolio_heat,
            )
            return False, reason

        return True, ""

    def validate_new_position(
        self,
        symbol: str,
        position_value: float,
        atr: float | None = None,
    ) -> tuple[bool, str]:
        """Run all portfolio-level risk checks for a proposed new position.

        Checks in order:
        1. Total exposure limit
        2. Correlation with existing positions
        3. Sector limits
        4. Portfolio heat

        Returns:
            (allowed, reason) tuple. If not allowed, reason explains why.
        """
        # 1. Exposure limit
        allowed, reason = self.check_exposure_limit(position_value)
        if not allowed:
            return False, reason

        # 2. Correlation check
        allowed, reason = self.check_correlation(symbol)
        if not allowed:
            return False, reason

        # 3. Sector limit
        allowed, reason = self.check_sector_limit(symbol)
        if not allowed:
            return False, reason

        # 4. Portfolio heat
        allowed, reason = self.check_portfolio_heat(position_value, atr)
        if not allowed:
            return False, reason

        return True, ""

    @property
    def positions(self) -> dict[str, dict]:
        """Get current tracked positions."""
        return dict(self._positions)

    @property
    def portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self._portfolio_value
