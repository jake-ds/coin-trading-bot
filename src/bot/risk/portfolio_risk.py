"""Portfolio-level risk management: correlation, exposure limits, heat map, VaR."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

from bot.quant.risk_metrics import (
    cornish_fisher_var,
    cvar,
    parametric_var,
)

if TYPE_CHECKING:
    from bot.models import OHLCV
    from bot.risk.volatility_service import VolatilityService

logger = structlog.get_logger()


class PortfolioRiskManager:
    """Portfolio-wide risk controls beyond per-trade limits.

    Manages:
    - Max total exposure as percentage of portfolio value
    - Correlation checks between positions to avoid redundant exposure
    - Sector/category limits for position grouping
    - Portfolio heat (aggregate risk) using position size * ATR
    - VaR/CVaR risk gates (parametric, Cornish-Fisher, stress)
    """

    def __init__(
        self,
        max_total_exposure_pct: float = 60.0,
        max_correlation: float = 0.8,
        correlation_window: int = 30,
        max_positions_per_sector: int = 3,
        max_portfolio_heat: float = 0.15,
        sector_map: dict[str, str] | None = None,
        var_enabled: bool = False,
        var_confidence: float = 0.95,
        max_portfolio_var_pct: float = 5.0,
        volatility_service: VolatilityService | None = None,
    ):
        self._max_total_exposure_pct = max_total_exposure_pct
        self._max_correlation = max_correlation
        self._correlation_window = correlation_window
        self._max_positions_per_sector = max_positions_per_sector
        self._max_portfolio_heat = max_portfolio_heat
        # Map symbol -> sector (e.g., "BTC/USDT" -> "large_cap")
        self._sector_map: dict[str, str] = sector_map or {}

        # VaR settings
        self._var_enabled = var_enabled
        self._var_confidence = var_confidence
        self._max_portfolio_var_pct = max_portfolio_var_pct

        # Optional volatility service for enhanced risk metrics
        self._volatility_service = volatility_service

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

    def calculate_portfolio_var(self) -> float | None:
        """Calculate portfolio VaR using historical returns.

        Returns:
            Portfolio VaR as a percentage, or None if insufficient data.
        """
        if not self._price_history or not self._positions:
            return None

        # Collect returns for current positions
        position_returns = []
        position_weights = []
        total_value = sum(p["value"] for p in self._positions.values())
        if total_value <= 0:
            return None

        for symbol, pos in self._positions.items():
            returns = self._price_history.get(symbol, [])
            if len(returns) < 10:
                continue
            position_returns.append(returns[-self._correlation_window :])
            position_weights.append(pos["value"] / total_value)

        if not position_returns:
            return None

        # Align lengths
        min_len = min(len(r) for r in position_returns)
        if min_len < 5:
            return None

        # Compute weighted portfolio returns
        port_returns = np.zeros(min_len)
        for ret, weight in zip(position_returns, position_weights):
            port_returns += np.array(ret[-min_len:]) * weight

        # Historical VaR
        percentile = (1 - self._var_confidence) * 100
        var_value = -float(np.percentile(port_returns, percentile))
        return max(var_value * 100, 0.0)  # Return as percentage

    def _get_portfolio_returns(self) -> np.ndarray | None:
        """Compute weighted portfolio returns from position data.

        Returns:
            Array of portfolio returns, or None if insufficient data.
        """
        if not self._price_history or not self._positions:
            return None

        position_returns = []
        position_weights = []
        total_value = sum(p["value"] for p in self._positions.values())
        if total_value <= 0:
            return None

        for symbol, pos in self._positions.items():
            returns = self._price_history.get(symbol, [])
            if len(returns) < 10:
                continue
            position_returns.append(returns[-self._correlation_window:])
            position_weights.append(pos["value"] / total_value)

        if not position_returns:
            return None

        min_len = min(len(r) for r in position_returns)
        if min_len < 5:
            return None

        port_returns = np.zeros(min_len)
        for ret, weight in zip(position_returns, position_weights):
            port_returns += np.array(ret[-min_len:]) * weight

        return port_returns

    def calculate_parametric_var(self) -> float | None:
        """Calculate parametric (Gaussian) VaR for the portfolio.

        Returns:
            VaR as a percentage, or None if insufficient data.
        """
        port_returns = self._get_portfolio_returns()
        if port_returns is None:
            return None

        var_val = parametric_var(port_returns, confidence=self._var_confidence)
        return max(var_val * 100, 0.0)

    def calculate_cornish_fisher_var(self) -> float | None:
        """Calculate Cornish-Fisher adjusted VaR for the portfolio.

        Accounts for skewness and kurtosis in the return distribution.

        Returns:
            VaR as a percentage, or None if insufficient data.
        """
        port_returns = self._get_portfolio_returns()
        if port_returns is None:
            return None

        var_val = cornish_fisher_var(port_returns, confidence=self._var_confidence)
        return max(var_val * 100, 0.0)

    def calculate_cvar(self) -> float | None:
        """Calculate Conditional VaR (Expected Shortfall) for the portfolio.

        CVaR is the expected loss given that the loss exceeds VaR.

        Returns:
            CVaR as a percentage, or None if insufficient data.
        """
        port_returns = self._get_portfolio_returns()
        if port_returns is None:
            return None

        cvar_val = cvar(port_returns, confidence=self._var_confidence)
        return max(cvar_val * 100, 0.0)

    def calculate_stress_var(self, n_simulations: int = 1000) -> float | None:
        """Calculate stress VaR using Monte Carlo simulation.

        Uses Cholesky decomposition of the correlation matrix to generate
        correlated random returns and computes VaR from the simulated
        portfolio return distribution.

        Args:
            n_simulations: Number of Monte Carlo simulations.

        Returns:
            Stress VaR as a percentage, or None if insufficient data.
        """
        if not self._price_history or not self._positions:
            return None

        symbols_with_data = []
        returns_matrix = []
        weights = []
        total_value = sum(p["value"] for p in self._positions.values())
        if total_value <= 0:
            return None

        for symbol, pos in self._positions.items():
            returns = self._price_history.get(symbol, [])
            if len(returns) < 10:
                continue
            symbols_with_data.append(symbol)
            returns_matrix.append(returns[-self._correlation_window:])
            weights.append(pos["value"] / total_value)

        if len(symbols_with_data) < 1:
            return None

        # Align lengths
        min_len = min(len(r) for r in returns_matrix)
        if min_len < 5:
            return None

        aligned = np.array([r[-min_len:] for r in returns_matrix])
        weights_arr = np.array(weights)

        # Mean and std per asset
        means = np.mean(aligned, axis=1)
        stds = np.std(aligned, axis=1, ddof=1)

        # Handle single-asset case
        if len(symbols_with_data) == 1:
            rng = np.random.default_rng(42)
            simulated = rng.normal(means[0], max(stds[0], 1e-10), n_simulations)
            port_sim = simulated * weights_arr[0]
            percentile = (1 - self._var_confidence) * 100
            var_val = -float(np.percentile(port_sim, percentile))
            return max(var_val * 100, 0.0)

        # Correlation matrix
        corr_matrix = np.corrcoef(aligned)

        # Fix any NaN in correlation matrix (can happen with constant series)
        if np.any(np.isnan(corr_matrix)):
            np.fill_diagonal(corr_matrix, 1.0)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Ensure positive semi-definite via eigenvalue clipping
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            # Re-normalize to correlation matrix
            d = np.sqrt(np.diag(corr_matrix))
            corr_matrix = corr_matrix / np.outer(d, d)
        except np.linalg.LinAlgError:
            # Fallback: use identity matrix
            corr_matrix = np.eye(len(symbols_with_data))

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            L = np.eye(len(symbols_with_data))

        # Generate correlated random returns
        rng = np.random.default_rng(42)
        z = rng.standard_normal((len(symbols_with_data), n_simulations))
        correlated_z = L @ z

        # Scale to actual return distributions
        simulated_returns = np.zeros((len(symbols_with_data), n_simulations))
        for i in range(len(symbols_with_data)):
            simulated_returns[i] = means[i] + stds[i] * correlated_z[i]

        # Weighted portfolio returns
        port_sim = weights_arr @ simulated_returns

        percentile = (1 - self._var_confidence) * 100
        var_val = -float(np.percentile(port_sim, percentile))
        return max(var_val * 100, 0.0)

    def pre_trade_var_check(
        self, symbol: str, position_value: float
    ) -> tuple[bool, str]:
        """Simulate adding a new position and check if VaR exceeds the limit.

        Creates a hypothetical portfolio with the new position added and
        computes VaR. If the resulting VaR exceeds max_portfolio_var_pct,
        the trade is rejected.

        Args:
            symbol: Symbol of the proposed position.
            position_value: Notional value of the proposed position.

        Returns:
            (allowed, reason) tuple.
        """
        if not self._var_enabled:
            return True, ""

        # Need returns for the symbol
        symbol_returns = self._price_history.get(symbol, [])
        if len(symbol_returns) < 10:
            return True, ""  # Can't calculate — allow

        # Build hypothetical portfolio returns including the new position
        total_value = sum(p["value"] for p in self._positions.values())
        new_total = total_value + position_value
        if new_total <= 0:
            return True, ""

        position_returns = []
        position_weights = []

        for sym, pos in self._positions.items():
            returns = self._price_history.get(sym, [])
            if len(returns) < 10:
                continue
            position_returns.append(returns[-self._correlation_window:])
            position_weights.append(pos["value"] / new_total)

        # Add the new position
        position_returns.append(symbol_returns[-self._correlation_window:])
        position_weights.append(position_value / new_total)

        if not position_returns:
            return True, ""

        min_len = min(len(r) for r in position_returns)
        if min_len < 5:
            return True, ""

        port_returns = np.zeros(min_len)
        for ret, weight in zip(position_returns, position_weights):
            port_returns += np.array(ret[-min_len:]) * weight

        # Use historical VaR on the hypothetical portfolio
        percentile = (1 - self._var_confidence) * 100
        var_val = -float(np.percentile(port_returns, percentile))
        projected_var = max(var_val * 100, 0.0)

        if projected_var > self._max_portfolio_var_pct:
            reason = (
                f"projected_var {projected_var:.2f}% would exceed "
                f"limit {self._max_portfolio_var_pct:.2f}%"
            )
            logger.warning(
                "portfolio_pre_trade_var_limit",
                symbol=symbol,
                projected_var_pct=round(projected_var, 2),
                limit_pct=self._max_portfolio_var_pct,
            )
            return False, reason

        return True, ""

    def check_var_limit(self, symbol: str, position_value: float) -> tuple[bool, str]:
        """Check if adding a position would breach portfolio VaR limit.

        Returns:
            (allowed, reason) tuple.
        """
        if not self._var_enabled:
            return True, ""

        current_var = self.calculate_portfolio_var()
        if current_var is None:
            return True, ""  # Can't calculate — allow

        if current_var > self._max_portfolio_var_pct:
            reason = (
                f"portfolio_var {current_var:.2f}% exceeds "
                f"limit {self._max_portfolio_var_pct:.2f}%"
            )
            logger.warning(
                "portfolio_var_limit",
                current_var_pct=round(current_var, 2),
                limit_pct=self._max_portfolio_var_pct,
                new_symbol=symbol,
            )
            return False, reason

        return True, ""

    def get_risk_metrics(self) -> dict:
        """Get current portfolio risk metrics summary.

        Returns:
            Dict with exposure, heat, VaR variants, CVaR, positions.
        """
        return {
            "exposure_pct": round(self.get_total_exposure(), 2),
            "heat": round(self.calculate_portfolio_heat(), 4),
            "var_pct": self.calculate_portfolio_var(),
            "parametric_var": self.calculate_parametric_var(),
            "cornish_fisher_var": self.calculate_cornish_fisher_var(),
            "cvar": self.calculate_cvar(),
            "stress_var": self.calculate_stress_var(),
            "n_positions": len(self._positions),
            "portfolio_value": self._portfolio_value,
            "var_enabled": self._var_enabled,
        }

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
        5. VaR limit (if enabled)
        6. Pre-trade VaR simulation (if enabled)

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

        # 5. VaR limit
        allowed, reason = self.check_var_limit(symbol, position_value)
        if not allowed:
            return False, reason

        # 6. Pre-trade VaR simulation
        allowed, reason = self.pre_trade_var_check(symbol, position_value)
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
