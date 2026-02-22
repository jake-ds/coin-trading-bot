"""GARCH volatility modeling for dynamic risk management.

Provides GARCH(1,1) fitting, volatility forecasting, dynamic stop-loss
calculation, and volatility regime classification.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import structlog
from numpy.typing import ArrayLike

logger = structlog.get_logger()


class VolatilityRegime(str, Enum):
    """Volatility regime classification."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class GARCHModel:
    """GARCH(1,1) model for volatility estimation and forecasting.

    Model: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
    """

    def __init__(self, p: int = 1, q: int = 1) -> None:
        self._p = p
        self._q = q
        self._omega: float = 0.0
        self._alpha: float = 0.0
        self._beta: float = 0.0
        self._fitted: bool = False
        self._conditional_volatility: np.ndarray | None = None
        self._last_resid: float = 0.0
        self._last_variance: float = 0.0

    def fit(self, returns: ArrayLike, rescale: bool = True) -> dict:
        """Fit GARCH(1,1) model to return series.

        Args:
            returns: Return series (log returns or simple returns).
            rescale: Whether to rescale returns for numerical stability.

        Returns:
            Dict with 'omega', 'alpha', 'beta', 'persistence',
            'unconditional_vol', and 'success'.
        """
        arr = np.asarray(returns, dtype=float)
        arr = arr[~np.isnan(arr)]

        if len(arr) < 30:
            logger.warning("garch_insufficient_data", n=len(arr))
            return self._empty_result()

        try:
            from arch import arch_model

            scale = 100.0 if rescale else 1.0
            scaled = arr * scale

            model = arch_model(scaled, vol="Garch", p=self._p, q=self._q, mean="Zero")
            result = model.fit(disp="off", show_warning=False)

            self._omega = float(result.params.get("omega", 0.0)) / (scale**2)
            self._alpha = float(result.params.get("alpha[1]", 0.0))
            self._beta = float(result.params.get("beta[1]", 0.0))
            self._fitted = True

            cond_vol = np.array(result.conditional_volatility) / scale
            self._conditional_volatility = cond_vol
            self._last_resid = float(arr[-1])
            self._last_variance = float(cond_vol[-1]) ** 2

            persistence = self._alpha + self._beta
            uncond_var = (
                self._omega / (1 - persistence) if persistence < 1 else float("inf")
            )

            return {
                "omega": self._omega,
                "alpha": self._alpha,
                "beta": self._beta,
                "persistence": persistence,
                "unconditional_vol": float(np.sqrt(abs(uncond_var))),
                "success": True,
            }
        except Exception as e:
            logger.warning("garch_fit_failed", error=str(e))
            return self._empty_result()

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Forecast volatility for given horizon.

        Args:
            horizon: Number of periods to forecast.

        Returns:
            Array of forecasted volatility (standard deviation) values.
        """
        if not self._fitted:
            return np.full(horizon, np.nan)

        forecasts = np.zeros(horizon)
        var_t = self._last_variance
        resid_sq = self._last_resid**2

        for h in range(horizon):
            if h == 0:
                var_t = self._omega + self._alpha * resid_sq + self._beta * var_t
            else:
                var_t = self._omega + (self._alpha + self._beta) * var_t
            forecasts[h] = np.sqrt(max(var_t, 0))

        return forecasts

    def dynamic_stop_loss(
        self, current_price: float, multiplier: float = 2.0, horizon: int = 1
    ) -> float:
        """Calculate dynamic stop-loss based on GARCH volatility forecast.

        Args:
            current_price: Current asset price.
            multiplier: Volatility multiplier for stop distance.
            horizon: Forecast horizon in periods.

        Returns:
            Stop-loss price level.
        """
        if not self._fitted:
            return current_price * 0.97  # fallback 3%

        vol_forecast = self.forecast(horizon)
        stop_distance = current_price * vol_forecast[0] * multiplier
        return current_price - stop_distance

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def conditional_volatility(self) -> np.ndarray | None:
        return self._conditional_volatility

    def _empty_result(self) -> dict:
        return {
            "omega": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "persistence": 0.0,
            "unconditional_vol": 0.0,
            "success": False,
        }


def classify_volatility_regime(
    returns: ArrayLike,
    window: int = 30,
    low_threshold: float = 0.5,
    high_threshold: float = 1.5,
) -> VolatilityRegime:
    """Classify current volatility regime relative to historical.

    Compares recent realized volatility to the longer-term average.

    Args:
        returns: Return series.
        window: Lookback window for recent volatility.
        low_threshold: Below this ratio -> LOW regime.
        high_threshold: Above this ratio -> HIGH regime.

    Returns:
        VolatilityRegime classification.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) < window * 2:
        return VolatilityRegime.NORMAL

    recent_vol = float(np.std(arr[-window:], ddof=1))
    long_vol = float(np.std(arr, ddof=1))

    if long_vol < 1e-10:
        return VolatilityRegime.NORMAL

    ratio = recent_vol / long_vol

    if ratio < low_threshold:
        return VolatilityRegime.LOW
    elif ratio > high_threshold:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.NORMAL
