"""Statistical utilities for quantitative trading.

Provides cointegration tests, z-score calculations, mean-reversion
half-life estimation, and rolling OLS hedge ratios.
"""

from __future__ import annotations

import numpy as np
import structlog
from numpy.typing import ArrayLike
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint

logger = structlog.get_logger()


def adf_test(series: ArrayLike, max_lag: int | None = None) -> dict:
    """Run Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Time series data.
        max_lag: Maximum lag order (None for automatic selection).

    Returns:
        Dict with 'statistic', 'pvalue', 'lags', 'nobs',
        'critical_values', and 'is_stationary' (at 5% level).
    """
    arr = np.asarray(series, dtype=float)
    if len(arr) < 10:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "lags": 0,
            "nobs": len(arr),
            "critical_values": {},
            "is_stationary": False,
        }

    result = adfuller(arr, maxlag=max_lag, autolag="AIC")
    return {
        "statistic": float(result[0]),
        "pvalue": float(result[1]),
        "lags": int(result[2]),
        "nobs": int(result[3]),
        "critical_values": {k: float(v) for k, v in result[4].items()},
        "is_stationary": bool(result[1] < 0.05),
    }


def engle_granger_cointegration(
    series_a: ArrayLike, series_b: ArrayLike
) -> dict:
    """Run Engle-Granger two-step cointegration test.

    Args:
        series_a: First price series.
        series_b: Second price series.

    Returns:
        Dict with 'statistic', 'pvalue', 'critical_values',
        and 'is_cointegrated' (at 5% level).
    """
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    min_len = min(len(a), len(b))
    if min_len < 20:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "critical_values": {},
            "is_cointegrated": False,
        }

    a = a[:min_len]
    b = b[:min_len]

    stat, pvalue, crit = coint(a, b)
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "critical_values": {
            "1%": float(crit[0]),
            "5%": float(crit[1]),
            "10%": float(crit[2]),
        },
        "is_cointegrated": bool(pvalue < 0.05),
    }


def calculate_zscore(spread: ArrayLike, window: int = 20) -> np.ndarray:
    """Calculate rolling z-score of a spread series.

    Args:
        spread: Spread time series.
        window: Rolling window size.

    Returns:
        Array of z-scores (NaN for insufficient data).
    """
    arr = np.asarray(spread, dtype=float)
    n = len(arr)
    zscores = np.full(n, np.nan)

    for i in range(window - 1, n):
        w = arr[i - window + 1 : i + 1]
        mean = np.mean(w)
        std = np.std(w, ddof=1)
        if std > 1e-10:
            zscores[i] = (arr[i] - mean) / std

    return zscores


def calculate_half_life(spread: ArrayLike) -> float:
    """Estimate mean-reversion half-life using OLS on lagged spread.

    Uses the Ornstein-Uhlenbeck model: dS = theta * (mu - S) * dt
    Half-life = -ln(2) / ln(1 + theta)

    Args:
        spread: Spread time series.

    Returns:
        Half-life in periods. Returns inf if no mean-reversion detected.
    """
    arr = np.asarray(spread, dtype=float)
    if len(arr) < 10:
        return float("inf")

    y = np.diff(arr)  # delta S
    x = arr[:-1]  # lagged S

    x_with_const = add_constant(x)
    model = OLS(y, x_with_const).fit()
    theta = model.params[1]  # coefficient on lagged spread

    if theta >= 0:
        return float("inf")  # No mean reversion

    half_life = -np.log(2) / np.log(1 + theta)
    return max(float(half_life), 0.0)


def estimate_ou_params(series: ArrayLike) -> dict:
    """Estimate Ornstein-Uhlenbeck process parameters.

    Model: dX = kappa * (theta - X) * dt + sigma * dW

    Args:
        series: Time series data.

    Returns:
        Dict with 'kappa' (speed of reversion), 'theta' (long-run mean),
        'sigma' (volatility), 'half_life'.
    """
    arr = np.asarray(series, dtype=float)
    if len(arr) < 10:
        return {
            "kappa": 0.0,
            "theta": float(np.mean(arr)) if len(arr) > 0 else 0.0,
            "sigma": 0.0,
            "half_life": float("inf"),
        }

    y = np.diff(arr)
    x = arr[:-1]

    x_with_const = add_constant(x)
    model = OLS(y, x_with_const).fit()

    # dX = a + b*X => kappa = -b, theta = -a/b
    b = model.params[1]
    a = model.params[0]

    kappa = -b
    theta = -a / b if abs(b) > 1e-10 else float(np.mean(arr))
    sigma = float(np.std(model.resid, ddof=1))

    half_life = np.log(2) / kappa if kappa > 0 else float("inf")

    return {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": sigma,
        "half_life": float(half_life),
    }


def rolling_ols_hedge_ratio(
    series_a: ArrayLike, series_b: ArrayLike, window: int = 60
) -> np.ndarray:
    """Calculate rolling OLS hedge ratio between two price series.

    For each window, regresses series_a on series_b:
        series_a = beta * series_b + alpha + epsilon

    The hedge ratio beta is used to construct the spread:
        spread = series_a - beta * series_b

    Args:
        series_a: Dependent variable (price series A).
        series_b: Independent variable (price series B).
        window: Rolling window size.

    Returns:
        Array of hedge ratios (NaN for insufficient data).
    """
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]

    ratios = np.full(min_len, np.nan)

    for i in range(window - 1, min_len):
        y = a[i - window + 1 : i + 1]
        x = b[i - window + 1 : i + 1]
        x_with_const = add_constant(x)
        try:
            model = OLS(y, x_with_const).fit()
            ratios[i] = model.params[1]
        except Exception:
            ratios[i] = np.nan

    return ratios
