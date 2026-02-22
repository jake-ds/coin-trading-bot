"""Risk metrics: VaR, CVaR, Sortino, Calmar, Information Ratio.

Provides parametric, historical, and Cornish-Fisher VaR/CVaR calculations
along with risk-adjusted performance ratios.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def parametric_var(
    returns: ArrayLike, confidence: float = 0.95, horizon: int = 1
) -> float:
    """Calculate parametric (Gaussian) Value at Risk.

    Args:
        returns: Return series.
        confidence: Confidence level (e.g., 0.95 for 95% VaR).
        horizon: Time horizon in periods.

    Returns:
        VaR as a positive number (maximum expected loss).
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return 0.0

    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    z = stats.norm.ppf(1 - confidence)

    var = -(mu * horizon + z * sigma * np.sqrt(horizon))
    return max(float(var), 0.0)


def historical_var(returns: ArrayLike, confidence: float = 0.95) -> float:
    """Calculate historical (empirical) Value at Risk.

    Args:
        returns: Return series.
        confidence: Confidence level.

    Returns:
        VaR as a positive number.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return 0.0

    percentile = (1 - confidence) * 100
    var = -float(np.percentile(arr, percentile))
    return max(var, 0.0)


def cornish_fisher_var(returns: ArrayLike, confidence: float = 0.95) -> float:
    """Calculate Cornish-Fisher adjusted VaR (accounts for skewness and kurtosis).

    Uses the Cornish-Fisher expansion to adjust the Gaussian quantile
    for non-normal return distributions.

    Args:
        returns: Return series.
        confidence: Confidence level.

    Returns:
        VaR as a positive number.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 10:
        return parametric_var(arr, confidence)

    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    if sigma < 1e-10:
        return 0.0

    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))  # excess kurtosis

    z = stats.norm.ppf(1 - confidence)

    # Cornish-Fisher expansion
    z_cf = (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * kurt / 24
        - (2 * z**3 - 5 * z) * skew**2 / 36
    )

    var = -(mu + z_cf * sigma)
    return max(float(var), 0.0)


def cvar(returns: ArrayLike, confidence: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall).

    CVaR is the expected loss given that the loss exceeds VaR.

    Args:
        returns: Return series.
        confidence: Confidence level.

    Returns:
        CVaR as a positive number.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return 0.0

    percentile = (1 - confidence) * 100
    var_threshold = float(np.percentile(arr, percentile))
    tail_losses = arr[arr <= var_threshold]

    if len(tail_losses) == 0:
        return historical_var(arr, confidence)

    return max(-float(np.mean(tail_losses)), 0.0)


def sortino_ratio(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> float:
    """Calculate Sortino ratio (downside-risk adjusted return).

    Args:
        returns: Return series.
        risk_free_rate: Risk-free rate per period.
        annualization: Annualization factor (252 for daily, 365*24 for hourly).

    Returns:
        Annualized Sortino ratio.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return 0.0

    excess = arr - risk_free_rate
    mean_excess = float(np.mean(excess))

    downside = arr[arr < risk_free_rate] - risk_free_rate
    if len(downside) == 0:
        return float("inf") if mean_excess > 0 else 0.0

    downside_std = float(np.std(downside, ddof=1))
    if downside_std < 1e-10:
        return float("inf") if mean_excess > 0 else 0.0

    return float(mean_excess / downside_std * np.sqrt(annualization))


def calmar_ratio(
    returns: ArrayLike, annualization: float = 252.0
) -> float:
    """Calculate Calmar ratio (return / max drawdown).

    Args:
        returns: Return series.
        annualization: Annualization factor.

    Returns:
        Calmar ratio.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return 0.0

    cumulative = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_dd = float(np.min(drawdown))

    if abs(max_dd) < 1e-10:
        return 0.0

    annualized_return = float(np.mean(arr)) * annualization
    return float(annualized_return / abs(max_dd))


def information_ratio(
    returns: ArrayLike,
    benchmark_returns: ArrayLike,
    annualization: float = 252.0,
) -> float:
    """Calculate Information Ratio (active return / tracking error).

    Args:
        returns: Strategy return series.
        benchmark_returns: Benchmark return series.
        annualization: Annualization factor.

    Returns:
        Annualized information ratio.
    """
    r = np.asarray(returns, dtype=float)
    b = np.asarray(benchmark_returns, dtype=float)
    min_len = min(len(r), len(b))
    if min_len < 5:
        return 0.0

    r = r[:min_len]
    b = b[:min_len]

    active = r - b
    mean_active = float(np.mean(active))
    tracking_error = float(np.std(active, ddof=1))

    if tracking_error < 1e-10:
        return 0.0

    return float(mean_active / tracking_error * np.sqrt(annualization))
