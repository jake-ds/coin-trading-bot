"""Portfolio optimization: Markowitz mean-variance, risk parity, efficient frontier.

Uses scipy.optimize for constrained portfolio optimization.
"""

from __future__ import annotations

import numpy as np
import structlog
from numpy.typing import ArrayLike
from scipy.optimize import minimize

logger = structlog.get_logger()


def _annualize(returns: np.ndarray, factor: float = 252.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute annualized mean returns and covariance matrix."""
    mean_returns = np.mean(returns, axis=0) * factor
    cov_matrix = np.cov(returns, rowvar=False) * factor
    return mean_returns, cov_matrix


def min_variance_portfolio(
    returns: ArrayLike, annualization: float = 252.0
) -> dict:
    """Find the minimum variance portfolio.

    Args:
        returns: 2D array of shape (n_periods, n_assets).
        annualization: Annualization factor.

    Returns:
        Dict with 'weights', 'expected_return', 'volatility', 'sharpe'.
    """
    ret = np.asarray(returns, dtype=float)
    if ret.ndim != 2 or ret.shape[0] < 10 or ret.shape[1] < 2:
        return _empty_portfolio(ret.shape[1] if ret.ndim == 2 else 0)

    mean_ret, cov = _annualize(ret, annualization)
    n = ret.shape[1]

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        logger.warning("min_variance_optimization_failed", message=result.message)
        return _empty_portfolio(n)

    weights = result.x
    return _build_result(weights, mean_ret, cov)


def max_sharpe_portfolio(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> dict:
    """Find the maximum Sharpe ratio portfolio.

    Args:
        returns: 2D array of shape (n_periods, n_assets).
        risk_free_rate: Annual risk-free rate.
        annualization: Annualization factor.

    Returns:
        Dict with 'weights', 'expected_return', 'volatility', 'sharpe'.
    """
    ret = np.asarray(returns, dtype=float)
    if ret.ndim != 2 or ret.shape[0] < 10 or ret.shape[1] < 2:
        return _empty_portfolio(ret.shape[1] if ret.ndim == 2 else 0)

    mean_ret, cov = _annualize(ret, annualization)
    n = ret.shape[1]

    def neg_sharpe(w: np.ndarray) -> float:
        port_return = float(w @ mean_ret)
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol < 1e-10:
            return 0.0
        return -(port_return - risk_free_rate) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        logger.warning("max_sharpe_optimization_failed", message=result.message)
        return _empty_portfolio(n)

    weights = result.x
    return _build_result(weights, mean_ret, cov, risk_free_rate)


def risk_parity_portfolio(
    returns: ArrayLike, annualization: float = 252.0
) -> dict:
    """Find the risk parity portfolio (equal risk contribution).

    Each asset contributes equally to the total portfolio variance.

    Args:
        returns: 2D array of shape (n_periods, n_assets).
        annualization: Annualization factor.

    Returns:
        Dict with 'weights', 'expected_return', 'volatility', 'sharpe',
        'risk_contributions'.
    """
    ret = np.asarray(returns, dtype=float)
    if ret.ndim != 2 or ret.shape[0] < 10 or ret.shape[1] < 2:
        return _empty_portfolio(ret.shape[1] if ret.ndim == 2 else 0)

    mean_ret, cov = _annualize(ret, annualization)
    n = ret.shape[1]
    target_risk = 1.0 / n

    def risk_parity_objective(w: np.ndarray) -> float:
        port_var = w @ cov @ w
        if port_var < 1e-20:
            return 0.0
        marginal_contrib = cov @ w
        risk_contrib = w * marginal_contrib / port_var
        return float(np.sum((risk_contrib - target_risk) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.01, 1.0)] * n  # Min 1% per asset for numerical stability
    x0 = np.ones(n) / n

    result = minimize(
        risk_parity_objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    if not result.success:
        logger.warning("risk_parity_optimization_failed", message=result.message)
        weights = np.ones(n) / n  # Fall back to equal weight
    else:
        weights = result.x

    port_result = _build_result(weights, mean_ret, cov)

    # Compute risk contributions
    port_var = weights @ cov @ weights
    if port_var > 1e-20:
        marginal = cov @ weights
        risk_contrib = weights * marginal / port_var
    else:
        risk_contrib = np.ones(n) / n

    port_result["risk_contributions"] = risk_contrib.tolist()
    return port_result


def efficient_frontier(
    returns: ArrayLike,
    n_points: int = 20,
    annualization: float = 252.0,
) -> list[dict]:
    """Compute the efficient frontier.

    Args:
        returns: 2D array of shape (n_periods, n_assets).
        n_points: Number of points on the frontier.
        annualization: Annualization factor.

    Returns:
        List of dicts with 'weights', 'expected_return', 'volatility', 'sharpe'.
    """
    ret = np.asarray(returns, dtype=float)
    if ret.ndim != 2 or ret.shape[0] < 10 or ret.shape[1] < 2:
        return []

    mean_ret, cov = _annualize(ret, annualization)
    n = ret.shape[1]

    # Find min and max feasible returns
    min_ret = float(np.min(mean_ret))
    max_ret = float(np.max(mean_ret))
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier = []
    for target in target_returns:

        def objective(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mean_ret - t},
        ]
        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            weights = result.x
            frontier.append(_build_result(weights, mean_ret, cov))

    return frontier


def _build_result(
    weights: np.ndarray,
    mean_ret: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float = 0.0,
) -> dict:
    """Build standardized portfolio result dict."""
    port_return = float(weights @ mean_ret)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (
        (port_return - risk_free_rate) / port_vol if port_vol > 1e-10 else 0.0
    )
    return {
        "weights": weights.tolist(),
        "expected_return": port_return,
        "volatility": port_vol,
        "sharpe": sharpe,
    }


def _empty_portfolio(n_assets: int) -> dict:
    """Return empty portfolio result."""
    return {
        "weights": [1.0 / max(n_assets, 1)] * max(n_assets, 1),
        "expected_return": 0.0,
        "volatility": 0.0,
        "sharpe": 0.0,
    }
