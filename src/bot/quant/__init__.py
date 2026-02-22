"""Quantitative finance utilities for statistical trading."""

from bot.quant.microstructure import (
    compute_orderbook_metrics,
    detect_walls,
    microprice,
    orderbook_imbalance,
    vwap_midprice,
)
from bot.quant.portfolio import (
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    risk_parity_portfolio,
)
from bot.quant.risk_metrics import (
    calmar_ratio,
    cornish_fisher_var,
    cvar,
    historical_var,
    information_ratio,
    parametric_var,
    sortino_ratio,
)
from bot.quant.statistics import (
    adf_test,
    calculate_half_life,
    calculate_zscore,
    engle_granger_cointegration,
    estimate_ou_params,
    rolling_ols_hedge_ratio,
)
from bot.quant.volatility import (
    GARCHModel,
    VolatilityRegime,
    classify_volatility_regime,
)

__all__ = [
    "GARCHModel",
    "VolatilityRegime",
    "adf_test",
    "calmar_ratio",
    "calculate_half_life",
    "calculate_zscore",
    "classify_volatility_regime",
    "compute_orderbook_metrics",
    "cornish_fisher_var",
    "cvar",
    "detect_walls",
    "efficient_frontier",
    "engle_granger_cointegration",
    "estimate_ou_params",
    "historical_var",
    "information_ratio",
    "max_sharpe_portfolio",
    "microprice",
    "min_variance_portfolio",
    "orderbook_imbalance",
    "parametric_var",
    "risk_parity_portfolio",
    "rolling_ols_hedge_ratio",
    "sortino_ratio",
    "vwap_midprice",
]
