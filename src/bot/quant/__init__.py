"""Quantitative finance utilities for statistical trading."""

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
    classify_volatility_regime,
)

__all__ = [
    "GARCHModel",
    "adf_test",
    "calmar_ratio",
    "calculate_half_life",
    "calculate_zscore",
    "classify_volatility_regime",
    "cornish_fisher_var",
    "cvar",
    "engle_granger_cointegration",
    "estimate_ou_params",
    "historical_var",
    "information_ratio",
    "parametric_var",
    "rolling_ols_hedge_ratio",
    "sortino_ratio",
]
