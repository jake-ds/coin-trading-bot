"""Mean reversion z-score strategy for single assets.

Uses ADF stationarity test and Ornstein-Uhlenbeck parameter estimation
to identify mean-reverting assets and trade z-score extremes.

Active in RANGING markets, disabled in TRENDING markets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.quant.statistics import adf_test, calculate_zscore, estimate_ou_params
from bot.strategies.base import BaseStrategy

logger = structlog.get_logger()


class MeanReversionStrategy(BaseStrategy):
    """Single-asset mean reversion strategy based on z-score."""

    def __init__(
        self,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        zscore_stop: float = 3.5,
        lookback: int = 100,
        zscore_window: int = 20,
        min_half_life: float = 2.0,
        max_half_life: float = 50.0,
        adf_pvalue: float = 0.05,
    ):
        self._zscore_entry = zscore_entry
        self._zscore_exit = zscore_exit
        self._zscore_stop = zscore_stop
        self._lookback = lookback
        self._zscore_window = zscore_window
        self._min_half_life = min_half_life
        self._max_half_life = max_half_life
        self._adf_pvalue = adf_pvalue
        self._regime_disabled = False

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def required_history_length(self) -> int:
        return self._lookback

    def adapt_to_regime(self, regime: Any) -> None:
        """Activate in RANGING, disable in TRENDING."""
        regime_str = str(regime) if regime else ""
        self._regime_disabled = "TRENDING" in regime_str

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        symbol = kwargs.get("symbol", ohlcv_data[-1].symbol if ohlcv_data else "UNKNOWN")

        if self._regime_disabled:
            return self._hold(symbol, {"reason": "regime_disabled"})

        if len(ohlcv_data) < self.required_history_length:
            return self._hold(symbol, {"reason": "insufficient_data"})

        closes = np.array([c.close for c in ohlcv_data[-self._lookback:]])

        # Test for stationarity (price returns, not levels)
        log_prices = np.log(closes)
        adf_result = adf_test(log_prices)

        # Estimate OU parameters
        ou_params = estimate_ou_params(log_prices)
        half_life = ou_params["half_life"]

        metadata = {
            "adf_statistic": round(adf_result["statistic"], 4),
            "adf_pvalue": round(adf_result["pvalue"], 4),
            "is_stationary": adf_result["is_stationary"],
            "ou_kappa": round(ou_params["kappa"], 6),
            "ou_theta": round(ou_params["theta"], 6),
            "half_life": round(half_life, 2),
        }

        # Check stationarity — if not stationary, still trade but with lower confidence
        stationarity_penalty = 1.0 if adf_result["is_stationary"] else 0.5

        # Check half-life is in tradeable range
        if not (self._min_half_life <= half_life <= self._max_half_life):
            metadata["reason"] = "half_life_out_of_range"
            return self._hold(symbol, metadata)

        # Calculate z-score on log prices
        zscores = calculate_zscore(log_prices, window=self._zscore_window)
        current_z = zscores[-1]
        if np.isnan(current_z):
            return self._hold(symbol, {"reason": "nan_zscore"})

        metadata["zscore"] = round(float(current_z), 4)

        # BUY: price below mean (z < -entry)
        if current_z < -self._zscore_entry:
            confidence = min(abs(current_z) / self._zscore_stop, 1.0) * stationarity_penalty
            metadata["signal_reason"] = "oversold_mean_reversion"
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=round(confidence, 4),
                metadata=metadata,
            )

        # SELL: price above mean (z > +entry)
        if current_z > self._zscore_entry:
            confidence = min(abs(current_z) / self._zscore_stop, 1.0) * stationarity_penalty
            metadata["signal_reason"] = "overbought_mean_reversion"
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.SELL,
                confidence=round(confidence, 4),
                metadata=metadata,
            )

        return self._hold(symbol, metadata)

    def _hold(self, symbol: str, metadata: dict) -> TradingSignal:
        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            metadata=metadata,
        )
