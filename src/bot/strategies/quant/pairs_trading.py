"""Statistical pairs trading strategy using cointegration and z-score.

Entry: z-score crosses ±2.0 threshold (spread expected to revert).
Exit: z-score crosses 0 (spread reverted) or ±3.0 (stop-loss).

Uses Engle-Granger cointegration for pair validation, rolling OLS
for dynamic hedge ratio, and half-life filter for tradeable pairs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.quant.statistics import (
    calculate_half_life,
    calculate_zscore,
    engle_granger_cointegration,
    rolling_ols_hedge_ratio,
)
from bot.strategies.base import BaseStrategy

logger = structlog.get_logger()


class PairsTradingStrategy(BaseStrategy):
    """Statistical pairs trading strategy.

    Requires two aligned price series passed via kwargs['pair_prices'].
    """

    def __init__(
        self,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        zscore_stop: float = 3.0,
        hedge_ratio_window: int = 60,
        zscore_window: int = 20,
        min_half_life: float = 2.0,
        max_half_life: float = 50.0,
        coint_pvalue: float = 0.05,
        recalc_interval: int = 20,
    ):
        self._zscore_entry = zscore_entry
        self._zscore_exit = zscore_exit
        self._zscore_stop = zscore_stop
        self._hedge_ratio_window = hedge_ratio_window
        self._zscore_window = zscore_window
        self._min_half_life = min_half_life
        self._max_half_life = max_half_life
        self._coint_pvalue = coint_pvalue
        self._recalc_interval = recalc_interval
        self._regime_disabled = False
        self._candles_since_recalc = 0
        self._cached_coint: dict | None = None

    @property
    def name(self) -> str:
        return "pairs_trading"

    @property
    def required_history_length(self) -> int:
        return max(self._hedge_ratio_window, self._zscore_window) + 10

    def adapt_to_regime(self, regime: Any) -> None:
        """Disable in HIGH_VOLATILITY regime."""
        regime_str = str(regime) if regime else ""
        self._regime_disabled = "HIGH_VOLATILITY" in regime_str

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        """Analyze pair spread for trading signals.

        Expected kwargs:
            pair_prices: dict with 'symbol_a', 'symbol_b', 'prices_a', 'prices_b'
            symbol: str (optional, defaults to pair name)
        """
        symbol = kwargs.get("symbol", ohlcv_data[-1].symbol if ohlcv_data else "UNKNOWN")
        pair_data = kwargs.get("pair_prices")

        if self._regime_disabled:
            return self._hold(symbol, {"reason": "regime_disabled"})

        if not pair_data:
            return self._hold(symbol, {"reason": "no_pair_data"})

        prices_a = np.asarray(pair_data["prices_a"], dtype=float)
        prices_b = np.asarray(pair_data["prices_b"], dtype=float)
        sym_a = pair_data.get("symbol_a", "A")
        sym_b = pair_data.get("symbol_b", "B")
        pair_symbol = f"{sym_a}/{sym_b}"

        min_len = min(len(prices_a), len(prices_b))
        if min_len < self.required_history_length:
            return self._hold(symbol, {"reason": "insufficient_data", "n": min_len})

        prices_a = prices_a[-min_len:]
        prices_b = prices_b[-min_len:]

        # Cointegration test (cached, recalculated periodically)
        self._candles_since_recalc += 1
        if self._cached_coint is None or self._candles_since_recalc >= self._recalc_interval:
            self._cached_coint = engle_granger_cointegration(prices_a, prices_b)
            self._candles_since_recalc = 0

        coint_result = self._cached_coint
        if not coint_result["is_cointegrated"] and coint_result["pvalue"] > self._coint_pvalue:
            return self._hold(symbol, {
                "reason": "not_cointegrated",
                "pair": pair_symbol,
                "coint_pvalue": round(coint_result["pvalue"], 4),
            })

        # Rolling hedge ratio
        hedge_ratios = rolling_ols_hedge_ratio(
            prices_a, prices_b, window=self._hedge_ratio_window
        )
        current_hedge = hedge_ratios[-1]
        if np.isnan(current_hedge):
            return self._hold(symbol, {"reason": "nan_hedge_ratio"})

        # Compute spread and z-score
        spread = prices_a - current_hedge * prices_b
        half_life = calculate_half_life(spread)

        if not (self._min_half_life <= half_life <= self._max_half_life):
            return self._hold(symbol, {
                "reason": "half_life_out_of_range",
                "half_life": round(half_life, 2),
                "range": [self._min_half_life, self._max_half_life],
            })

        zscores = calculate_zscore(spread, window=self._zscore_window)
        current_z = zscores[-1]
        if np.isnan(current_z):
            return self._hold(symbol, {"reason": "nan_zscore"})

        # Signal generation
        metadata = {
            "pair": pair_symbol,
            "hedge_ratio": round(float(current_hedge), 6),
            "zscore": round(float(current_z), 4),
            "half_life": round(half_life, 2),
            "coint_pvalue": round(coint_result["pvalue"], 4),
            "spread": round(float(spread[-1]), 6),
        }

        # BUY: z < -entry (spread compressed, expect reversion up)
        if current_z < -self._zscore_entry:
            confidence = min(abs(current_z) / self._zscore_stop, 1.0)
            metadata["signal_reason"] = "zscore_below_neg_entry"
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=round(confidence, 4),
                metadata=metadata,
            )

        # SELL: z > +entry (spread extended, expect reversion down)
        if current_z > self._zscore_entry:
            confidence = min(abs(current_z) / self._zscore_stop, 1.0)
            metadata["signal_reason"] = "zscore_above_pos_entry"
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
