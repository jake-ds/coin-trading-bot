"""Cross-timeframe momentum factor strategy.

Combines momentum signals from multiple timeframes (1h, 4h, 1d)
into a single weighted z-score composite.

Strengthened in TRENDING markets, weakened in RANGING markets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.quant.statistics import calculate_zscore
from bot.strategies.base import BaseStrategy

logger = structlog.get_logger()


class MomentumFactorStrategy(BaseStrategy):
    """Cross-timeframe momentum factor using weighted z-scores."""

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 50,
        zscore_window: int = 20,
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.5,
        timeframe_weights: dict[str, float] | None = None,
    ):
        self._short_window = short_window
        self._long_window = long_window
        self._zscore_window = zscore_window
        self._entry_threshold = entry_threshold
        self._exit_threshold = exit_threshold
        self._timeframe_weights = timeframe_weights or {
            "1h": 0.2,
            "4h": 0.3,
            "1d": 0.5,
        }
        self._regime_multiplier = 1.0

    @property
    def name(self) -> str:
        return "momentum_factor"

    @property
    def required_history_length(self) -> int:
        return self._long_window + self._zscore_window + 5

    def adapt_to_regime(self, regime: Any) -> None:
        """Strengthen in TRENDING, weaken in RANGING."""
        regime_str = str(regime) if regime else ""
        if "TRENDING" in regime_str:
            self._regime_multiplier = 1.3
        elif "RANGING" in regime_str:
            self._regime_multiplier = 0.5
        else:
            self._regime_multiplier = 1.0

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        """Analyze momentum across timeframes.

        Expected kwargs:
            multi_timeframe_candles: dict[str, list[OHLCV]] mapping timeframe -> candles
            symbol: str
        """
        symbol = kwargs.get("symbol", ohlcv_data[-1].symbol if ohlcv_data else "UNKNOWN")
        multi_tf = kwargs.get("multi_timeframe_candles", {})

        # Fall back to the provided ohlcv_data if no multi-TF data
        if not multi_tf:
            multi_tf = {"1h": ohlcv_data}

        composite_z = 0.0
        total_weight = 0.0
        tf_details = {}

        for tf, weight in self._timeframe_weights.items():
            candles = multi_tf.get(tf, [])
            if len(candles) < self.required_history_length:
                continue

            closes = np.array([c.close for c in candles])
            momentum = self._compute_momentum_zscore(closes)
            if np.isnan(momentum):
                continue

            composite_z += momentum * weight
            total_weight += weight
            tf_details[tf] = {
                "zscore": round(float(momentum), 4),
                "weight": weight,
            }

        if total_weight < 0.3:
            return self._hold(symbol, {"reason": "insufficient_timeframes"})

        composite_z /= total_weight
        composite_z *= self._regime_multiplier

        metadata = {
            "composite_zscore": round(float(composite_z), 4),
            "regime_multiplier": self._regime_multiplier,
            "timeframe_details": tf_details,
        }

        # Signal generation
        if composite_z > self._entry_threshold:
            confidence = min(abs(composite_z) / 3.0, 1.0)
            metadata["signal_reason"] = "strong_bullish_momentum"
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=round(confidence, 4),
                metadata=metadata,
            )

        if composite_z < -self._entry_threshold:
            confidence = min(abs(composite_z) / 3.0, 1.0)
            metadata["signal_reason"] = "strong_bearish_momentum"
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.SELL,
                confidence=round(confidence, 4),
                metadata=metadata,
            )

        return self._hold(symbol, metadata)

    def _compute_momentum_zscore(self, closes: np.ndarray) -> float:
        """Compute momentum z-score from close prices.

        Momentum = short MA / long MA - 1 (rate of change).
        Z-score normalizes the momentum signal.
        """
        if len(closes) < self._long_window + self._zscore_window:
            return float("nan")

        # Calculate rolling momentum
        n = len(closes)
        momentum = np.full(n, np.nan)
        for i in range(self._long_window - 1, n):
            short_ma = np.mean(closes[max(0, i - self._short_window + 1) : i + 1])
            long_ma = np.mean(closes[i - self._long_window + 1 : i + 1])
            if long_ma > 0:
                momentum[i] = (short_ma / long_ma) - 1.0

        # Z-score of momentum
        valid = momentum[~np.isnan(momentum)]
        if len(valid) < self._zscore_window:
            return float("nan")

        zscores = calculate_zscore(valid, window=self._zscore_window)
        return float(zscores[-1])

    def _hold(self, symbol: str, metadata: dict) -> TradingSignal:
        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            metadata=metadata,
        )
