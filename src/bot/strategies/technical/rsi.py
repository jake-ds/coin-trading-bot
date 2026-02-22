"""RSI (Relative Strength Index) strategy with divergence detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import ta

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy, strategy_registry

if TYPE_CHECKING:
    from bot.strategies.regime import MarketRegime


def _find_troughs(series: pd.Series, order: int = 3) -> list[tuple[int, float]]:
    """Find local troughs (minima) in a series.

    A trough at index i means series[i] <= all neighbors within `order` distance.

    Args:
        series: The data series to scan.
        order: Number of neighbors on each side to compare.

    Returns:
        List of (index, value) tuples for each trough found.
    """
    troughs = []
    values = series.values
    for i in range(order, len(values) - order):
        if np.isnan(values[i]):
            continue
        left = values[max(0, i - order) : i]
        right = values[i + 1 : i + order + 1]
        # Skip if any neighbor is NaN
        if np.any(np.isnan(left)) or np.any(np.isnan(right)):
            continue
        if values[i] <= np.min(left) and values[i] <= np.min(right):
            troughs.append((i, float(values[i])))
    return troughs


def _find_peaks(series: pd.Series, order: int = 3) -> list[tuple[int, float]]:
    """Find local peaks (maxima) in a series.

    A peak at index i means series[i] >= all neighbors within `order` distance.

    Args:
        series: The data series to scan.
        order: Number of neighbors on each side to compare.

    Returns:
        List of (index, value) tuples for each peak found.
    """
    peaks = []
    values = series.values
    for i in range(order, len(values) - order):
        if np.isnan(values[i]):
            continue
        left = values[max(0, i - order) : i]
        right = values[i + 1 : i + order + 1]
        if np.any(np.isnan(left)) or np.any(np.isnan(right)):
            continue
        if values[i] >= np.max(left) and values[i] >= np.max(right):
            peaks.append((i, float(values[i])))
    return peaks


class RSIStrategy(BaseStrategy):
    """RSI Strategy with divergence detection.

    Signals:
    - Bullish divergence: price lower low + RSI higher low → BUY (high confidence)
    - Bearish divergence: price higher high + RSI lower high → SELL (high confidence)
    - Oversold (RSI < oversold threshold) → BUY (lower confidence)
    - Overbought (RSI > overbought threshold) → SELL (lower confidence)
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        divergence_enabled: bool = False,
        divergence_lookback: int = 14,
        divergence_swing_order: int = 3,
    ):
        self._period = period
        self._overbought = overbought
        self._oversold = oversold
        self._original_overbought = overbought
        self._original_oversold = oversold
        self._divergence_enabled = divergence_enabled
        self._divergence_lookback = divergence_lookback
        self._divergence_swing_order = divergence_swing_order

    def adapt_to_regime(self, regime: MarketRegime) -> None:
        """Adapt RSI thresholds based on market regime.

        - RANGING: Use tighter bounds (35/65) for more sensitive signals.
        - TRENDING_UP/TRENDING_DOWN: Use standard bounds (30/70).
        - HIGH_VOLATILITY: Restore original bounds.
        """
        from bot.strategies.regime import MarketRegime

        if regime == MarketRegime.RANGING:
            self._oversold = 35.0
            self._overbought = 65.0
        elif regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            self._oversold = 30.0
            self._overbought = 70.0
        else:
            # HIGH_VOLATILITY or unknown: restore originals
            self._overbought = self._original_overbought
            self._oversold = self._original_oversold

    @property
    def name(self) -> str:
        return "rsi"

    @property
    def required_history_length(self) -> int:
        return self._period + 2

    def _detect_bullish_divergence(
        self, closes: pd.Series, rsi: pd.Series
    ) -> dict[str, Any] | None:
        """Detect bullish divergence: price lower low + RSI higher low.

        Looks for two troughs in the lookback window where:
        - Price at second trough < price at first trough (lower low)
        - RSI at second trough > RSI at first trough (higher low)
        """
        lookback_start = max(0, len(closes) - self._divergence_lookback)
        price_window = closes.iloc[lookback_start:]
        rsi_window = rsi.iloc[lookback_start:]

        price_troughs = _find_troughs(price_window, order=self._divergence_swing_order)
        rsi_troughs = _find_troughs(rsi_window, order=self._divergence_swing_order)

        if len(price_troughs) < 2 or len(rsi_troughs) < 2:
            return None

        # Use the last two troughs
        prev_price_trough = price_troughs[-2]
        curr_price_trough = price_troughs[-1]
        prev_rsi_trough = rsi_troughs[-2]
        curr_rsi_trough = rsi_troughs[-1]

        # Bullish divergence: price lower low, RSI higher low
        if (
            curr_price_trough[1] < prev_price_trough[1]
            and curr_rsi_trough[1] > prev_rsi_trough[1]
        ):
            return {
                "price_trough_1": prev_price_trough[1],
                "price_trough_2": curr_price_trough[1],
                "rsi_trough_1": prev_rsi_trough[1],
                "rsi_trough_2": curr_rsi_trough[1],
            }

        return None

    def _detect_bearish_divergence(
        self, closes: pd.Series, rsi: pd.Series
    ) -> dict[str, Any] | None:
        """Detect bearish divergence: price higher high + RSI lower high.

        Looks for two peaks in the lookback window where:
        - Price at second peak > price at first peak (higher high)
        - RSI at second peak < RSI at first peak (lower high)
        """
        lookback_start = max(0, len(closes) - self._divergence_lookback)
        price_window = closes.iloc[lookback_start:]
        rsi_window = rsi.iloc[lookback_start:]

        price_peaks = _find_peaks(price_window, order=self._divergence_swing_order)
        rsi_peaks = _find_peaks(rsi_window, order=self._divergence_swing_order)

        if len(price_peaks) < 2 or len(rsi_peaks) < 2:
            return None

        # Use the last two peaks
        prev_price_peak = price_peaks[-2]
        curr_price_peak = price_peaks[-1]
        prev_rsi_peak = rsi_peaks[-2]
        curr_rsi_peak = rsi_peaks[-1]

        # Bearish divergence: price higher high, RSI lower high
        if (
            curr_price_peak[1] > prev_price_peak[1]
            and curr_rsi_peak[1] < prev_rsi_peak[1]
        ):
            return {
                "price_peak_1": prev_price_peak[1],
                "price_peak_2": curr_price_peak[1],
                "rsi_peak_1": prev_rsi_peak[1],
                "rsi_peak_2": curr_rsi_peak[1],
            }

        return None

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        symbol = kwargs.get("symbol", ohlcv_data[-1].symbol if ohlcv_data else "UNKNOWN")

        if len(ohlcv_data) < self.required_history_length:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "insufficient_data"},
            )

        closes = pd.Series([c.close for c in ohlcv_data])
        rsi = ta.momentum.rsi(closes, window=self._period)

        current_rsi = rsi.iloc[-1]

        metadata: dict[str, Any] = {
            "rsi": float(current_rsi),
            "overbought": self._overbought,
            "oversold": self._oversold,
            "period": self._period,
        }

        # Check for divergence first (higher quality signals)
        if self._divergence_enabled:
            bullish_div = self._detect_bullish_divergence(closes, rsi)
            if bullish_div is not None:
                metadata["divergence_type"] = "bullish_divergence"
                metadata.update(bullish_div)
                return TradingSignal(
                    strategy_name=self.name,
                    symbol=symbol,
                    action=SignalAction.BUY,
                    confidence=0.8,
                    metadata=metadata,
                )

            bearish_div = self._detect_bearish_divergence(closes, rsi)
            if bearish_div is not None:
                metadata["divergence_type"] = "bearish_divergence"
                metadata.update(bearish_div)
                return TradingSignal(
                    strategy_name=self.name,
                    symbol=symbol,
                    action=SignalAction.SELL,
                    confidence=0.8,
                    metadata=metadata,
                )

        # Fallback: standard oversold/overbought signals
        if current_rsi <= self._oversold:
            confidence = min((self._oversold - current_rsi) / self._oversold, 1.0)
            action = SignalAction.BUY
            if self._divergence_enabled:
                metadata["divergence_type"] = "oversold"
        elif current_rsi >= self._overbought:
            confidence = min((current_rsi - self._overbought) / (100 - self._overbought), 1.0)
            action = SignalAction.SELL
            if self._divergence_enabled:
                metadata["divergence_type"] = "overbought"
        else:
            confidence = 0.0
            action = SignalAction.HOLD

        # When divergence is enabled, cap fallback confidence at 0.5
        if self._divergence_enabled and action != SignalAction.HOLD:
            confidence = min(confidence, 0.5)

        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=action,
            confidence=max(confidence, 0.1) if action != SignalAction.HOLD else 0.0,
            metadata=metadata,
        )


strategy_registry.register(RSIStrategy())
