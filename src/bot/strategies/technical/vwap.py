"""VWAP (Volume-Weighted Average Price) strategy with band analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy, strategy_registry

if TYPE_CHECKING:
    from bot.strategies.regime import MarketRegime


class VWAPStrategy(BaseStrategy):
    """Volume-Weighted Average Price strategy.

    Calculates VWAP and standard deviation bands. Generates signals when
    price crosses VWAP with volume confirmation (institutional accumulation/
    distribution).

    - BUY: price crosses above VWAP from below + volume increasing
    - SELL: price crosses below VWAP from above + volume increasing
    - Confidence scales with distance from VWAP and volume strength
    """

    def __init__(
        self,
        volume_period: int = 20,
        volume_multiplier: float = 1.2,
        band_std_count: int = 2,
    ):
        self._volume_period = volume_period
        self._volume_multiplier = volume_multiplier
        self._band_std_count = band_std_count
        self._regime_disabled = False

    @property
    def name(self) -> str:
        return "vwap"

    @property
    def required_history_length(self) -> int:
        return max(self._volume_period + 1, 21)

    def adapt_to_regime(self, regime: MarketRegime) -> None:
        """Adapt VWAP strategy based on market regime.

        VWAP works best in ranging/trending markets. Disable in high volatility
        where price whips through VWAP constantly.
        """
        from bot.strategies.regime import MarketRegime

        if regime == MarketRegime.HIGH_VOLATILITY:
            self._regime_disabled = True
        else:
            self._regime_disabled = False

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        symbol = kwargs.get("symbol", ohlcv_data[-1].symbol if ohlcv_data else "UNKNOWN")

        if self._regime_disabled:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "disabled_by_regime", "regime": "HIGH_VOLATILITY"},
            )

        if len(ohlcv_data) < self.required_history_length:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "insufficient_data"},
            )

        # Extract price/volume data
        closes = np.array([c.close for c in ohlcv_data], dtype=float)
        highs = np.array([c.high for c in ohlcv_data], dtype=float)
        lows = np.array([c.low for c in ohlcv_data], dtype=float)
        volumes = np.array([c.volume for c in ohlcv_data], dtype=float)

        # Calculate VWAP: cumulative(typical_price * volume) / cumulative(volume)
        typical_prices = (highs + lows + closes) / 3.0
        cum_tp_vol = np.cumsum(typical_prices * volumes)
        cum_vol = np.cumsum(volumes)

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, typical_prices)

        # Calculate VWAP standard deviation bands
        # Squared deviation of typical price from VWAP, weighted by volume
        sq_diff = (typical_prices - vwap) ** 2
        cum_sq_diff_vol = np.cumsum(sq_diff * volumes)
        with np.errstate(divide="ignore", invalid="ignore"):
            variance = np.where(cum_vol > 0, cum_sq_diff_vol / cum_vol, 0.0)
        std_dev = np.sqrt(variance)

        current_vwap = float(vwap[-1])
        current_std = float(std_dev[-1])
        current_close = float(closes[-1])
        prev_close = float(closes[-2])

        # VWAP bands
        upper_band_1 = current_vwap + current_std
        lower_band_1 = current_vwap - current_std
        upper_band_2 = current_vwap + 2 * current_std
        lower_band_2 = current_vwap - 2 * current_std

        # VWAP relative position for previous and current close
        prev_vwap = float(vwap[-2])

        # Volume analysis: compare current volume to rolling average
        vol_series = pd.Series(volumes)
        avg_volume = float(vol_series.rolling(window=self._volume_period).mean().iloc[-1])
        current_volume = float(volumes[-1])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0.0
        volume_increasing = volume_ratio >= self._volume_multiplier

        # Distance from VWAP as percentage
        if current_vwap > 0:
            vwap_distance_pct = (current_close - current_vwap) / current_vwap * 100
        else:
            vwap_distance_pct = 0.0

        metadata: dict[str, Any] = {
            "vwap": current_vwap,
            "close": current_close,
            "vwap_distance_pct": round(vwap_distance_pct, 4),
            "upper_band_1": round(upper_band_1, 4),
            "lower_band_1": round(lower_band_1, 4),
            "upper_band_2": round(upper_band_2, 4),
            "lower_band_2": round(lower_band_2, 4),
            "volume_ratio": round(volume_ratio, 4),
            "volume_increasing": bool(volume_increasing),
            "std_dev": round(current_std, 4),
        }

        # Crossover detection: price crosses VWAP
        crossed_above = prev_close <= prev_vwap and current_close > current_vwap
        crossed_below = prev_close >= prev_vwap and current_close < current_vwap

        if crossed_above and volume_increasing:
            # BUY: institutional accumulation
            confidence = self._calculate_confidence(
                vwap_distance_pct=abs(vwap_distance_pct),
                volume_ratio=volume_ratio,
                current_std=current_std,
                current_vwap=current_vwap,
            )
            metadata["crossover"] = "above"
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=confidence,
                metadata=metadata,
            )

        if crossed_below and volume_increasing:
            # SELL: institutional distribution
            confidence = self._calculate_confidence(
                vwap_distance_pct=abs(vwap_distance_pct),
                volume_ratio=volume_ratio,
                current_std=current_std,
                current_vwap=current_vwap,
            )
            metadata["crossover"] = "below"
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.SELL,
                confidence=confidence,
                metadata=metadata,
            )

        # No crossover or volume not confirming
        if crossed_above or crossed_below:
            metadata["reason"] = "volume_not_confirmed"
            metadata["crossover"] = "above" if crossed_above else "below"
        else:
            metadata["reason"] = "no_crossover"

        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            metadata=metadata,
        )

    def _calculate_confidence(
        self,
        vwap_distance_pct: float,
        volume_ratio: float,
        current_std: float,
        current_vwap: float,
    ) -> float:
        """Calculate confidence based on VWAP distance and volume strength.

        Components:
        - Volume strength: higher volume = stronger institutional activity
        - VWAP proximity: signals near VWAP are more reliable (not overextended)
        """
        # Volume component: caps at 1.0 for 3x average volume
        vol_conf = min(volume_ratio / 3.0, 1.0)

        # VWAP proximity: closer to VWAP = higher confidence (more reliable crossover)
        # Beyond 1 std dev = weaker signal (overextended)
        if current_std > 0 and current_vwap > 0:
            std_pct = current_std / current_vwap * 100
            if std_pct > 0:
                proximity_conf = max(1.0 - (vwap_distance_pct / std_pct), 0.2)
            else:
                proximity_conf = 0.5
        else:
            proximity_conf = 0.5

        # Weighted average
        confidence = 0.6 * vol_conf + 0.4 * proximity_conf
        return min(max(confidence, 0.1), 1.0)


strategy_registry.register(VWAPStrategy())
