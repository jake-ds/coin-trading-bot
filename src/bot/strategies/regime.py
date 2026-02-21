"""Market regime detector with strategy adaptation."""

from enum import Enum
from typing import Any

import pandas as pd
import structlog
import ta

from bot.models import OHLCV

logger = structlog.get_logger()


class MarketRegime(str, Enum):
    """Market regime classification."""

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


class MarketRegimeDetector:
    """Detects the current market regime from OHLCV data.

    Uses three indicators:
    - ADX + DI for trend strength and direction
    - ATR ratio for volatility detection
    - Bollinger Band width for ranging detection

    Classification logic:
    - ADX > 25 + positive DI > negative DI = TRENDING_UP
    - ADX > 25 + negative DI > positive DI = TRENDING_DOWN
    - ATR > 2x average ATR = HIGH_VOLATILITY
    - ADX < 20 + low BB width = RANGING
    """

    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        bb_period: int = 20,
        adx_trending_threshold: float = 25.0,
        adx_ranging_threshold: float = 20.0,
        atr_volatility_multiplier: float = 2.0,
    ):
        self._adx_period = adx_period
        self._atr_period = atr_period
        self._bb_period = bb_period
        self._adx_trending_threshold = adx_trending_threshold
        self._adx_ranging_threshold = adx_ranging_threshold
        self._atr_volatility_multiplier = atr_volatility_multiplier

    @property
    def required_history_length(self) -> int:
        """Minimum candles needed for regime detection."""
        return max(self._adx_period, self._atr_period, self._bb_period) + 5

    def detect(self, candles: list[OHLCV]) -> MarketRegime:
        """Detect the current market regime.

        Args:
            candles: OHLCV candles (oldest first).

        Returns:
            MarketRegime classification.
        """
        details = self.detect_with_details(candles)
        return details["regime"]

    def detect_with_details(self, candles: list[OHLCV]) -> dict[str, Any]:
        """Detect market regime with detailed indicator values.

        Args:
            candles: OHLCV candles (oldest first).

        Returns:
            Dict with regime, adx, plus_di, minus_di, atr_ratio, bb_width.
        """
        if len(candles) < self.required_history_length:
            logger.debug(
                "regime_detector_insufficient_data",
                candles=len(candles),
                required=self.required_history_length,
            )
            return {
                "regime": MarketRegime.RANGING,
                "adx": 0.0,
                "plus_di": 0.0,
                "minus_di": 0.0,
                "atr_ratio": 0.0,
                "bb_width": 0.0,
                "sufficient_data": False,
            }

        closes = pd.Series([c.close for c in candles])
        highs = pd.Series([c.high for c in candles])
        lows = pd.Series([c.low for c in candles])

        # ADX and directional indicators
        adx_indicator = ta.trend.ADXIndicator(
            high=highs, low=lows, close=closes, window=self._adx_period
        )
        adx = float(adx_indicator.adx().iloc[-1])
        plus_di = float(adx_indicator.adx_pos().iloc[-1])
        minus_di = float(adx_indicator.adx_neg().iloc[-1])

        # ATR for volatility
        atr_indicator = ta.volatility.AverageTrueRange(
            high=highs, low=lows, close=closes, window=self._atr_period
        )
        atr_series = atr_indicator.average_true_range()
        current_atr = float(atr_series.iloc[-1])
        avg_atr = float(atr_series.dropna().mean())
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

        # Bollinger Band width for ranging detection
        bb = ta.volatility.BollingerBands(
            close=closes, window=self._bb_period, window_dev=2.0
        )
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        bb_middle = bb.bollinger_mavg().iloc[-1]
        bb_width = (
            (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0.0
        )

        # Classification logic (order matters: volatility > trend > ranging)
        if atr_ratio > self._atr_volatility_multiplier:
            regime = MarketRegime.HIGH_VOLATILITY
        elif adx > self._adx_trending_threshold:
            if plus_di > minus_di:
                regime = MarketRegime.TRENDING_UP
            else:
                regime = MarketRegime.TRENDING_DOWN
        elif adx < self._adx_ranging_threshold:
            regime = MarketRegime.RANGING
        else:
            # ADX between ranging and trending thresholds — default to RANGING
            regime = MarketRegime.RANGING

        logger.debug(
            "regime_detected",
            regime=regime.value,
            adx=round(adx, 2),
            plus_di=round(plus_di, 2),
            minus_di=round(minus_di, 2),
            atr_ratio=round(atr_ratio, 2),
            bb_width=round(bb_width, 4),
        )

        return {
            "regime": regime,
            "adx": round(adx, 2),
            "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2),
            "atr_ratio": round(atr_ratio, 2),
            "bb_width": round(bb_width, 4),
            "sufficient_data": True,
        }
