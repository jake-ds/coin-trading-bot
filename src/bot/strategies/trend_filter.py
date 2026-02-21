"""Multi-timeframe trend filter using SMA slope and ADX."""

from enum import Enum
from typing import Any

import pandas as pd
import structlog
import ta

from bot.models import OHLCV

logger = structlog.get_logger()


class TrendDirection(str, Enum):
    """Market trend direction."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class TrendFilter:
    """Determines trend direction from higher-timeframe candles.

    Uses 20-period SMA slope + ADX to classify trend:
    - ADX > 25 = trending market
    - ADX < 20 = ranging market
    - SMA slope > 0 + ADX > 25 = BULLISH
    - SMA slope < 0 + ADX > 25 = BEARISH
    - Otherwise = NEUTRAL
    """

    def __init__(
        self,
        sma_period: int = 20,
        adx_period: int = 14,
        adx_trending_threshold: float = 25.0,
        adx_ranging_threshold: float = 20.0,
    ):
        self._sma_period = sma_period
        self._adx_period = adx_period
        self._adx_trending_threshold = adx_trending_threshold
        self._adx_ranging_threshold = adx_ranging_threshold

    @property
    def required_history_length(self) -> int:
        """Minimum candles needed for SMA + ADX calculation."""
        return max(self._sma_period, self._adx_period) + 5

    def get_trend(self, symbol: str, candles: list[OHLCV]) -> TrendDirection:
        """Determine trend direction from higher-timeframe candle data.

        Args:
            symbol: Trading pair symbol (for logging).
            candles: Higher-timeframe OHLCV candles (oldest first).

        Returns:
            TrendDirection indicating market trend.
        """
        if len(candles) < self.required_history_length:
            logger.debug(
                "trend_filter_insufficient_data",
                symbol=symbol,
                candles=len(candles),
                required=self.required_history_length,
            )
            return TrendDirection.NEUTRAL

        closes = pd.Series([c.close for c in candles])
        highs = pd.Series([c.high for c in candles])
        lows = pd.Series([c.low for c in candles])

        # Calculate 20-period SMA
        sma = closes.rolling(window=self._sma_period).mean()
        # SMA slope: difference between current and previous SMA values
        sma_slope = sma.iloc[-1] - sma.iloc[-2]

        # Calculate ADX
        adx_indicator = ta.trend.ADXIndicator(
            high=highs, low=lows, close=closes, window=self._adx_period
        )
        adx = adx_indicator.adx().iloc[-1]

        # Determine trend
        if pd.isna(adx) or pd.isna(sma_slope):
            direction = TrendDirection.NEUTRAL
        elif adx > self._adx_trending_threshold:
            if sma_slope > 0:
                direction = TrendDirection.BULLISH
            else:
                direction = TrendDirection.BEARISH
        else:
            direction = TrendDirection.NEUTRAL

        logger.debug(
            "trend_filter_result",
            symbol=symbol,
            direction=direction.value,
            adx=round(float(adx) if not pd.isna(adx) else 0.0, 2),
            sma_slope=round(float(sma_slope) if not pd.isna(sma_slope) else 0.0, 4),
        )

        return direction

    def get_trend_details(
        self, symbol: str, candles: list[OHLCV]
    ) -> dict[str, Any]:
        """Get trend direction with detailed indicator values.

        Args:
            symbol: Trading pair symbol.
            candles: Higher-timeframe OHLCV candles.

        Returns:
            Dict with direction, adx, sma_slope values.
        """
        if len(candles) < self.required_history_length:
            return {
                "direction": TrendDirection.NEUTRAL,
                "adx": 0.0,
                "sma_slope": 0.0,
                "sufficient_data": False,
            }

        closes = pd.Series([c.close for c in candles])
        highs = pd.Series([c.high for c in candles])
        lows = pd.Series([c.low for c in candles])

        sma = closes.rolling(window=self._sma_period).mean()
        sma_slope = float(sma.iloc[-1] - sma.iloc[-2])

        adx_indicator = ta.trend.ADXIndicator(
            high=highs, low=lows, close=closes, window=self._adx_period
        )
        adx_value = float(adx_indicator.adx().iloc[-1])

        direction = self.get_trend(symbol, candles)

        return {
            "direction": direction,
            "adx": round(adx_value, 2),
            "sma_slope": round(sma_slope, 4),
            "sufficient_data": True,
        }
