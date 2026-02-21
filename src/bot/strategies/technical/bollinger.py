"""Bollinger Bands strategy with squeeze breakout and mean reversion modes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import ta

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy, strategy_registry

if TYPE_CHECKING:
    from bot.strategies.regime import MarketRegime


class BollingerStrategy(BaseStrategy):
    """Bollinger Bands strategy with two modes.

    - mean_reversion (default): BUY at lower band, SELL at upper band.
    - squeeze: Detect BB squeeze (contracting bands), then trade breakouts
      with volume confirmation and cooldown.
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        mode: str = "mean_reversion",
        squeeze_candles_required: int = 5,
        breakout_volume_multiplier: float = 1.5,
        cooldown_candles: int = 10,
    ):
        self._period = period
        self._std_dev = std_dev
        self._mode = mode
        self._squeeze_candles_required = squeeze_candles_required
        self._breakout_volume_multiplier = breakout_volume_multiplier
        self._cooldown_candles = cooldown_candles
        self._candles_since_signal = cooldown_candles  # Start ready
        self._regime_disabled = False

    @property
    def name(self) -> str:
        return "bollinger"

    @property
    def required_history_length(self) -> int:
        if self._mode == "squeeze":
            # Need enough data for BB calculation + rolling average of BB width
            # + squeeze detection window
            return self._period * 2 + self._squeeze_candles_required
        return self._period + 1

    def adapt_to_regime(self, regime: MarketRegime) -> None:
        """Adapt Bollinger strategy based on market regime.

        - RANGING: Enable (mean reversion works well in ranges).
        - TRENDING_UP/TRENDING_DOWN: Disable (band touches are breakouts, not reversals).
        - HIGH_VOLATILITY: Enable (bands widen naturally).
        """
        from bot.strategies.regime import MarketRegime

        if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            self._regime_disabled = True
        else:
            # RANGING and HIGH_VOLATILITY: enable
            self._regime_disabled = False

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        symbol = kwargs.get("symbol", ohlcv_data[-1].symbol if ohlcv_data else "UNKNOWN")

        if self._regime_disabled:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "disabled_by_regime", "regime": "TRENDING"},
            )

        if len(ohlcv_data) < self.required_history_length:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "insufficient_data"},
            )

        if self._mode == "squeeze":
            return await self._analyze_squeeze(ohlcv_data, symbol)
        return await self._analyze_mean_reversion(ohlcv_data, symbol)

    async def _analyze_mean_reversion(
        self, ohlcv_data: list[OHLCV], symbol: str
    ) -> TradingSignal:
        """Original band-touch mean reversion logic."""
        closes = pd.Series([c.close for c in ohlcv_data])

        bb = ta.volatility.BollingerBands(
            closes, window=self._period, window_dev=self._std_dev
        )

        upper = bb.bollinger_hband().iloc[-1]
        lower = bb.bollinger_lband().iloc[-1]
        middle = bb.bollinger_mavg().iloc[-1]
        current_price = closes.iloc[-1]

        band_width = upper - lower if upper > lower else 1.0

        metadata = {
            "upper_band": float(upper),
            "lower_band": float(lower),
            "middle_band": float(middle),
            "price": float(current_price),
            "band_width": float(band_width),
            "mode": "mean_reversion",
        }

        if current_price <= lower:
            action = SignalAction.BUY
            confidence = min((lower - current_price) / band_width + 0.5, 1.0)
        elif current_price >= upper:
            action = SignalAction.SELL
            confidence = min((current_price - upper) / band_width + 0.5, 1.0)
        else:
            action = SignalAction.HOLD
            confidence = 0.0

        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=action,
            confidence=confidence,
            metadata=metadata,
        )

    async def _analyze_squeeze(
        self, ohlcv_data: list[OHLCV], symbol: str
    ) -> TradingSignal:
        """Squeeze breakout detection logic."""
        closes = pd.Series([c.close for c in ohlcv_data])
        volumes = pd.Series([c.volume for c in ohlcv_data])

        bb = ta.volatility.BollingerBands(
            closes, window=self._period, window_dev=self._std_dev
        )

        upper_series = bb.bollinger_hband()
        lower_series = bb.bollinger_lband()
        middle_series = bb.bollinger_mavg()

        # BB width series
        bb_width_series = upper_series - lower_series

        # Rolling average of BB width
        bb_width_avg = bb_width_series.rolling(window=self._period).mean()

        upper = float(upper_series.iloc[-1])
        lower = float(lower_series.iloc[-1])
        middle = float(middle_series.iloc[-1])
        current_price = float(closes.iloc[-1])
        current_width = float(bb_width_series.iloc[-1])
        avg_width = float(bb_width_avg.iloc[-1])

        # Count consecutive squeeze candles BEFORE the current candle
        squeeze_count = 0
        for i in range(len(bb_width_series) - 2, -1, -1):
            w = bb_width_series.iloc[i]
            a = bb_width_avg.iloc[i]
            if pd.isna(w) or pd.isna(a):
                break
            if w < a:
                squeeze_count += 1
            else:
                break

        in_squeeze = squeeze_count >= self._squeeze_candles_required

        # Volume confirmation
        avg_volume = float(np.mean(volumes.values[-self._period:]))
        current_volume = float(volumes.iloc[-1])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0.0
        volume_confirmed = volume_ratio >= self._breakout_volume_multiplier

        metadata: dict[str, Any] = {
            "upper_band": upper,
            "lower_band": lower,
            "middle_band": middle,
            "price": current_price,
            "band_width": current_width,
            "mode": "squeeze",
            "squeeze_count": squeeze_count,
            "in_squeeze": in_squeeze,
            "avg_band_width": avg_width,
            "volume_ratio": round(volume_ratio, 2),
            "volume_confirmed": volume_confirmed,
        }

        # Increment cooldown counter
        self._candles_since_signal += 1

        # Check cooldown
        if self._candles_since_signal <= self._cooldown_candles:
            metadata["reason"] = "cooldown"
            metadata["candles_since_signal"] = self._candles_since_signal
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata=metadata,
            )

        # Need a valid squeeze before we look for breakouts
        if not in_squeeze:
            metadata["reason"] = "no_squeeze"
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata=metadata,
            )

        # Breakout detection: price closes outside bands after squeeze
        action = SignalAction.HOLD
        confidence = 0.0

        if current_price > upper:
            # Breakout above upper band
            if volume_confirmed:
                action = SignalAction.BUY
                # Confidence based on breakout strength and volume
                if current_width > 0:
                    breakout_strength = (current_price - upper) / current_width
                else:
                    breakout_strength = 0.0
                confidence = min(
                    0.6 + breakout_strength * 0.2 + (volume_ratio - 1.0) * 0.1,
                    1.0,
                )
                metadata["breakout_direction"] = "up"
            else:
                metadata["reason"] = "breakout_no_volume"
                metadata["breakout_direction"] = "up"
        elif current_price < lower:
            # Breakout below lower band
            if volume_confirmed:
                action = SignalAction.SELL
                if current_width > 0:
                    breakout_strength = (lower - current_price) / current_width
                else:
                    breakout_strength = 0.0
                confidence = min(
                    0.6 + breakout_strength * 0.2 + (volume_ratio - 1.0) * 0.1,
                    1.0,
                )
                metadata["breakout_direction"] = "down"
            else:
                metadata["reason"] = "breakout_no_volume"
                metadata["breakout_direction"] = "down"
        else:
            metadata["reason"] = "squeeze_no_breakout"

        if action != SignalAction.HOLD:
            # Reset cooldown on signal
            self._candles_since_signal = 0

        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=action,
            confidence=confidence,
            metadata=metadata,
        )


strategy_registry.register(BollingerStrategy())
