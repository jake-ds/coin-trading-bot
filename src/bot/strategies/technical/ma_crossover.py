"""Moving Average Crossover strategy with optional confirmation filters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import ta

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy, strategy_registry

if TYPE_CHECKING:
    from bot.strategies.regime import MarketRegime


class MACrossoverStrategy(BaseStrategy):
    """MA Crossover with volume, momentum, ADX, trend-strength, and cooldown filters.

    All filters are disabled by default to maintain backward compatibility.
    Enable them for production use.
    """

    def __init__(
        self,
        short_period: int = 20,
        long_period: int = 50,
        volume_confirmation: bool = False,
        volume_multiplier: float = 1.5,
        volume_period: int = 20,
        momentum_confirmation: bool = False,
        adx_filter_enabled: bool = False,
        adx_threshold: float = 20.0,
        trend_strength_filter: bool = False,
        cooldown_candles: int = 0,
    ):
        self._short_period = short_period
        self._long_period = long_period
        self._original_short_period = short_period
        self._original_long_period = long_period
        self._regime_disabled = False

        # Confirmation filters
        self._volume_confirmation = volume_confirmation
        self._volume_multiplier = volume_multiplier
        self._volume_period = volume_period
        self._momentum_confirmation = momentum_confirmation
        self._adx_filter_enabled = adx_filter_enabled
        self._adx_threshold = adx_threshold
        self._trend_strength_filter = trend_strength_filter
        self._cooldown_candles = cooldown_candles

        # Cooldown state
        self._candles_since_last_signal = cooldown_candles  # Start ready

    @property
    def name(self) -> str:
        return "ma_crossover"

    @property
    def required_history_length(self) -> int:
        return self._long_period + 1

    def adapt_to_regime(self, regime: MarketRegime) -> None:
        """Adapt MA periods based on market regime.

        - TRENDING_UP/TRENDING_DOWN: Use shorter periods (10/30) for faster signals.
        - RANGING: Disable (crossovers whipsaw in ranges).
        - HIGH_VOLATILITY: Restore defaults.
        """
        from bot.strategies.regime import MarketRegime

        if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            self._short_period = 10
            self._long_period = 30
            self._regime_disabled = False
        elif regime == MarketRegime.RANGING:
            self._regime_disabled = True
        else:
            # HIGH_VOLATILITY or unknown: restore original parameters
            self._short_period = self._original_short_period
            self._long_period = self._original_long_period
            self._regime_disabled = False

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        symbol = kwargs.get("symbol", ohlcv_data[-1].symbol if ohlcv_data else "UNKNOWN")

        if self._regime_disabled:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "disabled_by_regime", "regime": "RANGING"},
            )

        if len(ohlcv_data) < self.required_history_length:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "insufficient_data"},
            )

        closes = pd.Series([c.close for c in ohlcv_data])
        opens = pd.Series([c.open for c in ohlcv_data])
        highs = pd.Series([c.high for c in ohlcv_data])
        lows = pd.Series([c.low for c in ohlcv_data])
        volumes = pd.Series([c.volume for c in ohlcv_data])

        short_ma = ta.trend.sma_indicator(closes, window=self._short_period)
        long_ma = ta.trend.sma_indicator(closes, window=self._long_period)

        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]

        # MA distance (current and previous for trend strength)
        current_distance = abs(current_short - current_long)
        prev_distance = abs(prev_short - prev_long)
        distance_ratio = current_distance / current_long if current_long > 0 else 0

        metadata: dict[str, Any] = {
            "short_ma": float(current_short),
            "long_ma": float(current_long),
            "short_period": self._short_period,
            "long_period": self._long_period,
        }

        # Crossover detection
        if prev_short <= prev_long and current_short > current_long:
            action = SignalAction.BUY
        elif prev_short >= prev_long and current_short < current_long:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD

        # If no crossover, increment cooldown counter and return HOLD
        if action == SignalAction.HOLD:
            self._candles_since_last_signal += 1
            confidence = min(distance_ratio * 10, 1.0)
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=action,
                confidence=confidence,
                metadata=metadata,
            )

        # --- Apply confirmation filters for BUY/SELL crossovers ---

        # Volume confirmation
        volume_confirmed = True
        volume_ratio = 0.0
        if self._volume_confirmation:
            avg_volume = volumes.rolling(window=self._volume_period).mean().iloc[-1]
            current_volume = volumes.iloc[-1]
            volume_ratio = float(current_volume / avg_volume if avg_volume > 0 else 0.0)
            volume_confirmed = volume_ratio >= self._volume_multiplier
            metadata["volume_ratio"] = volume_ratio
            metadata["volume_confirmed"] = bool(volume_confirmed)

        # Momentum confirmation (bullish candle for BUY, bearish for SELL)
        momentum_confirmed = True
        if self._momentum_confirmation:
            current_close = float(closes.iloc[-1])
            current_open = float(opens.iloc[-1])
            if action == SignalAction.BUY:
                momentum_confirmed = current_close > current_open
            else:  # SELL
                momentum_confirmed = current_close < current_open
            metadata["momentum_confirmed"] = bool(momentum_confirmed)

        # Trend strength filter (MA distance expanding)
        trend_strength_confirmed = True
        if self._trend_strength_filter:
            trend_strength_confirmed = current_distance > prev_distance
            metadata["ma_distance_expanding"] = bool(trend_strength_confirmed)
            metadata["current_ma_distance"] = float(current_distance)
            metadata["prev_ma_distance"] = float(prev_distance)

        # ADX filter
        adx_confirmed = True
        adx_value = 0.0
        if self._adx_filter_enabled:
            try:
                adx_indicator = ta.trend.ADXIndicator(
                    high=highs, low=lows, close=closes, window=14
                )
                adx_series = adx_indicator.adx()
                adx_value = float(adx_series.iloc[-1])
                adx_confirmed = adx_value >= self._adx_threshold
                metadata["adx"] = adx_value
                metadata["adx_confirmed"] = bool(adx_confirmed)
            except (IndexError, ValueError):
                # Insufficient data for ADX — skip filter
                metadata["adx"] = 0.0
                metadata["adx_confirmed"] = True

        # Cooldown check
        cooldown_ready = True
        if self._cooldown_candles > 0:
            cooldown_ready = self._candles_since_last_signal >= self._cooldown_candles
            metadata["cooldown_ready"] = cooldown_ready
            metadata["candles_since_last_signal"] = self._candles_since_last_signal

        # All filters must pass
        all_confirmed = (
            volume_confirmed
            and momentum_confirmed
            and trend_strength_confirmed
            and adx_confirmed
            and cooldown_ready
        )

        if not all_confirmed:
            # Crossover detected but filters rejected it
            reasons = []
            if not volume_confirmed:
                reasons.append("volume")
            if not momentum_confirmed:
                reasons.append("momentum")
            if not trend_strength_confirmed:
                reasons.append("trend_strength")
            if not adx_confirmed:
                reasons.append("adx")
            if not cooldown_ready:
                reasons.append("cooldown")
            metadata["rejected_by"] = reasons
            metadata["reason"] = "filters_rejected"

            self._candles_since_last_signal += 1
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata=metadata,
            )

        # All filters passed — calculate enhanced confidence
        confidence = self._calculate_confidence(
            distance_ratio=distance_ratio,
            volume_ratio=volume_ratio,
            adx_value=adx_value,
        )

        # Reset cooldown counter on successful signal
        self._candles_since_last_signal = 0

        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=action,
            confidence=confidence,
            metadata=metadata,
        )

    def _calculate_confidence(
        self,
        distance_ratio: float,
        volume_ratio: float,
        adx_value: float,
    ) -> float:
        """Calculate confidence from MA distance, volume ratio, and ADX.

        When filters are disabled, falls back to the original distance-based confidence.
        """
        components: list[float] = []
        weights: list[float] = []

        # MA distance always contributes
        ma_conf = min(distance_ratio * 10, 1.0)
        components.append(ma_conf)
        weights.append(1.0)

        # Volume ratio contribution
        if self._volume_confirmation and volume_ratio > 0:
            vol_conf = min(volume_ratio / 3.0, 1.0)  # 3x avg = max confidence
            components.append(vol_conf)
            weights.append(1.0)

        # ADX contribution
        if self._adx_filter_enabled and adx_value > 0:
            adx_conf = min(adx_value / 50.0, 1.0)  # ADX 50 = max confidence
            components.append(adx_conf)
            weights.append(1.0)

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(c * w for c, w in zip(components, weights))
        return min(weighted_sum / total_weight, 1.0)


strategy_registry.register(MACrossoverStrategy())
