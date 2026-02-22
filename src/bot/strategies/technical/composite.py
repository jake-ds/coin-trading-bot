"""Composite momentum strategy using RSI + MACD + Stochastic confirmation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import ta

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy, strategy_registry

if TYPE_CHECKING:
    from bot.strategies.regime import MarketRegime


class CompositeMomentumStrategy(BaseStrategy):
    """Triple-confirmation momentum strategy.

    Uses three indicators for high-quality signals:
    - RSI(14): momentum direction
    - MACD(12,26,9): trend momentum
    - Stochastic(14,3): overbought/oversold with crossover

    BUY requires ALL THREE:
    - RSI < rsi_buy_threshold (default 40)
    - MACD histogram turning positive (current > 0 and current > previous)
    - Stochastic %K crosses above %D from below stoch_buy_threshold (default 30)

    SELL requires ALL THREE:
    - RSI > rsi_sell_threshold (default 60)
    - MACD histogram turning negative (current < 0 and current < previous)
    - Stochastic %K crosses below %D from above stoch_sell_threshold (default 70)

    If only 2 out of 3 agree: signal with lower confidence (0.4).
    If all 3 agree: high confidence (0.8).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_buy_threshold: float = 40.0,
        rsi_sell_threshold: float = 60.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        stoch_period: int = 14,
        stoch_smooth: int = 3,
        stoch_buy_threshold: float = 30.0,
        stoch_sell_threshold: float = 70.0,
    ):
        self._rsi_period = rsi_period
        self._rsi_buy_threshold = rsi_buy_threshold
        self._rsi_sell_threshold = rsi_sell_threshold
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal = macd_signal
        self._stoch_period = stoch_period
        self._stoch_smooth = stoch_smooth
        self._stoch_buy_threshold = stoch_buy_threshold
        self._stoch_sell_threshold = stoch_sell_threshold
        self._regime_disabled = False

    @property
    def name(self) -> str:
        return "composite_momentum"

    @property
    def required_history_length(self) -> int:
        # Need enough data for all indicators to produce valid values
        # MACD needs slow_period + signal_period, Stochastic needs stoch_period + smooth
        # RSI needs rsi_period. Take the max and add a buffer.
        return max(
            self._macd_slow + self._macd_signal + 1,
            self._stoch_period + self._stoch_smooth + 1,
            self._rsi_period + 2,
        )

    def adapt_to_regime(self, regime: MarketRegime) -> None:
        """Adapt composite strategy based on market regime.

        Disable in RANGING markets where momentum indicators whipsaw.
        """
        from bot.strategies.regime import MarketRegime

        if regime == MarketRegime.RANGING:
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
        highs = pd.Series([c.high for c in ohlcv_data])
        lows = pd.Series([c.low for c in ohlcv_data])

        # Calculate RSI
        rsi_series = ta.momentum.rsi(closes, window=self._rsi_period)
        current_rsi = float(rsi_series.iloc[-1])

        # Calculate MACD
        macd_indicator = ta.trend.MACD(
            closes,
            window_fast=self._macd_fast,
            window_slow=self._macd_slow,
            window_sign=self._macd_signal,
        )
        histogram = macd_indicator.macd_diff()
        current_hist = float(histogram.iloc[-1])
        prev_hist = float(histogram.iloc[-2])
        current_macd = float(macd_indicator.macd().iloc[-1])
        current_macd_signal = float(macd_indicator.macd_signal().iloc[-1])

        # Calculate Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=highs,
            low=lows,
            close=closes,
            window=self._stoch_period,
            smooth_window=self._stoch_smooth,
        )
        stoch_k = stoch.stoch()
        stoch_d = stoch.stoch_signal()
        current_k = float(stoch_k.iloc[-1])
        current_d = float(stoch_d.iloc[-1])
        prev_k = float(stoch_k.iloc[-2])
        prev_d = float(stoch_d.iloc[-2])

        # Determine each indicator's confirmation
        rsi_buy = current_rsi < self._rsi_buy_threshold
        rsi_sell = current_rsi > self._rsi_sell_threshold

        macd_buy = current_hist > 0 and current_hist > prev_hist
        macd_sell = current_hist < 0 and current_hist < prev_hist

        stoch_buy = (
            prev_k <= prev_d
            and current_k > current_d
            and current_k < self._stoch_buy_threshold
        )
        stoch_sell = (
            prev_k >= prev_d
            and current_k < current_d
            and current_k > self._stoch_sell_threshold
        )

        metadata: dict[str, Any] = {
            "rsi": round(current_rsi, 4),
            "rsi_buy_confirmed": bool(rsi_buy),
            "rsi_sell_confirmed": bool(rsi_sell),
            "macd": round(current_macd, 4),
            "macd_signal": round(current_macd_signal, 4),
            "macd_histogram": round(current_hist, 4),
            "macd_prev_histogram": round(prev_hist, 4),
            "macd_buy_confirmed": bool(macd_buy),
            "macd_sell_confirmed": bool(macd_sell),
            "stoch_k": round(current_k, 4),
            "stoch_d": round(current_d, 4),
            "stoch_prev_k": round(prev_k, 4),
            "stoch_prev_d": round(prev_d, 4),
            "stoch_buy_confirmed": bool(stoch_buy),
            "stoch_sell_confirmed": bool(stoch_sell),
        }

        # Count confirmations
        buy_count = sum([rsi_buy, macd_buy, stoch_buy])
        sell_count = sum([rsi_sell, macd_sell, stoch_sell])

        metadata["buy_confirmations"] = buy_count
        metadata["sell_confirmations"] = sell_count

        if buy_count == 3:
            metadata["signal_type"] = "triple_buy"
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=0.8,
                metadata=metadata,
            )

        if sell_count == 3:
            metadata["signal_type"] = "triple_sell"
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.SELL,
                confidence=0.8,
                metadata=metadata,
            )

        if buy_count == 2 and sell_count == 0:
            metadata["signal_type"] = "partial_buy"
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=0.4,
                metadata=metadata,
            )

        if sell_count == 2 and buy_count == 0:
            metadata["signal_type"] = "partial_sell"
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.SELL,
                confidence=0.4,
                metadata=metadata,
            )

        metadata["signal_type"] = "no_agreement"
        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            metadata=metadata,
        )


strategy_registry.register(CompositeMomentumStrategy())
