"""RSI (Relative Strength Index) strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import ta

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy, strategy_registry

if TYPE_CHECKING:
    from bot.strategies.regime import MarketRegime


class RSIStrategy(BaseStrategy):
    """RSI Strategy: BUY when RSI crosses below oversold, SELL when above overbought."""

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
    ):
        self._period = period
        self._overbought = overbought
        self._oversold = oversold
        self._original_overbought = overbought
        self._original_oversold = oversold

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

        # Confidence based on distance from thresholds
        if current_rsi <= self._oversold:
            confidence = min((self._oversold - current_rsi) / self._oversold, 1.0)
            action = SignalAction.BUY
        elif current_rsi >= self._overbought:
            confidence = min((current_rsi - self._overbought) / (100 - self._overbought), 1.0)
            action = SignalAction.SELL
        else:
            confidence = 0.0
            action = SignalAction.HOLD

        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=action,
            confidence=max(confidence, 0.1) if action != SignalAction.HOLD else 0.0,
            metadata={
                "rsi": float(current_rsi),
                "overbought": self._overbought,
                "oversold": self._oversold,
                "period": self._period,
            },
        )


strategy_registry.register(RSIStrategy())
