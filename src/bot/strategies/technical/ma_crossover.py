"""Moving Average Crossover strategy."""

from typing import Any

import pandas as pd
import ta

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy, strategy_registry


class MACrossoverStrategy(BaseStrategy):
    """MA Crossover: BUY when short MA crosses above long MA, SELL when below."""

    def __init__(self, short_period: int = 20, long_period: int = 50):
        self._short_period = short_period
        self._long_period = long_period

    @property
    def name(self) -> str:
        return "ma_crossover"

    @property
    def required_history_length(self) -> int:
        return self._long_period + 1

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

        short_ma = ta.trend.sma_indicator(closes, window=self._short_period)
        long_ma = ta.trend.sma_indicator(closes, window=self._long_period)

        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]

        # Calculate confidence based on distance between MAs
        distance = abs(current_short - current_long) / current_long if current_long > 0 else 0
        confidence = min(distance * 10, 1.0)  # Scale to 0-1

        metadata = {
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

        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=action,
            confidence=confidence,
            metadata=metadata,
        )


strategy_registry.register(MACrossoverStrategy())
