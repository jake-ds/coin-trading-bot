"""MACD (Moving Average Convergence Divergence) strategy."""

from typing import Any

import pandas as pd
import ta

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy, strategy_registry


class MACDStrategy(BaseStrategy):
    """MACD: BUY when MACD crosses above signal line, SELL when below."""

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._signal_period = signal_period

    @property
    def name(self) -> str:
        return "macd"

    @property
    def required_history_length(self) -> int:
        return self._slow_period + self._signal_period + 1

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

        macd_indicator = ta.trend.MACD(
            closes,
            window_fast=self._fast_period,
            window_slow=self._slow_period,
            window_sign=self._signal_period,
        )

        macd_line = macd_indicator.macd()
        signal_line = macd_indicator.macd_signal()
        histogram = macd_indicator.macd_diff()

        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        current_hist = histogram.iloc[-1]

        # Confidence from histogram magnitude
        price = closes.iloc[-1]
        confidence = min(abs(current_hist) / price * 100, 1.0) if price > 0 else 0.0

        metadata = {
            "macd": float(current_macd),
            "signal": float(current_signal),
            "histogram": float(current_hist),
        }

        if prev_macd <= prev_signal and current_macd > current_signal:
            action = SignalAction.BUY
        elif prev_macd >= prev_signal and current_macd < current_signal:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD

        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=action,
            confidence=max(confidence, 0.1) if action != SignalAction.HOLD else 0.0,
            metadata=metadata,
        )


strategy_registry.register(MACDStrategy())
