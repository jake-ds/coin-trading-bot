"""Dollar Cost Averaging strategy."""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import ta

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy, strategy_registry


class DCAStrategy(BaseStrategy):
    """Dollar Cost Averaging: buy at regular intervals, optionally enhanced with RSI."""

    def __init__(
        self,
        interval: str = "daily",
        buy_amount: float = 100.0,
        use_rsi_enhancement: bool = True,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_bonus_multiplier: float = 1.5,
    ):
        self._interval = interval
        self._buy_amount = buy_amount
        self._use_rsi = use_rsi_enhancement
        self._rsi_period = rsi_period
        self._rsi_oversold = rsi_oversold
        self._rsi_bonus = rsi_bonus_multiplier
        self._last_buy_time: datetime | None = None
        self._total_invested: float = 0.0
        self._total_quantity: float = 0.0

    @property
    def name(self) -> str:
        return "dca"

    @property
    def required_history_length(self) -> int:
        return self._rsi_period + 2 if self._use_rsi else 1

    def _get_interval_delta(self) -> timedelta:
        intervals = {
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
            "monthly": timedelta(days=30),
        }
        return intervals.get(self._interval, timedelta(days=1))

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        symbol = kwargs.get("symbol", ohlcv_data[-1].symbol if ohlcv_data else "UNKNOWN")

        if not ohlcv_data:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
            )

        current_time = ohlcv_data[-1].timestamp
        current_price = ohlcv_data[-1].close
        interval_delta = self._get_interval_delta()

        # Check if it's time to buy
        should_buy = (
            self._last_buy_time is None
            or current_time - self._last_buy_time >= interval_delta
        )

        if not should_buy:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={
                    "reason": "not_time_yet",
                    "last_buy": str(self._last_buy_time),
                    "total_invested": self._total_invested,
                },
            )

        # Calculate buy amount and confidence
        amount = self._buy_amount
        confidence = 0.5  # Base confidence for DCA

        # Enhanced DCA: buy more when RSI indicates oversold
        rsi_value = None
        if self._use_rsi and len(ohlcv_data) >= self._rsi_period + 2:
            closes = pd.Series([c.close for c in ohlcv_data])
            rsi_series = ta.momentum.rsi(closes, window=self._rsi_period)
            rsi_value = float(rsi_series.iloc[-1])

            if rsi_value < self._rsi_oversold:
                amount *= self._rsi_bonus
                confidence = 0.8

        # Update tracking
        self._last_buy_time = current_time
        quantity = amount / current_price if current_price > 0 else 0
        self._total_invested += amount
        self._total_quantity += quantity

        avg_price = self._total_invested / self._total_quantity if self._total_quantity > 0 else 0

        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=SignalAction.BUY,
            confidence=confidence,
            metadata={
                "buy_amount": amount,
                "quantity": quantity,
                "total_invested": self._total_invested,
                "total_quantity": self._total_quantity,
                "average_price": avg_price,
                "rsi": rsi_value,
                "interval": self._interval,
            },
        )


strategy_registry.register(DCAStrategy())
