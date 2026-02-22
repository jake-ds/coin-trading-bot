"""GARCH volatility breakout strategy.

Generates signals when realized volatility significantly exceeds
GARCH-forecasted volatility, indicating a volatility regime change.

Uses dynamic stop-loss based on GARCH forecasts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.quant.volatility import GARCHModel
from bot.strategies.base import BaseStrategy

logger = structlog.get_logger()


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility breakout strategy using GARCH forecasting."""

    def __init__(
        self,
        lookback: int = 120,
        realized_window: int = 10,
        breakout_multiplier: float = 2.0,
        stop_loss_vol_multiplier: float = 2.0,
        min_data_points: int = 60,
    ):
        self._lookback = lookback
        self._realized_window = realized_window
        self._breakout_multiplier = breakout_multiplier
        self._stop_loss_vol_multiplier = stop_loss_vol_multiplier
        self._min_data_points = min_data_points
        self._garch = GARCHModel()

    @property
    def name(self) -> str:
        return "volatility_breakout"

    @property
    def required_history_length(self) -> int:
        return self._lookback

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        symbol = kwargs.get("symbol", ohlcv_data[-1].symbol if ohlcv_data else "UNKNOWN")

        if len(ohlcv_data) < self._min_data_points:
            return self._hold(symbol, {"reason": "insufficient_data"})

        closes = np.array([c.close for c in ohlcv_data[-self._lookback :]])
        returns = np.diff(np.log(np.maximum(closes, 1e-10)))

        if len(returns) < self._min_data_points:
            return self._hold(symbol, {"reason": "insufficient_returns"})

        # Fit GARCH model
        fit_result = self._garch.fit(returns)
        if not fit_result["success"]:
            return self._hold(symbol, {"reason": "garch_fit_failed"})

        # Forecast next-period volatility
        vol_forecast = self._garch.forecast(horizon=1)
        forecasted_vol = float(vol_forecast[0])

        # Realized volatility (recent window)
        recent_returns = returns[-self._realized_window :]
        realized_vol = float(np.std(recent_returns, ddof=1))

        # Breakout detection
        vol_ratio = realized_vol / forecasted_vol if forecasted_vol > 1e-10 else 0.0

        # Dynamic stop-loss
        current_price = closes[-1]
        stop_loss = self._garch.dynamic_stop_loss(
            current_price, multiplier=self._stop_loss_vol_multiplier
        )

        metadata = {
            "forecasted_vol": round(forecasted_vol, 8),
            "realized_vol": round(realized_vol, 8),
            "vol_ratio": round(vol_ratio, 4),
            "breakout_threshold": self._breakout_multiplier,
            "garch_persistence": round(fit_result["persistence"], 4),
            "dynamic_stop_loss": round(stop_loss, 2),
            "unconditional_vol": round(fit_result["unconditional_vol"], 8),
        }

        if vol_ratio >= self._breakout_multiplier:
            # Volatility breakout detected
            # Direction from recent return sign
            recent_return = float(np.sum(recent_returns))
            confidence = min(vol_ratio / (self._breakout_multiplier * 2), 1.0)

            if recent_return > 0:
                metadata["signal_reason"] = "upside_vol_breakout"
                return TradingSignal(
                    strategy_name=self.name,
                    symbol=symbol,
                    action=SignalAction.BUY,
                    confidence=round(confidence, 4),
                    metadata=metadata,
                )
            else:
                metadata["signal_reason"] = "downside_vol_breakout"
                return TradingSignal(
                    strategy_name=self.name,
                    symbol=symbol,
                    action=SignalAction.SELL,
                    confidence=round(confidence, 4),
                    metadata=metadata,
                )

        return self._hold(symbol, metadata)

    def _hold(self, symbol: str, metadata: dict) -> TradingSignal:
        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            metadata=metadata,
        )
