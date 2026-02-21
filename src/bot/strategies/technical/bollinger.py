"""Bollinger Bands strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import ta

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy, strategy_registry

if TYPE_CHECKING:
    from bot.strategies.regime import MarketRegime


class BollingerStrategy(BaseStrategy):
    """Bollinger Bands: BUY at lower band, SELL at upper band."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self._period = period
        self._std_dev = std_dev
        self._regime_disabled = False

    @property
    def name(self) -> str:
        return "bollinger"

    @property
    def required_history_length(self) -> int:
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


strategy_registry.register(BollingerStrategy())
