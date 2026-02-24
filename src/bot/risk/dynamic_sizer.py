"""Dynamic position sizing based on GARCH volatility forecasts and ATR.

Uses VolatilityService GARCH forecasts to scale positions inversely
with volatility: high vol → smaller positions, low vol → larger positions.
Falls back to ATR-based sizing or fixed sizing when GARCH unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from bot.risk.portfolio_risk import PortfolioRiskManager
    from bot.risk.volatility_service import VolatilityService

logger = structlog.get_logger(__name__)


@dataclass
class PositionSize:
    """Result of dynamic position sizing calculation."""

    quantity: float
    notional_value: float
    risk_amount: float
    vol_multiplier: float
    method: str  # 'garch' | 'atr' | 'fixed'


class DynamicPositionSizer:
    """Volatility-based dynamic position sizer.

    Sizing logic priority:
    1. GARCH forecast → vol_multiplier = median_conditional_vol / forecast
    2. ATR fallback → vol_multiplier = target_atr / actual_atr
    3. Fixed fallback → vol_multiplier = 1.0

    vol_multiplier is clamped to [0.25, 2.0].
    """

    def __init__(
        self,
        volatility_service: VolatilityService | None = None,
        portfolio_risk: PortfolioRiskManager | None = None,
        base_risk_pct: float = 1.0,
        vol_scale_factor: float = 1.0,
    ) -> None:
        self._volatility_service = volatility_service
        self._portfolio_risk = portfolio_risk
        self._base_risk_pct = base_risk_pct
        self._vol_scale_factor = vol_scale_factor

    def calculate_size(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        atr: float | None = None,
    ) -> PositionSize:
        """Calculate position size adjusted for volatility.

        Args:
            symbol: Trading pair symbol.
            price: Current asset price.
            portfolio_value: Total portfolio value.
            atr: ATR value (absolute, not ratio). Used as fallback if GARCH unavailable.

        Returns:
            PositionSize with calculated values.
        """
        vol_multiplier = 1.0
        method = "fixed"

        # Try GARCH-based sizing
        if self._volatility_service is not None:
            forecast = self._volatility_service.get_forecast(symbol)
            if forecast is not None and forecast > 0:
                model = self._volatility_service.get_model(symbol)
                if model is not None and model.conditional_volatility is not None:
                    cond_vol = model.conditional_volatility
                    median_vol = float(np.median(cond_vol))
                    if median_vol > 0:
                        vol_multiplier = median_vol / forecast
                        method = "garch"

        # Fallback to ATR-based sizing
        if method == "fixed" and atr is not None and atr > 0 and price > 0:
            # Normalize ATR as fraction of price
            atr_pct = atr / price
            target_atr = self._base_risk_pct / 100.0
            if atr_pct > 0:
                vol_multiplier = target_atr / atr_pct
                method = "atr"

        # Clamp vol_multiplier to [0.25, 2.0]
        vol_multiplier = max(0.25, min(2.0, vol_multiplier))

        # Calculate risk amount and quantity
        risk_amount = portfolio_value * (self._base_risk_pct / 100.0) * vol_multiplier

        if price > 0 and self._vol_scale_factor > 0:
            quantity = risk_amount / (price * self._vol_scale_factor)
        else:
            quantity = 0.0

        notional_value = quantity * price

        return PositionSize(
            quantity=quantity,
            notional_value=notional_value,
            risk_amount=risk_amount,
            vol_multiplier=vol_multiplier,
            method=method,
        )

    def validate_size(
        self,
        position_size: PositionSize,
        portfolio_value: float,
        max_pct: float,
    ) -> PositionSize:
        """Validate and clip position size to max_pct of portfolio.

        Args:
            position_size: Original calculated position size.
            portfolio_value: Total portfolio value.
            max_pct: Maximum position size as percentage of portfolio.

        Returns:
            PositionSize, possibly clipped.
        """
        if portfolio_value <= 0 or max_pct <= 0:
            return position_size

        max_notional = portfolio_value * (max_pct / 100.0)
        if position_size.notional_value > max_notional and position_size.notional_value > 0:
            clip_ratio = max_notional / position_size.notional_value
            return PositionSize(
                quantity=position_size.quantity * clip_ratio,
                notional_value=max_notional,
                risk_amount=position_size.risk_amount * clip_ratio,
                vol_multiplier=position_size.vol_multiplier,
                method=position_size.method,
            )

        return position_size
