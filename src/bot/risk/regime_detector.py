"""MarketRegimeDetector — real-time market regime classification with CRISIS detection.

Extends VolatilityService regime classification with CRISIS detection
(extreme volatility or large BTC drawdown). Tracks regime transition history.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

import structlog

from bot.quant.volatility import VolatilityRegime

if TYPE_CHECKING:
    from bot.risk.volatility_service import VolatilityService

logger = structlog.get_logger(__name__)


class MarketRegime(str, Enum):
    """Market regime classification with CRISIS level."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRISIS = "CRISIS"


class MarketRegimeDetector:
    """Detects market regime using VolatilityService + CRISIS heuristics.

    CRISIS is triggered when:
    - Recent volatility exceeds long-term average by crisis_threshold (OR)
    - BTC 24h change is <= -10%
    """

    def __init__(
        self,
        volatility_service: VolatilityService | None = None,
        crisis_threshold: float = 2.5,
        lookback_window: int = 30,
    ):
        self._volatility_service = volatility_service
        self._crisis_threshold = crisis_threshold
        self._lookback_window = lookback_window

        self._current_regime: MarketRegime = MarketRegime.NORMAL
        self._regime_since: datetime = datetime.now(timezone.utc)
        self._regime_history: list[dict] = []
        self._btc_24h_change: float | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect_regime(self) -> MarketRegime:
        """Detect current market regime from VolatilityService + CRISIS check."""
        if self._volatility_service is None:
            return self._current_regime

        # Check for CRISIS conditions first
        if self._is_crisis():
            new_regime = MarketRegime.CRISIS
        else:
            # Map VolatilityRegime to MarketRegime
            vol_regime = self._volatility_service.get_market_regime()
            new_regime = self._map_vol_regime(vol_regime)

        # Record transition if regime changed
        if new_regime != self._current_regime:
            now = datetime.now(timezone.utc)
            duration = (now - self._regime_since).total_seconds() / 60.0
            trigger = self._get_trigger_reason(new_regime)
            self._regime_history.append({
                "timestamp": now.isoformat(),
                "from_regime": self._current_regime.value,
                "to_regime": new_regime.value,
                "duration_minutes": round(duration, 1),
                "trigger": trigger,
            })
            # Cap history at 100 entries
            if len(self._regime_history) > 100:
                self._regime_history = self._regime_history[-100:]

            logger.info(
                "regime_transition",
                from_regime=self._current_regime.value,
                to_regime=new_regime.value,
                duration_minutes=round(duration, 1),
                trigger=trigger,
            )
            self._current_regime = new_regime
            self._regime_since = now

        return self._current_regime

    def get_current_regime(self) -> MarketRegime:
        """Return the cached current regime."""
        return self._current_regime

    def get_regime_history(self) -> list[dict]:
        """Return regime transition history (up to 100 entries)."""
        return list(self._regime_history)

    def get_regime_duration(self) -> float:
        """Return minutes the current regime has been active."""
        now = datetime.now(timezone.utc)
        return (now - self._regime_since).total_seconds() / 60.0

    def is_crisis(self) -> bool:
        """Quick check if current regime is CRISIS."""
        return self._current_regime == MarketRegime.CRISIS

    def set_btc_24h_change(self, change_pct: float) -> None:
        """Update the BTC 24h price change (called externally)."""
        self._btc_24h_change = change_pct

    # ------------------------------------------------------------------
    # Detection loop
    # ------------------------------------------------------------------

    async def _detection_loop(
        self, interval_seconds: float = 300.0
    ) -> None:
        """Periodically run detect_regime(). Meant for background task."""
        await asyncio.sleep(60)  # Initial 1-minute delay
        while True:
            try:
                self.detect_regime()
            except Exception as e:
                logger.error("regime_detection_error", error=str(e))
            await asyncio.sleep(interval_seconds)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_crisis(self) -> bool:
        """Check CRISIS conditions."""
        # Condition 1: BTC 24h change <= -10%
        if self._btc_24h_change is not None and self._btc_24h_change <= -10.0:
            return True

        # Condition 2: volatility exceeds crisis_threshold × long-term avg
        if self._volatility_service is None:
            return False

        forecast = self._volatility_service.get_forecast("BTC/USDT")
        if forecast is None:
            return False

        model = self._volatility_service.get_model("BTC/USDT")
        if model is None:
            return False

        cond_vol = getattr(model, "conditional_volatility", None)
        if cond_vol is None or len(cond_vol) < 2:
            return False

        import numpy as np

        median_vol = float(np.median(cond_vol))
        if median_vol <= 0:
            return False

        ratio = forecast / median_vol
        return ratio >= self._crisis_threshold

    @staticmethod
    def _map_vol_regime(vol_regime: VolatilityRegime) -> MarketRegime:
        """Map VolatilityRegime enum to MarketRegime."""
        mapping = {
            VolatilityRegime.LOW: MarketRegime.LOW,
            VolatilityRegime.NORMAL: MarketRegime.NORMAL,
            VolatilityRegime.HIGH: MarketRegime.HIGH,
        }
        return mapping.get(vol_regime, MarketRegime.NORMAL)

    def _get_trigger_reason(self, new_regime: MarketRegime) -> str:
        """Describe what triggered the regime change."""
        if new_regime == MarketRegime.CRISIS:
            reasons = []
            if (
                self._btc_24h_change is not None
                and self._btc_24h_change <= -10.0
            ):
                reasons.append(
                    f"BTC 24h change: {self._btc_24h_change:.1f}%"
                )
            reasons.append("volatility spike")
            return "; ".join(reasons)
        return f"volatility regime: {new_regime.value}"
