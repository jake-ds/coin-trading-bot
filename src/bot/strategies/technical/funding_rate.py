"""Funding rate strategy for perpetual futures markets.

Generates signals based on extreme funding rates in perpetual futures:
- Extreme positive funding (> threshold) = market overheated → SELL signal
- Extreme negative funding (< threshold) = market oversold → BUY signal

Also tracks perpetual-spot spread for delta-neutral yield opportunities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy, strategy_registry

if TYPE_CHECKING:
    from bot.data.funding import FundingRateMonitor
    from bot.strategies.regime import MarketRegime


class FundingRateStrategy(BaseStrategy):
    """Funding rate-based trading strategy.

    Uses perpetual futures funding rates as a contrarian indicator:
    - Very high positive funding means the market is overleveraged long
      → likely to correct down → SELL signal
    - Very negative funding means the market is overleveraged short
      → likely to bounce up → BUY signal

    Confidence scales with the magnitude of the funding rate deviation
    from neutral (0%).

    Optional spread tracking identifies delta-neutral arbitrage
    opportunities when the perpetual-spot spread is wide.
    """

    def __init__(
        self,
        extreme_positive_rate: float = 0.0005,
        extreme_negative_rate: float = -0.0003,
        confidence_scale_factor: float = 10.0,
        min_confidence: float = 0.3,
        max_confidence: float = 0.9,
        spread_threshold_pct: float = 0.5,
        use_rate_trend: bool = True,
        rate_trend_periods: int = 3,
    ):
        """Initialize the funding rate strategy.

        Args:
            extreme_positive_rate: Funding rate above this triggers SELL
                (0.0005 = 0.05% per period, ~0.15% daily at 3 settlements).
            extreme_negative_rate: Funding rate below this triggers BUY
                (-0.0003 = -0.03% per period).
            confidence_scale_factor: Multiplier for scaling rate deviation
                into confidence (higher = more sensitive).
            min_confidence: Minimum confidence for any signal.
            max_confidence: Maximum confidence cap.
            spread_threshold_pct: Perp-spot spread threshold for
                spread-based signals (percentage).
            use_rate_trend: Whether to consider rate trend direction.
            rate_trend_periods: Number of periods for rate trend analysis.
        """
        self._extreme_positive_rate = extreme_positive_rate
        self._extreme_negative_rate = extreme_negative_rate
        self._confidence_scale_factor = confidence_scale_factor
        self._min_confidence = min_confidence
        self._max_confidence = max_confidence
        self._spread_threshold_pct = spread_threshold_pct
        self._use_rate_trend = use_rate_trend
        self._rate_trend_periods = rate_trend_periods
        self._regime_disabled = False
        self._funding_monitor: FundingRateMonitor | None = None

    @property
    def name(self) -> str:
        return "funding_rate"

    @property
    def required_history_length(self) -> int:
        return 1

    def set_funding_monitor(self, monitor: FundingRateMonitor) -> None:
        """Set the funding rate monitor to use for data access."""
        self._funding_monitor = monitor

    def adapt_to_regime(self, regime: MarketRegime) -> None:
        """Adapt funding rate strategy based on market regime.

        Funding rate signals work in all regimes but are less reliable
        in high volatility where rates can be temporarily extreme
        without meaning a reversal.
        """
        from bot.strategies.regime import MarketRegime

        if regime == MarketRegime.HIGH_VOLATILITY:
            self._regime_disabled = True
        else:
            self._regime_disabled = False

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        """Analyze funding rate data and generate a trading signal.

        The strategy primarily uses funding rate data from the FundingRateMonitor
        rather than OHLCV candles. OHLCV data is used to extract the symbol
        and current price context.

        Kwargs:
            symbol: Trading pair symbol.
            funding_rate: Current funding rate (float). If provided directly,
                takes precedence over monitor data.
            funding_data: Full funding data dict from monitor.
        """
        symbol = kwargs.get(
            "symbol", ohlcv_data[-1].symbol if ohlcv_data else "UNKNOWN"
        )

        if self._regime_disabled:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "disabled_by_regime", "regime": "HIGH_VOLATILITY"},
            )

        # Get funding rate data from kwargs or monitor
        funding_rate = kwargs.get("funding_rate")
        funding_data = kwargs.get("funding_data")

        if funding_rate is None and funding_data is not None:
            funding_rate = funding_data.get("funding_rate")

        if funding_rate is None and self._funding_monitor is not None:
            rate_data = self._funding_monitor.get_latest_rate(symbol)
            if rate_data is not None:
                funding_rate = rate_data.get("funding_rate")
                if funding_data is None:
                    funding_data = rate_data

        if funding_rate is None:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "no_funding_data"},
            )

        # Extract additional data
        mark_price = 0.0
        spot_price = 0.0
        spread_pct = 0.0
        if funding_data:
            mark_price = funding_data.get("mark_price", 0.0)
            spot_price = funding_data.get("spot_price", 0.0)
            spread_pct = funding_data.get("spread_pct", 0.0)

        # Get rate trend if monitor is available
        rate_trend = self._analyze_rate_trend(symbol)

        # Build metadata
        metadata: dict[str, Any] = {
            "funding_rate": funding_rate,
            "extreme_positive_threshold": self._extreme_positive_rate,
            "extreme_negative_threshold": self._extreme_negative_rate,
            "mark_price": mark_price,
            "spot_price": spot_price,
            "spread_pct": round(spread_pct, 4),
            "rate_trend": rate_trend,
        }

        # Determine signal based on funding rate
        if funding_rate > self._extreme_positive_rate:
            # Market is overleveraged long → contrarian SELL
            deviation = funding_rate - self._extreme_positive_rate
            confidence = self._calculate_confidence(deviation)

            # Rate trend confirmation: if rates are rising, stronger signal
            if self._use_rate_trend and rate_trend == "rising":
                confidence = min(confidence * 1.2, self._max_confidence)

            metadata["signal_type"] = "extreme_positive_funding"
            metadata["deviation"] = round(deviation, 6)

            # Check for spread-based signal enhancement
            if spread_pct > self._spread_threshold_pct:
                metadata["spread_signal"] = "perp_premium"
                confidence = min(confidence * 1.1, self._max_confidence)

            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.SELL,
                confidence=confidence,
                metadata=metadata,
            )

        if funding_rate < self._extreme_negative_rate:
            # Market is overleveraged short → contrarian BUY
            deviation = abs(funding_rate - self._extreme_negative_rate)
            confidence = self._calculate_confidence(deviation)

            # Rate trend confirmation: if rates are falling, stronger signal
            if self._use_rate_trend and rate_trend == "falling":
                confidence = min(confidence * 1.2, self._max_confidence)

            metadata["signal_type"] = "extreme_negative_funding"
            metadata["deviation"] = round(deviation, 6)

            # Check for spread-based signal enhancement
            if spread_pct < -self._spread_threshold_pct:
                metadata["spread_signal"] = "perp_discount"
                confidence = min(confidence * 1.1, self._max_confidence)

            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=confidence,
                metadata=metadata,
            )

        # Funding rate within normal range
        metadata["reason"] = "funding_rate_normal"
        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            metadata=metadata,
        )

    def _calculate_confidence(self, deviation: float) -> float:
        """Calculate signal confidence from funding rate deviation.

        Higher deviation from threshold = higher confidence.
        """
        raw_confidence = deviation * self._confidence_scale_factor
        return min(max(raw_confidence + self._min_confidence, self._min_confidence),
                   self._max_confidence)

    def _analyze_rate_trend(self, symbol: str) -> str | None:
        """Analyze the trend of recent funding rates.

        Returns:
            "rising": rates increasing (more positive/less negative)
            "falling": rates decreasing (less positive/more negative)
            None: insufficient data or no monitor
        """
        if not self._use_rate_trend or self._funding_monitor is None:
            return None

        history = self._funding_monitor.get_rate_history(symbol)
        if len(history) < self._rate_trend_periods:
            return None

        recent = history[-self._rate_trend_periods :]
        rates = [h.get("funding_rate", 0.0) for h in recent]

        # Simple trend: compare average of first half to second half
        mid = len(rates) // 2
        if mid == 0:
            return None

        first_half_avg = sum(rates[:mid]) / mid
        second_half_avg = sum(rates[mid:]) / (len(rates) - mid)

        if second_half_avg > first_half_avg:
            return "rising"
        elif second_half_avg < first_half_avg:
            return "falling"
        return None


strategy_registry.register(FundingRateStrategy())
