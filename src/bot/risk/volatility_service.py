"""VolatilityService — GARCH-based real-time volatility forecasting service.

Wraps GARCHModel and classify_volatility_regime to provide per-symbol
periodic GARCH fitting, volatility forecasts, and regime classification.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import structlog

from bot.quant.volatility import GARCHModel, VolatilityRegime, classify_volatility_regime

if TYPE_CHECKING:
    from bot.research.data_provider import HistoricalDataProvider

logger = structlog.get_logger()


class VolatilityService:
    """Provides per-symbol GARCH volatility forecasts and regime classification.

    Periodically fits GARCH models to return series obtained from
    HistoricalDataProvider and caches forecasts and regime classifications
    for fast lookup by risk modules and engines.

    If data_provider is None, all forecasts return None and all regimes
    return NORMAL (graceful degradation for backward compatibility).
    """

    def __init__(self, data_provider: HistoricalDataProvider | None = None) -> None:
        self._data_provider = data_provider
        self._models: dict[str, GARCHModel] = {}
        self._forecasts: dict[str, float] = {}
        self._regimes: dict[str, VolatilityRegime] = {}
        self._last_fit: dict[str, datetime] = {}

    async def fit_symbol(self, symbol: str, lookback_days: int = 60) -> bool:
        """Fit GARCH model for a single symbol.

        Args:
            symbol: Trading pair symbol (e.g. 'BTC/USDT').
            lookback_days: Number of days of return data to use.

        Returns:
            True if fit succeeded, False otherwise.
        """
        if self._data_provider is None:
            logger.warning("volatility_fit_no_provider", symbol=symbol)
            return False

        try:
            returns = await self._data_provider.get_returns(
                symbol, timeframe="1h", lookback_days=lookback_days
            )
            if len(returns) < 30:
                logger.warning(
                    "volatility_fit_insufficient_data",
                    symbol=symbol,
                    n_returns=len(returns),
                )
                return False

            model = GARCHModel()
            result = model.fit(returns)

            if not result.get("success", False):
                logger.warning("volatility_fit_failed", symbol=symbol)
                return False

            # Store model and cache forecast
            self._models[symbol] = model
            forecast_arr = model.forecast(horizon=1)
            self._forecasts[symbol] = float(forecast_arr[0])

            # Classify regime from returns
            self._regimes[symbol] = classify_volatility_regime(returns)

            self._last_fit[symbol] = datetime.now(timezone.utc)

            logger.info(
                "volatility_fit_success",
                symbol=symbol,
                forecast=round(self._forecasts[symbol], 6),
                regime=self._regimes[symbol].value,
            )
            return True

        except Exception as e:
            logger.error("volatility_fit_error", symbol=symbol, error=str(e))
            return False

    async def fit_all(self, symbols: list[str]) -> dict[str, bool]:
        """Fit GARCH models for all given symbols.

        Symbols that fail fitting are skipped with a warning.

        Returns:
            Dict mapping symbol → success boolean.
        """
        results: dict[str, bool] = {}
        for symbol in symbols:
            success = await self.fit_symbol(symbol)
            results[symbol] = success
        return results

    def get_forecast(self, symbol: str, horizon: int = 1) -> float | None:
        """Get cached volatility forecast for a symbol.

        Args:
            symbol: Trading pair symbol.
            horizon: Forecast horizon (only horizon=1 uses cache; >1 recomputes).

        Returns:
            Forecasted volatility (standard deviation), or None if not fitted.
        """
        if horizon == 1:
            return self._forecasts.get(symbol)

        model = self._models.get(symbol)
        if model is None or not model.is_fitted:
            return None

        forecast_arr = model.forecast(horizon=horizon)
        val = float(forecast_arr[-1])
        if np.isnan(val):
            return None
        return val

    def get_regime(self, symbol: str) -> VolatilityRegime:
        """Get cached volatility regime for a symbol.

        Returns NORMAL if no regime has been computed.
        """
        return self._regimes.get(symbol, VolatilityRegime.NORMAL)

    def get_all_regimes(self) -> dict[str, VolatilityRegime]:
        """Get all cached regime classifications."""
        return dict(self._regimes)

    def get_market_regime(self) -> VolatilityRegime:
        """Get market-wide regime using BTC/USDT as proxy.

        Returns NORMAL if BTC regime is not available.
        """
        return self._regimes.get("BTC/USDT", VolatilityRegime.NORMAL)

    def needs_refit(self, symbol: str, max_age_hours: float = 6.0) -> bool:
        """Check whether a symbol's model needs refitting.

        Returns True if the symbol has never been fitted or if the last
        fit is older than max_age_hours.
        """
        last = self._last_fit.get(symbol)
        if last is None:
            return True
        age = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        return age > max_age_hours

    def get_model(self, symbol: str) -> GARCHModel | None:
        """Get the fitted GARCH model for a symbol (if any)."""
        return self._models.get(symbol)

    async def _fit_loop(
        self, symbols: list[str], interval_hours: float = 6.0
    ) -> None:
        """Background loop that periodically refits all symbols.

        Intended to be started as an asyncio task by EngineManager.
        First execution after a 60-second startup delay.
        """
        await asyncio.sleep(60)  # Initial delay
        while True:
            try:
                results = await self.fit_all(symbols)
                fitted = sum(1 for v in results.values() if v)
                logger.info(
                    "volatility_fit_loop_completed",
                    total=len(symbols),
                    fitted=fitted,
                    failed=len(symbols) - fitted,
                )
            except Exception as e:
                logger.error("volatility_fit_loop_error", error=str(e))
            await asyncio.sleep(interval_hours * 3600)
