"""HistoricalDataProvider — thin wrapper over DataStore for research experiments."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from bot.data.store import DataStore
from bot.models import OHLCV

# Timeframe → candles per day mapping
_CANDLES_PER_DAY: dict[str, int] = {
    "1m": 1440,
    "5m": 288,
    "15m": 96,
    "30m": 48,
    "1h": 24,
    "2h": 12,
    "4h": 6,
    "6h": 4,
    "8h": 3,
    "12h": 2,
    "1d": 1,
}


class HistoricalDataProvider:
    """Provides historical market data from DataStore for research experiments."""

    def __init__(self, data_store: DataStore) -> None:
        self._data_store = data_store

    def _calculate_limit(self, timeframe: str, lookback_days: int) -> int:
        """Calculate the number of candles for a given timeframe and lookback."""
        cpd = _CANDLES_PER_DAY.get(timeframe, 24)
        return cpd * lookback_days

    def _start_time(self, lookback_days: int) -> datetime:
        """Calculate start datetime from lookback_days."""
        return datetime.now(timezone.utc) - timedelta(days=lookback_days)

    async def get_prices(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback_days: int = 30,
    ) -> list[float]:
        """Get close prices for a symbol, sorted oldest→newest.

        Returns empty list if no data available.
        """
        candles = await self._data_store.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            start=self._start_time(lookback_days),
            limit=self._calculate_limit(timeframe, lookback_days),
        )
        return [c.close for c in candles]

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback_days: int = 30,
    ) -> list[OHLCV]:
        """Get raw OHLCV models for a symbol, sorted oldest→newest."""
        return await self._data_store.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            start=self._start_time(lookback_days),
            limit=self._calculate_limit(timeframe, lookback_days),
        )

    async def get_returns(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback_days: int = 30,
    ) -> list[float]:
        """Get close-to-close returns. len(returns) == len(prices) - 1."""
        prices = await self.get_prices(symbol, timeframe, lookback_days)
        if len(prices) < 2:
            return []
        return [
            (prices[i] - prices[i - 1]) / prices[i - 1]
            for i in range(1, len(prices))
        ]

    async def get_funding_rates(
        self,
        symbol: str,
        lookback_days: int = 30,
    ) -> list[dict]:
        """Get funding rate history as list of dicts.

        Each dict has: timestamp, funding_rate, mark_price, spot_price.
        """
        start = self._start_time(lookback_days)
        # Funding rates come every 8h → 3 per day
        limit = lookback_days * 3
        records = await self._data_store.get_funding_rates(
            symbol=symbol,
            start=start,
            limit=limit,
        )
        return [
            {
                "timestamp": r["timestamp"],
                "funding_rate": r["funding_rate"],
                "mark_price": r["mark_price"],
                "spot_price": r["spot_price"],
            }
            for r in records
        ]

    async def get_multi_prices(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        lookback_days: int = 30,
    ) -> dict[str, list[float]]:
        """Get prices for multiple symbols concurrently."""
        tasks = [
            self.get_prices(sym, timeframe, lookback_days) for sym in symbols
        ]
        results = await asyncio.gather(*tasks)
        return dict(zip(symbols, results))

    async def get_available_symbols(
        self,
        timeframe: str = "1h",
        min_candles: int = 100,
    ) -> list[str]:
        """Get symbols with sufficient data in the DataStore."""
        return await self._data_store.get_available_symbols(
            timeframe=timeframe,
            min_count=min_candles,
        )
