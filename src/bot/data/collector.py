"""DataCollector - fetches market data from exchanges and stores it."""

import asyncio

import structlog

from bot.data.store import DataStore
from bot.exchanges.base import ExchangeAdapter

logger = structlog.get_logger()


class DataCollector:
    """Collects OHLCV data from configured exchanges and stores it."""

    def __init__(
        self,
        exchanges: list[ExchangeAdapter],
        store: DataStore,
        symbols: list[str],
        timeframes: list[str] | None = None,
        collection_interval: int = 60,
    ):
        self._exchanges = exchanges
        self._store = store
        self._symbols = symbols
        self._timeframes = timeframes or ["1h"]
        self._collection_interval = collection_interval
        self._running = False

    async def collect_once(self) -> int:
        """Run a single collection cycle across all exchanges/symbols/timeframes.

        Returns the total number of candles collected.
        """
        total = 0
        for exchange in self._exchanges:
            for symbol in self._symbols:
                for timeframe in self._timeframes:
                    try:
                        candles = await exchange.get_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            limit=100,
                        )
                        if candles:
                            await self._store.save_candles(candles)
                            total += len(candles)
                            logger.info(
                                "collected_candles",
                                exchange=exchange.name,
                                symbol=symbol,
                                timeframe=timeframe,
                                count=len(candles),
                            )
                    except (ValueError, ConnectionError, RuntimeError) as e:
                        logger.error(
                            "collection_error",
                            exchange=exchange.name,
                            symbol=symbol,
                            timeframe=timeframe,
                            error=str(e),
                        )
        return total

    async def backfill(
        self,
        exchange: ExchangeAdapter,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> int:
        """Backfill missing historical candles for a symbol.

        Detects gaps and fetches missing data.

        Returns the number of candles backfilled.
        """
        existing = await self._store.get_candles(
            symbol=symbol, timeframe=timeframe, limit=limit
        )
        if not existing:
            # No data at all — fetch full history
            try:
                candles = await exchange.get_ohlcv(
                    symbol=symbol, timeframe=timeframe, limit=limit
                )
                if candles:
                    await self._store.save_candles(candles)
                    logger.info(
                        "backfill_complete",
                        symbol=symbol,
                        timeframe=timeframe,
                        count=len(candles),
                    )
                    return len(candles)
            except (ValueError, ConnectionError, RuntimeError) as e:
                logger.error("backfill_error", symbol=symbol, error=str(e))
            return 0

        # Check for gaps: fetch latest data and fill in missing candles
        latest_ts = existing[-1].timestamp
        try:
            new_candles = await exchange.get_ohlcv(
                symbol=symbol, timeframe=timeframe, limit=limit
            )
            # Filter to only new candles
            if latest_ts.tzinfo is None:
                new_only = [c for c in new_candles if c.timestamp.replace(tzinfo=None) > latest_ts]
            else:
                new_only = [c for c in new_candles if c.timestamp > latest_ts]
            if new_only:
                await self._store.save_candles(new_only)
                logger.info(
                    "backfill_gaps",
                    symbol=symbol,
                    timeframe=timeframe,
                    count=len(new_only),
                )
                return len(new_only)
        except (ValueError, ConnectionError, RuntimeError) as e:
            logger.error("backfill_error", symbol=symbol, error=str(e))
        return 0

    async def run(self) -> None:
        """Run continuous data collection."""
        self._running = True
        logger.info(
            "collector_started",
            symbols=self._symbols,
            timeframes=self._timeframes,
            interval=self._collection_interval,
        )
        while self._running:
            await self.collect_once()
            await asyncio.sleep(self._collection_interval)

    def stop(self) -> None:
        """Stop the collection loop."""
        self._running = False
        logger.info("collector_stopped")
