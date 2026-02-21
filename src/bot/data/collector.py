"""DataCollector - fetches market data from exchanges and stores it."""

import asyncio
from datetime import datetime, timezone

import structlog

from bot.data.store import DataStore
from bot.exchanges.base import ExchangeAdapter
from bot.models import OHLCV

logger = structlog.get_logger()

# Map timeframe strings to their duration in seconds
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "1w": 604800,
}


def validate_candle(candle: OHLCV) -> bool:
    """Validate candle data quality.

    Rejects candles where:
    - high < low
    - close > high or close < low
    - volume < 0
    """
    if candle.high < candle.low:
        return False
    if candle.close > candle.high or candle.close < candle.low:
        return False
    if candle.volume < 0:
        return False
    return True


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

    @staticmethod
    def _filter_valid_candles(candles: list[OHLCV]) -> list[OHLCV]:
        """Filter out candles that fail data quality validation."""
        valid = []
        for candle in candles:
            if validate_candle(candle):
                valid.append(candle)
            else:
                logger.warning(
                    "invalid_candle_rejected",
                    symbol=candle.symbol,
                    timeframe=candle.timeframe,
                    timestamp=str(candle.timestamp),
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                )
        return valid

    @staticmethod
    def _check_staleness(candles: list[OHLCV], timeframe: str) -> None:
        """Warn if the latest candle is stale (older than 2x the timeframe)."""
        if not candles:
            return
        tf_seconds = TIMEFRAME_SECONDS.get(timeframe)
        if tf_seconds is None:
            return
        latest = candles[-1].timestamp
        # Ensure we compare in UTC
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_seconds = (now - latest).total_seconds()
        staleness_threshold = tf_seconds * 2
        if age_seconds > staleness_threshold:
            logger.warning(
                "stale_candle_data",
                symbol=candles[-1].symbol,
                timeframe=timeframe,
                latest_timestamp=str(latest),
                age_seconds=age_seconds,
                threshold_seconds=staleness_threshold,
            )

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
                            valid_candles = self._filter_valid_candles(candles)
                            self._check_staleness(valid_candles, timeframe)
                            if valid_candles:
                                await self._store.save_candles(valid_candles)
                                total += len(valid_candles)
                            logger.info(
                                "collected_candles",
                                exchange=exchange.name,
                                symbol=symbol,
                                timeframe=timeframe,
                                count=len(valid_candles),
                                rejected=len(candles) - len(valid_candles),
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
