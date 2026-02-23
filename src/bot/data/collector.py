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
        self._dynamic_symbols: set[str] = set()

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
        all_symbols = list(dict.fromkeys(self._symbols + sorted(self._dynamic_symbols)))
        total = 0
        for exchange in self._exchanges:
            for symbol in all_symbols:
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

    async def bulk_backfill(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        days: int = 30,
    ) -> int:
        """Backfill historical data for multiple symbols with rate limiting.

        Respects exchange API rate limits by inserting a 0.5s delay between
        symbols. Uses INSERT OR IGNORE so already-stored data is skipped.

        Returns total number of candles backfilled.
        """
        if not symbols or not self._exchanges:
            return 0

        tf_seconds = TIMEFRAME_SECONDS.get(timeframe, 3600)
        candles_per_day = 86400 // tf_seconds
        limit = candles_per_day * days

        exchange = self._exchanges[0]
        total = 0
        n = len(symbols)

        for i, symbol in enumerate(symbols):
            try:
                count = await self.backfill(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                )
                total += count
                logger.info(
                    "bulk_backfill_progress",
                    symbol=symbol,
                    progress=f"{i + 1}/{n}",
                    candles=count,
                )
            except Exception as e:
                logger.error(
                    "bulk_backfill_symbol_error",
                    symbol=symbol,
                    error=str(e),
                )
            # Rate limit: 0.5s between symbols
            if i < n - 1:
                await asyncio.sleep(0.5)

        logger.info(
            "bulk_backfill_complete",
            symbols=n,
            total_candles=total,
            timeframe=timeframe,
            days=days,
        )
        return total

    def auto_discover_symbols(
        self,
        registry: object,
        min_score: float = 30.0,
    ) -> list[str]:
        """Discover symbols from OpportunityRegistry and add to dynamic set.

        Args:
            registry: OpportunityRegistry instance.
            min_score: Minimum opportunity score to include.

        Returns:
            List of newly discovered symbols.
        """
        from bot.engines.opportunity_registry import OpportunityType

        new_symbols: list[str] = []
        existing = set(self._symbols) | self._dynamic_symbols

        for op_type in OpportunityType:
            discovered = registry.get_symbols(op_type, n=20, min_score=min_score)
            for sym in discovered:
                if sym not in existing:
                    self._dynamic_symbols.add(sym)
                    existing.add(sym)
                    new_symbols.append(sym)

        if new_symbols:
            logger.info(
                "auto_discover_new_symbols",
                count=len(new_symbols),
                symbols=new_symbols[:10],
            )
        return new_symbols

    async def _backfill_loop(
        self,
        registry: object | None = None,
        settings: object | None = None,
    ) -> None:
        """Background loop: periodically discover symbols and backfill data.

        Runs every data_backfill_interval_hours (default 6h). On each cycle:
        1. Discovers new symbols from OpportunityRegistry
        2. Backfills dynamic symbols with recent data (7 days)
        """
        await asyncio.sleep(300)  # 5 min initial delay

        while True:
            try:
                enabled = getattr(settings, "data_backfill_enabled", True) if settings else True
                if not enabled:
                    interval = (
                        getattr(settings, "data_backfill_interval_hours", 6.0) * 3600
                        if settings
                        else 21600
                    )
                    await asyncio.sleep(interval)
                    continue

                # 1. Discover new symbols from registry
                if registry:
                    self.auto_discover_symbols(registry)

                # 2. Backfill dynamic symbols (recent 7 days)
                if self._dynamic_symbols:
                    await self.bulk_backfill(
                        symbols=sorted(self._dynamic_symbols),
                        timeframe="1h",
                        days=7,
                    )

                logger.info(
                    "backfill_loop_completed",
                    dynamic_symbols=len(self._dynamic_symbols),
                )
            except Exception as e:
                logger.error("backfill_loop_error", error=str(e))

            interval = (
                getattr(settings, "data_backfill_interval_hours", 6.0) * 3600
                if settings
                else 21600
            )
            await asyncio.sleep(interval)

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
