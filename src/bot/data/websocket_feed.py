"""WebSocket real-time data feed with REST polling fallback.

Provides real-time price updates for position monitoring (stop-loss checks)
and candle close callbacks for triggering strategy analysis.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger()


class WebSocketFeed:
    """Real-time data feed supporting both WebSocket and REST polling.

    - WebSocket mode: uses ccxt pro's watch_ticker/watch_ohlcv if supported
    - REST polling mode: falls back to exchange adapter's get_ticker periodically
    - Stores latest prices in a dict accessible by PositionManager
    - Graceful reconnection with exponential backoff on WebSocket errors
    """

    def __init__(
        self,
        exchange: Any,
        symbols: list[str],
        timeframes: list[str] | None = None,
        on_candle_close: (
            Callable[[str, str], Coroutine[Any, Any, None]] | None
        ) = None,
        poll_interval: float = 5.0,
        max_reconnect_delay: float = 60.0,
    ):
        self._exchange = exchange
        self._symbols = symbols
        self._timeframes = timeframes or ["1h"]
        self._on_candle_close = on_candle_close
        self._poll_interval = poll_interval
        self._max_reconnect_delay = max_reconnect_delay

        # Latest prices for each symbol
        self._latest_prices: dict[str, float] = {}

        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._ws_supported = self._check_ws_support()

        # Track last candle timestamps for close detection
        self._last_candle_timestamps: dict[tuple[str, str], float] = {}

        # Initial reconnection delay (doubles on each failure, resets on success)
        self._initial_reconnect_delay = 1.0

    def _check_ws_support(self) -> bool:
        """Check if the exchange supports WebSocket via ccxt pro."""
        ccxt_exchange = self._get_ccxt_exchange()
        if ccxt_exchange is None:
            return False
        has = getattr(ccxt_exchange, "has", None)
        if has and isinstance(has, dict):
            return bool(has.get("watchTicker", False))
        return False

    def _get_ccxt_exchange(self) -> Any | None:
        """Get the underlying ccxt exchange object for WebSocket calls.

        Traverses the adapter chain:
        ResilientExchange._exchange (adapter) → adapter._exchange (ccxt)
        """
        inner = self._exchange
        # ResilientExchange stores adapter as _exchange
        adapter = getattr(inner, "_exchange", None)
        if adapter is None:
            return None
        # ExchangeAdapter (e.g. BinanceAdapter) stores ccxt as _exchange
        ccxt_ex = getattr(adapter, "_exchange", None)
        if ccxt_ex is not None:
            return ccxt_ex
        # If the exchange itself is an adapter with _exchange (direct adapter)
        return adapter if hasattr(adapter, "has") else None

    @property
    def ws_supported(self) -> bool:
        """Whether WebSocket is supported by the exchange."""
        return self._ws_supported

    @property
    def latest_prices(self) -> dict[str, float]:
        """Get a copy of all latest prices."""
        return dict(self._latest_prices)

    @property
    def is_running(self) -> bool:
        """Whether the feed is currently running."""
        return self._running

    def get_latest_price(self, symbol: str) -> float | None:
        """Get the latest price for a specific symbol."""
        return self._latest_prices.get(symbol)

    def update_price(self, symbol: str, price: float) -> None:
        """Manually update a price (used by REST polling and tests)."""
        self._latest_prices[symbol] = price

    async def start(self) -> None:
        """Start the data feed.

        Uses WebSocket if supported, otherwise falls back to REST polling.
        """
        self._running = True
        logger.info(
            "websocket_feed_starting",
            symbols=self._symbols,
            timeframes=self._timeframes,
            ws_supported=self._ws_supported,
        )

        if self._ws_supported:
            # Start WebSocket watchers for each symbol
            for symbol in self._symbols:
                task = asyncio.create_task(
                    self._watch_ticker_loop(symbol),
                    name=f"ws_ticker_{symbol}",
                )
                self._tasks.append(task)
                for timeframe in self._timeframes:
                    task = asyncio.create_task(
                        self._watch_ohlcv_loop(symbol, timeframe),
                        name=f"ws_ohlcv_{symbol}_{timeframe}",
                    )
                    self._tasks.append(task)
            logger.info("websocket_feed_started_ws_mode")
        else:
            # Fallback to REST polling
            task = asyncio.create_task(
                self._poll_prices_loop(),
                name="rest_poll_prices",
            )
            self._tasks.append(task)
            logger.info("websocket_feed_started_rest_mode")

    async def stop(self) -> None:
        """Stop the data feed and cancel all background tasks."""
        self._running = False
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._tasks.clear()
        logger.info("websocket_feed_stopped")

    async def _poll_prices_loop(self) -> None:
        """REST polling fallback: periodically fetch ticker prices."""
        while self._running:
            for symbol in self._symbols:
                try:
                    ticker = await self._exchange.get_ticker(symbol)
                    if ticker and "last" in ticker:
                        price = float(ticker["last"])
                        if price > 0:
                            self._latest_prices[symbol] = price
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(
                        "poll_price_error",
                        symbol=symbol,
                        error=str(e),
                    )
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                raise

    async def _watch_ticker_loop(self, symbol: str) -> None:
        """WebSocket ticker watching with exponential backoff reconnection."""
        delay = self._initial_reconnect_delay
        while self._running:
            try:
                ccxt_exchange = self._get_ccxt_exchange()
                if ccxt_exchange is None:
                    logger.warning(
                        "ws_no_ccxt_exchange",
                        symbol=symbol,
                    )
                    break

                while self._running:
                    ticker = await ccxt_exchange.watch_ticker(symbol)
                    if ticker and "last" in ticker:
                        price = float(ticker["last"])
                        if price > 0:
                            self._latest_prices[symbol] = price
                    # Reset delay on successful watch
                    delay = self._initial_reconnect_delay

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "ws_ticker_reconnecting",
                    symbol=symbol,
                    error=str(e),
                    reconnect_delay=delay,
                )
                if not self._running:
                    break
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    raise
                delay = min(delay * 2, self._max_reconnect_delay)

    async def _watch_ohlcv_loop(self, symbol: str, timeframe: str) -> None:
        """WebSocket OHLCV watching with candle close detection."""
        delay = self._initial_reconnect_delay
        while self._running:
            try:
                ccxt_exchange = self._get_ccxt_exchange()
                if ccxt_exchange is None:
                    logger.warning(
                        "ws_no_ccxt_exchange_ohlcv",
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                    break

                while self._running:
                    candles = await ccxt_exchange.watch_ohlcv(
                        symbol, timeframe
                    )
                    if candles:
                        # Latest candle data
                        latest = candles[-1]
                        # ccxt returns [timestamp, open, high, low, close, volume]
                        ts_ms = latest[0]
                        key = (symbol, timeframe)

                        if key in self._last_candle_timestamps:
                            if ts_ms > self._last_candle_timestamps[key]:
                                # New candle appeared → previous candle closed
                                self._last_candle_timestamps[key] = ts_ms
                                if self._on_candle_close:
                                    try:
                                        await self._on_candle_close(
                                            symbol, timeframe
                                        )
                                    except Exception as cb_err:
                                        logger.warning(
                                            "candle_close_callback_error",
                                            symbol=symbol,
                                            timeframe=timeframe,
                                            error=str(cb_err),
                                        )
                        else:
                            self._last_candle_timestamps[key] = ts_ms

                        # Also update latest price from candle close
                        close_price = float(latest[4])
                        if close_price > 0:
                            self._latest_prices[symbol] = close_price

                    # Reset delay on success
                    delay = self._initial_reconnect_delay

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "ws_ohlcv_reconnecting",
                    symbol=symbol,
                    timeframe=timeframe,
                    error=str(e),
                    reconnect_delay=delay,
                )
                if not self._running:
                    break
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    raise
                delay = min(delay * 2, self._max_reconnect_delay)
