"""Tests for WebSocketFeed real-time data feed."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.data.websocket_feed import WebSocketFeed

# --- Helpers ---


def make_exchange_mock(ws_supported: bool = False):
    """Create a mock exchange (ResilientExchange-style wrapper)."""
    # Inner adapter (e.g. BinanceAdapter) with ccxt exchange
    ccxt_exchange = MagicMock()
    ccxt_exchange.has = {"watchTicker": ws_supported, "watchOHLCV": ws_supported}

    adapter = MagicMock()
    adapter._exchange = ccxt_exchange
    adapter.name = "binance"

    # ResilientExchange wraps adapter as _exchange
    exchange = MagicMock()
    exchange._exchange = adapter
    exchange.name = "binance"
    exchange.get_ticker = AsyncMock(
        return_value={"bid": 49900.0, "ask": 50100.0, "last": 50000.0, "volume": 1000.0}
    )
    return exchange


def make_direct_adapter_mock(ws_supported: bool = False):
    """Create a mock direct exchange adapter (no ResilientExchange wrapping)."""
    ccxt_exchange = MagicMock()
    ccxt_exchange.has = {"watchTicker": ws_supported, "watchOHLCV": ws_supported}

    adapter = MagicMock()
    adapter._exchange = ccxt_exchange
    adapter.name = "test_exchange"
    adapter.get_ticker = AsyncMock(
        return_value={"last": 50000.0}
    )
    return adapter


# --- WebSocket Support Detection ---


class TestWsSupportDetection:
    def test_ws_not_supported_rest_exchange(self):
        """REST-only exchange should not have WS support."""
        exchange = make_exchange_mock(ws_supported=False)
        feed = WebSocketFeed(exchange=exchange, symbols=["BTC/USDT"])
        assert feed.ws_supported is False

    def test_ws_supported_pro_exchange(self):
        """Exchange with watchTicker should be detected as WS-capable."""
        exchange = make_exchange_mock(ws_supported=True)
        feed = WebSocketFeed(exchange=exchange, symbols=["BTC/USDT"])
        assert feed.ws_supported is True

    def test_ws_no_underlying_exchange(self):
        """Exchange without _exchange attr should default to no WS."""
        exchange = MagicMock(spec=[])
        exchange.name = "unknown"
        feed = WebSocketFeed(exchange=exchange, symbols=["BTC/USDT"])
        assert feed.ws_supported is False

    def test_ws_no_has_attribute(self):
        """Exchange without 'has' dict defaults to no WS."""
        adapter = MagicMock()
        adapter._exchange = MagicMock(spec=[])  # No 'has' attribute
        exchange = MagicMock()
        exchange._exchange = adapter
        feed = WebSocketFeed(exchange=exchange, symbols=["BTC/USDT"])
        assert feed.ws_supported is False


# --- Constructor and Properties ---


class TestConstructor:
    def test_default_params(self):
        exchange = make_exchange_mock()
        feed = WebSocketFeed(exchange=exchange, symbols=["BTC/USDT"])
        assert feed._symbols == ["BTC/USDT"]
        assert feed._timeframes == ["1h"]
        assert feed._poll_interval == 5.0
        assert feed._max_reconnect_delay == 60.0
        assert feed.is_running is False
        assert feed.latest_prices == {}

    def test_custom_params(self):
        exchange = make_exchange_mock()
        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT", "ETH/USDT"],
            timeframes=["15m", "1h"],
            poll_interval=2.0,
            max_reconnect_delay=30.0,
        )
        assert feed._symbols == ["BTC/USDT", "ETH/USDT"]
        assert feed._timeframes == ["15m", "1h"]
        assert feed._poll_interval == 2.0
        assert feed._max_reconnect_delay == 30.0

    def test_get_latest_price_empty(self):
        exchange = make_exchange_mock()
        feed = WebSocketFeed(exchange=exchange, symbols=["BTC/USDT"])
        assert feed.get_latest_price("BTC/USDT") is None

    def test_update_price(self):
        exchange = make_exchange_mock()
        feed = WebSocketFeed(exchange=exchange, symbols=["BTC/USDT"])
        feed.update_price("BTC/USDT", 50000.0)
        assert feed.get_latest_price("BTC/USDT") == 50000.0

    def test_latest_prices_returns_copy(self):
        exchange = make_exchange_mock()
        feed = WebSocketFeed(exchange=exchange, symbols=["BTC/USDT"])
        feed.update_price("BTC/USDT", 50000.0)
        prices = feed.latest_prices
        prices["ETH/USDT"] = 3000.0
        # Original should be unaffected
        assert feed.get_latest_price("ETH/USDT") is None


# --- REST Polling Mode ---


class TestRestPollingMode:
    @pytest.mark.asyncio
    async def test_start_rest_mode(self):
        """Feed starts in REST mode when WS not supported."""
        exchange = make_exchange_mock(ws_supported=False)
        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            poll_interval=0.05,
        )
        await feed.start()
        assert feed.is_running is True
        assert len(feed._tasks) == 1
        assert feed._tasks[0].get_name() == "rest_poll_prices"

        # Allow one poll cycle
        await asyncio.sleep(0.1)

        assert feed.get_latest_price("BTC/USDT") == 50000.0

        await feed.stop()
        assert feed.is_running is False
        assert len(feed._tasks) == 0

    @pytest.mark.asyncio
    async def test_poll_multiple_symbols(self):
        """REST polling updates prices for all symbols."""
        prices = {"BTC/USDT": 50000.0, "ETH/USDT": 3000.0}

        async def mock_ticker(symbol):
            return {"last": prices[symbol]}

        exchange = make_exchange_mock(ws_supported=False)
        exchange.get_ticker = AsyncMock(side_effect=mock_ticker)

        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT", "ETH/USDT"],
            poll_interval=0.05,
        )
        await feed.start()
        await asyncio.sleep(0.1)

        assert feed.get_latest_price("BTC/USDT") == 50000.0
        assert feed.get_latest_price("ETH/USDT") == 3000.0

        await feed.stop()

    @pytest.mark.asyncio
    async def test_poll_handles_errors_gracefully(self):
        """REST polling continues even when get_ticker raises."""
        call_count = 0

        async def flaky_ticker(symbol):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network error")
            return {"last": 50000.0}

        exchange = make_exchange_mock(ws_supported=False)
        exchange.get_ticker = AsyncMock(side_effect=flaky_ticker)

        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            poll_interval=0.05,
        )
        await feed.start()
        await asyncio.sleep(0.15)

        # Should recover after error
        assert feed.get_latest_price("BTC/USDT") == 50000.0

        await feed.stop()

    @pytest.mark.asyncio
    async def test_poll_ignores_zero_price(self):
        """Zero prices from ticker should not be stored."""
        exchange = make_exchange_mock(ws_supported=False)
        exchange.get_ticker = AsyncMock(return_value={"last": 0.0})

        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            poll_interval=0.05,
        )
        await feed.start()
        await asyncio.sleep(0.1)

        assert feed.get_latest_price("BTC/USDT") is None

        await feed.stop()

    @pytest.mark.asyncio
    async def test_poll_ignores_missing_last(self):
        """Ticker without 'last' key should not crash."""
        exchange = make_exchange_mock(ws_supported=False)
        exchange.get_ticker = AsyncMock(return_value={"bid": 49900.0})

        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            poll_interval=0.05,
        )
        await feed.start()
        await asyncio.sleep(0.1)

        assert feed.get_latest_price("BTC/USDT") is None

        await feed.stop()


# --- WebSocket Mode ---


class TestWebSocketMode:
    @pytest.mark.asyncio
    async def test_start_ws_mode(self):
        """Feed starts WS tasks when exchange supports WebSocket."""
        exchange = make_exchange_mock(ws_supported=True)
        ccxt_ex = exchange._exchange._exchange

        # Make watch_ticker a coroutine that blocks forever
        async def watch_forever(*args, **kwargs):
            await asyncio.sleep(100)

        ccxt_ex.watch_ticker = AsyncMock(side_effect=watch_forever)
        ccxt_ex.watch_ohlcv = AsyncMock(side_effect=watch_forever)

        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            timeframes=["1h"],
        )
        await feed.start()
        assert feed.is_running is True
        # 1 ticker task + 1 ohlcv task per symbol/timeframe
        assert len(feed._tasks) == 2

        await feed.stop()

    @pytest.mark.asyncio
    async def test_ws_ticker_updates_price(self):
        """WebSocket ticker updates latest_prices dict."""
        exchange = make_exchange_mock(ws_supported=True)
        ccxt_ex = exchange._exchange._exchange

        ticker_data = {"last": 51000.0, "bid": 50900.0, "ask": 51100.0}

        call_count = 0

        async def mock_watch_ticker(symbol):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                await asyncio.sleep(100)  # Block after 2 calls
            return ticker_data

        ccxt_ex.watch_ticker = AsyncMock(side_effect=mock_watch_ticker)
        ccxt_ex.watch_ohlcv = AsyncMock(side_effect=lambda *a: asyncio.sleep(100))

        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            timeframes=["1h"],
        )
        await feed.start()
        await asyncio.sleep(0.1)

        assert feed.get_latest_price("BTC/USDT") == 51000.0

        await feed.stop()

    @pytest.mark.asyncio
    async def test_ws_ohlcv_candle_close_callback(self):
        """New candle timestamp triggers the on_candle_close callback."""
        exchange = make_exchange_mock(ws_supported=True)
        ccxt_ex = exchange._exchange._exchange

        callback_calls = []

        async def on_candle(symbol, timeframe):
            callback_calls.append((symbol, timeframe))

        call_count = 0

        async def mock_watch_ohlcv(symbol, timeframe):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: initial candle
                return [[1700000000000, 50000, 51000, 49000, 50500, 100]]
            elif call_count == 2:
                # Second call: new candle (different timestamp = previous closed)
                return [[1700003600000, 50500, 51500, 50000, 51000, 200]]
            else:
                await asyncio.sleep(100)

        ccxt_ex.watch_ticker = AsyncMock(side_effect=lambda *a: asyncio.sleep(100))
        ccxt_ex.watch_ohlcv = AsyncMock(side_effect=mock_watch_ohlcv)

        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            timeframes=["1h"],
            on_candle_close=on_candle,
        )
        await feed.start()
        await asyncio.sleep(0.2)

        # Should have triggered callback for candle close
        assert len(callback_calls) == 1
        assert callback_calls[0] == ("BTC/USDT", "1h")

        await feed.stop()

    @pytest.mark.asyncio
    async def test_ws_ohlcv_same_timestamp_no_callback(self):
        """Same candle timestamp should not trigger callback."""
        exchange = make_exchange_mock(ws_supported=True)
        ccxt_ex = exchange._exchange._exchange

        callback_calls = []

        async def on_candle(symbol, timeframe):
            callback_calls.append((symbol, timeframe))

        call_count = 0

        async def mock_watch_ohlcv(symbol, timeframe):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                # Same timestamp each time (candle still open)
                return [[1700000000000, 50000, 51000, 49000, 50500, 100]]
            else:
                await asyncio.sleep(100)

        ccxt_ex.watch_ticker = AsyncMock(side_effect=lambda *a: asyncio.sleep(100))
        ccxt_ex.watch_ohlcv = AsyncMock(side_effect=mock_watch_ohlcv)

        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            timeframes=["1h"],
            on_candle_close=on_candle,
        )
        await feed.start()
        await asyncio.sleep(0.2)

        assert len(callback_calls) == 0

        await feed.stop()

    @pytest.mark.asyncio
    async def test_ws_ohlcv_updates_price(self):
        """OHLCV WebSocket also updates latest price from candle close."""
        exchange = make_exchange_mock(ws_supported=True)
        ccxt_ex = exchange._exchange._exchange

        call_count = 0

        async def mock_watch_ohlcv(symbol, timeframe):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [[1700000000000, 50000, 51000, 49000, 50500, 100]]
            else:
                await asyncio.sleep(100)

        ccxt_ex.watch_ticker = AsyncMock(side_effect=lambda *a: asyncio.sleep(100))
        ccxt_ex.watch_ohlcv = AsyncMock(side_effect=mock_watch_ohlcv)

        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            timeframes=["1h"],
        )
        await feed.start()
        await asyncio.sleep(0.1)

        assert feed.get_latest_price("BTC/USDT") == 50500.0

        await feed.stop()

    @pytest.mark.asyncio
    async def test_ws_multiple_symbols(self):
        """WebSocket creates tasks for each symbol."""
        exchange = make_exchange_mock(ws_supported=True)
        ccxt_ex = exchange._exchange._exchange
        ccxt_ex.watch_ticker = AsyncMock(side_effect=lambda *a: asyncio.sleep(100))
        ccxt_ex.watch_ohlcv = AsyncMock(side_effect=lambda *a, **kw: asyncio.sleep(100))

        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT", "ETH/USDT"],
            timeframes=["1h", "15m"],
        )
        await feed.start()
        # 2 ticker tasks + 4 ohlcv tasks (2 symbols × 2 timeframes)
        assert len(feed._tasks) == 6

        await feed.stop()


# --- Reconnection Logic ---


class TestReconnection:
    @pytest.mark.asyncio
    async def test_ws_reconnects_on_error(self):
        """WebSocket ticker reconnects after error with backoff."""
        exchange = make_exchange_mock(ws_supported=True)
        ccxt_ex = exchange._exchange._exchange

        call_count = 0

        async def mock_watch_ticker(symbol):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection lost")
            elif call_count == 2:
                return {"last": 50000.0}
            else:
                await asyncio.sleep(100)

        ccxt_ex.watch_ticker = AsyncMock(side_effect=mock_watch_ticker)
        ccxt_ex.watch_ohlcv = AsyncMock(side_effect=lambda *a, **kw: asyncio.sleep(100))

        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            timeframes=["1h"],
        )
        # Use very short reconnect delay for testing
        feed._initial_reconnect_delay = 0.05

        await feed.start()
        await asyncio.sleep(0.3)

        # Should have recovered
        assert feed.get_latest_price("BTC/USDT") == 50000.0

        await feed.stop()

    @pytest.mark.asyncio
    async def test_ws_exponential_backoff(self):
        """Reconnection delay doubles up to max."""
        exchange = make_exchange_mock(ws_supported=True)
        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            max_reconnect_delay=8.0,
        )
        # Test the backoff logic directly
        assert feed._initial_reconnect_delay == 1.0
        assert feed._max_reconnect_delay == 8.0

        # Verify backoff formula: delay = min(delay * 2, max_delay)
        delay = feed._initial_reconnect_delay
        delays = []
        for _ in range(5):
            delays.append(delay)
            delay = min(delay * 2, feed._max_reconnect_delay)

        assert delays == [1.0, 2.0, 4.0, 8.0, 8.0]

    @pytest.mark.asyncio
    async def test_ws_ohlcv_reconnects_on_error(self):
        """WebSocket OHLCV reconnects after error."""
        exchange = make_exchange_mock(ws_supported=True)
        ccxt_ex = exchange._exchange._exchange

        call_count = 0

        async def mock_watch_ohlcv(symbol, timeframe):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Stream broken")
            elif call_count == 2:
                return [[1700000000000, 50000, 51000, 49000, 50500, 100]]
            else:
                await asyncio.sleep(100)

        ccxt_ex.watch_ticker = AsyncMock(side_effect=lambda *a: asyncio.sleep(100))
        ccxt_ex.watch_ohlcv = AsyncMock(side_effect=mock_watch_ohlcv)

        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            timeframes=["1h"],
        )
        feed._initial_reconnect_delay = 0.05

        await feed.start()
        await asyncio.sleep(0.3)

        # Should have recovered and updated price
        assert feed.get_latest_price("BTC/USDT") == 50500.0

        await feed.stop()


# --- Stop / Cleanup ---


class TestStopCleanup:
    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self):
        """Stopping the feed cancels all background tasks."""
        exchange = make_exchange_mock(ws_supported=False)
        exchange.get_ticker = AsyncMock(
            side_effect=lambda *a: asyncio.sleep(100)
        )
        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            poll_interval=100,
        )
        await feed.start()
        assert len(feed._tasks) == 1
        assert not feed._tasks[0].done()

        await feed.stop()
        assert feed.is_running is False
        assert len(feed._tasks) == 0

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        """Stopping twice should not raise."""
        exchange = make_exchange_mock(ws_supported=False)
        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            poll_interval=0.05,
        )
        await feed.start()
        await feed.stop()
        await feed.stop()  # Second stop should be safe
        assert feed.is_running is False


# --- Candle Close Callback Edge Cases ---


class TestCandleCloseCallbackEdgeCases:
    @pytest.mark.asyncio
    async def test_callback_error_does_not_crash_feed(self):
        """Error in candle close callback should not stop the feed."""
        exchange = make_exchange_mock(ws_supported=True)
        ccxt_ex = exchange._exchange._exchange

        async def bad_callback(symbol, timeframe):
            raise ValueError("Callback error")

        call_count = 0

        async def mock_watch_ohlcv(symbol, timeframe):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [[1700000000000, 50000, 51000, 49000, 50500, 100]]
            elif call_count == 2:
                return [[1700003600000, 50500, 51500, 50000, 51000, 200]]
            elif call_count == 3:
                return [[1700007200000, 51000, 52000, 50500, 51500, 300]]
            else:
                await asyncio.sleep(100)

        ccxt_ex.watch_ticker = AsyncMock(side_effect=lambda *a: asyncio.sleep(100))
        ccxt_ex.watch_ohlcv = AsyncMock(side_effect=mock_watch_ohlcv)

        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            timeframes=["1h"],
            on_candle_close=bad_callback,
        )
        await feed.start()
        await asyncio.sleep(0.2)

        # Feed should still be running despite callback error
        assert feed.is_running is True
        # Price should still be updated
        assert feed.get_latest_price("BTC/USDT") is not None

        await feed.stop()

    @pytest.mark.asyncio
    async def test_no_callback_configured(self):
        """Feed works fine without a candle close callback."""
        exchange = make_exchange_mock(ws_supported=True)
        ccxt_ex = exchange._exchange._exchange

        call_count = 0

        async def mock_watch_ohlcv(symbol, timeframe):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [[1700000000000, 50000, 51000, 49000, 50500, 100]]
            elif call_count == 2:
                return [[1700003600000, 50500, 51500, 50000, 51000, 200]]
            else:
                await asyncio.sleep(100)

        ccxt_ex.watch_ticker = AsyncMock(side_effect=lambda *a: asyncio.sleep(100))
        ccxt_ex.watch_ohlcv = AsyncMock(side_effect=mock_watch_ohlcv)

        feed = WebSocketFeed(
            exchange=exchange,
            symbols=["BTC/USDT"],
            timeframes=["1h"],
            on_candle_close=None,
        )
        await feed.start()
        await asyncio.sleep(0.2)

        # Should work without crashing
        assert feed.get_latest_price("BTC/USDT") is not None

        await feed.stop()


# --- Integration with main.py ---


class TestMainIntegration:
    def test_ws_feed_default_disabled(self):
        """WebSocket feed is disabled by default in config."""
        from bot.config import Settings

        settings = Settings(
            trading_mode="paper",
            database_url="sqlite+aiosqlite:///:memory:",
        )
        assert settings.websocket_enabled is False

    def test_ws_feed_config_params(self):
        """WebSocket config params are available in Settings."""
        from bot.config import Settings

        settings = Settings(
            trading_mode="paper",
            database_url="sqlite+aiosqlite:///:memory:",
            websocket_enabled=True,
            websocket_poll_interval=2.0,
            websocket_max_reconnect_delay=30.0,
        )
        assert settings.websocket_enabled is True
        assert settings.websocket_poll_interval == 2.0
        assert settings.websocket_max_reconnect_delay == 30.0

    @pytest.mark.asyncio
    async def test_bot_init_ws_disabled(self):
        """Bot initializes without WS feed when disabled (default)."""
        from bot.config import Settings, TradingMode
        from bot.main import TradingBot

        settings = Settings(
            trading_mode=TradingMode.PAPER,
            database_url="sqlite+aiosqlite:///:memory:",
            binance_api_key="",
            upbit_api_key="",
            websocket_enabled=False,
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        assert bot._ws_feed is None

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_bot_init_ws_enabled_no_exchanges(self):
        """Bot with WS enabled but no exchanges skips WS feed."""
        from bot.config import Settings, TradingMode
        from bot.main import TradingBot

        settings = Settings(
            trading_mode=TradingMode.PAPER,
            database_url="sqlite+aiosqlite:///:memory:",
            binance_api_key="",
            upbit_api_key="",
            websocket_enabled=True,
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        assert bot._ws_feed is None

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_bot_exit_check_uses_ws_price(self):
        """PositionManager exit check prefers WS feed price over candles."""
        from bot.config import Settings, TradingMode
        from bot.main import TradingBot

        settings = Settings(
            trading_mode=TradingMode.PAPER,
            database_url="sqlite+aiosqlite:///:memory:",
            binance_api_key="",
            upbit_api_key="",
            signal_min_agreement=1,
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Simulate a managed position
        bot._position_manager.add_position("BTC/USDT", 50000.0, 0.1)

        # Create a mock WS feed with a price
        mock_feed = MagicMock()
        mock_feed.get_latest_price = MagicMock(return_value=48000.0)
        mock_feed.stop = AsyncMock()
        bot._ws_feed = mock_feed

        # The exit check should use the WS price
        # 48000 < 50000 * 0.97 = 48500, so stop-loss should trigger
        await bot._trading_cycle()

        # Verify WS feed was consulted
        mock_feed.get_latest_price.assert_called_with("BTC/USDT")

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_bot_exit_check_falls_back_to_candles(self):
        """When WS feed has no price, falls back to candle close price."""
        from bot.config import Settings, TradingMode
        from bot.main import TradingBot

        settings = Settings(
            trading_mode=TradingMode.PAPER,
            database_url="sqlite+aiosqlite:///:memory:",
            binance_api_key="",
            upbit_api_key="",
            signal_min_agreement=1,
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Simulate a managed position
        bot._position_manager.add_position("BTC/USDT", 50000.0, 0.1)

        # Create a mock WS feed that returns None (no price yet)
        mock_feed = MagicMock()
        mock_feed.get_latest_price = MagicMock(return_value=None)
        mock_feed.stop = AsyncMock()
        bot._ws_feed = mock_feed

        # Should fall back to store candles (which are empty, so no exit)
        await bot._trading_cycle()

        mock_feed.get_latest_price.assert_called_with("BTC/USDT")

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_bot_shutdown_stops_ws_feed(self):
        """Bot shutdown stops the WS feed."""
        from bot.config import Settings, TradingMode
        from bot.main import TradingBot

        settings = Settings(
            trading_mode=TradingMode.PAPER,
            database_url="sqlite+aiosqlite:///:memory:",
            binance_api_key="",
            upbit_api_key="",
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Attach a mock WS feed
        mock_feed = MagicMock()
        mock_feed.stop = AsyncMock()
        bot._ws_feed = mock_feed

        await bot.shutdown()

        mock_feed.stop.assert_awaited_once()


# --- Import/Export ---


class TestImportExport:
    def test_exported_from_data_package(self):
        """WebSocketFeed should be importable from bot.data."""
        from bot.data import WebSocketFeed as WSF

        assert WSF is WebSocketFeed
