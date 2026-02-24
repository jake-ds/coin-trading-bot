"""Tests for V6-002: DataCollector batch backfill + scanner auto-discovery."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.data.collector import DataCollector
from bot.models import OHLCV


def _make_candles(n: int, symbol: str = "BTC/USDT") -> list[OHLCV]:
    """Create n mock OHLCV candles."""
    return [
        OHLCV(
            timestamp=datetime(2025, 1, 1, i, 0, tzinfo=timezone.utc),
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=1000.0,
            symbol=symbol,
            timeframe="1h",
        )
        for i in range(n)
    ]


def _make_collector(
    symbols: list[str] | None = None,
    exchange_candles: list[OHLCV] | None = None,
    store_candles: list[OHLCV] | None = None,
) -> tuple[DataCollector, MagicMock, MagicMock]:
    """Create a DataCollector with mock exchange and store."""
    mock_exchange = MagicMock()
    mock_exchange.name = "mock_exchange"
    mock_exchange.get_ohlcv = AsyncMock(return_value=exchange_candles or [])

    mock_store = MagicMock()
    mock_store.get_candles = AsyncMock(return_value=store_candles or [])
    mock_store.save_candles = AsyncMock()

    collector = DataCollector(
        exchanges=[mock_exchange],
        store=mock_store,
        symbols=symbols or ["BTC/USDT"],
    )
    return collector, mock_exchange, mock_store


class TestDynamicSymbols:
    def test_dynamic_symbols_initialized_empty(self):
        collector, _, _ = _make_collector()
        assert collector._dynamic_symbols == set()

    def test_dynamic_symbols_merged_in_collect_once(self):
        """collect_once() should iterate over both static and dynamic symbols."""
        collector, mock_exchange, mock_store = _make_collector(
            symbols=["BTC/USDT"]
        )
        collector._dynamic_symbols = {"ETH/USDT", "SOL/USDT"}

        # Run collect
        asyncio.get_event_loop().run_until_complete(collector.collect_once())

        # Exchange should be called for all 3 symbols
        symbols_called = [
            call.kwargs["symbol"]
            for call in mock_exchange.get_ohlcv.call_args_list
        ]
        assert "BTC/USDT" in symbols_called
        assert "ETH/USDT" in symbols_called
        assert "SOL/USDT" in symbols_called

    def test_no_duplicate_symbols(self):
        """If a dynamic symbol matches a static one, it shouldn't be duplicated."""
        collector, mock_exchange, _ = _make_collector(symbols=["BTC/USDT"])
        collector._dynamic_symbols = {"BTC/USDT", "ETH/USDT"}

        asyncio.get_event_loop().run_until_complete(collector.collect_once())

        symbols_called = [
            call.kwargs["symbol"]
            for call in mock_exchange.get_ohlcv.call_args_list
        ]
        assert symbols_called.count("BTC/USDT") == 1
        assert "ETH/USDT" in symbols_called


class TestBulkBackfill:
    @pytest.mark.asyncio
    async def test_backfills_multiple_symbols(self):
        collector, mock_exchange, mock_store = _make_collector()
        candles = _make_candles(10)
        mock_exchange.get_ohlcv = AsyncMock(return_value=candles)
        mock_store.get_candles = AsyncMock(return_value=[])

        total = await collector.bulk_backfill(
            symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            timeframe="1h",
            days=1,
        )

        assert total == 30  # 10 candles × 3 symbols
        assert mock_store.save_candles.call_count == 3

    @pytest.mark.asyncio
    async def test_empty_symbols_returns_zero(self):
        collector, _, _ = _make_collector()
        total = await collector.bulk_backfill(symbols=[], timeframe="1h", days=1)
        assert total == 0

    @pytest.mark.asyncio
    async def test_no_exchanges_returns_zero(self):
        collector = DataCollector(
            exchanges=[], store=MagicMock(), symbols=["BTC/USDT"]
        )
        total = await collector.bulk_backfill(
            symbols=["BTC/USDT"], timeframe="1h", days=1
        )
        assert total == 0

    @pytest.mark.asyncio
    async def test_rate_limit_delay(self):
        """bulk_backfill should insert delays between symbols."""
        collector, mock_exchange, mock_store = _make_collector()
        mock_exchange.get_ohlcv = AsyncMock(return_value=_make_candles(5))
        mock_store.get_candles = AsyncMock(return_value=[])

        with patch("bot.data.collector.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await collector.bulk_backfill(
                symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                timeframe="1h",
                days=1,
            )

            # Should have rate-limit sleeps between symbols (n-1 times)
            sleep_calls = [
                c for c in mock_sleep.call_args_list if c.args[0] == 0.5
            ]
            assert len(sleep_calls) == 2  # 3 symbols → 2 delays

    @pytest.mark.asyncio
    async def test_limit_calculation_1h(self):
        """1h timeframe, 30 days → 24*30 = 720 candles limit."""
        collector, mock_exchange, mock_store = _make_collector()
        mock_exchange.get_ohlcv = AsyncMock(return_value=[])
        mock_store.get_candles = AsyncMock(return_value=[])

        await collector.bulk_backfill(
            symbols=["BTC/USDT"], timeframe="1h", days=30
        )

        # backfill() is called with limit=720
        call_kwargs = mock_exchange.get_ohlcv.call_args[1]
        assert call_kwargs["limit"] == 720

    @pytest.mark.asyncio
    async def test_limit_calculation_4h(self):
        """4h timeframe, 10 days → 6*10 = 60 candles limit."""
        collector, mock_exchange, mock_store = _make_collector()
        mock_exchange.get_ohlcv = AsyncMock(return_value=[])
        mock_store.get_candles = AsyncMock(return_value=[])

        await collector.bulk_backfill(
            symbols=["BTC/USDT"], timeframe="4h", days=10
        )

        call_kwargs = mock_exchange.get_ohlcv.call_args[1]
        assert call_kwargs["limit"] == 60

    @pytest.mark.asyncio
    async def test_error_in_one_symbol_continues(self):
        """If one symbol fails, others should still be processed."""
        collector, mock_exchange, mock_store = _make_collector()
        call_count = 0

        async def _side_effect(symbol, timeframe="1h", limit=500):
            nonlocal call_count
            call_count += 1
            if symbol == "BAD/USDT":
                raise ConnectionError("API error")
            return _make_candles(5, symbol=symbol)

        mock_exchange.get_ohlcv = AsyncMock(side_effect=_side_effect)
        mock_store.get_candles = AsyncMock(return_value=[])

        total = await collector.bulk_backfill(
            symbols=["BTC/USDT", "BAD/USDT", "ETH/USDT"],
            timeframe="1h",
            days=1,
        )

        # BTC and ETH should succeed (5 each), BAD should fail
        assert total == 10
        assert mock_store.save_candles.call_count == 2


class TestAutoDiscoverSymbols:
    def test_discovers_new_symbols(self):
        collector, _, _ = _make_collector(symbols=["BTC/USDT"])

        mock_registry = MagicMock()
        mock_registry.get_symbols = MagicMock(
            side_effect=lambda op_type, n=10, min_score=0.0: (
                ["ETH/USDT", "SOL/USDT"]
                if op_type.value == "funding_rate"
                else []
            )
        )

        new = collector.auto_discover_symbols(mock_registry, min_score=30.0)

        assert "ETH/USDT" in new
        assert "SOL/USDT" in new
        assert "ETH/USDT" in collector._dynamic_symbols
        assert "SOL/USDT" in collector._dynamic_symbols

    def test_no_duplicates_with_static_symbols(self):
        collector, _, _ = _make_collector(symbols=["BTC/USDT"])

        mock_registry = MagicMock()
        mock_registry.get_symbols = MagicMock(
            return_value=["BTC/USDT", "ETH/USDT"]
        )

        new = collector.auto_discover_symbols(mock_registry)

        # BTC/USDT already in static — should not be in new
        assert "BTC/USDT" not in new
        assert "ETH/USDT" in new

    def test_no_duplicates_with_dynamic_symbols(self):
        collector, _, _ = _make_collector(symbols=["BTC/USDT"])
        collector._dynamic_symbols = {"ETH/USDT"}

        mock_registry = MagicMock()
        mock_registry.get_symbols = MagicMock(
            return_value=["ETH/USDT", "SOL/USDT"]
        )

        new = collector.auto_discover_symbols(mock_registry)

        # ETH/USDT already dynamic — should not appear again
        assert "ETH/USDT" not in new
        assert "SOL/USDT" in new
        assert len(collector._dynamic_symbols) == 2

    def test_empty_registry_returns_empty(self):
        collector, _, _ = _make_collector()

        mock_registry = MagicMock()
        mock_registry.get_symbols = MagicMock(return_value=[])

        new = collector.auto_discover_symbols(mock_registry)
        assert new == []

    def test_min_score_passed_to_registry(self):
        collector, _, _ = _make_collector()

        mock_registry = MagicMock()
        mock_registry.get_symbols = MagicMock(return_value=[])

        collector.auto_discover_symbols(mock_registry, min_score=50.0)

        for call in mock_registry.get_symbols.call_args_list:
            assert call.kwargs["min_score"] == 50.0


class TestBackfillLoop:
    @pytest.mark.asyncio
    async def test_loop_discovers_and_backfills(self):
        """_backfill_loop should discover symbols and backfill them."""
        collector, mock_exchange, mock_store = _make_collector()
        mock_exchange.get_ohlcv = AsyncMock(return_value=_make_candles(5))
        mock_store.get_candles = AsyncMock(return_value=[])

        mock_registry = MagicMock()
        mock_registry.get_symbols = MagicMock(return_value=["ETH/USDT"])

        mock_settings = MagicMock()
        mock_settings.data_backfill_enabled = True
        mock_settings.data_backfill_interval_hours = 6.0
        mock_settings.data_backfill_days = 30

        # Patch sleep to avoid waiting, break after first iteration
        call_count = 0

        async def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                raise asyncio.CancelledError()

        with patch("bot.data.collector.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await collector._backfill_loop(
                    registry=mock_registry, settings=mock_settings
                )

        # Symbols should have been discovered
        assert "ETH/USDT" in collector._dynamic_symbols

    @pytest.mark.asyncio
    async def test_loop_skips_when_disabled(self):
        """When data_backfill_enabled=False, loop should skip work."""
        collector, mock_exchange, _ = _make_collector()

        mock_settings = MagicMock()
        mock_settings.data_backfill_enabled = False
        mock_settings.data_backfill_interval_hours = 0.001

        call_count = 0

        async def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                raise asyncio.CancelledError()

        with patch("bot.data.collector.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await collector._backfill_loop(
                    registry=None, settings=mock_settings
                )

        # Exchange should not have been called (no discover or backfill)
        mock_exchange.get_ohlcv.assert_not_called()

    @pytest.mark.asyncio
    async def test_loop_handles_errors(self):
        """Loop should not crash on errors."""
        collector, mock_exchange, _ = _make_collector()
        collector._dynamic_symbols = {"ERR/USDT"}
        mock_exchange.get_ohlcv = AsyncMock(side_effect=RuntimeError("API down"))

        mock_store = MagicMock()
        mock_store.get_candles = AsyncMock(return_value=[])
        collector._store = mock_store

        mock_settings = MagicMock()
        mock_settings.data_backfill_enabled = True
        mock_settings.data_backfill_interval_hours = 0.001

        call_count = 0

        async def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                raise asyncio.CancelledError()

        with patch("bot.data.collector.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await collector._backfill_loop(settings=mock_settings)

        # Should not crash — loop continues despite errors


class TestConfigFields:
    def test_default_backfill_settings(self):
        from bot.config import Settings

        s = Settings()
        assert s.data_backfill_enabled is True
        assert s.data_backfill_interval_hours == 6.0
        assert s.data_backfill_days == 30

    def test_settings_metadata_exists(self):
        from bot.config import SETTINGS_METADATA

        assert "data_backfill_enabled" in SETTINGS_METADATA
        assert "data_backfill_interval_hours" in SETTINGS_METADATA
        assert "data_backfill_days" in SETTINGS_METADATA

        for key in (
            "data_backfill_enabled",
            "data_backfill_interval_hours",
            "data_backfill_days",
        ):
            meta = SETTINGS_METADATA[key]
            assert meta["section"] == "Data Collection"
            assert meta["requires_restart"] is False


class TestEngineManagerBackfillWiring:
    def test_set_collector(self):
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        em = EngineManager(portfolio_manager=pm)

        mock_collector = MagicMock()
        em.set_collector(mock_collector)

        assert em._collector is mock_collector

    @pytest.mark.asyncio
    async def test_start_background_loops_creates_backfill_task(self):
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        mock_settings = MagicMock()
        mock_settings.data_backfill_enabled = True
        mock_settings.tuner_enabled = False
        mock_settings.engine_rebalance_enabled = False
        mock_settings.research_enabled = False

        em = EngineManager(portfolio_manager=pm, settings=mock_settings)

        mock_collector = MagicMock()
        mock_collector._backfill_loop = AsyncMock()
        em.set_collector(mock_collector)

        await em.start_background_loops()

        assert em._backfill_task is not None
        # Clean up the task
        em._backfill_task.cancel()
        try:
            await em._backfill_task
        except asyncio.CancelledError:
            pass
