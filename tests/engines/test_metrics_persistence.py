"""Tests for MetricsPersistence — save/restore engine trades and metrics."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.engines.tracker import EngineTracker, TradeRecord

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _sample_trade(
    engine: str = "funding_rate_arb",
    symbol: str = "BTC/USDT",
    pnl: float = 50.0,
    exit_time: str = "2026-02-23T10:00:00Z",
) -> TradeRecord:
    return TradeRecord(
        engine_name=engine,
        symbol=symbol,
        side="long",
        entry_price=40000,
        exit_price=40500,
        quantity=0.1,
        pnl=pnl + 2.0,  # gross = net + cost
        cost=2.0,
        net_pnl=pnl,
        entry_time="2026-02-23T08:00:00Z",
        exit_time=exit_time,
        hold_time_seconds=7200,
    )


def _mock_data_store():
    """Create mock DataStore with async session factory."""
    store = MagicMock()
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()

    # Context manager support
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    store._session_factory.return_value = session
    return store, session


# ──────────────────────────────────────────────────────────────
# save_trade
# ──────────────────────────────────────────────────────────────


class TestSaveTrade:
    @pytest.mark.asyncio
    async def test_save_trade_success(self):
        from bot.engines.metrics_persistence import MetricsPersistence

        store, session = _mock_data_store()
        tracker = EngineTracker()
        persistence = MetricsPersistence(data_store=store, tracker=tracker)

        trade = _sample_trade()
        await persistence.save_trade("funding_rate_arb", trade)

        session.add.assert_called_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_trade_error_handled(self):
        from bot.engines.metrics_persistence import MetricsPersistence

        store, session = _mock_data_store()
        session.commit = AsyncMock(side_effect=Exception("DB error"))
        tracker = EngineTracker()
        persistence = MetricsPersistence(data_store=store, tracker=tracker)

        trade = _sample_trade()
        # Should not raise
        await persistence.save_trade("funding_rate_arb", trade)


# ──────────────────────────────────────────────────────────────
# save_metrics_snapshot
# ──────────────────────────────────────────────────────────────


class TestSaveMetricsSnapshot:
    @pytest.mark.asyncio
    async def test_snapshot_saves_all_engines(self):
        from bot.engines.metrics_persistence import MetricsPersistence

        store, session = _mock_data_store()
        tracker = EngineTracker()
        tracker.record_trade("engine_a", _sample_trade("engine_a"))
        tracker.record_trade("engine_b", _sample_trade("engine_b"))

        persistence = MetricsPersistence(data_store=store, tracker=tracker)
        await persistence.save_metrics_snapshot()

        # 2 engines → 2 snapshot records added
        assert session.add.call_count == 2
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_snapshot_empty_tracker(self):
        from bot.engines.metrics_persistence import MetricsPersistence

        store, session = _mock_data_store()
        tracker = EngineTracker()

        persistence = MetricsPersistence(data_store=store, tracker=tracker)
        await persistence.save_metrics_snapshot()

        # No engines → no records
        assert session.add.call_count == 0

    @pytest.mark.asyncio
    async def test_snapshot_error_handled(self):
        from bot.engines.metrics_persistence import MetricsPersistence

        store, session = _mock_data_store()
        session.commit = AsyncMock(side_effect=Exception("DB error"))
        tracker = EngineTracker()
        tracker.record_trade("engine_a", _sample_trade("engine_a"))

        persistence = MetricsPersistence(data_store=store, tracker=tracker)
        # Should not raise
        await persistence.save_metrics_snapshot()


# ──────────────────────────────────────────────────────────────
# load_trades
# ──────────────────────────────────────────────────────────────


class TestLoadTrades:
    @pytest.mark.asyncio
    async def test_load_trades_success(self):
        from bot.engines.metrics_persistence import MetricsPersistence

        store, session = _mock_data_store()
        tracker = EngineTracker()

        # Mock DB rows
        row = MagicMock()
        row.engine_name = "funding_rate_arb"
        row.symbol = "BTC/USDT"
        row.side = "long"
        row.entry_price = 40000
        row.exit_price = 40500
        row.quantity = 0.1
        row.pnl = 52.0
        row.cost = 2.0
        row.net_pnl = 50.0
        row.entry_time = "2026-02-23T08:00:00Z"
        row.exit_time = "2026-02-23T10:00:00Z"
        row.hold_time_seconds = 7200

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [row]
        session.execute = AsyncMock(return_value=result_mock)

        persistence = MetricsPersistence(data_store=store, tracker=tracker)
        trades = await persistence.load_trades(since_hours=24)

        assert "funding_rate_arb" in trades
        assert len(trades["funding_rate_arb"]) == 1
        loaded = trades["funding_rate_arb"][0]
        assert loaded.symbol == "BTC/USDT"
        assert loaded.net_pnl == 50.0

    @pytest.mark.asyncio
    async def test_load_trades_empty(self):
        from bot.engines.metrics_persistence import MetricsPersistence

        store, session = _mock_data_store()
        tracker = EngineTracker()

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result_mock)

        persistence = MetricsPersistence(data_store=store, tracker=tracker)
        trades = await persistence.load_trades()
        assert trades == {}

    @pytest.mark.asyncio
    async def test_load_trades_error(self):
        from bot.engines.metrics_persistence import MetricsPersistence

        store, session = _mock_data_store()
        session.execute = AsyncMock(side_effect=Exception("DB error"))
        tracker = EngineTracker()

        persistence = MetricsPersistence(data_store=store, tracker=tracker)
        trades = await persistence.load_trades()
        assert trades == {}


# ──────────────────────────────────────────────────────────────
# restore_tracker
# ──────────────────────────────────────────────────────────────


class TestRestoreTracker:
    @pytest.mark.asyncio
    async def test_restore_populates_tracker(self):
        from bot.engines.metrics_persistence import MetricsPersistence

        store, session = _mock_data_store()
        tracker = EngineTracker()

        # Mock DB rows
        row = MagicMock()
        row.engine_name = "grid_trading"
        row.symbol = "ETH/USDT"
        row.side = "buy"
        row.entry_price = 2500
        row.exit_price = 2550
        row.quantity = 1.0
        row.pnl = 51.0
        row.cost = 1.0
        row.net_pnl = 50.0
        row.entry_time = "2026-02-23T08:00:00Z"
        row.exit_time = "2026-02-23T10:00:00Z"
        row.hold_time_seconds = 7200

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [row]
        session.execute = AsyncMock(return_value=result_mock)

        persistence = MetricsPersistence(data_store=store, tracker=tracker)
        await persistence.restore_tracker()

        assert "grid_trading" in tracker._trades
        assert len(tracker._trades["grid_trading"]) == 1
        assert tracker._trades["grid_trading"][0].net_pnl == 50.0

    @pytest.mark.asyncio
    async def test_restore_empty_db(self):
        from bot.engines.metrics_persistence import MetricsPersistence

        store, session = _mock_data_store()
        tracker = EngineTracker()

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result_mock)

        persistence = MetricsPersistence(data_store=store, tracker=tracker)
        await persistence.restore_tracker()

        assert tracker._trades == {}


# ──────────────────────────────────────────────────────────────
# cleanup
# ──────────────────────────────────────────────────────────────


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_executes_deletes(self):
        from bot.engines.metrics_persistence import MetricsPersistence

        store, session = _mock_data_store()
        tracker = EngineTracker()

        persistence = MetricsPersistence(data_store=store, tracker=tracker)
        await persistence.cleanup(max_days=90)

        # 2 delete statements (trades + snapshots) + 1 commit
        assert session.execute.await_count == 2
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_error_handled(self):
        from bot.engines.metrics_persistence import MetricsPersistence

        store, session = _mock_data_store()
        session.execute = AsyncMock(side_effect=Exception("DB error"))
        tracker = EngineTracker()

        persistence = MetricsPersistence(data_store=store, tracker=tracker)
        # Should not raise
        await persistence.cleanup()


# ──────────────────────────────────────────────────────────────
# bulk_load_trades (EngineTracker)
# ──────────────────────────────────────────────────────────────


class TestBulkLoadTrades:
    def test_bulk_load_populates_trades(self):
        tracker = EngineTracker()
        trades = [
            _sample_trade("engine_a", pnl=10.0),
            _sample_trade("engine_a", pnl=20.0),
        ]
        tracker.bulk_load_trades("engine_a", trades)
        assert len(tracker._trades["engine_a"]) == 2

    def test_bulk_load_updates_pnl_history(self):
        tracker = EngineTracker()
        trades = [
            _sample_trade("engine_a", pnl=10.0),
            _sample_trade("engine_a", pnl=20.0),
        ]
        tracker.bulk_load_trades("engine_a", trades)
        history = tracker._pnl_history["engine_a"]
        assert len(history) == 2
        assert history[-1]["cumulative_pnl"] == 30.0

    def test_bulk_load_appends_to_existing(self):
        tracker = EngineTracker()
        tracker.record_trade(
            "engine_a", _sample_trade("engine_a", pnl=5.0)
        )
        assert len(tracker._trades["engine_a"]) == 1

        tracker.bulk_load_trades(
            "engine_a",
            [_sample_trade("engine_a", pnl=15.0)],
        )
        assert len(tracker._trades["engine_a"]) == 2
        history = tracker._pnl_history["engine_a"]
        assert history[-1]["cumulative_pnl"] == 20.0

    def test_bulk_load_empty_list(self):
        tracker = EngineTracker()
        tracker.bulk_load_trades("engine_a", [])
        assert tracker._trades.get("engine_a", []) == []


# ──────────────────────────────────────────────────────────────
# _snapshot_loop
# ──────────────────────────────────────────────────────────────


class TestSnapshotLoop:
    @pytest.mark.asyncio
    async def test_snapshot_loop_calls_save(self):
        from bot.engines.metrics_persistence import MetricsPersistence

        store, _ = _mock_data_store()
        tracker = EngineTracker()
        persistence = MetricsPersistence(data_store=store, tracker=tracker)

        call_count = 0

        async def mock_save():
            nonlocal call_count
            call_count += 1

        persistence.save_metrics_snapshot = mock_save

        with patch("bot.engines.metrics_persistence.asyncio") as mock_asyncio:
            # Make sleep raise CancelledError after first call
            sleep_calls = 0

            async def mock_sleep(seconds):
                nonlocal sleep_calls
                sleep_calls += 1
                if sleep_calls > 1:
                    raise asyncio.CancelledError()

            import asyncio

            mock_asyncio.sleep = mock_sleep
            mock_asyncio.CancelledError = asyncio.CancelledError

            with pytest.raises(asyncio.CancelledError):
                await persistence._snapshot_loop(interval_minutes=1.0)

            assert call_count >= 1


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────


class TestConfig:
    def test_config_defaults(self):
        from bot.config import Settings

        s = Settings(
            _env_file=None,
            exchange="binance",
            api_key="test",
            api_secret="test",
        )
        assert s.metrics_persistence_enabled is True
        assert s.metrics_snapshot_interval_minutes == 5.0
        assert s.metrics_retention_days == 90

    def test_settings_metadata(self):
        from bot.config import SETTINGS_METADATA

        assert "metrics_persistence_enabled" in SETTINGS_METADATA
        assert "metrics_snapshot_interval_minutes" in SETTINGS_METADATA
        assert "metrics_retention_days" in SETTINGS_METADATA
        for key in [
            "metrics_persistence_enabled",
            "metrics_snapshot_interval_minutes",
            "metrics_retention_days",
        ]:
            assert SETTINGS_METADATA[key]["section"] == "Metrics"


# ──────────────────────────────────────────────────────────────
# EngineManager integration
# ──────────────────────────────────────────────────────────────


class TestEngineManagerIntegration:
    def test_set_metrics_persistence(self):
        from bot.engines.manager import EngineManager
        from bot.engines.portfolio_manager import PortfolioManager

        pm = PortfolioManager(total_capital=10000)
        mgr = EngineManager(pm)

        mock_persistence = MagicMock()
        mgr.set_metrics_persistence(mock_persistence)
        assert mgr._metrics_persistence is mock_persistence

    def test_default_none(self):
        from bot.engines.manager import EngineManager
        from bot.engines.portfolio_manager import PortfolioManager

        pm = PortfolioManager(total_capital=10000)
        mgr = EngineManager(pm)
        assert mgr._metrics_persistence is None


# ──────────────────────────────────────────────────────────────
# DB Models
# ──────────────────────────────────────────────────────────────


class TestDBModels:
    def test_engine_trade_record_import(self):
        from bot.data.models import EngineTradeRecord

        assert EngineTradeRecord.__tablename__ == "engine_trades"

    def test_engine_metric_snapshot_import(self):
        from bot.data.models import EngineMetricSnapshot

        assert EngineMetricSnapshot.__tablename__ == "engine_metric_snapshots"
