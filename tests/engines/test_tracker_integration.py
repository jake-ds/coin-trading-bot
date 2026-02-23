"""Tests for V5-006: EngineTracker integration in EngineManager + API."""

import pytest
from httpx import ASGITransport, AsyncClient

from bot.config import load_settings
from bot.dashboard.app import app, set_engine_manager, set_settings
from bot.engines.base import EngineCycleResult
from bot.engines.portfolio_manager import PortfolioManager
from bot.engines.tracker import EngineTracker, TradeRecord

# ------------------------------------------------------------------ #
# EngineManager tracker integration
# ------------------------------------------------------------------ #


class TestEngineManagerTracker:
    """Verify EngineManager creates tracker and records cycles/trades."""

    def test_manager_has_tracker(self):
        from bot.engines.manager import EngineManager

        pm = PortfolioManager(total_capital=10000)
        mgr = EngineManager(portfolio_manager=pm)
        assert isinstance(mgr.tracker, EngineTracker)

    def test_register_wires_cycle_callback(self):
        from bot.engines.funding_arb import FundingRateArbEngine
        from bot.engines.manager import EngineManager

        pm = PortfolioManager(
            total_capital=10000,
            engine_allocations={"funding_rate_arb": 0.3},
        )
        mgr = EngineManager(portfolio_manager=pm)
        engine = FundingRateArbEngine(
            portfolio_manager=pm, exchanges=[], paper_mode=True,
        )
        mgr.register(engine)
        # Callback should be set
        assert engine._on_cycle_complete is not None

    def test_cycle_callback_records_cycle(self):
        from bot.engines.funding_arb import FundingRateArbEngine
        from bot.engines.manager import EngineManager

        pm = PortfolioManager(
            total_capital=10000,
            engine_allocations={"funding_rate_arb": 0.3},
        )
        mgr = EngineManager(portfolio_manager=pm)
        engine = FundingRateArbEngine(
            portfolio_manager=pm, exchanges=[], paper_mode=True,
        )
        mgr.register(engine)

        # Simulate a cycle result
        result = EngineCycleResult(
            engine_name="funding_rate_arb",
            cycle_num=1,
            timestamp="2026-02-23T12:00:00Z",
            duration_ms=100.0,
            pnl_update=0.0,
        )
        engine._on_cycle_complete(result)
        assert len(mgr.tracker._cycles.get("funding_rate_arb", [])) == 1

    def test_cycle_with_pnl_records_trade(self):
        from bot.engines.funding_arb import FundingRateArbEngine
        from bot.engines.manager import EngineManager

        pm = PortfolioManager(
            total_capital=10000,
            engine_allocations={"funding_rate_arb": 0.3},
        )
        mgr = EngineManager(portfolio_manager=pm)
        engine = FundingRateArbEngine(
            portfolio_manager=pm, exchanges=[], paper_mode=True,
        )
        mgr.register(engine)

        # Simulate a cycle with trade action
        result = EngineCycleResult(
            engine_name="funding_rate_arb",
            cycle_num=1,
            timestamp="2026-02-23T12:00:00Z",
            duration_ms=100.0,
            pnl_update=5.0,
            actions_taken=[{
                "action": "close",
                "symbol": "BTC/USDT",
                "pnl": 5.0,
                "cost": 1.0,
                "gross_pnl": 6.0,
                "entry_price": 50000,
                "quantity": 0.01,
            }],
        )
        engine._on_cycle_complete(result)

        trades = mgr.tracker._trades.get("funding_rate_arb", [])
        assert len(trades) == 1
        assert trades[0].net_pnl == 5.0
        assert trades[0].cost == 1.0
        assert trades[0].symbol == "BTC/USDT"

    def test_preserves_existing_callback(self):
        from bot.engines.funding_arb import FundingRateArbEngine
        from bot.engines.manager import EngineManager

        pm = PortfolioManager(
            total_capital=10000,
            engine_allocations={"funding_rate_arb": 0.3},
        )
        mgr = EngineManager(portfolio_manager=pm)
        engine = FundingRateArbEngine(
            portfolio_manager=pm, exchanges=[], paper_mode=True,
        )
        # Set existing callback
        calls = []
        engine.set_on_cycle_complete(lambda r: calls.append(r))
        mgr.register(engine)

        result = EngineCycleResult(
            engine_name="funding_rate_arb",
            cycle_num=1,
            timestamp="2026-02-23T12:00:00Z",
            duration_ms=100.0,
            pnl_update=0.0,
        )
        engine._on_cycle_complete(result)
        # Both tracker and original callback should have been called
        assert len(mgr.tracker._cycles.get("funding_rate_arb", [])) == 1
        assert len(calls) == 1


# ------------------------------------------------------------------ #
# API endpoints
# ------------------------------------------------------------------ #


class MockEngineManagerWithTracker:
    """EngineManager mock that includes a tracker with sample data."""

    def __init__(self):
        self.tracker = EngineTracker()
        self._status = {
            "funding_rate_arb": {
                "name": "funding_rate_arb",
                "description": "Funding rate arb",
                "status": "running",
                "cycle_count": 10,
                "total_pnl": 25.0,
                "allocated_capital": 3000.0,
                "position_count": 1,
                "max_positions": 3,
                "loop_interval": 300.0,
                "error": None,
            },
        }
        # Add sample trades
        from datetime import datetime, timezone

        for pnl in [10.0, -3.0, 8.0, 5.0, -2.0]:
            trade = TradeRecord(
                engine_name="funding_rate_arb",
                symbol="BTC/USDT",
                side="buy",
                entry_price=50000.0,
                exit_price=50010.0,
                quantity=0.01,
                pnl=pnl + 1.0,  # gross
                cost=1.0,
                net_pnl=pnl,
                entry_time=datetime.now(timezone.utc).isoformat(),
                exit_time=datetime.now(timezone.utc).isoformat(),
                hold_time_seconds=3600.0,
            )
            self.tracker.record_trade("funding_rate_arb", trade)

    def get_status(self):
        return {k: dict(v) for k, v in self._status.items()}


@pytest.fixture
def mock_manager_with_tracker():
    return MockEngineManagerWithTracker()


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture(autouse=True)
def setup_for_api(mock_manager_with_tracker, settings):
    set_engine_manager(mock_manager_with_tracker)
    set_settings(settings)
    yield
    set_engine_manager(None)
    set_settings(None)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestEngineMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_data(self, client):
        resp = await client.get("/api/engines/funding_rate_arb/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["engine"] == "funding_rate_arb"
        assert data["metrics"]["total_trades"] == 5
        assert data["metrics"]["winning_trades"] == 3
        assert data["metrics"]["total_pnl"] > 0

    @pytest.mark.asyncio
    async def test_metrics_custom_window(self, client):
        resp = await client.get("/api/engines/funding_rate_arb/metrics?hours=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["window_hours"] == 1.0

    @pytest.mark.asyncio
    async def test_metrics_unknown_engine(self, client):
        resp = await client.get("/api/engines/nonexistent/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["metrics"]["total_trades"] == 0


class TestPerformanceSummaryEndpoint:
    @pytest.mark.asyncio
    async def test_summary_returns_totals(self, client):
        resp = await client.get("/api/performance/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "engines" in data
        assert "totals" in data
        assert data["totals"]["total_trades"] == 5
        assert data["totals"]["total_pnl"] > 0

    @pytest.mark.asyncio
    async def test_summary_includes_engine_metrics(self, client):
        resp = await client.get("/api/performance/summary")
        data = resp.json()
        assert "funding_rate_arb" in data["engines"]
        eng = data["engines"]["funding_rate_arb"]
        assert eng["win_rate"] > 0

    @pytest.mark.asyncio
    async def test_summary_custom_window(self, client):
        resp = await client.get("/api/performance/summary?hours=48")
        assert resp.status_code == 200
        data = resp.json()
        assert data["window_hours"] == 48.0
