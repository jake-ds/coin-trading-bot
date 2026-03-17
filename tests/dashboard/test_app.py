"""Tests for FastAPI dashboard."""

import pytest
from httpx import ASGITransport, AsyncClient

from bot.dashboard.app import app, get_state, update_state


@pytest.fixture(autouse=True)
def reset_state():
    """Reset bot state before each test."""
    update_state(
        status="stopped",
        started_at=None,
        trades=[],
        metrics={},
        portfolio={"balances": {}, "positions": [], "total_value": 0.0},
        cycle_metrics={
            "cycle_count": 0,
            "average_cycle_duration": 0.0,
            "last_cycle_time": None,
        },
        strategy_stats={},
        equity_curve=[],
        open_positions=[],
        regime=None,
    )


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestDashboardAPI:
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_get_status_default(self, client):
        resp = await client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_get_status_running(self, client):
        update_state(status="running")
        resp = await client.get("/api/status")
        data = resp.json()
        assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_trades_empty(self, client):
        resp = await client.get("/api/trades")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trades"] == []

    @pytest.mark.asyncio
    async def test_get_trades_with_data(self, client):
        update_state(trades=[
            {"symbol": "BTC/USDT", "side": "BUY", "quantity": 0.1, "price": 50000},
        ])
        resp = await client.get("/api/trades")
        data = resp.json()
        assert len(data["trades"]) == 1
        assert data["trades"][0]["symbol"] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_get_metrics_default(self, client):
        resp = await client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["metrics"] == {}

    @pytest.mark.asyncio
    async def test_get_metrics_with_data(self, client):
        update_state(metrics={"total_return_pct": 5.0, "win_rate": 60.0})
        resp = await client.get("/api/metrics")
        data = resp.json()
        assert data["metrics"]["total_return_pct"] == 5.0

    @pytest.mark.asyncio
    async def test_get_portfolio(self, client):
        update_state(portfolio={
            "balances": {"USDT": 10000},
            "positions": [],
            "total_value": 10000.0,
        })
        resp = await client.get("/api/portfolio")
        data = resp.json()
        assert data["portfolio"]["total_value"] == 10000.0

    @pytest.mark.asyncio
    async def test_legacy_dashboard_html(self, client):
        resp = await client.get("/legacy")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Trading Bot Dashboard" in resp.text

    @pytest.mark.asyncio
    async def test_legacy_dashboard_shows_status(self, client):
        """Legacy HTML now serves the SPA, so just check it returns 200."""
        update_state(status="running")
        resp = await client.get("/legacy")
        assert resp.status_code == 200


class TestStateManagement:
    def test_get_state(self):
        state = get_state()
        assert "status" in state
        assert "trades" in state

    def test_update_state(self):
        update_state(status="running")
        state = get_state()
        assert state["status"] == "running"
