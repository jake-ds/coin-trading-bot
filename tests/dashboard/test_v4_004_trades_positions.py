"""Tests for V4-004: Trades and positions page with live updates."""

import pytest
from httpx import ASGITransport, AsyncClient

from bot.dashboard.app import app, update_state


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


def _make_trades(n: int) -> list[dict]:
    """Generate n sample trade dicts."""
    trades = []
    for i in range(n):
        trades.append({
            "timestamp": f"2026-02-22T10:{i:02d}:00Z",
            "symbol": "BTC/USDT" if i % 2 == 0 else "ETH/USDT",
            "side": "BUY" if i % 3 != 0 else "SELL",
            "quantity": 0.1 + i * 0.01,
            "price": 50000 + i * 100,
            "pnl": (i - n // 2) * 10.0,
            "strategy": "ma_crossover" if i % 2 == 0 else "rsi",
        })
    return trades


class TestTradesPagination:
    """Test /api/trades with pagination."""

    @pytest.mark.asyncio
    async def test_trades_empty(self, client):
        resp = await client.get("/api/trades")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trades"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["total_pages"] == 1

    @pytest.mark.asyncio
    async def test_trades_default_page(self, client):
        update_state(trades=_make_trades(5))
        resp = await client.get("/api/trades")
        data = resp.json()
        assert len(data["trades"]) == 5
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["total_pages"] == 1

    @pytest.mark.asyncio
    async def test_trades_returns_newest_first(self, client):
        trades = _make_trades(5)
        update_state(trades=trades)
        resp = await client.get("/api/trades")
        data = resp.json()
        # Newest trade (last in original list) should be first in response
        assert data["trades"][0]["timestamp"] == trades[-1]["timestamp"]
        assert data["trades"][-1]["timestamp"] == trades[0]["timestamp"]

    @pytest.mark.asyncio
    async def test_trades_pagination_page_1(self, client):
        update_state(trades=_make_trades(50))
        resp = await client.get("/api/trades", params={"page": 1, "limit": 20})
        data = resp.json()
        assert len(data["trades"]) == 20
        assert data["total"] == 50
        assert data["page"] == 1
        assert data["total_pages"] == 3

    @pytest.mark.asyncio
    async def test_trades_pagination_page_2(self, client):
        update_state(trades=_make_trades(50))
        resp = await client.get("/api/trades", params={"page": 2, "limit": 20})
        data = resp.json()
        assert len(data["trades"]) == 20
        assert data["page"] == 2

    @pytest.mark.asyncio
    async def test_trades_pagination_last_page(self, client):
        update_state(trades=_make_trades(50))
        resp = await client.get("/api/trades", params={"page": 3, "limit": 20})
        data = resp.json()
        assert len(data["trades"]) == 10  # 50 - 20 - 20 = 10
        assert data["page"] == 3

    @pytest.mark.asyncio
    async def test_trades_pagination_beyond_last_page(self, client):
        update_state(trades=_make_trades(5))
        resp = await client.get("/api/trades", params={"page": 10, "limit": 20})
        data = resp.json()
        assert len(data["trades"]) == 0
        assert data["page"] == 10

    @pytest.mark.asyncio
    async def test_trades_custom_limit(self, client):
        update_state(trades=_make_trades(30))
        resp = await client.get("/api/trades", params={"limit": 10})
        data = resp.json()
        assert len(data["trades"]) == 10
        assert data["total_pages"] == 3

    @pytest.mark.asyncio
    async def test_trades_symbol_filter(self, client):
        update_state(trades=_make_trades(10))
        resp = await client.get("/api/trades", params={"symbol": "BTC/USDT"})
        data = resp.json()
        # Only BTC/USDT trades (even-indexed: 0, 2, 4, 6, 8 = 5 trades)
        assert all(t["symbol"] == "BTC/USDT" for t in data["trades"])
        assert data["total"] == 5

    @pytest.mark.asyncio
    async def test_trades_symbol_filter_no_match(self, client):
        update_state(trades=_make_trades(5))
        resp = await client.get("/api/trades", params={"symbol": "DOGE/USDT"})
        data = resp.json()
        assert data["trades"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_trades_symbol_filter_with_pagination(self, client):
        update_state(trades=_make_trades(50))
        resp = await client.get(
            "/api/trades", params={"symbol": "BTC/USDT", "page": 1, "limit": 5}
        )
        data = resp.json()
        assert len(data["trades"]) == 5
        assert all(t["symbol"] == "BTC/USDT" for t in data["trades"])
        assert data["total"] == 25  # Half of 50 are BTC/USDT

    @pytest.mark.asyncio
    async def test_trades_limit_validation_min(self, client):
        resp = await client.get("/api/trades", params={"limit": 0})
        assert resp.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_trades_limit_validation_max(self, client):
        resp = await client.get("/api/trades", params={"limit": 200})
        assert resp.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_trades_page_validation(self, client):
        resp = await client.get("/api/trades", params={"page": 0})
        assert resp.status_code == 422  # Validation error


class TestPositionsEndpoint:
    """Test /api/positions endpoint."""

    @pytest.mark.asyncio
    async def test_positions_empty(self, client):
        resp = await client.get("/api/positions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["positions"] == []

    @pytest.mark.asyncio
    async def test_positions_with_data(self, client):
        positions = [
            {
                "symbol": "BTC/USDT",
                "quantity": 0.5,
                "entry_price": 50000.0,
                "current_price": 52000.0,
                "unrealized_pnl": 1000.0,
                "stop_loss": 48000.0,
                "take_profit": 55000.0,
                "opened_at": "2026-02-22T08:00:00Z",
                "strategy": "ma_crossover",
            },
            {
                "symbol": "ETH/USDT",
                "quantity": 5.0,
                "entry_price": 3000.0,
                "current_price": 2900.0,
                "unrealized_pnl": -500.0,
                "stop_loss": 2800.0,
                "take_profit": 3500.0,
                "opened_at": "2026-02-22T09:00:00Z",
                "strategy": "rsi",
            },
        ]
        update_state(open_positions=positions)
        resp = await client.get("/api/positions")
        data = resp.json()
        assert len(data["positions"]) == 2
        assert data["positions"][0]["symbol"] == "BTC/USDT"
        assert data["positions"][0]["unrealized_pnl"] == 1000.0
        assert data["positions"][1]["symbol"] == "ETH/USDT"
        assert data["positions"][1]["unrealized_pnl"] == -500.0

    @pytest.mark.asyncio
    async def test_positions_matches_open_positions(self, client):
        """Verify /api/positions returns same data as /api/open-positions."""
        positions = [
            {
                "symbol": "BTC/USDT",
                "quantity": 1.0,
                "entry_price": 50000.0,
                "current_price": 51000.0,
                "unrealized_pnl": 1000.0,
                "stop_loss": 49000.0,
                "take_profit": 55000.0,
            },
        ]
        update_state(open_positions=positions)
        resp_new = await client.get("/api/positions")
        resp_old = await client.get("/api/open-positions")
        assert resp_new.json()["positions"] == resp_old.json()["positions"]
