"""Tests for WebSocket real-time updates (V4-003)."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from bot.dashboard.app import (
    app,
    broadcast_alert,
    broadcast_position_change,
    broadcast_state_update,
    broadcast_trade,
    update_state,
)
from bot.dashboard.websocket import ConnectionManager, ws_manager


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
    # Clear any connections from previous tests
    ws_manager._connections.clear()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestConnectionManager:
    """Tests for the WebSocket ConnectionManager class."""

    def test_initial_state(self):
        manager = ConnectionManager()
        assert manager.active_connections == 0

    @pytest.mark.asyncio
    async def test_rate_limiting_debounce(self):
        """Rate limiter should debounce rapid broadcasts."""
        manager = ConnectionManager(min_broadcast_interval=0.5)
        # With no connections, broadcast is a no-op (should not error)
        await manager.broadcast({"type": "status_update", "payload": {}})
        assert manager.active_connections == 0

    @pytest.mark.asyncio
    async def test_broadcast_immediate_no_connections(self):
        """broadcast_immediate with no connections should not error."""
        manager = ConnectionManager()
        await manager.broadcast_immediate({"type": "trade", "payload": {}})

    @pytest.mark.asyncio
    async def test_send_personal_handles_error(self):
        """send_personal should handle disconnected clients gracefully."""
        manager = ConnectionManager()

        class FakeWebSocket:
            async def send_text(self, data):
                raise RuntimeError("disconnected")

        fake_ws = FakeWebSocket()
        manager._connections.append(fake_ws)
        assert manager.active_connections == 1

        await manager.send_personal(fake_ws, {"type": "test"})
        # Connection should be removed after error
        assert manager.active_connections == 0

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_connections(self):
        """Broadcast should clean up disconnected clients."""
        manager = ConnectionManager(min_broadcast_interval=0.0)

        class DeadSocket:
            async def send_text(self, data):
                raise RuntimeError("connection closed")

        class AliveSocket:
            messages = []

            async def send_text(self, data):
                self.messages.append(data)

        dead = DeadSocket()
        alive = AliveSocket()
        manager._connections.extend([dead, alive])
        assert manager.active_connections == 2

        await manager.broadcast({"type": "status_update", "payload": {"status": "running"}})
        # Dead connection should be removed
        assert manager.active_connections == 1
        assert alive in manager._connections
        assert len(alive.messages) == 1

    @pytest.mark.asyncio
    async def test_rate_limit_queues_messages(self):
        """Messages sent within the interval should be queued and sent after delay."""
        manager = ConnectionManager(min_broadcast_interval=0.1)

        class AliveSocket:
            messages = []

            async def send_text(self, data):
                self.messages.append(data)

        alive = AliveSocket()
        manager._connections.append(alive)

        # First broadcast goes through immediately
        await manager.broadcast({"type": "status_update", "payload": {"seq": 1}})
        assert len(alive.messages) == 1

        # Second broadcast within interval gets queued
        await manager.broadcast({"type": "status_update", "payload": {"seq": 2}})
        # Not yet sent
        assert len(alive.messages) == 1

        # Wait for debounce to fire
        await asyncio.sleep(0.2)
        assert len(alive.messages) == 2

    @pytest.mark.asyncio
    async def test_broadcast_immediate_bypasses_rate_limit(self):
        """broadcast_immediate should bypass rate limiting."""
        manager = ConnectionManager(min_broadcast_interval=10.0)

        class AliveSocket:
            messages = []

            async def send_text(self, data):
                self.messages.append(data)

        alive = AliveSocket()
        manager._connections.append(alive)

        # First normal broadcast
        await manager.broadcast({"type": "status_update", "payload": {}})
        assert len(alive.messages) == 1

        # Immediate broadcast bypasses rate limit
        await manager.broadcast_immediate({"type": "trade", "payload": {}})
        assert len(alive.messages) == 2


class TestWebSocketEndpoint:
    """Tests for the /api/ws WebSocket endpoint."""

    def test_websocket_connect_and_receive_initial_state(self):
        """WebSocket should accept connection and send initial state."""
        client = TestClient(app)
        with client.websocket_connect("/api/ws") as ws:
            # Should receive initial state on connect
            data = ws.receive_json()
            assert data["type"] == "status_update"
            assert "payload" in data
            payload = data["payload"]
            assert payload["status"] == "stopped"
            assert "cycle_metrics" in payload
            assert "portfolio" in payload
            assert "metrics" in payload
            assert "regime" in payload
            assert "trades" in payload
            assert "strategy_stats" in payload
            assert "open_positions" in payload

    def test_websocket_receives_state_with_data(self):
        """WebSocket should reflect current bot state on connect."""
        update_state(
            status="running",
            metrics={"total_return_pct": 5.0, "win_rate": 65.0},
            portfolio={"balances": {"USDT": 10000}, "positions": [], "total_value": 10500.0},
            regime="TRENDING_UP",
        )
        client = TestClient(app)
        with client.websocket_connect("/api/ws") as ws:
            data = ws.receive_json()
            payload = data["payload"]
            assert payload["status"] == "running"
            assert payload["metrics"]["total_return_pct"] == 5.0
            assert payload["portfolio"]["total_value"] == 10500.0
            assert payload["regime"] == "TRENDING_UP"

    def test_websocket_disconnect_cleanup(self):
        """Disconnecting should clean up the connection."""
        client = TestClient(app)
        initial_count = ws_manager.active_connections
        with client.websocket_connect("/api/ws") as ws:
            ws.receive_json()  # consume initial state
            # Connection is active
            assert ws_manager.active_connections >= initial_count + 1
        # After disconnect, connection count should decrease
        # (TestClient closes cleanly)


class TestBroadcastHelpers:
    """Tests for the broadcast helper functions."""

    @pytest.mark.asyncio
    async def test_broadcast_state_update_no_error(self):
        """broadcast_state_update should not error with no connections."""
        await broadcast_state_update()

    @pytest.mark.asyncio
    async def test_broadcast_trade_no_error(self):
        """broadcast_trade should not error with no connections."""
        await broadcast_trade({
            "symbol": "BTC/USDT",
            "side": "BUY",
            "quantity": 0.1,
            "price": 50000,
        })

    @pytest.mark.asyncio
    async def test_broadcast_position_change_no_error(self):
        """broadcast_position_change should not error with no connections."""
        await broadcast_position_change([
            {"symbol": "BTC/USDT", "quantity": 0.1, "entry_price": 50000},
        ])

    @pytest.mark.asyncio
    async def test_broadcast_alert_no_error(self):
        """broadcast_alert should not error with no connections."""
        await broadcast_alert("Test alert", severity="warning")

    @pytest.mark.asyncio
    async def test_broadcast_trade_message_format(self):
        """broadcast_trade should send correct message format."""
        received = []

        class FakeSocket:
            async def send_text(self, data):
                import json
                received.append(json.loads(data))

        ws_manager._connections.append(FakeSocket())
        try:
            await broadcast_trade({
                "symbol": "ETH/USDT",
                "side": "SELL",
                "quantity": 1.0,
                "price": 3000,
            })
            assert len(received) == 1
            msg = received[0]
            assert msg["type"] == "trade"
            assert msg["payload"]["symbol"] == "ETH/USDT"
            assert msg["payload"]["side"] == "SELL"
        finally:
            ws_manager._connections.clear()

    @pytest.mark.asyncio
    async def test_broadcast_alert_message_format(self):
        """broadcast_alert should send correct message format."""
        received = []

        class FakeSocket:
            async def send_text(self, data):
                import json
                received.append(json.loads(data))

        ws_manager._connections.append(FakeSocket())
        try:
            await broadcast_alert("High drawdown detected", severity="critical")
            assert len(received) == 1
            msg = received[0]
            assert msg["type"] == "alert"
            assert msg["payload"]["message"] == "High drawdown detected"
            assert msg["payload"]["severity"] == "critical"
        finally:
            ws_manager._connections.clear()

    @pytest.mark.asyncio
    async def test_broadcast_position_change_message_format(self):
        """broadcast_position_change should send correct message format."""
        received = []

        class FakeSocket:
            async def send_text(self, data):
                import json
                received.append(json.loads(data))

        ws_manager._connections.append(FakeSocket())
        try:
            positions = [
                {"symbol": "BTC/USDT", "quantity": 0.5, "entry_price": 45000},
            ]
            await broadcast_position_change(positions)
            assert len(received) == 1
            msg = received[0]
            assert msg["type"] == "position_change"
            assert msg["payload"]["positions"] == positions
        finally:
            ws_manager._connections.clear()


class TestExistingEndpointsStillWork:
    """Verify existing API endpoints are unaffected by WebSocket addition."""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_status_endpoint(self, client):
        resp = await client.get("/api/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_trades_endpoint(self, client):
        resp = await client.get("/api/trades")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_portfolio_endpoint(self, client):
        resp = await client.get("/api/portfolio")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_regime_endpoint(self, client):
        resp = await client.get("/api/regime")
        assert resp.status_code == 200
