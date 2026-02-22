"""Tests for V4-014: Live mode integration test.

End-to-end test simulating a complete live trading session with all V4
features active: pre-flight checks, rate limiting, position reconciliation,
emergency stop/close-all/resume, audit trail, WebSocket broadcasts, JWT auth,
and settings hot-reload.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from bot.config import Settings
from bot.dashboard import auth as auth_module
from bot.dashboard.app import (
    app,
    get_state,
    set_audit_logger,
    set_settings,
    set_strategy_registry,
    set_trading_bot,
    update_state,
)
from bot.data.store import DataStore
from bot.exchanges.rate_limiter import RateLimiter
from bot.execution.preflight import CheckStatus, PreFlightChecker
from bot.execution.reconciler import PositionReconciler
from bot.monitoring.audit import AuditLogger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides):
    """Create a Settings-like namespace for testing (non-Settings endpoints)."""
    defaults = {
        "trading_mode": SimpleNamespace(value="live"),
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "stop_loss_pct": 5.0,
        "daily_loss_limit_pct": 3.0,
        "dashboard_password": "secure-password-123",
        "dashboard_username": "admin",
        "jwt_secret": "test-jwt-secret-for-integration",
        "rate_limit_enabled": True,
        "reconciliation_enabled": True,
        "reconciliation_interval_cycles": 10,
        "reconciliation_auto_fix": False,
        "loop_interval_seconds": 1,
        "log_level": "INFO",
        "database_url": "sqlite+aiosqlite:///:memory:",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_real_settings(**overrides):
    """Create a real Settings instance for settings endpoint tests."""
    defaults = {
        "trading_mode": "paper",
        "binance_api_key": "",
        "binance_secret_key": "",
        "binance_testnet": True,
        "upbit_api_key": "",
        "upbit_secret_key": "",
        "database_url": "sqlite+aiosqlite:///test.db",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "dashboard_username": "admin",
        "dashboard_password": "secure-password-123",
        "jwt_secret": "test-jwt-secret-for-integration",
        "config_file": "__nonexistent__",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_exchange(
    name="binance",
    balance=None,
    ticker_prices=None,
):
    """Create a mock exchange adapter."""
    ex = AsyncMock()
    ex.name = name
    ex.get_balance = AsyncMock(
        return_value=balance or {"USDT": 10000.0, "BTC": 0.5}
    )

    def _ticker(symbol):
        prices = ticker_prices or {"BTC/USDT": 50000, "ETH/USDT": 3000}
        return {"last": prices.get(symbol, 1000)}

    ex.get_ticker = AsyncMock(side_effect=_ticker)
    return ex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings():
    return _make_settings()


@pytest.fixture
def exchange():
    return _make_exchange()


@pytest.fixture
async def store():
    s = DataStore("sqlite+aiosqlite:///:memory:")
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def audit_logger(store):
    al = AuditLogger(store=store)
    return al


@pytest.fixture(autouse=True)
def _reset_dashboard_state(settings):
    """Reset dashboard module state before/after each test."""
    set_settings(settings)
    set_trading_bot(None)
    set_audit_logger(None)
    set_strategy_registry(None)
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
        cycle_log=[],
        reconciliation={},
        preflight={},
        emergency={"active": False, "activated_at": None, "reason": None},
    )
    yield
    set_settings(None)
    set_trading_bot(None)
    set_audit_logger(None)
    set_strategy_registry(None)
    auth_module.clear_blacklist()


@pytest.fixture
def auth_token(settings):
    """Create a valid JWT access token for authenticated requests."""
    return auth_module.create_access_token(settings, "admin")


@pytest.fixture
def auth_headers(auth_token):
    """Authorization headers for authenticated requests."""
    return {"Authorization": f"Bearer {auth_token}"}


# ---------------------------------------------------------------------------
# 1. Pre-flight checks pass before startup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_checks_pass_for_live_mode(settings, exchange):
    """Pre-flight checks should pass with properly configured live settings."""
    checker = PreFlightChecker(
        min_balance_usd=100.0,
        validation_report_dir="__nonexistent__",
    )
    result = await checker.run_all_checks(
        settings=settings,
        exchanges=[exchange],
        symbols=["BTC/USDT", "ETH/USDT"],
        rate_limit_enabled=True,
    )
    # Overall should be WARN (password changed but no validation report)
    # — never FAIL since all critical checks pass
    assert result.overall != CheckStatus.FAIL

    # Critical checks should all pass
    check_map = {c.name: c for c in result.checks}
    assert check_map["api_key_validity"].status == CheckStatus.PASS
    assert check_map["sufficient_balance"].status == CheckStatus.PASS
    assert check_map["symbol_availability"].status == CheckStatus.PASS
    assert check_map["rate_limit_configured"].status == CheckStatus.PASS
    assert check_map["stop_loss_configured"].status == CheckStatus.PASS
    assert check_map["daily_loss_limit_configured"].status == CheckStatus.PASS
    assert check_map["password_changed"].status == CheckStatus.PASS


@pytest.mark.asyncio
async def test_preflight_fail_blocks_startup(settings, exchange):
    """Pre-flight FAIL (no stop-loss) should block live startup."""
    bad_settings = _make_settings(stop_loss_pct=0)
    checker = PreFlightChecker(
        min_balance_usd=100.0,
        validation_report_dir="__nonexistent__",
    )
    result = await checker.run_all_checks(
        settings=bad_settings,
        exchanges=[exchange],
        symbols=["BTC/USDT"],
        rate_limit_enabled=True,
    )
    assert result.overall == CheckStatus.FAIL
    assert result.has_failures is True


# ---------------------------------------------------------------------------
# 2. Rate limiter throttles rapid API calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiter_throttles_rapid_calls():
    """Rate limiter should throttle when burst is exhausted."""
    limiter = RateLimiter(
        max_requests_per_second=100.0,
        burst_size=3,
        name="test",
    )
    # First 3 should be instant (burst)
    for _ in range(3):
        wait = await limiter.acquire()
        assert wait == 0.0

    # 4th should require waiting
    wait = await limiter.acquire()
    assert wait > 0.0
    assert limiter.metrics.throttled_requests >= 1
    assert limiter.metrics.total_requests == 4


@pytest.mark.asyncio
async def test_rate_limiter_metrics_tracked():
    """Rate limiter metrics should track requests and throttle events."""
    limiter = RateLimiter(
        max_requests_per_second=1000.0,
        burst_size=5,
        name="metrics-test",
    )
    for _ in range(5):
        await limiter.acquire()

    assert limiter.metrics.total_requests == 5
    assert limiter.metrics.throttled_requests == 0

    d = limiter.to_dict()
    assert d["name"] == "metrics-test"
    assert d["max_requests_per_second"] == 1000.0
    assert d["burst_size"] == 5


# ---------------------------------------------------------------------------
# 3. Position reconciliation detects and reports discrepancy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconciliation_detects_discrepancy():
    """Reconciler should detect local-only and exchange-only positions."""
    reconciler = PositionReconciler(tolerance_pct=1.0)

    exchange = AsyncMock()
    exchange.name = "binance"
    # Exchange has ETH but not BTC
    exchange.get_balance = AsyncMock(
        return_value={"ETH": 2.0, "USDT": 5000.0}
    )

    # Local has BTC and ETH (with qty mismatch)
    local_positions = {
        "BTC/USDT": {"quantity": 0.5, "entry_price": 50000},
        "ETH/USDT": {"quantity": 1.0, "entry_price": 3000},
    }

    result = await reconciler.reconcile(exchange, local_positions)

    assert result.has_discrepancies is True
    # BTC is local-only (not on exchange)
    assert len(result.local_only) == 1
    assert result.local_only[0].symbol == "BTC/USDT"
    # ETH has qty mismatch (local=1.0, exchange=2.0)
    assert len(result.qty_mismatch) == 1
    assert result.qty_mismatch[0].symbol == "ETH/USDT"


@pytest.mark.asyncio
async def test_reconciliation_clean_when_matching():
    """Reconciler should report no discrepancies when positions match."""
    reconciler = PositionReconciler(tolerance_pct=1.0)

    exchange = AsyncMock()
    exchange.name = "binance"
    exchange.get_balance = AsyncMock(
        return_value={"BTC": 0.5, "USDT": 5000.0}
    )

    local_positions = {
        "BTC/USDT": {"quantity": 0.5, "entry_price": 50000},
    }

    result = await reconciler.reconcile(exchange, local_positions)
    assert result.has_discrepancies is False
    assert "BTC/USDT" in result.matched


# ---------------------------------------------------------------------------
# 4-6. Emergency stop, close-all, resume (via API)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emergency_stop_halts_trading(auth_headers):
    """POST /api/emergency/stop should halt the trading bot."""
    bot = AsyncMock()
    bot._emergency_stopped = False

    async def mock_stop(reason="manual"):
        bot._emergency_stopped = True
        update_state(
            emergency={
                "active": True,
                "activated_at": "2026-02-22T10:00:00+00:00",
                "reason": reason,
                "cancelled_orders": 0,
            },
        )
        return {
            "success": True,
            "cancelled_orders": 0,
            "activated_at": "2026-02-22T10:00:00+00:00",
        }

    bot.emergency_stop = AsyncMock(side_effect=mock_stop)
    set_trading_bot(bot)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/emergency/stop",
            headers=auth_headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    bot.emergency_stop.assert_called_once()

    # Dashboard state should reflect emergency
    state = get_state()
    assert state["emergency"]["active"] is True


@pytest.mark.asyncio
async def test_emergency_close_all_sells_positions(auth_headers):
    """POST /api/emergency/close-all should close all positions."""
    bot = AsyncMock()
    bot._emergency_stopped = False

    async def mock_close_all(reason="manual"):
        bot._emergency_stopped = True
        update_state(
            emergency={
                "active": True,
                "activated_at": "2026-02-22T10:00:00+00:00",
                "reason": reason,
            },
        )
        return {
            "success": True,
            "closed_positions": [
                {"symbol": "BTC/USDT", "pnl": 200.0},
            ],
        }

    bot.emergency_close_all = AsyncMock(side_effect=mock_close_all)
    set_trading_bot(bot)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/emergency/close-all",
            headers=auth_headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert len(data["closed_positions"]) == 1
    assert data["closed_positions"][0]["pnl"] == 200.0


@pytest.mark.asyncio
async def test_emergency_resume_restarts_trading(auth_headers):
    """POST /api/emergency/resume should resume trading after stop."""
    bot = AsyncMock()
    bot._emergency_stopped = True

    async def mock_resume():
        bot._emergency_stopped = False
        update_state(
            emergency={
                "active": False,
                "activated_at": None,
                "reason": None,
            },
        )
        return {"success": True}

    bot.emergency_resume = AsyncMock(side_effect=mock_resume)
    set_trading_bot(bot)

    # First set emergency state
    update_state(
        emergency={
            "active": True,
            "activated_at": "2026-02-22T10:00:00+00:00",
            "reason": "test",
        },
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/emergency/resume",
            headers=auth_headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True

    state = get_state()
    assert state["emergency"]["active"] is False


@pytest.mark.asyncio
async def test_emergency_full_lifecycle(auth_headers):
    """Full emergency lifecycle: stop -> verify halted -> resume."""
    bot = AsyncMock()
    bot._emergency_stopped = False

    async def mock_stop(reason="manual"):
        bot._emergency_stopped = True
        update_state(
            emergency={
                "active": True,
                "activated_at": "2026-02-22T10:00:00+00:00",
                "reason": reason,
                "cancelled_orders": 2,
            },
        )
        return {"success": True, "cancelled_orders": 2}

    async def mock_resume():
        bot._emergency_stopped = False
        update_state(
            emergency={"active": False, "activated_at": None, "reason": None},
        )
        return {"success": True}

    bot.emergency_stop = AsyncMock(side_effect=mock_stop)
    bot.emergency_resume = AsyncMock(side_effect=mock_resume)
    set_trading_bot(bot)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Step 1: Stop
        resp = await ac.post(
            "/api/emergency/stop", headers=auth_headers
        )
        assert resp.status_code == 200

        # Step 2: Verify stopped state
        resp = await ac.get(
            "/api/emergency", headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["emergency"]["active"] is True

        # Step 3: Resume
        resp = await ac.post(
            "/api/emergency/resume", headers=auth_headers
        )
        assert resp.status_code == 200

        # Step 4: Verify resumed
        resp = await ac.get(
            "/api/emergency", headers=auth_headers
        )
        assert resp.json()["emergency"]["active"] is False


# ---------------------------------------------------------------------------
# 7. Audit trail records all events from the session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_trail_records_events(store, audit_logger):
    """Audit trail should persist events across various actions."""
    # Log a variety of events
    await audit_logger.log_bot_started(mode="live", symbols=["BTC/USDT"])
    await audit_logger.log_trade(
        symbol="BTC/USDT", side="BUY", quantity=0.1, price=50000,
        strategy="ma_crossover",
    )
    await audit_logger.log_strategy_toggled(
        name="rsi", active=False, actor="user"
    )
    await audit_logger.log_config_changed(
        changed=["stop_loss_pct"], previous={"stop_loss_pct": 3.0},
    )
    await audit_logger.log_emergency_stop(reason="test", cancelled_orders=2)
    await audit_logger.log_emergency_resume(previous_reason="test")
    await audit_logger.log_bot_stopped()

    # Query all events
    result = await store.get_audit_logs(limit=50)
    logs = result["logs"]

    assert result["total"] == 7
    event_types = {log["event_type"] for log in logs}
    assert "bot_started" in event_types
    assert "trade_executed" in event_types
    assert "strategy_toggled" in event_types
    assert "config_changed" in event_types
    assert "emergency_stop" in event_types
    assert "emergency_resume" in event_types
    assert "bot_stopped" in event_types


@pytest.mark.asyncio
async def test_audit_trail_api_endpoint(store, audit_logger, auth_headers):
    """GET /api/audit should return persisted audit logs."""
    set_audit_logger(audit_logger)

    await audit_logger.log_bot_started(mode="live", symbols=["BTC/USDT"])
    await audit_logger.log_trade(
        symbol="BTC/USDT", side="BUY", quantity=0.1, price=50000,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/audit", headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["logs"]) == 2


@pytest.mark.asyncio
async def test_audit_trail_filter_by_severity(store, audit_logger):
    """Audit logs should be filterable by severity."""
    await audit_logger.log_event(
        "normal_event", severity="info"
    )
    await audit_logger.log_emergency_stop(reason="critical_issue")

    # Filter for critical events
    result = await store.get_audit_logs(severity="critical")
    assert result["total"] == 1
    assert result["logs"][0]["event_type"] == "emergency_stop"


# ---------------------------------------------------------------------------
# 8. WebSocket broadcasts state changes to connected client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_websocket_state_broadcast(auth_headers):
    """WebSocket should broadcast state on connect."""
    update_state(status="running")

    from starlette.testclient import TestClient

    client = TestClient(app)
    with client.websocket_connect("/api/ws") as ws:
        data = ws.receive_json()
        assert data["type"] == "status_update"
        assert data["payload"]["status"] == "running"


# ---------------------------------------------------------------------------
# 9. JWT auth blocks unauthenticated API requests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_blocks_unauthenticated_requests():
    """API should reject requests without a valid JWT when auth is enabled."""
    # Auth is enabled via _reset_dashboard_state fixture (password != changeme)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/status")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_allows_authenticated_requests(auth_headers):
    """API should accept requests with a valid JWT."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/status", headers=auth_headers)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_login_returns_tokens(settings):
    """Login endpoint should return access and refresh tokens."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/auth/login",
            json={"username": "admin", "password": "secure-password-123"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data

    # Verify the access token works
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/status",
            headers={"Authorization": f"Bearer {data['access_token']}"},
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_refresh_token_flow(settings):
    """Refresh token should issue a new access token."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Login
        login_resp = await ac.post(
            "/api/auth/login",
            json={"username": "admin", "password": "secure-password-123"},
        )
        tokens = login_resp.json()

        # Refresh
        refresh_resp = await ac.post(
            "/api/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )
    assert refresh_resp.status_code == 200
    new_tokens = refresh_resp.json()
    assert "access_token" in new_tokens


@pytest.mark.asyncio
async def test_auth_logout_invalidates_refresh(settings):
    """Logout should blacklist the refresh token."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Login
        login_resp = await ac.post(
            "/api/auth/login",
            json={"username": "admin", "password": "secure-password-123"},
        )
        tokens = login_resp.json()

        # Logout
        logout_resp = await ac.post(
            "/api/auth/logout",
            json={"refresh_token": tokens["refresh_token"]},
            headers={
                "Authorization": f"Bearer {tokens['access_token']}"
            },
        )
        assert logout_resp.status_code == 200

        # Refresh should now fail
        refresh_resp = await ac.post(
            "/api/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )
    assert refresh_resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_wrong_password_rejected(settings):
    """Login with wrong password should be rejected."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/auth/login",
            json={"username": "admin", "password": "wrong-password"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_health_accessible_without_auth():
    """Health endpoint should be accessible without authentication."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 10. Settings hot-reload changes take effect without restart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settings_hot_reload(auth_headers):
    """PUT /api/settings should apply changes to safe settings."""
    real_settings = _make_real_settings()
    set_settings(real_settings)
    token = auth_module.create_access_token(real_settings, "admin")
    headers = {"Authorization": f"Bearer {token}"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.put(
            "/api/settings",
            json={"stop_loss_pct": 7.5},
            headers=headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "stop_loss_pct" in data.get("changed", [])

    # Verify the setting was actually changed
    assert real_settings.stop_loss_pct == 7.5


@pytest.mark.asyncio
async def test_settings_unsafe_rejected(auth_headers):
    """PUT /api/settings should reject unsafe (restart-required) settings."""
    real_settings = _make_real_settings()
    set_settings(real_settings)
    token = auth_module.create_access_token(real_settings, "admin")
    headers = {"Authorization": f"Bearer {token}"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.put(
            "/api/settings",
            json={"trading_mode": "paper"},
            headers=headers,
        )

    # Should fail because trading_mode requires restart
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_settings_get_returns_config(auth_headers):
    """GET /api/settings should return current configuration."""
    real_settings = _make_real_settings()
    set_settings(real_settings)
    token = auth_module.create_access_token(real_settings, "admin")
    headers = {"Authorization": f"Bearer {token}"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/settings", headers=headers)

    assert resp.status_code == 200
    data = resp.json()
    assert "settings" in data
    # Should contain known fields
    setting_keys = [s["key"] for s in data["settings"]]
    assert "stop_loss_pct" in setting_keys


# ---------------------------------------------------------------------------
# 11. Full integrated lifecycle simulation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_live_session_lifecycle(
    settings, exchange, store, audit_logger, auth_headers,
):
    """Simulate a complete live trading session with all V4 features.

    Steps:
    1. Pre-flight checks pass
    2. Audit: bot_started logged
    3. Rate limiter tracks requests
    4. Reconciliation detects discrepancy
    5. Emergency stop halts trading
    6. Emergency resume restarts
    7. Audit trail records all events
    8. Auth protects endpoints
    """
    set_audit_logger(audit_logger)

    # --- Step 1: Pre-flight checks ---
    checker = PreFlightChecker(
        min_balance_usd=100.0,
        validation_report_dir="__nonexistent__",
    )
    preflight_result = await checker.run_all_checks(
        settings=settings,
        exchanges=[exchange],
        symbols=["BTC/USDT", "ETH/USDT"],
        rate_limit_enabled=True,
    )
    assert preflight_result.overall != CheckStatus.FAIL
    await audit_logger.log_preflight_result(
        overall=preflight_result.overall.value,
        warnings=["paper_validation"],
    )

    # --- Step 2: Audit bot started ---
    await audit_logger.log_bot_started(
        mode="live", symbols=["BTC/USDT", "ETH/USDT"]
    )
    update_state(status="running")

    # --- Step 3: Rate limiter tracks requests ---
    limiter = RateLimiter(
        max_requests_per_second=100.0, burst_size=10, name="binance"
    )
    for _ in range(5):
        await limiter.acquire()
    assert limiter.metrics.total_requests == 5

    # --- Step 4: Reconciliation detects discrepancy ---
    reconciler = PositionReconciler(tolerance_pct=1.0)
    recon_exchange = AsyncMock()
    recon_exchange.name = "binance"
    recon_exchange.get_balance = AsyncMock(
        return_value={"BTC": 0.3, "USDT": 5000.0}
    )
    recon_result = await reconciler.reconcile(
        recon_exchange,
        {"BTC/USDT": {"quantity": 0.5, "entry_price": 50000}},
    )
    assert recon_result.has_discrepancies is True
    await audit_logger.log_reconciliation_result(
        has_discrepancies=True,
        matched=0,
        discrepancies=recon_result.total_discrepancies,
    )

    # --- Step 5: Emergency stop ---
    await audit_logger.log_emergency_stop(
        reason="integration_test", cancelled_orders=1
    )
    update_state(
        emergency={
            "active": True,
            "activated_at": "2026-02-22T10:00:00+00:00",
            "reason": "integration_test",
        },
    )

    # --- Step 6: Emergency resume ---
    await audit_logger.log_emergency_resume(
        previous_reason="integration_test"
    )
    update_state(
        emergency={"active": False, "activated_at": None, "reason": None},
    )

    # --- Step 7: Verify audit trail ---
    result = await store.get_audit_logs(limit=50)
    logs = result["logs"]
    event_types = [log["event_type"] for log in logs]

    assert "preflight_result" in event_types
    assert "bot_started" in event_types
    assert "reconciliation_result" in event_types
    assert "emergency_stop" in event_types
    assert "emergency_resume" in event_types
    assert result["total"] == 5

    # --- Step 8: Auth protects endpoints ---
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Unauthenticated should fail
        resp = await ac.get("/api/status")
        assert resp.status_code == 401

        # Authenticated should succeed
        resp = await ac.get("/api/status", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    # --- Step 9: Bot shutdown ---
    await audit_logger.log_bot_stopped()
    final_logs = await store.get_audit_logs(limit=50)
    assert final_logs["total"] == 6
    final_types = [log["event_type"] for log in final_logs["logs"]]
    assert "bot_stopped" in final_types


# ---------------------------------------------------------------------------
# 12. TradingBot unit integration (emergency methods with audit)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bot_emergency_stop_with_audit():
    """TradingBot.emergency_stop should log to audit trail."""
    from bot.main import TradingBot

    bot = TradingBot.__new__(TradingBot)
    bot._emergency_stopped = False
    bot._emergency_stopped_at = None
    bot._emergency_reason = None
    bot._execution_engines = {}
    bot._telegram = None
    bot._risk_manager = None
    bot._position_manager = None
    bot._portfolio_risk = None
    bot._audit_logger = AsyncMock()

    await bot.emergency_stop(reason="api_request")

    bot._audit_logger.log_emergency_stop.assert_called_once()
    call_kwargs = bot._audit_logger.log_emergency_stop.call_args
    assert call_kwargs[1]["actor"] == "user"  # api_request -> user


@pytest.mark.asyncio
async def test_bot_emergency_resume_with_audit():
    """TradingBot.emergency_resume should log to audit trail."""
    from bot.main import TradingBot

    bot = TradingBot.__new__(TradingBot)
    bot._emergency_stopped = True
    bot._emergency_stopped_at = "2026-01-01T00:00:00+00:00"
    bot._emergency_reason = "test"
    bot._telegram = None
    bot._audit_logger = AsyncMock()

    await bot.emergency_resume()

    bot._audit_logger.log_emergency_resume.assert_called_once()


# ---------------------------------------------------------------------------
# 13. Cross-feature: audit + auth + API integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_logs_protected_by_auth():
    """GET /api/audit should require authentication when auth is enabled."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/audit")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_emergency_endpoints_protected_by_auth():
    """Emergency endpoints should require authentication."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/emergency/stop")
        assert resp.status_code == 401

        resp = await ac.post("/api/emergency/close-all")
        assert resp.status_code == 401

        resp = await ac.post("/api/emergency/resume")
        assert resp.status_code == 401

        resp = await ac.get("/api/emergency")
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_settings_endpoints_protected_by_auth():
    """Settings endpoints should require authentication."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/settings")
        assert resp.status_code == 401

        resp = await ac.put(
            "/api/settings",
            json={"stop_loss_pct": 5.0},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 14. Preflight results accessible via API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_api_endpoint(auth_headers):
    """GET /api/preflight should return pre-flight results."""
    update_state(
        preflight={
            "overall": "WARN",
            "checks": [
                {"name": "api_key_validity", "status": "PASS"},
            ],
        },
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/preflight", headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["preflight"]["overall"] == "WARN"


# ---------------------------------------------------------------------------
# 15. Reconciliation results accessible via API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconciliation_api_endpoint(auth_headers):
    """GET /api/reconciliation should return reconciliation results."""
    update_state(
        reconciliation={
            "timestamp": "2026-02-22T10:00:00+00:00",
            "exchange_name": "binance",
            "matched": ["BTC/USDT"],
            "has_discrepancies": False,
        },
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/reconciliation", headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["reconciliation"]["has_discrepancies"] is False
    assert "BTC/USDT" in data["reconciliation"]["matched"]
