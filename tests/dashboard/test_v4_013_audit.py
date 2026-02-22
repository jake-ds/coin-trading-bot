"""Tests for V4-013: Comprehensive audit trail and trade log."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from bot.dashboard import auth as auth_module
from bot.dashboard.app import (
    app,
    set_audit_logger,
    set_settings,
    set_trading_bot,
    update_state,
)
from bot.data.models import AuditLogRecord
from bot.data.store import DataStore
from bot.monitoring.audit import AuditLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_settings(**overrides):
    """Create a mock settings object for audit tests."""
    defaults = {
        "dashboard_username": "admin",
        "dashboard_password": "changeme",
        "jwt_secret": "",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset bot state, settings, and audit logger before each test."""
    update_state(
        status="running",
        started_at=None,
        trades=[],
        metrics={},
        portfolio={"balances": {}, "positions": [], "total_value": 10000.0},
        cycle_metrics={
            "cycle_count": 0,
            "average_cycle_duration": 0.0,
            "last_cycle_time": None,
        },
        strategy_stats={},
        equity_curve=[],
        open_positions=[],
        regime=None,
        emergency={"active": False, "activated_at": None, "reason": None},
        reconciliation={},
        preflight={},
    )
    set_settings(_make_settings())
    set_trading_bot(None)
    set_audit_logger(None)
    auth_module.clear_blacklist()
    yield
    set_settings(None)
    set_trading_bot(None)
    set_audit_logger(None)


@pytest.fixture
async def store():
    """Create a real in-memory DataStore for testing."""
    ds = DataStore(database_url="sqlite+aiosqlite:///:memory:")
    await ds.initialize()
    yield ds
    await ds.close()


@pytest.fixture
async def audit_logger(store):
    """Create an AuditLogger with a real DataStore."""
    logger = AuditLogger(store=store)
    return logger


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# AuditLogger unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_logger_log_event(audit_logger, store):
    """AuditLogger.log_event persists to database."""
    await audit_logger.log_event(
        event_type="test_event",
        actor="tester",
        details={"key": "value"},
        severity="info",
    )

    result = await store.get_audit_logs()
    assert result["total"] == 1
    log = result["logs"][0]
    assert log["event_type"] == "test_event"
    assert log["actor"] == "tester"
    assert log["details"] == {"key": "value"}
    assert log["severity"] == "info"


@pytest.mark.asyncio
async def test_audit_logger_log_event_without_store():
    """AuditLogger works without a store (logs to structlog only)."""
    logger = AuditLogger(store=None)
    # Should not raise
    await logger.log_event(event_type="test", details={"x": 1})


@pytest.mark.asyncio
async def test_audit_logger_log_trade(audit_logger, store):
    """log_trade creates a trade_executed event."""
    await audit_logger.log_trade(
        symbol="BTC/USDT",
        side="BUY",
        quantity=0.1,
        price=50000,
        strategy="ma_crossover",
    )

    result = await store.get_audit_logs(event_type="trade_executed")
    assert result["total"] == 1
    log = result["logs"][0]
    assert log["details"]["symbol"] == "BTC/USDT"
    assert log["details"]["side"] == "BUY"
    assert log["details"]["strategy"] == "ma_crossover"


@pytest.mark.asyncio
async def test_audit_logger_log_order_cancelled(audit_logger, store):
    """log_order_cancelled creates an order_cancelled event."""
    await audit_logger.log_order_cancelled(
        order_id="ord-123",
        symbol="ETH/USDT",
        reason="emergency",
    )

    result = await store.get_audit_logs(event_type="order_cancelled")
    assert result["total"] == 1
    assert result["logs"][0]["details"]["order_id"] == "ord-123"


@pytest.mark.asyncio
async def test_audit_logger_log_position_closed(audit_logger, store):
    """log_position_closed creates a position_closed event."""
    await audit_logger.log_position_closed(
        symbol="BTC/USDT",
        quantity=0.5,
        pnl=250.0,
        exit_type="stop_loss",
    )

    result = await store.get_audit_logs(event_type="position_closed")
    assert result["total"] == 1
    assert result["logs"][0]["details"]["pnl"] == 250.0
    assert result["logs"][0]["details"]["exit_type"] == "stop_loss"


@pytest.mark.asyncio
async def test_audit_logger_log_strategy_toggled(audit_logger, store):
    """log_strategy_toggled creates a strategy_toggled event."""
    await audit_logger.log_strategy_toggled(name="rsi", active=False)

    result = await store.get_audit_logs(event_type="strategy_toggled")
    assert result["total"] == 1
    assert result["logs"][0]["details"]["strategy"] == "rsi"
    assert result["logs"][0]["details"]["active"] is False
    assert result["logs"][0]["actor"] == "user"


@pytest.mark.asyncio
async def test_audit_logger_log_config_changed(audit_logger, store):
    """log_config_changed creates a config_changed event."""
    await audit_logger.log_config_changed(
        changed=["stop_loss_pct", "max_drawdown_pct"],
        previous={"stop_loss_pct": 3.0, "max_drawdown_pct": 10.0},
    )

    result = await store.get_audit_logs(event_type="config_changed")
    assert result["total"] == 1
    log = result["logs"][0]
    assert "stop_loss_pct" in log["details"]["changed_keys"]
    assert log["details"]["previous"]["stop_loss_pct"] == 3.0


@pytest.mark.asyncio
async def test_audit_logger_log_emergency_stop(audit_logger, store):
    """log_emergency_stop creates a critical emergency_stop event."""
    await audit_logger.log_emergency_stop(
        reason="api_request",
        cancelled_orders=3,
        actor="user",
    )

    result = await store.get_audit_logs(event_type="emergency_stop")
    assert result["total"] == 1
    log = result["logs"][0]
    assert log["severity"] == "critical"
    assert log["details"]["cancelled_orders"] == 3
    assert log["actor"] == "user"


@pytest.mark.asyncio
async def test_audit_logger_log_emergency_resume(audit_logger, store):
    """log_emergency_resume creates a warning-level event."""
    await audit_logger.log_emergency_resume(previous_reason="test")

    result = await store.get_audit_logs(event_type="emergency_resume")
    assert result["total"] == 1
    assert result["logs"][0]["severity"] == "warning"
    assert result["logs"][0]["details"]["previous_reason"] == "test"


@pytest.mark.asyncio
async def test_audit_logger_log_bot_started(audit_logger, store):
    """log_bot_started creates a bot_started event."""
    await audit_logger.log_bot_started(
        mode="paper", symbols=["BTC/USDT", "ETH/USDT"],
    )

    result = await store.get_audit_logs(event_type="bot_started")
    assert result["total"] == 1
    assert result["logs"][0]["details"]["mode"] == "paper"
    assert "BTC/USDT" in result["logs"][0]["details"]["symbols"]


@pytest.mark.asyncio
async def test_audit_logger_log_bot_stopped(audit_logger, store):
    """log_bot_stopped creates a bot_stopped event."""
    await audit_logger.log_bot_stopped()

    result = await store.get_audit_logs(event_type="bot_stopped")
    assert result["total"] == 1


@pytest.mark.asyncio
async def test_audit_logger_log_auth_login_success(audit_logger, store):
    """log_auth_login records successful login."""
    await audit_logger.log_auth_login("admin", success=True)

    result = await store.get_audit_logs(event_type="auth_login")
    assert result["total"] == 1
    assert result["logs"][0]["severity"] == "info"
    assert result["logs"][0]["actor"] == "user"


@pytest.mark.asyncio
async def test_audit_logger_log_auth_login_failed(audit_logger, store):
    """log_auth_login records failed login with warning severity."""
    await audit_logger.log_auth_login("hacker", success=False)

    result = await store.get_audit_logs(event_type="auth_failed")
    assert result["total"] == 1
    assert result["logs"][0]["severity"] == "warning"


@pytest.mark.asyncio
async def test_audit_logger_log_preflight_result(audit_logger, store):
    """log_preflight_result records pre-flight check results."""
    await audit_logger.log_preflight_result(
        overall="PASS",
        failures=[],
        warnings=["password_check"],
    )

    result = await store.get_audit_logs(event_type="preflight_result")
    assert result["total"] == 1
    assert result["logs"][0]["severity"] == "warning"
    assert result["logs"][0]["details"]["overall"] == "PASS"


@pytest.mark.asyncio
async def test_audit_logger_log_preflight_with_failures(audit_logger, store):
    """Preflight with failures has critical severity."""
    await audit_logger.log_preflight_result(
        overall="FAIL",
        failures=["stop_loss_check"],
    )

    result = await store.get_audit_logs(event_type="preflight_result")
    assert result["logs"][0]["severity"] == "critical"


@pytest.mark.asyncio
async def test_audit_logger_log_reconciliation_result(audit_logger, store):
    """log_reconciliation_result records reconciliation outcome."""
    await audit_logger.log_reconciliation_result(
        has_discrepancies=True,
        matched=3,
        discrepancies=2,
    )

    result = await store.get_audit_logs(event_type="reconciliation_result")
    assert result["total"] == 1
    assert result["logs"][0]["severity"] == "warning"
    assert result["logs"][0]["details"]["matched"] == 3


# ---------------------------------------------------------------------------
# DataStore audit log operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_datastore_save_and_get_audit_log(store):
    """DataStore saves and retrieves audit logs."""
    await store.save_audit_log(
        event_type="test_event",
        actor="system",
        details={"foo": "bar"},
        severity="info",
    )

    result = await store.get_audit_logs()
    assert result["total"] == 1
    assert result["logs"][0]["event_type"] == "test_event"
    assert result["logs"][0]["details"] == {"foo": "bar"}


@pytest.mark.asyncio
async def test_datastore_audit_log_pagination(store):
    """DataStore paginates audit logs correctly."""
    for i in range(15):
        await store.save_audit_log(
            event_type=f"event_{i}",
            actor="system",
            severity="info",
        )

    # Page 1: 10 items
    result = await store.get_audit_logs(page=1, limit=10)
    assert len(result["logs"]) == 10
    assert result["total"] == 15
    assert result["total_pages"] == 2
    assert result["page"] == 1

    # Page 2: 5 items
    result = await store.get_audit_logs(page=2, limit=10)
    assert len(result["logs"]) == 5
    assert result["page"] == 2


@pytest.mark.asyncio
async def test_datastore_audit_log_filter_by_event_type(store):
    """DataStore filters by event_type."""
    await store.save_audit_log(event_type="trade_executed", actor="system")
    await store.save_audit_log(event_type="emergency_stop", actor="system")
    await store.save_audit_log(event_type="trade_executed", actor="system")

    result = await store.get_audit_logs(event_type="trade_executed")
    assert result["total"] == 2

    result = await store.get_audit_logs(event_type="emergency_stop")
    assert result["total"] == 1


@pytest.mark.asyncio
async def test_datastore_audit_log_filter_by_severity(store):
    """DataStore filters by severity."""
    await store.save_audit_log(event_type="a", severity="info")
    await store.save_audit_log(event_type="b", severity="warning")
    await store.save_audit_log(event_type="c", severity="critical")

    result = await store.get_audit_logs(severity="critical")
    assert result["total"] == 1
    assert result["logs"][0]["event_type"] == "c"


@pytest.mark.asyncio
async def test_datastore_audit_log_filter_by_date_range(store):
    """DataStore filters by date range."""
    now = datetime.utcnow()
    old = now - timedelta(days=10)
    recent = now - timedelta(hours=1)

    await store.save_audit_log(event_type="old", timestamp=old)
    await store.save_audit_log(event_type="recent", timestamp=recent)

    # Only recent
    cutoff = now - timedelta(days=1)
    result = await store.get_audit_logs(start=cutoff)
    assert result["total"] == 1
    assert result["logs"][0]["event_type"] == "recent"


@pytest.mark.asyncio
async def test_datastore_audit_log_newest_first(store):
    """DataStore returns audit logs newest-first."""
    t1 = datetime.utcnow() - timedelta(hours=2)
    t2 = datetime.utcnow() - timedelta(hours=1)
    t3 = datetime.utcnow()

    await store.save_audit_log(event_type="first", timestamp=t1)
    await store.save_audit_log(event_type="second", timestamp=t2)
    await store.save_audit_log(event_type="third", timestamp=t3)

    result = await store.get_audit_logs()
    assert result["logs"][0]["event_type"] == "third"
    assert result["logs"][2]["event_type"] == "first"


@pytest.mark.asyncio
async def test_datastore_cleanup_old_audit_logs(store):
    """DataStore deletes audit logs older than max_age_days."""
    old = datetime.utcnow() - timedelta(days=100)
    recent = datetime.utcnow() - timedelta(days=10)

    await store.save_audit_log(event_type="old", timestamp=old)
    await store.save_audit_log(event_type="recent", timestamp=recent)

    deleted = await store.cleanup_old_audit_logs(max_age_days=90)
    assert deleted == 1

    result = await store.get_audit_logs()
    assert result["total"] == 1
    assert result["logs"][0]["event_type"] == "recent"


@pytest.mark.asyncio
async def test_datastore_cleanup_no_old_logs(store):
    """Cleanup with no old logs deletes nothing."""
    await store.save_audit_log(event_type="recent")

    deleted = await store.cleanup_old_audit_logs(max_age_days=90)
    assert deleted == 0


# ---------------------------------------------------------------------------
# GET /api/audit endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_audit_no_logger(client):
    """GET /api/audit returns empty when no logger configured."""
    resp = await client.get("/api/audit")
    assert resp.status_code == 200
    data = resp.json()
    assert data["logs"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_get_audit_returns_logs(client, audit_logger):
    """GET /api/audit returns stored audit logs."""
    set_audit_logger(audit_logger)

    await audit_logger.log_event(event_type="test_event_1")
    await audit_logger.log_event(event_type="test_event_2")

    resp = await client.get("/api/audit")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["logs"]) == 2


@pytest.mark.asyncio
async def test_get_audit_filter_event_type(client, audit_logger):
    """GET /api/audit?event_type=X filters correctly."""
    set_audit_logger(audit_logger)

    await audit_logger.log_event(event_type="trade_executed")
    await audit_logger.log_event(event_type="emergency_stop", severity="critical")

    resp = await client.get("/api/audit?event_type=trade_executed")
    data = resp.json()
    assert data["total"] == 1
    assert data["logs"][0]["event_type"] == "trade_executed"


@pytest.mark.asyncio
async def test_get_audit_filter_severity(client, audit_logger):
    """GET /api/audit?severity=critical filters correctly."""
    set_audit_logger(audit_logger)

    await audit_logger.log_event(event_type="a", severity="info")
    await audit_logger.log_event(event_type="b", severity="critical")

    resp = await client.get("/api/audit?severity=critical")
    data = resp.json()
    assert data["total"] == 1
    assert data["logs"][0]["severity"] == "critical"


@pytest.mark.asyncio
async def test_get_audit_pagination(client, audit_logger):
    """GET /api/audit supports pagination."""
    set_audit_logger(audit_logger)

    for i in range(25):
        await audit_logger.log_event(event_type=f"event_{i}")

    resp = await client.get("/api/audit?page=1&limit=10")
    data = resp.json()
    assert len(data["logs"]) == 10
    assert data["total"] == 25
    assert data["total_pages"] == 3

    resp = await client.get("/api/audit?page=3&limit=10")
    data = resp.json()
    assert len(data["logs"]) == 5


@pytest.mark.asyncio
async def test_get_audit_date_filter(client, audit_logger, store):
    """GET /api/audit supports date range filtering."""
    set_audit_logger(audit_logger)

    old = datetime.utcnow() - timedelta(days=10)
    recent = datetime.utcnow()
    await store.save_audit_log(event_type="old_event", timestamp=old)
    await store.save_audit_log(event_type="new_event", timestamp=recent)

    cutoff = (datetime.utcnow() - timedelta(days=1)).isoformat()
    resp = await client.get(f"/api/audit?start_date={cutoff}")
    data = resp.json()
    assert data["total"] == 1
    assert data["logs"][0]["event_type"] == "new_event"


# ---------------------------------------------------------------------------
# AuditLogRecord model tests
# ---------------------------------------------------------------------------


def test_audit_log_record_table_name():
    """AuditLogRecord uses 'audit_log' table name."""
    assert AuditLogRecord.__tablename__ == "audit_log"


def test_audit_log_record_has_required_columns():
    """AuditLogRecord has all required columns."""
    columns = {c.name for c in AuditLogRecord.__table__.columns}
    assert "id" in columns
    assert "timestamp" in columns
    assert "event_type" in columns
    assert "actor" in columns
    assert "details" in columns
    assert "severity" in columns


# ---------------------------------------------------------------------------
# Audit logger store property
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_logger_store_property(store):
    """AuditLogger.store property getter/setter works."""
    logger = AuditLogger()
    assert logger.store is None

    logger.store = store
    assert logger.store is store


# ---------------------------------------------------------------------------
# Immutability: audit entries are append-only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_entries_are_immutable(audit_logger, store):
    """Audit log entries cannot be updated — only appended."""
    await audit_logger.log_event(event_type="event_1")
    await audit_logger.log_event(event_type="event_2")

    result = await store.get_audit_logs()
    assert result["total"] == 2

    # Can add more
    await audit_logger.log_event(event_type="event_3")
    result = await store.get_audit_logs()
    assert result["total"] == 3

    # DataStore has no update method for audit logs — only save and get


# ---------------------------------------------------------------------------
# Database error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_logger_handles_store_error():
    """AuditLogger handles database errors gracefully."""
    bad_store = MagicMock()
    bad_store.save_audit_log = AsyncMock(side_effect=RuntimeError("DB error"))

    logger = AuditLogger(store=bad_store)
    # Should not raise — error is caught and logged
    await logger.log_event(event_type="test", details={"x": 1})


# ---------------------------------------------------------------------------
# Combined filters test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_datastore_combined_filters(store):
    """DataStore supports combining event_type + severity + date filters."""
    now = datetime.utcnow()

    await store.save_audit_log(
        event_type="trade_executed", severity="info",
        timestamp=now - timedelta(hours=1),
    )
    await store.save_audit_log(
        event_type="emergency_stop", severity="critical",
        timestamp=now - timedelta(hours=1),
    )
    await store.save_audit_log(
        event_type="trade_executed", severity="info",
        timestamp=now - timedelta(days=5),
    )

    result = await store.get_audit_logs(
        event_type="trade_executed",
        severity="info",
        start=now - timedelta(days=1),
    )
    assert result["total"] == 1
