"""Tests for V4-008: Web-based settings panel with hot-reload."""

import pytest
from httpx import ASGITransport, AsyncClient

from bot.config import SETTINGS_METADATA, Settings
from bot.dashboard.app import app, set_settings, update_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_settings(**overrides):
    """Create a real Settings instance with test defaults."""
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
        "dashboard_password": "changeme",
        "jwt_secret": "",
        "config_file": "__nonexistent__",
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset bot state and settings before each test."""
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
    set_settings(None)
    yield
    set_settings(None)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def settings():
    """A real Settings instance for testing."""
    return _make_settings()


# ---------------------------------------------------------------------------
# GET /api/settings tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_settings_returns_settings_list(client, settings):
    """GET /api/settings returns list of all settings with metadata."""
    set_settings(settings)
    resp = await client.get("/api/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "settings" in data
    assert isinstance(data["settings"], list)
    assert len(data["settings"]) > 0


@pytest.mark.asyncio
async def test_get_settings_without_settings_returns_500(client):
    """GET /api/settings returns 500 when settings not configured."""
    resp = await client.get("/api/settings")
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_get_settings_has_correct_fields(client, settings):
    """Each setting item has key, value, default, section, description, type, requires_restart."""
    set_settings(settings)
    resp = await client.get("/api/settings")
    data = resp.json()
    for item in data["settings"]:
        assert "key" in item
        assert "value" in item
        assert "default" in item
        assert "section" in item
        assert "description" in item
        assert "type" in item
        assert "requires_restart" in item


@pytest.mark.asyncio
async def test_get_settings_masks_secrets(client):
    """Secret fields like API keys show masked values, never the real key."""
    s = _make_settings(binance_api_key="my-real-api-key-12345")
    set_settings(s)
    resp = await client.get("/api/settings")
    data = resp.json()
    for item in data["settings"]:
        if item["key"] == "binance_api_key":
            assert item["value"] == "***configured***"
            assert "my-real-api-key" not in str(item)
            break
    else:
        pytest.fail("binance_api_key not found in settings")


@pytest.mark.asyncio
async def test_get_settings_masks_empty_secrets(client, settings):
    """Empty secret fields show empty string, not masked."""
    set_settings(settings)
    resp = await client.get("/api/settings")
    data = resp.json()
    for item in data["settings"]:
        if item["key"] == "binance_api_key":
            assert item["value"] == ""
            break


@pytest.mark.asyncio
async def test_get_settings_grouped_by_section(client, settings):
    """Settings have section metadata for frontend grouping."""
    set_settings(settings)
    resp = await client.get("/api/settings")
    data = resp.json()
    sections = {item["section"] for item in data["settings"]}
    assert "Risk Management" in sections
    assert "Trading" in sections
    assert "Strategies" in sections


@pytest.mark.asyncio
async def test_get_settings_restart_fields_flagged(client, settings):
    """Settings requiring restart have requires_restart=True."""
    set_settings(settings)
    resp = await client.get("/api/settings")
    data = resp.json()

    restart_fields = {
        item["key"] for item in data["settings"] if item["requires_restart"]
    }
    assert "trading_mode" in restart_fields
    assert "binance_api_key" in restart_fields
    assert "database_url" in restart_fields
    assert "dashboard_port" in restart_fields

    safe_fields = {
        item["key"] for item in data["settings"] if not item["requires_restart"]
    }
    assert "stop_loss_pct" in safe_fields
    assert "take_profit_pct" in safe_fields
    assert "signal_min_agreement" in safe_fields


@pytest.mark.asyncio
async def test_get_settings_shows_current_values(client):
    """Settings show actual current values, not just defaults."""
    s = _make_settings(stop_loss_pct=7.5, take_profit_pct=12.0)
    set_settings(s)
    resp = await client.get("/api/settings")
    data = resp.json()
    for item in data["settings"]:
        if item["key"] == "stop_loss_pct":
            assert item["value"] == 7.5
        if item["key"] == "take_profit_pct":
            assert item["value"] == 12.0


@pytest.mark.asyncio
async def test_get_settings_includes_defaults(client, settings):
    """Settings include default values for reference."""
    set_settings(settings)
    resp = await client.get("/api/settings")
    data = resp.json()
    for item in data["settings"]:
        if item["key"] == "stop_loss_pct":
            assert item["default"] == 3.0
        if item["key"] == "max_concurrent_positions":
            assert item["default"] == 5


@pytest.mark.asyncio
async def test_get_settings_select_fields_have_options(client, settings):
    """Select-type settings include list of allowed options."""
    set_settings(settings)
    resp = await client.get("/api/settings")
    data = resp.json()
    for item in data["settings"]:
        if item["key"] == "log_level":
            assert item["type"] == "select"
            assert "options" in item
            assert "INFO" in item["options"]
            assert "DEBUG" in item["options"]


# ---------------------------------------------------------------------------
# PUT /api/settings tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_settings_updates_safe_values(client, settings):
    """PUT /api/settings updates safe (hot-reloadable) settings."""
    set_settings(settings)
    resp = await client.put("/api/settings", json={
        "stop_loss_pct": 5.0,
        "take_profit_pct": 10.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "stop_loss_pct" in data["changed"]
    assert "take_profit_pct" in data["changed"]
    # Verify settings actually changed
    assert settings.stop_loss_pct == 5.0
    assert settings.take_profit_pct == 10.0


@pytest.mark.asyncio
async def test_put_settings_rejects_unsafe_values(client, settings):
    """PUT /api/settings rejects settings that require restart."""
    set_settings(settings)
    resp = await client.put("/api/settings", json={
        "trading_mode": "live",
    })
    assert resp.status_code == 400
    data = resp.json()
    assert "requires restart" in data["detail"]


@pytest.mark.asyncio
async def test_put_settings_rejects_unknown_fields(client, settings):
    """PUT /api/settings rejects unknown field names."""
    set_settings(settings)
    resp = await client.put("/api/settings", json={
        "nonexistent_field": 42,
    })
    assert resp.status_code == 400
    assert "Unknown" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_put_settings_returns_previous_values(client, settings):
    """PUT /api/settings returns previous values for undo support."""
    set_settings(settings)
    original_sl = settings.stop_loss_pct
    resp = await client.put("/api/settings", json={
        "stop_loss_pct": 8.0,
    })
    data = resp.json()
    assert data["previous"]["stop_loss_pct"] == original_sl


@pytest.mark.asyncio
async def test_put_settings_empty_body_returns_400(client, settings):
    """PUT /api/settings with empty body returns 400."""
    set_settings(settings)
    resp = await client.put("/api/settings", json={})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_put_settings_without_settings_returns_500(client):
    """PUT /api/settings returns 500 when settings not configured."""
    resp = await client.put("/api/settings", json={"stop_loss_pct": 5.0})
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_put_settings_unchanged_value_not_in_changed(client, settings):
    """Setting a value to its current value should not appear in changed list."""
    set_settings(settings)
    current_sl = settings.stop_loss_pct
    resp = await client.put("/api/settings", json={
        "stop_loss_pct": current_sl,
    })
    data = resp.json()
    assert "stop_loss_pct" not in data["changed"]


@pytest.mark.asyncio
async def test_put_settings_multiple_safe_values(client, settings):
    """PUT /api/settings can update multiple safe values at once."""
    set_settings(settings)
    resp = await client.put("/api/settings", json={
        "max_concurrent_positions": 3,
        "signal_min_agreement": 3,
        "trailing_stop_enabled": True,
    })
    data = resp.json()
    assert data["success"] is True
    assert len(data["changed"]) == 3
    assert settings.max_concurrent_positions == 3
    assert settings.signal_min_agreement == 3
    assert settings.trailing_stop_enabled is True


@pytest.mark.asyncio
async def test_put_settings_mixed_safe_unsafe_rejected(client, settings):
    """PUT /api/settings rejects entire request if any field is unsafe."""
    set_settings(settings)
    original_sl = settings.stop_loss_pct
    resp = await client.put("/api/settings", json={
        "stop_loss_pct": 9.0,
        "trading_mode": "live",
    })
    assert resp.status_code == 400
    # Safe value should NOT have been applied since request was rejected
    assert settings.stop_loss_pct == original_sl


# ---------------------------------------------------------------------------
# Settings.reload() unit tests
# ---------------------------------------------------------------------------


def test_reload_changes_safe_settings():
    """reload() updates safe settings and returns changed list."""
    s = _make_settings()
    changed = s.reload({"stop_loss_pct": 7.0, "take_profit_pct": 15.0})
    assert "stop_loss_pct" in changed
    assert "take_profit_pct" in changed
    assert s.stop_loss_pct == 7.0
    assert s.take_profit_pct == 15.0


def test_reload_rejects_unsafe_settings():
    """reload() raises ValueError for settings that require restart."""
    s = _make_settings()
    with pytest.raises(ValueError, match="requires restart"):
        s.reload({"trading_mode": "live"})


def test_reload_rejects_unknown_settings():
    """reload() raises ValueError for unknown fields."""
    s = _make_settings()
    with pytest.raises(ValueError, match="Unknown"):
        s.reload({"totally_bogus_field": 42})


def test_reload_unchanged_value_not_reported():
    """reload() does not report a field as changed if value is the same."""
    s = _make_settings(stop_loss_pct=3.0)
    changed = s.reload({"stop_loss_pct": 3.0})
    assert "stop_loss_pct" not in changed


def test_reload_multiple_fields():
    """reload() handles multiple field updates at once."""
    s = _make_settings()
    changed = s.reload({
        "max_concurrent_positions": 10,
        "signal_min_agreement": 4,
        "trailing_stop_enabled": True,
        "log_level": "DEBUG",
    })
    assert len(changed) == 4
    assert s.max_concurrent_positions == 10
    assert s.signal_min_agreement == 4
    assert s.trailing_stop_enabled is True
    assert s.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# SETTINGS_METADATA tests
# ---------------------------------------------------------------------------


def test_settings_metadata_covers_key_fields():
    """SETTINGS_METADATA covers essential fields."""
    assert "stop_loss_pct" in SETTINGS_METADATA
    assert "take_profit_pct" in SETTINGS_METADATA
    assert "trading_mode" in SETTINGS_METADATA
    assert "signal_min_agreement" in SETTINGS_METADATA
    assert "binance_api_key" in SETTINGS_METADATA
    assert "log_level" in SETTINGS_METADATA


def test_settings_metadata_safe_fields_are_reloadable():
    """Safe fields in SETTINGS_METADATA have requires_restart=False."""
    safe_keys = [
        "stop_loss_pct",
        "take_profit_pct",
        "max_concurrent_positions",
        "signal_min_agreement",
        "trailing_stop_enabled",
        "loop_interval_seconds",
    ]
    for key in safe_keys:
        assert key in SETTINGS_METADATA
        assert SETTINGS_METADATA[key]["requires_restart"] is False, f"{key} should be safe"


def test_settings_metadata_unsafe_fields_require_restart():
    """Unsafe fields in SETTINGS_METADATA have requires_restart=True."""
    unsafe_keys = [
        "trading_mode",
        "binance_api_key",
        "database_url",
        "dashboard_port",
    ]
    for key in unsafe_keys:
        assert key in SETTINGS_METADATA
        assert SETTINGS_METADATA[key]["requires_restart"] is True, f"{key} should require restart"


def test_settings_metadata_all_have_section():
    """Every entry in SETTINGS_METADATA has a section."""
    for key, meta in SETTINGS_METADATA.items():
        assert "section" in meta, f"{key} missing section"
        assert meta["section"], f"{key} has empty section"


def test_settings_metadata_all_have_description():
    """Every entry in SETTINGS_METADATA has a description."""
    for key, meta in SETTINGS_METADATA.items():
        assert "description" in meta, f"{key} missing description"
        assert meta["description"], f"{key} has empty description"
