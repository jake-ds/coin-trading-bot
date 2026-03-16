"""Tests for V4-012: Emergency kill switch via API and Telegram."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from bot.dashboard import auth as auth_module
from bot.dashboard.app import (
    app,
    set_settings,
    set_trading_bot,
    update_state,
)
from bot.monitoring.telegram import TelegramNotifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_settings(**overrides):
    """Create a mock settings object for emergency tests."""
    defaults = {
        "dashboard_username": "admin",
        "dashboard_password": "changeme",
        "jwt_secret": "",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset bot state, settings, and trading_bot before each test."""
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
    auth_module.clear_blacklist()
    yield
    set_settings(None)
    set_trading_bot(None)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _make_mock_bot(
    emergency_stopped=False,
    open_positions=None,
    pending_orders=None,
):
    """Create a mock TradingBot with emergency methods."""
    bot = AsyncMock()
    bot._emergency_stopped = emergency_stopped

    async def mock_stop(reason="manual"):
        bot._emergency_stopped = True
        update_state(
            emergency={
                "active": True,
                "activated_at": "2026-02-22T10:00:00+00:00",
                "reason": reason,
                "cancelled_orders": len(pending_orders or []),
            },
        )
        return {
            "success": True,
            "cancelled_orders": len(pending_orders or []),
            "activated_at": "2026-02-22T10:00:00+00:00",
        }

    async def mock_close_all(reason="manual"):
        bot._emergency_stopped = True
        positions = open_positions or []
        closed = [
            {
                "symbol": p["symbol"],
                "quantity": p["quantity"],
                "entry_price": p["entry_price"],
                "exit_price": p.get("current_price", 0),
                "pnl": round(
                    (p.get("current_price", 0) - p["entry_price"])
                    * p["quantity"],
                    2,
                ),
            }
            for p in positions
        ]
        update_state(
            emergency={
                "active": True,
                "activated_at": "2026-02-22T10:00:00+00:00",
                "reason": reason,
                "closed_positions": closed,
            },
            open_positions=[],
        )
        return {"success": True, "closed_positions": closed}

    async def mock_resume():
        if not bot._emergency_stopped:
            return {"success": False, "error": "Not in emergency stop state"}
        bot._emergency_stopped = False
        update_state(
            emergency={"active": False, "activated_at": None, "reason": None},
        )
        return {"success": True, "previous_reason": "api_request"}

    bot.emergency_stop = AsyncMock(side_effect=mock_stop)
    bot.emergency_close_all = AsyncMock(side_effect=mock_close_all)
    bot.emergency_resume = AsyncMock(side_effect=mock_resume)
    bot.emergency_state = {
        "active": emergency_stopped,
        "activated_at": None,
        "reason": None,
    }
    return bot


# ---------------------------------------------------------------------------
# GET /api/emergency tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_emergency_state_default(client):
    """GET /api/emergency returns inactive state by default."""
    resp = await client.get("/api/emergency")
    assert resp.status_code == 200
    data = resp.json()
    assert data["emergency"]["active"] is False
    assert data["emergency"]["reason"] is None


@pytest.mark.asyncio
async def test_get_emergency_state_when_active(client):
    """GET /api/emergency reflects active emergency state."""
    update_state(
        emergency={
            "active": True,
            "activated_at": "2026-02-22T10:00:00+00:00",
            "reason": "test",
        }
    )
    resp = await client.get("/api/emergency")
    assert resp.status_code == 200
    data = resp.json()
    assert data["emergency"]["active"] is True
    assert data["emergency"]["reason"] == "test"


# ---------------------------------------------------------------------------
# POST /api/emergency/stop tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emergency_stop_no_bot(client):
    """POST /api/emergency/stop returns 503 when bot not available."""
    resp = await client.post("/api/emergency/stop")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_emergency_stop_success(client):
    """POST /api/emergency/stop halts trading and cancels pending orders."""
    bot = _make_mock_bot(pending_orders=["order1", "order2"])
    set_trading_bot(bot)

    resp = await client.post("/api/emergency/stop")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["cancelled_orders"] == 2
    assert data["activated_at"] is not None

    bot.emergency_stop.assert_called_once_with(reason="api_request")


@pytest.mark.asyncio
async def test_emergency_stop_updates_dashboard_state(client):
    """Emergency stop updates the dashboard emergency state."""
    bot = _make_mock_bot()
    set_trading_bot(bot)

    await client.post("/api/emergency/stop")

    # Check state was updated
    resp = await client.get("/api/emergency")
    data = resp.json()
    assert data["emergency"]["active"] is True
    assert data["emergency"]["reason"] == "api_request"


# ---------------------------------------------------------------------------
# POST /api/emergency/close-all tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emergency_close_all_no_bot(client):
    """POST /api/emergency/close-all returns 503 when bot not available."""
    resp = await client.post("/api/emergency/close-all")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_emergency_close_all_success(client):
    """POST /api/emergency/close-all closes all positions."""
    positions = [
        {
            "symbol": "BTC/USDT",
            "quantity": 0.1,
            "entry_price": 50000,
            "current_price": 51000,
        },
        {
            "symbol": "ETH/USDT",
            "quantity": 1.0,
            "entry_price": 3000,
            "current_price": 3100,
        },
    ]
    update_state(open_positions=positions)
    bot = _make_mock_bot(open_positions=positions)
    set_trading_bot(bot)

    resp = await client.post("/api/emergency/close-all")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert len(data["closed_positions"]) == 2

    # Check PnL calculation
    btc_close = next(
        p for p in data["closed_positions"] if p["symbol"] == "BTC/USDT"
    )
    assert btc_close["pnl"] == 100.0  # (51000-50000)*0.1

    eth_close = next(
        p for p in data["closed_positions"] if p["symbol"] == "ETH/USDT"
    )
    assert eth_close["pnl"] == 100.0  # (3100-3000)*1.0


@pytest.mark.asyncio
async def test_emergency_close_all_clears_positions(client):
    """Emergency close all clears open_positions in dashboard state."""
    positions = [
        {
            "symbol": "BTC/USDT",
            "quantity": 0.1,
            "entry_price": 50000,
            "current_price": 51000,
        },
    ]
    update_state(open_positions=positions)
    bot = _make_mock_bot(open_positions=positions)
    set_trading_bot(bot)

    await client.post("/api/emergency/close-all")

    resp = await client.get("/api/positions")
    data = resp.json()
    assert data["positions"] == []


@pytest.mark.asyncio
async def test_emergency_close_all_with_no_positions(client):
    """Emergency close all works with no open positions."""
    bot = _make_mock_bot(open_positions=[])
    set_trading_bot(bot)

    resp = await client.post("/api/emergency/close-all")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert len(data["closed_positions"]) == 0


# ---------------------------------------------------------------------------
# POST /api/emergency/resume tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emergency_resume_no_bot(client):
    """POST /api/emergency/resume returns 503 when bot not available."""
    resp = await client.post("/api/emergency/resume")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_emergency_resume_success(client):
    """POST /api/emergency/resume resumes trading after stop."""
    bot = _make_mock_bot(emergency_stopped=True)
    set_trading_bot(bot)

    # First stop
    await client.post("/api/emergency/stop")

    # Then resume
    resp = await client.post("/api/emergency/resume")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_emergency_resume_clears_state(client):
    """Resume clears emergency state in dashboard."""
    bot = _make_mock_bot(emergency_stopped=True)
    set_trading_bot(bot)

    await client.post("/api/emergency/stop")
    await client.post("/api/emergency/resume")

    resp = await client.get("/api/emergency")
    data = resp.json()
    assert data["emergency"]["active"] is False


@pytest.mark.asyncio
async def test_emergency_resume_when_not_stopped(client):
    """Resume returns failure when not in emergency state."""
    bot = _make_mock_bot(emergency_stopped=False)
    set_trading_bot(bot)

    resp = await client.post("/api/emergency/resume")
    data = resp.json()
    assert data["success"] is False
    assert "error" in data


# ---------------------------------------------------------------------------
# TradingBot emergency_stop unit tests
# ---------------------------------------------------------------------------


def _make_bot():
    """Create a TradingBot with minimal attributes for emergency tests."""
    from bot.main import TradingBot

    bot = TradingBot.__new__(TradingBot)
    bot._emergency_stopped = False
    bot._emergency_stopped_at = None
    bot._emergency_reason = None
    bot._telegram = None
    bot._audit_logger = AsyncMock()

    # Engine manager with mock engine
    engine = AsyncMock()
    engine.pause = AsyncMock()
    engine.resume = AsyncMock()
    engine.positions = {}
    engine._remove_position = MagicMock()

    mgr = MagicMock()
    mgr.engines = {"onchain_trader": engine}
    bot._engine_manager = mgr
    return bot


@pytest.mark.asyncio
async def test_bot_emergency_stop_sets_flag():
    """TradingBot.emergency_stop sets _emergency_stopped flag."""
    bot = _make_bot()
    result = await bot.emergency_stop(reason="test")
    assert "cancelled_orders" in result
    assert bot._emergency_stopped is True
    assert bot._emergency_stopped_at is not None
    assert bot._emergency_reason == "test"


@pytest.mark.asyncio
async def test_bot_emergency_stop_pauses_engines():
    """TradingBot.emergency_stop pauses all engines."""
    bot = _make_bot()
    await bot.emergency_stop(reason="test")
    engine = bot._engine_manager.engines["onchain_trader"]
    engine.pause.assert_awaited_once()


@pytest.mark.asyncio
async def test_bot_emergency_stop_notifies_telegram():
    """TradingBot.emergency_stop sends Telegram notification."""
    bot = _make_bot()
    telegram = AsyncMock()
    telegram.send_message = AsyncMock(return_value=True)
    bot._telegram = telegram

    await bot.emergency_stop(reason="test")
    telegram.send_message.assert_called_once()
    msg = telegram.send_message.call_args[0][0]
    assert "EMERGENCY STOP" in msg


# ---------------------------------------------------------------------------
# TradingBot emergency_close_all unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bot_emergency_close_all_removes_positions():
    """TradingBot.emergency_close_all removes all engine positions."""
    bot = _make_bot()
    engine = bot._engine_manager.engines["onchain_trader"]
    engine.positions = {
        "BTC/USDT": {"symbol": "BTC/USDT", "quantity": 0.01},
        "ETH/USDT": {"symbol": "ETH/USDT", "quantity": 0.1},
    }

    result = await bot.emergency_close_all(reason="test")
    assert len(result["closed_positions"]) == 2
    assert bot._emergency_stopped is True


# ---------------------------------------------------------------------------
# TradingBot emergency_resume unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bot_emergency_resume_clears_flag():
    """TradingBot.emergency_resume clears emergency state."""
    bot = _make_bot()
    bot._emergency_stopped = True
    bot._emergency_stopped_at = "2026-01-01T00:00:00+00:00"
    bot._emergency_reason = "test"

    result = await bot.emergency_resume()
    assert result["success"] is True
    assert bot._emergency_stopped is False
    assert bot._emergency_stopped_at is None
    assert bot._emergency_reason is None


@pytest.mark.asyncio
async def test_bot_emergency_resume_fails_when_not_stopped():
    """TradingBot.emergency_resume fails when not in emergency state."""
    bot = _make_bot()
    result = await bot.emergency_resume()
    assert result["success"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# TradingBot emergency_state property tests
# ---------------------------------------------------------------------------


def test_bot_emergency_state_property():
    """emergency_state property returns current state dict."""
    from bot.main import TradingBot

    bot = TradingBot.__new__(TradingBot)
    bot._emergency_stopped = True
    bot._emergency_stopped_at = "2026-01-01T00:00:00+00:00"
    bot._emergency_reason = "test_reason"

    state = bot.emergency_state
    assert state["active"] is True
    assert state["activated_at"] == "2026-01-01T00:00:00+00:00"
    assert state["reason"] == "test_reason"


def test_bot_emergency_state_inactive():
    """emergency_state property shows inactive by default."""
    from bot.main import TradingBot

    bot = TradingBot.__new__(TradingBot)
    bot._emergency_stopped = False
    bot._emergency_stopped_at = None
    bot._emergency_reason = None

    state = bot.emergency_state
    assert state["active"] is False


# ---------------------------------------------------------------------------
# Telegram command handler tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_telegram_register_command():
    """TelegramNotifier.register_command stores callback."""
    notifier = TelegramNotifier(bot_token="test", chat_id="123")

    async def handler():
        return "ok"

    notifier.register_command("test", handler)
    assert "test" in notifier._commands


@pytest.mark.asyncio
async def test_telegram_command_handler_dispatches():
    """TelegramNotifier dispatches registered commands from updates."""
    notifier = TelegramNotifier(bot_token="test", chat_id="123")

    callback_called = False

    async def stop_handler():
        nonlocal callback_called
        callback_called = True
        return "Stopped!"

    notifier.register_command("stop", stop_handler)

    # Create mock update
    update = MagicMock()
    update.update_id = 1
    update.message = MagicMock()
    update.message.text = "/stop"
    update.message.chat_id = 123

    # Mock send_message
    notifier.send_message = AsyncMock(return_value=True)

    await notifier._handle_update(update)
    assert callback_called
    notifier.send_message.assert_called_once_with("Stopped!")


@pytest.mark.asyncio
async def test_telegram_command_ignores_wrong_chat():
    """TelegramNotifier ignores commands from non-configured chats."""
    notifier = TelegramNotifier(bot_token="test", chat_id="123")

    callback_called = False

    async def handler():
        nonlocal callback_called
        callback_called = True
        return "ok"

    notifier.register_command("stop", handler)

    update = MagicMock()
    update.update_id = 1
    update.message = MagicMock()
    update.message.text = "/stop"
    update.message.chat_id = 999  # Wrong chat

    await notifier._handle_update(update)
    assert not callback_called


@pytest.mark.asyncio
async def test_telegram_unknown_command_lists_available():
    """Unknown commands get a response listing available commands."""
    notifier = TelegramNotifier(bot_token="test", chat_id="123")

    async def handler():
        return "ok"

    notifier.register_command("stop", handler)
    notifier.register_command("resume", handler)
    notifier.send_message = AsyncMock(return_value=True)

    update = MagicMock()
    update.update_id = 1
    update.message = MagicMock()
    update.message.text = "/unknown"
    update.message.chat_id = 123

    await notifier._handle_update(update)
    msg = notifier.send_message.call_args[0][0]
    assert "Unknown command" in msg
    assert "/stop" in msg
    assert "/resume" in msg


@pytest.mark.asyncio
async def test_telegram_command_with_bot_mention():
    """Commands with @botname suffix are parsed correctly."""
    notifier = TelegramNotifier(bot_token="test", chat_id="123")

    callback_called = False

    async def handler():
        nonlocal callback_called
        callback_called = True
        return "ok"

    notifier.register_command("stop", handler)
    notifier.send_message = AsyncMock(return_value=True)

    update = MagicMock()
    update.update_id = 1
    update.message = MagicMock()
    update.message.text = "/stop@my_trading_bot"
    update.message.chat_id = 123

    await notifier._handle_update(update)
    assert callback_called


@pytest.mark.asyncio
async def test_telegram_ignores_non_command_messages():
    """Non-command messages (no leading /) are ignored."""
    notifier = TelegramNotifier(bot_token="test", chat_id="123")

    callback_called = False

    async def handler():
        nonlocal callback_called
        callback_called = True
        return "ok"

    notifier.register_command("stop", handler)

    update = MagicMock()
    update.update_id = 1
    update.message = MagicMock()
    update.message.text = "hello there"
    update.message.chat_id = 123

    await notifier._handle_update(update)
    assert not callback_called


@pytest.mark.asyncio
async def test_telegram_start_stop_polling():
    """Start and stop command polling lifecycle."""
    notifier = TelegramNotifier(bot_token="test", chat_id="123")

    # Mock _process_updates to avoid actual API calls
    notifier._process_updates = AsyncMock()

    await notifier.start_command_polling(interval=0.01)
    assert notifier._polling is True
    assert notifier._polling_task is not None

    await asyncio.sleep(0.05)
    await notifier.stop_command_polling()
    assert notifier._polling is False


@pytest.mark.asyncio
async def test_telegram_command_error_sends_error_message():
    """Command handler errors send error message to chat."""
    notifier = TelegramNotifier(bot_token="test", chat_id="123")

    async def bad_handler():
        raise RuntimeError("boom")

    notifier.register_command("bad", bad_handler)
    notifier.send_message = AsyncMock(return_value=True)

    update = MagicMock()
    update.update_id = 1
    update.message = MagicMock()
    update.message.text = "/bad"
    update.message.chat_id = 123

    await notifier._handle_update(update)
    msg = notifier.send_message.call_args[0][0]
    assert "Error" in msg


@pytest.mark.asyncio
async def test_telegram_ignores_empty_messages():
    """Updates without message text are ignored."""
    notifier = TelegramNotifier(bot_token="test", chat_id="123")

    # No message
    update1 = MagicMock()
    update1.update_id = 1
    update1.message = None
    await notifier._handle_update(update1)

    # Message without text
    update2 = MagicMock()
    update2.update_id = 2
    update2.message = MagicMock()
    update2.message.text = None
    await notifier._handle_update(update2)


# ---------------------------------------------------------------------------
# WebSocket broadcast includes emergency state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_state_includes_emergency(client):
    """WebSocket state payload includes emergency state."""
    from bot.dashboard.app import _build_full_state_payload

    update_state(
        emergency={
            "active": True,
            "activated_at": "2026-02-22T10:00:00+00:00",
            "reason": "test",
        }
    )
    payload = await _build_full_state_payload()
    assert "emergency" in payload
    assert payload["emergency"]["active"] is True
    assert payload["emergency"]["reason"] == "test"


# ---------------------------------------------------------------------------
# Integration: full stop -> close-all -> resume flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_emergency_flow(client):
    """Full lifecycle: stop -> verify state -> resume -> verify state."""
    positions = [
        {
            "symbol": "BTC/USDT",
            "quantity": 0.5,
            "entry_price": 40000,
            "current_price": 42000,
        },
    ]
    update_state(open_positions=positions)
    bot = _make_mock_bot(open_positions=positions)
    set_trading_bot(bot)

    # 1. Emergency stop
    resp = await client.post("/api/emergency/stop")
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    # 2. Verify emergency state active
    resp = await client.get("/api/emergency")
    assert resp.json()["emergency"]["active"] is True

    # 3. Resume
    resp = await client.post("/api/emergency/resume")
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    # 4. Verify emergency state cleared
    resp = await client.get("/api/emergency")
    assert resp.json()["emergency"]["active"] is False


@pytest.mark.asyncio
async def test_close_all_flow(client):
    """Close-all flow: close positions then resume."""
    positions = [
        {
            "symbol": "BTC/USDT",
            "quantity": 0.1,
            "entry_price": 50000,
            "current_price": 51000,
        },
        {
            "symbol": "ETH/USDT",
            "quantity": 2.0,
            "entry_price": 3000,
            "current_price": 2900,
        },
    ]
    update_state(open_positions=positions)
    bot = _make_mock_bot(open_positions=positions)
    set_trading_bot(bot)

    # Close all
    resp = await client.post("/api/emergency/close-all")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert len(data["closed_positions"]) == 2

    # Positions should be cleared
    resp = await client.get("/api/positions")
    assert resp.json()["positions"] == []

    # Emergency should be active
    resp = await client.get("/api/emergency")
    assert resp.json()["emergency"]["active"] is True

    # Resume trading
    resp = await client.post("/api/emergency/resume")
    assert resp.json()["success"] is True
