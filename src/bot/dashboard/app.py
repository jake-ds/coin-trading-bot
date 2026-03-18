"""FastAPI monitoring dashboard with React SPA frontend."""

import asyncio
import collections
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bot.config import ENGINE_DESCRIPTIONS, SETTINGS_METADATA  # noqa: F401
from bot.dashboard.auth import (
    blacklist_refresh_token,
    create_access_token,
    create_refresh_token,
    is_auth_enabled,
    validate_access_token,
    validate_refresh_token,
    verify_credentials,
)
from bot.dashboard.websocket import ws_manager

logger = structlog.get_logger()

app = FastAPI(title="Coin Trading Bot Dashboard")

# Track application start time for uptime calculation
_app_start_time: float = time.time()

# CORS configuration — allowed_origins can be overridden via configure_cors()
_default_origins = ["http://localhost", "http://localhost:8000", "http://localhost:5173"]


def configure_cors(allowed_origins: list[str] | None = None) -> None:
    """Configure CORS middleware with the given origins."""
    origins = allowed_origins or _default_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Apply default CORS on import
configure_cors()

# Shared state (set by the TradingBot orchestrator)
_bot_state = {
    "status": "stopped",
    "started_at": None,
    "trades": [],
    "metrics": {},
    "portfolio": {"balances": {}, "positions": [], "total_value": 0.0},
    "cycle_metrics": {
        "cycle_count": 0,
        "average_cycle_duration": 0.0,
        "last_cycle_time": None,
    },
    "strategy_stats": {},
    "equity_curve": [],
    "open_positions": [],
    "regime": None,
    "cycle_log": [],
    "reconciliation": {},
    "preflight": {},
    "emergency": {"active": False, "activated_at": None, "reason": None},
}

# Reference to settings — set by main.py via set_settings()
_settings = None

# Reference to TradingBot — set by main.py via set_trading_bot()
_trading_bot = None

# Reference to AuditLogger — set by main.py via set_audit_logger()
_audit_logger = None

# Reference to DataStore for health check DB connectivity
_store_ref = None

# Reference to futures exchange adapter for live position fetching
_futures_exchange = None
# Cached exchange positions (to avoid hammering the API on every WS broadcast)
_exchange_positions_cache: list[dict] = []
_exchange_positions_cache_ts: float = 0.0
_EXCHANGE_POSITIONS_CACHE_TTL: float = 10.0  # seconds

# Reference to EngineManager — set by main.py via set_engine_manager()
_engine_manager = None

# Signal history — 24 hours at 5-minute intervals (288 slots)
_signal_history: collections.deque = collections.deque(maxlen=288)


def record_signal_snapshot() -> None:
    """Record current onchain signals into the history deque.

    Called after each engine cycle by the TradingBot orchestrator.
    """
    if _engine_manager is None:
        return
    engine = _engine_manager.get_engine("onchain_trader")
    if engine is None or not hasattr(engine, "latest_signals"):
        return
    signals = engine.latest_signals
    if not signals:
        return
    _signal_history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signals": signals,
    })


# Keep set_strategy_registry for backward compatibility with main.py
def set_strategy_registry(registry) -> None:
    """Set the strategy registry reference (no-op, kept for compatibility)."""
    pass


def set_settings(settings) -> None:
    """Set the settings reference for auth and config endpoints."""
    global _settings
    _settings = settings


def get_settings():
    """Get the current settings object."""
    return _settings


def set_trading_bot(bot) -> None:
    """Set the TradingBot reference for emergency endpoints."""
    global _trading_bot
    _trading_bot = bot


def get_trading_bot():
    """Get the current TradingBot reference."""
    return _trading_bot


def set_audit_logger(audit_logger) -> None:
    """Set the AuditLogger reference for audit endpoints and event logging."""
    global _audit_logger
    _audit_logger = audit_logger


def get_audit_logger():
    """Get the current AuditLogger reference."""
    return _audit_logger


def set_store_ref(store) -> None:
    """Set the DataStore reference for health check DB connectivity."""
    global _store_ref
    _store_ref = store


def set_futures_exchange(exchange) -> None:
    """Set the futures exchange adapter for live position fetching."""
    global _futures_exchange
    _futures_exchange = exchange


def set_engine_manager(manager) -> None:
    """Set the EngineManager reference for engine control endpoints."""
    global _engine_manager
    _engine_manager = manager


def get_engine_manager():
    """Get the current EngineManager reference."""
    return _engine_manager


def _get_cycle_metrics() -> dict:
    """Build cycle metrics from the engine manager's actual state."""
    if _engine_manager is None:
        return _bot_state["cycle_metrics"]
    total_cycles = 0
    total_duration = 0.0
    duration_count = 0
    last_cycle_time = None
    for engine in _engine_manager.engines.values():
        total_cycles += engine.cycle_count
        for result in engine.cycle_history[-10:]:
            total_duration += result.duration_ms
            duration_count += 1
            ts = result.timestamp
            if last_cycle_time is None or ts > last_cycle_time:
                last_cycle_time = ts
    avg_duration = (total_duration / duration_count) if duration_count else 0.0
    return {
        "cycle_count": total_cycles,
        "average_cycle_duration": round(avg_duration, 1),
        "last_cycle_time": last_cycle_time,
    }


# ---------------------------------------------------------------------------
# Auth request/response models
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str
    password: str
    remember_me: bool = False


class RefreshRequest(BaseModel):
    refresh_token: str


# ---------------------------------------------------------------------------
# Auth dependency — skipped when auth is disabled (password='changeme')
# ---------------------------------------------------------------------------


async def require_auth(request: Request) -> str | None:
    """Validate JWT token on protected routes. Returns username or None.

    Auth is disabled when dashboard_password='changeme', allowing all requests.
    """
    if _settings is None or not is_auth_enabled(_settings):
        return None  # Auth disabled — allow all

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return None  # Will be checked by routes that need it

    token = auth_header[7:]
    username = validate_access_token(_settings, token)
    return username


async def require_auth_strict(request: Request) -> str:
    """Strict auth: returns 401 if auth is enabled and token is invalid."""
    if _settings is None or not is_auth_enabled(_settings):
        return "anonymous"

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise _auth_error()

    token = auth_header[7:]
    username = validate_access_token(_settings, token)
    if username is None:
        raise _auth_error()
    return username


def _auth_error():
    """Create an HTTP 401 exception."""
    from fastapi import HTTPException

    return HTTPException(
        status_code=401,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


# API router — all data endpoints live under /api/*
# Protected by require_auth_strict (no-op when auth is disabled)
api_router = APIRouter(prefix="/api", dependencies=[Depends(require_auth_strict)])


def get_state() -> dict:
    """Get the current bot state."""
    return _bot_state


def update_state(**kwargs) -> None:
    """Update the bot state."""
    _bot_state.update(kwargs)


# ---------------------------------------------------------------------------
# Auth endpoints (under /api/auth/ — no auth required)
# ---------------------------------------------------------------------------

auth_router = APIRouter(prefix="/api/auth")


@auth_router.post("/login")
async def login(body: LoginRequest):
    """Authenticate and return JWT tokens."""
    if _settings is None:
        return JSONResponse(
            status_code=500,
            content={"detail": "Settings not configured"},
        )

    if not is_auth_enabled(_settings):
        return JSONResponse(
            status_code=400,
            content={"detail": "Auth is disabled (default password). "
                     "Set DASHBOARD_PASSWORD to enable."},
        )

    if not verify_credentials(_settings, body.username, body.password):
        logger.warning("auth_login_failed", username=body.username)
        if _audit_logger:
            asyncio.ensure_future(_audit_logger.log_auth_login(body.username, success=False))
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid credentials"},
        )

    access_token = create_access_token(_settings, body.username)
    refresh_token = create_refresh_token(_settings, body.username, remember_me=body.remember_me)
    logger.info("auth_login_success", username=body.username)
    if _audit_logger:
        asyncio.ensure_future(_audit_logger.log_auth_login(body.username, success=True))
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@auth_router.post("/refresh")
async def refresh(body: RefreshRequest):
    """Refresh an access token using a refresh token."""
    if _settings is None:
        return JSONResponse(
            status_code=500,
            content={"detail": "Settings not configured"},
        )

    username = validate_refresh_token(_settings, body.refresh_token)
    if username is None:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or expired refresh token"},
        )

    access_token = create_access_token(_settings, username)
    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@auth_router.post("/logout")
async def logout(body: RefreshRequest):
    """Invalidate a refresh token (logout)."""
    if _settings is None:
        return {"success": True}

    blacklist_refresh_token(_settings, body.refresh_token)
    logger.info("auth_logout")
    return {"success": True}


@auth_router.get("/status")
async def auth_status():
    """Check if authentication is enabled and current auth state."""
    enabled = _settings is not None and is_auth_enabled(_settings)
    return {
        "auth_enabled": enabled,
        "dev_mode": not enabled,
    }


app.include_router(auth_router)


# ---------------------------------------------------------------------------
# API endpoints (under /api/ prefix)
# ---------------------------------------------------------------------------


@api_router.get("/status")
async def get_status():
    """Get bot status."""
    result = {
        "status": _bot_state["status"],
        "started_at": _bot_state["started_at"],
        "cycle_metrics": _get_cycle_metrics(),
    }
    # Include rate limit info if available
    rate_limits = _bot_state.get("rate_limits")
    if rate_limits:
        result["rate_limits"] = rate_limits
    return result


@api_router.get("/trades")
async def get_trades(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    symbol: str | None = Query(None, description="Filter by symbol"),
):
    """Get trades with pagination and optional symbol filter."""
    all_trades: list[dict] = []

    # Read from EngineTracker (in-memory, restored from DB on startup)
    if _engine_manager is not None:
        for engine_name, trades in _engine_manager.tracker._trades.items():
            for t in trades:
                trade_dict = {
                    "engine": t.engine_name,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": round(t.pnl, 4),
                    "cost": round(t.cost, 4),
                    "net_pnl": round(t.net_pnl, 4),
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "hold_time_seconds": t.hold_time_seconds,
                    "mode": t.mode,
                }
                all_trades.append(trade_dict)

    # Sort by exit_time
    all_trades.sort(key=lambda t: t.get("exit_time", ""))

    # Apply symbol filter
    if symbol:
        all_trades = [t for t in all_trades if t.get("symbol") == symbol]

    total = len(all_trades)

    # Reverse for newest-first, then paginate
    reversed_trades = list(reversed(all_trades))
    start = (page - 1) * limit
    end = start + limit
    page_trades = reversed_trades[start:end]

    return {
        "trades": page_trades,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": max(1, (total + limit - 1) // limit),
    }


def _get_all_positions() -> list[dict]:
    """Return positions from engine internal state, fallback to bot state."""
    positions = []

    # Add engine-tracked positions
    if _engine_manager is not None:
        for engine_name, engine in _engine_manager.engines.items():
            for pos in engine.positions.values():
                sym = pos.get("symbol", "")
                pos_data = {
                    "symbol": sym,
                    "quantity": pos.get("quantity", 0),
                    "entry_price": pos.get("entry_price", 0),
                    "current_price": pos.get("current_price", pos.get("entry_price", 0)),
                    "unrealized_pnl": pos.get("unrealized_pnl", 0.0),
                    "stop_loss": 0,
                    "take_profit": 0,
                    "opened_at": pos.get("opened_at", ""),
                    "strategy": f"engine:{engine_name}",
                    "signal_score": pos.get("signal_score", 0),
                    "signal_confidence": pos.get("signal_confidence", 0),
                }
                positions.append(pos_data)

    # Fallback to bot state if no engine positions
    if not positions:
        positions = list(_bot_state.get("open_positions", []))

    return positions


@api_router.get("/positions")
async def get_positions():
    """Get current open positions with SL/TP info."""
    return {"positions": _get_all_positions()}


@api_router.get("/onchain-signals")
async def get_onchain_signals():
    """Get current on-chain composite signals for all symbols."""
    if _engine_manager is None:
        return {"signals": {}}
    engine = _engine_manager.get_engine("onchain_trader")
    if engine is None or not hasattr(engine, "latest_signals"):
        return {"signals": {}}
    return {"signals": engine.latest_signals}


@api_router.get("/signals/history")
async def get_signal_history(
    hours: float = Query(24, ge=1, le=48, description="Hours of history"),
):
    """Get signal history time-series data."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cutoff_str = cutoff.isoformat()
    filtered = [
        entry for entry in _signal_history
        if entry["timestamp"] >= cutoff_str
    ]
    return {"history": filtered}


@api_router.get("/portfolio")
async def get_portfolio():
    """Get current portfolio."""
    return {"portfolio": _bot_state["portfolio"]}


@api_router.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    return {"metrics": _bot_state["metrics"]}


@api_router.get("/emergency")
async def get_emergency_state():
    """Get current emergency stop state."""
    return {"emergency": _bot_state.get("emergency", {"active": False})}


@api_router.post("/emergency/stop")
async def emergency_stop():
    """Emergency stop: halt trading, cancel pending orders, keep positions open."""
    if _trading_bot is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Trading bot not available"},
        )
    result = await _trading_bot.emergency_stop(reason="api_request")
    return result


@api_router.post("/emergency/close-all")
async def emergency_close_all():
    """Emergency close all: halt trading + close all positions at market price."""
    if _trading_bot is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Trading bot not available"},
        )
    result = await _trading_bot.emergency_close_all(reason="api_request")
    return result


@api_router.post("/emergency/resume")
async def emergency_resume():
    """Resume trading after emergency stop."""
    if _trading_bot is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Trading bot not available"},
        )
    result = await _trading_bot.emergency_resume()
    return result


# ---------------------------------------------------------------------------
# Settings endpoints
# ---------------------------------------------------------------------------

# Track the last saved config snapshot for undo
_settings_previous: dict | None = None


@api_router.get("/settings")
async def get_settings_endpoint():
    """Get current configuration with metadata. Sensitive fields are masked."""
    if _settings is None:
        return JSONResponse(
            status_code=500,
            content={"detail": "Settings not configured"},
        )

    settings_list = []
    for field_name, meta in SETTINGS_METADATA.items():
        if not hasattr(_settings, field_name):
            continue
        current_value = getattr(_settings, field_name)

        # Get default value from model fields (access from class, not instance)
        field_info = type(_settings).model_fields.get(field_name)
        default_value = None
        if field_info is not None:
            default_value = field_info.default
            # Handle default_factory
            if default_value is None and field_info.default_factory is not None:
                default_value = field_info.default_factory()

        # Mask sensitive fields
        display_value = current_value
        if meta.get("type") == "secret":
            if current_value:
                display_value = "***configured***"
            else:
                display_value = ""

        # Convert enums to string
        if hasattr(display_value, "value"):
            display_value = display_value.value
        if hasattr(default_value, "value"):
            default_value = default_value.value

        settings_list.append({
            "key": field_name,
            "value": display_value,
            "default": default_value,
            "section": meta.get("section", "Other"),
            "description": meta.get("description", ""),
            "type": meta.get("type", "str"),
            "requires_restart": meta.get("requires_restart", False),
            "options": meta.get("options"),
        })

    return {"settings": settings_list}


@api_router.put("/settings")
async def update_settings_endpoint(body: dict):
    """Update bot settings at runtime (hot-reload safe settings only)."""
    global _settings_previous

    if _settings is None:
        return JSONResponse(
            status_code=500,
            content={"detail": "Settings not configured"},
        )

    if not body:
        return JSONResponse(
            status_code=400,
            content={"detail": "No settings provided"},
        )

    # Validate fields exist and are not unsafe
    errors = []
    for key, value in body.items():
        if key not in SETTINGS_METADATA:
            errors.append(f"Unknown setting: {key}")
            continue
        meta = SETTINGS_METADATA[key]
        if meta.get("requires_restart", False):
            errors.append(
                f"Setting '{key}' requires restart and cannot be changed at runtime"
            )

    if errors:
        return JSONResponse(
            status_code=400,
            content={"detail": "; ".join(errors), "errors": errors},
        )

    # Snapshot current values for undo
    snapshot = {}
    for key in body:
        if hasattr(_settings, key):
            val = getattr(_settings, key)
            if hasattr(val, "value"):
                val = val.value
            snapshot[key] = val
    _settings_previous = snapshot

    # Apply changes via hot-reload
    try:
        changed = _settings.reload(body)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"detail": str(e)},
        )

    logger.info("settings_updated", changed=changed)
    # Audit log
    if _audit_logger:
        asyncio.ensure_future(_audit_logger.log_config_changed(
            changed=changed, previous=snapshot,
        ))
    return {
        "success": True,
        "changed": changed,
        "previous": snapshot,
    }


@api_router.get("/health")
async def api_health_check(
    detailed: bool = False,
):
    """Health check endpoint under /api prefix."""
    return await health_check(detailed=detailed)


# Include the API router
app.include_router(api_router)


# ---------------------------------------------------------------------------
# Engine management endpoints (under /api/engines)
# ---------------------------------------------------------------------------

engine_router = APIRouter(
    prefix="/api/engines", dependencies=[Depends(require_auth_strict)]
)


@engine_router.get("")
async def list_engines():
    """List all engines with status, descriptions, and symbols."""
    if _engine_manager is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Engine mode not enabled"},
        )
    status = _engine_manager.get_status()
    for name, info in status.items():
        desc = ENGINE_DESCRIPTIONS.get(name, {})
        info["role_ko"] = desc.get("role_ko", "")
        info["role_en"] = desc.get("role_en", "")
        info["description_ko"] = desc.get("description_ko", "")
        info["key_params"] = desc.get("key_params", "")
        # Add tracked symbols from settings
        if name == "onchain_trader" and _settings:
            info["symbols"] = getattr(_settings, "onchain_symbols", [])
        else:
            info["symbols"] = []
        # Add onchain signals if available
        engine_obj = _engine_manager.get_engine(name)
        if engine_obj and hasattr(engine_obj, "latest_signals"):
            info["onchain_signals"] = engine_obj.latest_signals
    return status


@engine_router.post("/{name}/start")
async def start_engine(name: str):
    """Start a specific engine."""
    if _engine_manager is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Engine mode not enabled"},
        )
    success = await _engine_manager.start_engine(name)
    if not success:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Engine '{name}' not found or already running"},
        )
    return {"success": True, "engine": name, "action": "started"}


@engine_router.post("/{name}/stop")
async def stop_engine(name: str):
    """Stop a specific engine."""
    if _engine_manager is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Engine mode not enabled"},
        )
    success = await _engine_manager.stop_engine(name)
    if not success:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Engine '{name}' not found"},
        )
    return {"success": True, "engine": name, "action": "stopped"}


@engine_router.post("/{name}/pause")
async def pause_engine(name: str):
    """Pause a specific engine."""
    if _engine_manager is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Engine mode not enabled"},
        )
    success = await _engine_manager.pause_engine(name)
    if not success:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Engine '{name}' not found"},
        )
    return {"success": True, "engine": name, "action": "paused"}


@engine_router.post("/{name}/resume")
async def resume_engine(name: str):
    """Resume a paused engine."""
    if _engine_manager is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Engine mode not enabled"},
        )
    success = await _engine_manager.resume_engine(name)
    if not success:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Engine '{name}' not found"},
        )
    return {"success": True, "engine": name, "action": "resumed"}


@engine_router.get("/{name}/cycle-log")
async def engine_cycle_log(name: str):
    """Get cycle log for a specific engine."""
    if _engine_manager is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Engine mode not enabled"},
        )
    logs = _engine_manager.get_engine_cycle_log(name)
    return {"engine": name, "cycle_log": logs}


@engine_router.get("/{name}/params")
async def engine_params(name: str):
    """Get engine-specific config params, description metadata, and symbols."""
    if _engine_manager is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Engine mode not enabled"},
        )
    # Engine param prefix mapping
    _prefix_map = {
        "onchain_trader": "onchain_",
    }
    prefix = _prefix_map.get(name)
    if prefix is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Unknown engine: {name}"},
        )

    # Gather config params for this engine
    params = {}
    if _settings:
        for field_name in type(_settings).model_fields:
            if field_name.startswith(prefix):
                params[field_name] = getattr(_settings, field_name, None)

    desc = ENGINE_DESCRIPTIONS.get(name, {})
    return {
        "engine": name,
        "description": desc,
        "params": params,
    }


@engine_router.get("/{name}/positions")
async def engine_positions(name: str):
    """Get current positions for a specific engine (live from exchange)."""
    if _engine_manager is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Engine mode not enabled"},
        )
    positions = await _engine_manager.get_live_positions(name)
    return {"engine": name, "positions": positions}


@engine_router.get("/{name}/metrics")
async def engine_metrics(
    name: str,
    hours: float = Query(24, ge=0, description="Time window in hours"),
):
    """Get performance metrics for a specific engine."""
    if _engine_manager is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Engine mode not enabled"},
        )
    metrics = _engine_manager.tracker.get_metrics(name, window_hours=hours)
    return {"engine": name, "metrics": metrics.to_dict(), "window_hours": hours}


app.include_router(engine_router)


# ---------------------------------------------------------------------------
# WebSocket endpoint (at /api/ws — registered directly on app)
# ---------------------------------------------------------------------------


@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    await ws_manager.connect(websocket)
    # Send current state immediately on connect
    await ws_manager.send_personal(websocket, {
        "type": "status_update",
        "payload": _build_full_state_payload(),
    })
    try:
        while True:
            # Keep connection alive; client may send pings or commands
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


def _build_engine_performance_summary() -> dict:
    """Build abbreviated per-engine performance for WebSocket broadcasts."""
    if _engine_manager is None:
        return {}
    try:
        all_metrics = _engine_manager.tracker.get_all_metrics(window_hours=24)
        return {
            name: {
                "pnl": round(m.total_pnl, 4),
                "win_rate": round(m.win_rate, 4),
                "total_trades": m.total_trades,
            }
            for name, m in all_metrics.items()
        }
    except Exception:
        return {}


def _build_onchain_signals() -> dict:
    """Get current onchain signals from engine for WebSocket payload."""
    if _engine_manager is None:
        return {}
    engine = _engine_manager.get_engine("onchain_trader")
    if engine is None or not hasattr(engine, "latest_signals"):
        return {}
    return engine.latest_signals


def _build_recent_trades(limit: int = 50) -> list[dict]:
    """Get recent trades from EngineTracker for WebSocket/API."""
    if _engine_manager is None:
        return []
    trades = []
    for engine_name, trade_list in _engine_manager.tracker._trades.items():
        for t in trade_list:
            trades.append({
                "engine": t.engine_name,
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": round(t.net_pnl, 4),
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "mode": t.mode,
            })
    trades.sort(key=lambda t: t.get("exit_time", ""), reverse=True)
    return trades[:limit]


def _build_full_state_payload() -> dict:
    """Build a complete state payload for WebSocket broadcast."""
    return {
        "status": _bot_state["status"],
        "started_at": _bot_state["started_at"],
        "cycle_metrics": _get_cycle_metrics(),
        "portfolio": _bot_state["portfolio"],
        "metrics": _bot_state["metrics"],
        "trades": _build_recent_trades(50),
        "open_positions": _get_all_positions(),
        "emergency": _bot_state.get("emergency", {"active": False}),
        "engine_performance": _build_engine_performance_summary(),
        "onchain_signals": _build_onchain_signals(),
    }


async def broadcast_state_update() -> None:
    """Broadcast the full dashboard state to all WebSocket clients (rate-limited)."""
    await ws_manager.broadcast({
        "type": "status_update",
        "payload": _build_full_state_payload(),
    })


async def broadcast_trade(trade_info: dict) -> None:
    """Broadcast a trade event immediately to all WebSocket clients."""
    await ws_manager.broadcast_immediate({
        "type": "trade",
        "payload": trade_info,
    })


async def broadcast_position_change(positions: list[dict]) -> None:
    """Broadcast a position change event immediately."""
    await ws_manager.broadcast_immediate({
        "type": "position_change",
        "payload": {"positions": positions},
    })


async def broadcast_alert(message: str, severity: str = "info") -> None:
    """Broadcast an alert event immediately."""
    await ws_manager.broadcast_immediate({
        "type": "alert",
        "payload": {"message": message, "severity": severity},
    })


# ---------------------------------------------------------------------------
# Root-level routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check(
    detailed: bool = False,
):
    """Health check endpoint for Docker (no auth required).

    Status logic:
    - healthy: all engines RUNNING or no engine mode
    - degraded: some engines in ERROR state
    - unhealthy: all engines STOPPED or last cycle stale
    """
    now = datetime.now(timezone.utc).isoformat()
    uptime = round(time.time() - _app_start_time, 1)
    last_cycle = _bot_state["cycle_metrics"].get("last_cycle_time")
    bot_status = _bot_state["status"]

    # Determine health status based on engines
    status = "healthy"
    engines_info: dict = {}

    if _engine_manager is not None:
        engine_status_map = _engine_manager.get_status()
        running_count = 0
        error_count = 0
        stopped_count = 0
        total_engines = 0

        for name, info in engine_status_map.items():
            eng_status = info.get("status", "unknown")
            engines_info[name] = eng_status
            total_engines += 1
            if eng_status == "running":
                running_count += 1
            elif eng_status == "error":
                error_count += 1
            elif eng_status == "stopped":
                stopped_count += 1

        if total_engines > 0:
            if error_count > 0:
                status = "degraded"
            if stopped_count == total_engines:
                status = "unhealthy"

    # Stale cycle check (non-engine mode)
    if bot_status == "running" and last_cycle is not None:
        elapsed = time.time() - last_cycle
        if elapsed > 300:  # 5 minutes
            status = "unhealthy"

    # Check database connectivity
    db_connected = _store_ref is not None

    result: dict = {
        "status": status,
        "uptime_seconds": uptime,
        "engines": engines_info,
        "database_connected": db_connected,
        "timestamp": now,
    }

    if detailed:
        import os
        import shutil

        # Disk space
        try:
            disk = shutil.disk_usage(os.getcwd())
            result["disk_space_mb"] = round(disk.free / (1024 * 1024), 1)
        except Exception:
            result["disk_space_mb"] = None

        # Memory usage
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            # maxrss is in bytes on macOS, KB on Linux
            import platform
            rss = usage.ru_maxrss
            if platform.system() == "Darwin":
                result["memory_usage_mb"] = round(rss / (1024 * 1024), 1)
            else:
                result["memory_usage_mb"] = round(rss / 1024, 1)
        except Exception:
            result["memory_usage_mb"] = None

    return result


# ---------------------------------------------------------------------------
# SPA fallback — serve React frontend for non-API routes
# ---------------------------------------------------------------------------

_static_dir = Path(__file__).parent / "static"

# Mount static assets if the build directory exists
if _static_dir.exists() and (_static_dir / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_static_dir / "assets")), name="static-assets")


def _serve_index() -> FileResponse | HTMLResponse:
    """Serve index.html with no-cache headers to prevent stale dashboard."""
    index_file = _static_dir / "index.html"
    if index_file.exists():
        return FileResponse(
            str(index_file),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return HTMLResponse(
        "<h1>Frontend not built</h1>"
        "<p>Run <code>cd frontend && npm run build</code> to build the React dashboard.</p>"
    )


@app.get("/", response_class=HTMLResponse)
async def serve_root():
    """Serve React SPA index.html or fallback message."""
    return _serve_index()


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """SPA fallback: serve index.html for all unmatched routes."""
    # Try to serve a static file first
    static_file = _static_dir / full_path
    if static_file.exists() and static_file.is_file():
        return FileResponse(str(static_file))
    # Fall back to index.html for client-side routing
    return _serve_index()
