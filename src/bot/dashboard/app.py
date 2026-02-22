"""FastAPI monitoring dashboard with React SPA frontend."""

import asyncio
import html
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog
from fastapi import APIRouter, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from bot.dashboard.websocket import ws_manager

logger = structlog.get_logger()

app = FastAPI(title="Coin Trading Bot Dashboard")

# API router — all data endpoints live under /api/*
api_router = APIRouter(prefix="/api")

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
}

# Reference to strategy_registry — set by main.py via set_strategy_registry()
_strategy_registry = None


def set_strategy_registry(registry) -> None:
    """Set the strategy registry reference for toggle endpoint."""
    global _strategy_registry
    _strategy_registry = registry


def get_state() -> dict:
    """Get the current bot state."""
    return _bot_state


def update_state(**kwargs) -> None:
    """Update the bot state."""
    _bot_state.update(kwargs)


# ---------------------------------------------------------------------------
# API endpoints (under /api/ prefix)
# ---------------------------------------------------------------------------


@api_router.get("/status")
async def get_status():
    """Get bot status."""
    return {
        "status": _bot_state["status"],
        "started_at": _bot_state["started_at"],
        "cycle_metrics": _bot_state["cycle_metrics"],
    }


@api_router.get("/trades")
async def get_trades(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    symbol: str | None = Query(None, description="Filter by symbol"),
):
    """Get trades with pagination and optional symbol filter."""
    all_trades = _bot_state["trades"]

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


@api_router.get("/positions")
async def get_positions():
    """Get current open positions with SL/TP info."""
    return {"positions": _bot_state["open_positions"]}


@api_router.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    return {"metrics": _bot_state["metrics"]}


@api_router.get("/portfolio")
async def get_portfolio():
    """Get current portfolio."""
    return {"portfolio": _bot_state["portfolio"]}


@api_router.get("/strategies")
async def get_strategies():
    """Get per-strategy performance stats with active status from registry."""
    stats = _bot_state["strategy_stats"]
    # Merge active status from registry if available
    if _strategy_registry is not None:
        enriched = {}
        for name, s in stats.items():
            entry = dict(s) if isinstance(s, dict) else s
            entry["active"] = _strategy_registry.is_active(name)
            enriched[name] = entry
        return {"strategies": enriched}
    return {"strategies": stats}


@api_router.get("/equity-curve")
async def get_equity_curve():
    """Get equity curve time-series data for charting."""
    return {"equity_curve": _bot_state["equity_curve"]}


@api_router.get("/open-positions")
async def get_open_positions():
    """Get current open positions with SL/TP info."""
    return {"positions": _bot_state["open_positions"]}


@api_router.get("/regime")
async def get_regime():
    """Get current market regime."""
    return {"regime": _bot_state["regime"]}


@api_router.get("/analytics")
async def get_analytics(
    range: str = Query("all", description="Date range: 7d, 30d, 90d, all"),
):
    """Get performance analytics: equity curve, drawdown, monthly returns, stats."""
    equity_curve = _bot_state.get("equity_curve", [])
    trades = _bot_state.get("trades", [])
    metrics = _bot_state.get("metrics", {})

    # Filter by date range
    filtered_curve = _filter_by_range(equity_curve, range)
    filtered_trades = _filter_by_range(trades, range)

    # Compute drawdown series from equity curve
    drawdown_series = _compute_drawdown_series(filtered_curve)

    # Map trade markers to equity curve
    trade_markers = _compute_trade_markers(filtered_trades, filtered_curve)

    # Compute monthly returns
    monthly_returns = _compute_monthly_returns(equity_curve)

    # Compute extended stats from trade PnLs
    trade_pnls = [t.get("pnl", 0) for t in trades if "pnl" in t]
    stats = _compute_analytics_stats(metrics, trade_pnls)

    return {
        "equity_curve": filtered_curve,
        "drawdown": drawdown_series,
        "trade_markers": trade_markers,
        "monthly_returns": monthly_returns,
        "stats": stats,
        "range": range,
    }


@api_router.get("/quant/risk-metrics")
async def get_quant_risk_metrics():
    """Get quantitative risk metrics (VaR, CVaR, Sortino, etc.)."""
    return {"risk_metrics": _bot_state.get("quant_risk_metrics", {})}


@api_router.get("/quant/correlation-matrix")
async def get_correlation_matrix():
    """Get correlation matrix between traded symbols."""
    return {"correlation_matrix": _bot_state.get("correlation_matrix", {})}


@api_router.get("/quant/portfolio-optimization")
async def get_portfolio_optimization():
    """Get current portfolio optimization results."""
    return {"optimization": _bot_state.get("portfolio_optimization", {})}


@api_router.get("/quant/garch")
async def get_garch_metrics():
    """Get GARCH volatility model metrics."""
    return {"garch": _bot_state.get("garch_metrics", {})}


@api_router.post("/strategies/{name}/toggle")
async def toggle_strategy(
    name: str,
    force: bool = Query(False, description="Force disable"),
):
    """Toggle a strategy's active state (enable/disable)."""
    if _strategy_registry is None:
        return {"error": "Strategy registry not available", "success": False}

    strategy = _strategy_registry.get(name)
    if strategy is None:
        return {"error": f"Strategy '{name}' not found", "success": False}

    if _strategy_registry.is_active(name):
        # Check for open positions when disabling
        if not force:
            open_positions = _bot_state.get("open_positions", [])
            strategy_positions = [
                p for p in open_positions
                if p.get("strategy") == name
            ]
            if strategy_positions:
                count = len(strategy_positions)
                return {
                    "name": name,
                    "active": True,
                    "success": False,
                    "has_open_positions": True,
                    "open_position_count": count,
                    "warning": (
                        f"Strategy '{name}' has {count} "
                        f"open position(s). "
                        f"Use force=true to disable anyway."
                    ),
                }
        _strategy_registry.disable(name)
        new_state = "disabled"
    else:
        _strategy_registry.enable(name)
        new_state = "enabled"

    logger.info("strategy_toggled", name=name, state=new_state)
    # Broadcast strategy toggle via WebSocket
    asyncio.ensure_future(broadcast_state_update())
    return {"name": name, "active": new_state == "enabled", "success": True}


@api_router.get("/health")
async def api_health_check():
    """Health check endpoint under /api prefix."""
    return await health_check()


# Include the API router
app.include_router(api_router)


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


def _build_full_state_payload() -> dict:
    """Build a complete state payload for WebSocket broadcast."""
    return {
        "status": _bot_state["status"],
        "started_at": _bot_state["started_at"],
        "cycle_metrics": _bot_state["cycle_metrics"],
        "portfolio": _bot_state["portfolio"],
        "metrics": _bot_state["metrics"],
        "regime": _bot_state["regime"],
        "trades": _bot_state["trades"][-50:],
        "strategy_stats": _bot_state["strategy_stats"],
        "open_positions": _bot_state["open_positions"],
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
async def health_check():
    """Health check endpoint for Docker.

    Returns unhealthy if the bot is running but the last cycle was more than
    5 minutes ago, indicating the trading loop may be stuck.
    """
    now = datetime.now(timezone.utc).isoformat()
    last_cycle = _bot_state["cycle_metrics"].get("last_cycle_time")
    bot_status = _bot_state["status"]

    # If the bot is running and we have a last_cycle_time, check staleness
    if bot_status == "running" and last_cycle is not None:
        elapsed = time.time() - last_cycle
        if elapsed > 300:  # 5 minutes
            return {
                "status": "unhealthy",
                "reason": "last_cycle_stale",
                "last_cycle_seconds_ago": round(elapsed, 1),
                "timestamp": now,
            }

    return {"status": "healthy", "timestamp": now}


@app.get("/legacy", response_class=HTMLResponse)
async def legacy_dashboard():
    """Render legacy HTML dashboard page (backward compatibility)."""
    status = html.escape(str(_bot_state["status"]))
    metrics = _bot_state["metrics"]
    trades = _bot_state["trades"][-10:]
    portfolio = _bot_state["portfolio"]
    regime = _bot_state.get("regime")
    open_positions = _bot_state.get("open_positions", [])
    strategy_stats = _bot_state.get("strategy_stats", {})
    equity_curve = _bot_state.get("equity_curve", [])

    # Trades table
    no_trades = '<tr><td colspan="5">No trades yet</td></tr>'
    trades_html = ""
    for trade in reversed(trades):
        side_val = html.escape(str(trade.get('side', '')))
        side_class = "buy" if side_val == "BUY" else "sell"
        trades_html += (
            f"<tr><td>{html.escape(str(trade.get('timestamp', '')))}</td>"
            f"<td>{html.escape(str(trade.get('symbol', '')))}</td>"
            f'<td class="{side_class}">{side_val}</td>'
            f"<td>{html.escape(str(trade.get('quantity', '')))}</td>"
            f"<td>{html.escape(str(trade.get('price', '')))}</td></tr>"
        )
    tbody_content = trades_html if trades_html else no_trades

    # Open positions table
    no_positions = '<tr><td colspan="7">No open positions</td></tr>'
    positions_html = ""
    for pos in open_positions:
        entry_p = pos.get('entry_price', 0)
        current_p = pos.get('current_price', 0)
        upnl = pos.get('unrealized_pnl', 0)
        sl_price = pos.get('stop_loss', 0)
        tp_price = pos.get('take_profit', 0)
        pnl_class = "positive" if upnl >= 0 else "negative"
        positions_html += (
            f"<tr><td>{html.escape(str(pos.get('symbol', '')))}</td>"
            f"<td>{html.escape(str(pos.get('quantity', '')))}</td>"
            f"<td>{html.escape(f'{entry_p:,.2f}')}</td>"
            f"<td>{html.escape(f'{current_p:,.2f}')}</td>"
            f'<td class="{pnl_class}">{html.escape(f"{upnl:+,.2f}")}</td>'
            f"<td>{html.escape(f'{sl_price:,.2f}')}</td>"
            f"<td>{html.escape(f'{tp_price:,.2f}')}</td></tr>"
        )
    positions_tbody = positions_html if positions_html else no_positions

    # Market regime
    regime_display = html.escape(str(regime)) if regime else "UNKNOWN"
    regime_color = _regime_color(regime)

    # Metrics
    total_return = html.escape(str(metrics.get('total_return_pct', 0)))
    win_rate = html.escape(str(metrics.get('win_rate', 0)))
    total_trades = html.escape(str(metrics.get('total_trades', 0)))
    max_drawdown = html.escape(str(metrics.get('max_drawdown_pct', 0)))
    total_value = html.escape(f"{portfolio.get('total_value', 0):,.2f}")

    # Strategy stats for bar chart
    strat_names_js = _build_strategy_names_js(strategy_stats)
    strat_pnl_js = _build_strategy_pnl_js(strategy_stats)
    strat_colors_js = _build_strategy_colors_js(strategy_stats)

    # Equity curve data for chart
    eq_labels_js = _build_equity_labels_js(equity_curve)
    eq_values_js = _build_equity_values_js(equity_curve)

    # Trade markers for equity curve chart
    trade_markers_js = _build_trade_markers_js(trades, equity_curve)

    # Strategy list with toggle buttons
    strategies_list_html = _build_strategies_list_html(strategy_stats)

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        body {{ font-family: sans-serif; margin: 2em; background: #f5f5f5; }}
        .card {{ background: white; padding: 1.5em; margin: 1em 0; border-radius: 8px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .status {{ font-size: 1.2em; font-weight: bold;
                   color: {'green' if status == 'running' else 'red'}; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; }}
        .metric {{ display: inline-block; margin: 0 2em 1em 0; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .buy {{ color: #22c55e; font-weight: bold; }}
        .sell {{ color: #ef4444; font-weight: bold; }}
        .positive {{ color: #22c55e; }}
        .negative {{ color: #ef4444; }}
        .regime-badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px;
                         color: white; font-weight: bold; font-size: 0.9em; }}
        .chart-container {{ position: relative; height: 300px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }}
        .strategy-item {{ display: flex; justify-content: space-between; align-items: center;
                          padding: 8px 0; border-bottom: 1px solid #eee; }}
        .toggle-btn {{ padding: 4px 12px; border: none; border-radius: 4px; cursor: pointer;
                       font-size: 0.85em; }}
        .toggle-btn.active {{ background: #22c55e; color: white; }}
        .toggle-btn.inactive {{ background: #ef4444; color: white; }}
    </style>
</head>
<body>
    <h1>Trading Bot Dashboard (Legacy)</h1>

    <div class="card">
        <h2>Status</h2>
        <p class="status">{status.upper()}</p>
        <span class="regime-badge" style="background:{regime_color}">
            Regime: {regime_display}
        </span>
    </div>

    <div class="card">
        <h2>Key Metrics</h2>
        <div class="metric">
            <div class="metric-value">{total_return}%</div>
            <div class="metric-label">Total Return</div>
        </div>
        <div class="metric">
            <div class="metric-value">{win_rate}%</div>
            <div class="metric-label">Win Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{total_trades}</div>
            <div class="metric-label">Total Trades</div>
        </div>
        <div class="metric">
            <div class="metric-value">{max_drawdown}%</div>
            <div class="metric-label">Max Drawdown</div>
        </div>
    </div>

    <div class="card">
        <h2>Portfolio</h2>
        <p>Total Value: <strong>${total_value}</strong></p>
    </div>

    <div class="card">
        <h2>Equity Curve</h2>
        <div class="chart-container">
            <canvas id="equityChart"></canvas>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Strategy Performance</h2>
            <div class="chart-container">
                <canvas id="strategyChart"></canvas>
            </div>
        </div>
        <div class="card">
            <h2>Active Strategies</h2>
            {strategies_list_html}
        </div>
    </div>

    <div class="card">
        <h2>Open Positions</h2>
        <table>
            <thead><tr>
                <th>Symbol</th><th>Qty</th><th>Entry</th><th>Current</th>
                <th>Unrealized PnL</th><th>Stop Loss</th><th>Take Profit</th>
            </tr></thead>
            <tbody>{positions_tbody}</tbody>
        </table>
    </div>

    <div class="card">
        <h2>Recent Trades</h2>
        <table>
            <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th></tr></thead>
            <tbody>{tbody_content}</tbody>
        </table>
    </div>

    <script>
    // Equity Curve Chart
    const eqCtx = document.getElementById('equityChart');
    if (eqCtx) {{
        new Chart(eqCtx, {{
            type: 'line',
            data: {{
                labels: {eq_labels_js},
                datasets: [
                    {{
                        label: 'Portfolio Value',
                        data: {eq_values_js},
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59,130,246,0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                    }},
                    {trade_markers_js}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ display: true, title: {{ display: false }} }},
                    y: {{ display: true, title: {{ display: true, text: 'Value ($)' }} }}
                }},
                plugins: {{ legend: {{ display: true }} }}
            }}
        }});
    }}

    // Strategy Performance Bar Chart
    const stratCtx = document.getElementById('strategyChart');
    if (stratCtx) {{
        new Chart(stratCtx, {{
            type: 'bar',
            data: {{
                labels: {strat_names_js},
                datasets: [{{
                    label: 'Total PnL',
                    data: {strat_pnl_js},
                    backgroundColor: {strat_colors_js},
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ title: {{ display: true, text: 'PnL ($)' }} }}
                }},
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});
    }}

    // Strategy toggle
    async function toggleStrategy(name) {{
        try {{
            const resp = await fetch('/api/strategies/' + name + '/toggle', {{ method: 'POST' }});
            const data = await resp.json();
            if (data.success) {{
                location.reload();
            }}
        }} catch(e) {{
            console.error('Toggle failed', e);
        }}
    }}
    </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# SPA fallback — serve React frontend for non-API routes
# ---------------------------------------------------------------------------

_static_dir = Path(__file__).parent / "static"

# Mount static assets if the build directory exists
if _static_dir.exists() and (_static_dir / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_static_dir / "assets")), name="static-assets")


@app.get("/", response_class=HTMLResponse)
async def serve_root():
    """Serve React SPA index.html or fallback message."""
    index_file = _static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse(
        "<h1>Frontend not built</h1>"
        "<p>Run <code>make frontend-build</code> to build the React dashboard.</p>"
        '<p>Or visit <a href="/legacy">/legacy</a> for the legacy HTML dashboard.</p>'
    )


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """SPA fallback: serve index.html for all unmatched routes."""
    # Try to serve a static file first
    static_file = _static_dir / full_path
    if static_file.exists() and static_file.is_file():
        return FileResponse(str(static_file))
    # Fall back to index.html for client-side routing
    index_file = _static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse(
        "<h1>Frontend not built</h1>"
        "<p>Run <code>make frontend-build</code> to build the React dashboard.</p>"
        '<p>Or visit <a href="/legacy">/legacy</a> for the legacy HTML dashboard.</p>'
    )


# ---------------------------------------------------------------------------
# Helper functions (private)
# ---------------------------------------------------------------------------


def _regime_color(regime) -> str:
    """Return CSS color for market regime."""
    if regime is None:
        return "#6b7280"
    regime_str = str(regime).upper()
    if "TRENDING_UP" in regime_str:
        return "#22c55e"
    if "TRENDING_DOWN" in regime_str:
        return "#ef4444"
    if "RANGING" in regime_str:
        return "#f59e0b"
    if "HIGH_VOLATILITY" in regime_str or "VOLATILE" in regime_str:
        return "#8b5cf6"
    return "#6b7280"


def _build_strategy_names_js(strategy_stats: dict) -> str:
    """Build JS array of strategy names."""
    if not strategy_stats:
        return "[]"
    names = [html.escape(str(k)) for k in strategy_stats.keys()]
    return "[" + ",".join(f'"{n}"' for n in names) + "]"


def _build_strategy_pnl_js(strategy_stats: dict) -> str:
    """Build JS array of strategy PnL values."""
    if not strategy_stats:
        return "[]"
    values = []
    for stats in strategy_stats.values():
        if isinstance(stats, dict):
            values.append(str(stats.get("total_pnl", 0)))
        else:
            values.append("0")
    return "[" + ",".join(values) + "]"


def _build_strategy_colors_js(strategy_stats: dict) -> str:
    """Build JS array of bar colors (green for positive, red for negative)."""
    if not strategy_stats:
        return "[]"
    colors = []
    for stats in strategy_stats.values():
        pnl = 0
        if isinstance(stats, dict):
            pnl = stats.get("total_pnl", 0)
        colors.append('"#22c55e"' if pnl >= 0 else '"#ef4444"')
    return "[" + ",".join(colors) + "]"


def _build_equity_labels_js(equity_curve: list) -> str:
    """Build JS array of equity curve timestamps."""
    if not equity_curve:
        return "[]"
    labels = []
    for point in equity_curve:
        ts = point.get("timestamp", "")
        labels.append(f'"{html.escape(str(ts))}"')
    return "[" + ",".join(labels) + "]"


def _build_equity_values_js(equity_curve: list) -> str:
    """Build JS array of equity curve values."""
    if not equity_curve:
        return "[]"
    values = [str(point.get("total_value", 0)) for point in equity_curve]
    return "[" + ",".join(values) + "]"


def _build_trade_markers_js(trades: list, equity_curve: list) -> str:
    """Build Chart.js dataset for trade markers on equity curve.

    Creates scatter-style points: green triangles for BUY, red for SELL.
    Maps trades to equity curve indices by timestamp proximity.
    """
    if not trades or not equity_curve:
        return ""

    buy_points = []
    sell_points = []

    eq_timestamps = [p.get("timestamp", "") for p in equity_curve]

    for trade in trades:
        trade_ts = str(trade.get("timestamp", ""))
        side = str(trade.get("side", "")).upper()

        # Find closest equity curve index
        best_idx = _find_closest_index(trade_ts, eq_timestamps)
        if best_idx is not None and best_idx < len(equity_curve):
            val = equity_curve[best_idx].get("total_value", 0)
            point = f'{{x:"{html.escape(eq_timestamps[best_idx])}",y:{val}}}'
            if side == "BUY":
                buy_points.append(point)
            elif side == "SELL":
                sell_points.append(point)

    datasets = []
    if buy_points:
        datasets.append(
            "{"
            'label:"BUY",'
            "data:[" + ",".join(buy_points) + "],"
            'borderColor:"#22c55e",'
            'backgroundColor:"#22c55e",'
            'pointStyle:"triangle",'
            "pointRadius:8,"
            "showLine:false"
            "}"
        )
    if sell_points:
        datasets.append(
            "{"
            'label:"SELL",'
            "data:[" + ",".join(sell_points) + "],"
            'borderColor:"#ef4444",'
            'backgroundColor:"#ef4444",'
            'pointStyle:"triangle",'
            "pointRadius:8,"
            "pointRotation:180,"
            "showLine:false"
            "}"
        )

    return ",".join(datasets)


def _find_closest_index(target_ts: str, timestamps: list[str]) -> int | None:
    """Find the index of the closest timestamp in the list."""
    if not timestamps or not target_ts:
        return None
    # Exact match first
    if target_ts in timestamps:
        return timestamps.index(target_ts)
    # Simple linear scan — last index before or equal
    for i in range(len(timestamps) - 1, -1, -1):
        if timestamps[i] <= target_ts:
            return i
    return 0 if timestamps else None


def _build_strategies_list_html(strategy_stats: dict) -> str:
    """Build HTML list of strategies with toggle buttons."""
    if not strategy_stats:
        return '<p style="color:#666">No strategy data available</p>'

    items = []
    for name, stats in strategy_stats.items():
        escaped_name = html.escape(str(name))
        active = True
        win_rate = 0
        total_pnl = 0
        if isinstance(stats, dict):
            active = stats.get("active", True)
            win_rate = stats.get("win_rate", 0)
            total_pnl = stats.get("total_pnl", 0)

        btn_class = "active" if active else "inactive"
        btn_text = "Enabled" if active else "Disabled"
        pnl_class = "positive" if total_pnl >= 0 else "negative"

        items.append(
            f'<div class="strategy-item">'
            f"<div>"
            f"<strong>{escaped_name}</strong><br>"
            f'<small>Win: {html.escape(str(win_rate))}% | '
            f'PnL: <span class="{pnl_class}">{html.escape(f"{total_pnl:+.2f}")}</span></small>'
            f"</div>"
            f'<button class="toggle-btn {btn_class}" '
            f"onclick=\"toggleStrategy('{escaped_name}')\">{btn_text}</button>"
            f"</div>"
        )

    return "\n".join(items)


# ---------------------------------------------------------------------------
# Analytics helper functions
# ---------------------------------------------------------------------------


def _filter_by_range(data: list[dict], range_str: str) -> list[dict]:
    """Filter a list of dicts with 'timestamp' by date range."""
    if range_str == "all" or not data:
        return data

    days_map = {"7d": 7, "30d": 30, "90d": 90}
    days = days_map.get(range_str)
    if days is None:
        return data

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_str = cutoff.isoformat()

    filtered = []
    for item in data:
        ts = item.get("timestamp", "")
        if ts >= cutoff_str:
            filtered.append(item)
    return filtered


def _compute_drawdown_series(equity_curve: list[dict]) -> list[dict]:
    """Compute drawdown percentage series from equity curve."""
    if not equity_curve:
        return []

    series = []
    peak = 0.0
    for point in equity_curve:
        value = point.get("total_value", 0)
        if value > peak:
            peak = value
        dd_pct = ((peak - value) / peak * 100) if peak > 0 else 0.0
        series.append({
            "timestamp": point.get("timestamp", ""),
            "drawdown_pct": round(dd_pct, 2),
        })
    return series


def _compute_trade_markers(
    trades: list[dict], equity_curve: list[dict]
) -> list[dict]:
    """Map trades to equity curve positions for chart overlay."""
    if not trades or not equity_curve:
        return []

    eq_timestamps = [p.get("timestamp", "") for p in equity_curve]
    markers = []

    for trade in trades:
        trade_ts = str(trade.get("timestamp", ""))
        side = str(trade.get("side", "")).upper()
        if side not in ("BUY", "SELL"):
            continue

        best_idx = _find_closest_index(trade_ts, eq_timestamps)
        if best_idx is not None and best_idx < len(equity_curve):
            markers.append({
                "timestamp": equity_curve[best_idx].get("timestamp", ""),
                "value": equity_curve[best_idx].get("total_value", 0),
                "side": side,
                "symbol": trade.get("symbol", ""),
                "price": trade.get("price", 0),
            })

    return markers


def _compute_monthly_returns(equity_curve: list[dict]) -> list[dict]:
    """Compute monthly return percentages from equity curve."""
    if len(equity_curve) < 2:
        return []

    # Group equity curve by year-month, take first and last value per month
    monthly: dict[str, dict] = {}
    for point in equity_curve:
        ts = point.get("timestamp", "")
        value = point.get("total_value", 0)
        # Extract YYYY-MM from ISO timestamp
        month_key = ts[:7] if len(ts) >= 7 else ""
        if not month_key:
            continue
        if month_key not in monthly:
            monthly[month_key] = {"first": value, "last": value}
        else:
            monthly[month_key]["last"] = value

    results = []
    for month_key in sorted(monthly.keys()):
        data = monthly[month_key]
        first_val = data["first"]
        last_val = data["last"]
        ret_pct = (
            ((last_val - first_val) / first_val * 100) if first_val > 0 else 0.0
        )
        results.append({
            "month": month_key,
            "return_pct": round(ret_pct, 2),
        })

    return results


def _compute_analytics_stats(
    metrics: dict, trade_pnls: list[float]
) -> dict:
    """Compute extended analytics stats including Sortino ratio."""
    sharpe = metrics.get("sharpe_ratio", 0.0)
    max_dd = metrics.get("max_drawdown_pct", 0.0)
    win_rate = metrics.get("win_rate", 0.0)
    total_return = metrics.get("total_return_pct", 0.0)
    total_trades = metrics.get("total_trades", 0)
    winning_trades = metrics.get("winning_trades", 0)
    losing_trades = metrics.get("losing_trades", 0)

    # Profit factor
    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    # Average trade PnL
    avg_trade_pnl = (
        sum(trade_pnls) / len(trade_pnls) if trade_pnls else 0.0
    )

    # Best and worst trade
    best_trade = max(trade_pnls) if trade_pnls else 0.0
    worst_trade = min(trade_pnls) if trade_pnls else 0.0

    # Sortino ratio (uses only downside deviation)
    sortino = 0.0
    if len(trade_pnls) > 1:
        mean_return = sum(trade_pnls) / len(trade_pnls)
        downside_returns = [r for r in trade_pnls if r < 0]
        if downside_returns:
            downside_sq = sum(r ** 2 for r in downside_returns) / len(trade_pnls)
            downside_dev = math.sqrt(downside_sq)
            sortino = (
                mean_return / downside_dev if downside_dev > 0 else 0.0
            )

    return {
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "best_trade": round(best_trade, 2),
        "worst_trade": round(worst_trade, 2),
        "total_return_pct": round(total_return, 2),
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": round(win_rate, 2),
    }
