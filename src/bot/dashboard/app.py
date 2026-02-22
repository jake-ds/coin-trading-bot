"""FastAPI monitoring dashboard."""

import html
import time
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

logger = structlog.get_logger()

app = FastAPI(title="Coin Trading Bot Dashboard")

# CORS configuration — allowed_origins can be overridden via configure_cors()
_default_origins = ["http://localhost", "http://localhost:8000"]


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


@app.get("/status")
async def get_status():
    """Get bot status."""
    return {
        "status": _bot_state["status"],
        "started_at": _bot_state["started_at"],
        "cycle_metrics": _bot_state["cycle_metrics"],
    }


@app.get("/trades")
async def get_trades():
    """Get recent trades."""
    return {"trades": _bot_state["trades"][-50:]}


@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    return {"metrics": _bot_state["metrics"]}


@app.get("/portfolio")
async def get_portfolio():
    """Get current portfolio."""
    return {"portfolio": _bot_state["portfolio"]}


@app.get("/strategies")
async def get_strategies():
    """Get per-strategy performance stats."""
    return {"strategies": _bot_state["strategy_stats"]}


@app.get("/equity-curve")
async def get_equity_curve():
    """Get equity curve time-series data for charting."""
    return {"equity_curve": _bot_state["equity_curve"]}


@app.get("/open-positions")
async def get_open_positions():
    """Get current open positions with SL/TP info."""
    return {"positions": _bot_state["open_positions"]}


@app.get("/regime")
async def get_regime():
    """Get current market regime."""
    return {"regime": _bot_state["regime"]}


@app.get("/quant/risk-metrics")
async def get_quant_risk_metrics():
    """Get quantitative risk metrics (VaR, CVaR, Sortino, etc.)."""
    return {"risk_metrics": _bot_state.get("quant_risk_metrics", {})}


@app.get("/quant/correlation-matrix")
async def get_correlation_matrix():
    """Get correlation matrix between traded symbols."""
    return {"correlation_matrix": _bot_state.get("correlation_matrix", {})}


@app.get("/quant/portfolio-optimization")
async def get_portfolio_optimization():
    """Get current portfolio optimization results."""
    return {"optimization": _bot_state.get("portfolio_optimization", {})}


@app.get("/quant/garch")
async def get_garch_metrics():
    """Get GARCH volatility model metrics."""
    return {"garch": _bot_state.get("garch_metrics", {})}


@app.post("/strategies/{name}/toggle")
async def toggle_strategy(name: str):
    """Toggle a strategy's active state (enable/disable)."""
    if _strategy_registry is None:
        return {"error": "Strategy registry not available", "success": False}

    strategy = _strategy_registry.get(name)
    if strategy is None:
        return {"error": f"Strategy '{name}' not found", "success": False}

    if _strategy_registry.is_active(name):
        _strategy_registry.disable(name)
        new_state = "disabled"
    else:
        _strategy_registry.enable(name)
        new_state = "enabled"

    logger.info("strategy_toggled", name=name, state=new_state)
    return {"name": name, "active": new_state == "enabled", "success": True}


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


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Render HTML dashboard page."""
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
    <h1>Trading Bot Dashboard</h1>

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
            const resp = await fetch('/strategies/' + name + '/toggle', {{ method: 'POST' }});
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
