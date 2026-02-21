"""FastAPI monitoring dashboard."""

from datetime import datetime, timezone

import structlog
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

logger = structlog.get_logger()

app = FastAPI(title="Coin Trading Bot Dashboard")

# Shared state (set by the TradingBot orchestrator)
_bot_state = {
    "status": "stopped",
    "started_at": None,
    "trades": [],
    "metrics": {},
    "portfolio": {"balances": {}, "positions": [], "total_value": 0.0},
}


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


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Render HTML dashboard page."""
    status = _bot_state["status"]
    metrics = _bot_state["metrics"]
    trades = _bot_state["trades"][-10:]
    portfolio = _bot_state["portfolio"]

    no_trades = '<tr><td colspan="5">No trades yet</td></tr>'
    trades_html = ""
    for trade in reversed(trades):
        trades_html += (
            f"<tr><td>{trade.get('timestamp', '')}</td>"
            f"<td>{trade.get('symbol', '')}</td>"
            f"<td>{trade.get('side', '')}</td>"
            f"<td>{trade.get('quantity', '')}</td>"
            f"<td>{trade.get('price', '')}</td></tr>"
        )
    tbody_content = trades_html if trades_html else no_trades

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Dashboard</title>
    <meta http-equiv="refresh" content="10">
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
    </style>
</head>
<body>
    <h1>Trading Bot Dashboard</h1>
    <div class="card">
        <h2>Status</h2>
        <p class="status">{status.upper()}</p>
    </div>
    <div class="card">
        <h2>Key Metrics</h2>
        <div class="metric">
            <div class="metric-value">{metrics.get('total_return_pct', 0)}%</div>
            <div class="metric-label">Total Return</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('win_rate', 0)}%</div>
            <div class="metric-label">Win Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('total_trades', 0)}</div>
            <div class="metric-label">Total Trades</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('max_drawdown_pct', 0)}%</div>
            <div class="metric-label">Max Drawdown</div>
        </div>
    </div>
    <div class="card">
        <h2>Portfolio</h2>
        <p>Total Value: <strong>${portfolio.get('total_value', 0):,.2f}</strong></p>
    </div>
    <div class="card">
        <h2>Recent Trades</h2>
        <table>
            <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th></tr></thead>
            <tbody>{tbody_content}</tbody>
        </table>
    </div>
</body>
</html>"""
