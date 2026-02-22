# Dashboard Guide

The trading bot includes a React-based web dashboard for real-time monitoring and control.

## Accessing the Dashboard

The dashboard is served at `http://localhost:8000` (or your configured `DASHBOARD_PORT`).

### Authentication

When `DASHBOARD_PASSWORD` is set to something other than `changeme`, JWT authentication is required. Log in with your configured username and password.

Default credentials (dev mode, auth disabled):
- Username: `admin`
- Password: `changeme`

## Pages

### Dashboard (Overview)

The main page shows:
- **Bot status**: Running/stopped indicator with uptime
- **Portfolio value**: Total value and return percentage
- **Key metrics**: Win rate, total trades, max drawdown, Sharpe ratio
- **Market regime**: Current market regime badge (trending up/down, ranging, volatile)
- **Cycle metrics**: Cycle count, average duration, last cycle time

Data updates in real-time via WebSocket. Falls back to 10-second polling if WebSocket disconnects.

### Positions

Shows all currently open positions:
- Symbol, quantity, entry price, current price
- Unrealized PnL (color-coded green/red) with percentage and absolute value
- Stop-loss and take-profit levels
- Duration held

### Trades

Complete trade history with:
- Sortable table (click column headers)
- Pagination (20 trades per page)
- Symbol filter
- CSV export button
- New trade toast notifications via WebSocket

### Strategies

Strategy management page:
- Strategy cards with toggle switch (enable/disable)
- Stats: PnL, win rate, total trades, consecutive losses, Sharpe ratio
- Per-strategy mini chart: cumulative PnL over last 50 trades
- Strategy comparison bar chart
- Auto-disabled strategies show warning badge with reason
- Confirmation dialog before disabling strategy with open positions

### Analytics

Performance analytics with interactive charts:
- **Equity curve**: Zoomable line chart with BUY/SELL markers
- **Drawdown chart**: Red filled area showing drawdown from peak
- **Monthly returns heatmap**: Color-coded grid by year and month
- **Stats sidebar**: Sharpe, Sortino, max drawdown, profit factor, win rate, avg/best/worst trade
- Date range selector: 7d, 30d, 90d, All Time

### Settings

Bot configuration management:
- Grouped by section: Trading, Risk Management, Strategies, Exchange, Dashboard, Notifications
- Each setting shows: current value, description, type, whether restart is required
- Safe settings (risk params, position sizing, etc.) can be changed at runtime
- Unsafe settings (API keys, trading mode) shown as read-only
- Save button with confirmation dialog
- Undo button to revert changes

### Login

Login form for JWT authentication. Only shown when auth is enabled (password changed from default).

## Real-Time Updates

The dashboard connects to the WebSocket endpoint (`/api/ws`) for real-time updates:
- Connection indicator: green dot (Live) / red dot (Offline) in the header
- Automatic reconnection with exponential backoff (up to 30 seconds)
- Falls back to polling when WebSocket is disconnected

## Emergency Controls

The dashboard provides emergency controls (accessible from the API):
- **Emergency Stop**: Halts trading, cancels pending orders
- **Close All**: Halts trading and closes all positions at market price
- **Resume**: Restarts trading after an emergency stop

## Development

For frontend development with hot reload:

```bash
# Install dependencies
make frontend-install

# Start dev server (proxies API to localhost:8003)
make frontend-dev

# Build for production
make frontend-build
```

The frontend is built with:
- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS (styling)
- recharts (charts)
- axios (API client)
- react-router-dom (routing)
