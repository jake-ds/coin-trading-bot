# Coin Trading Bot

A Python-based cryptocurrency automated trading system with multiple strategies, risk management, backtesting, a real-time React dashboard, JWT authentication, and comprehensive live trading infrastructure.

## Features

- **12+ Strategies**: MA Crossover (with filters), RSI (with divergence), MACD, Bollinger Squeeze, VWAP, Composite Momentum, Funding Rate, DCA, Cross-Exchange Arbitrage, ML Prediction
- **Signal Pipeline**: Ensemble voting, multi-timeframe trend filtering, market regime detection
- **Risk Management**: Position sizing (ATR-based), stop-loss/take-profit (multi-level), trailing stop, portfolio-level exposure/correlation limits, daily loss limits, max drawdown
- **Backtesting**: Realistic simulation with SL/TP enforcement, dynamic slippage, walk-forward analysis, strategy comparison CLI
- **React Dashboard**: Real-time SPA with equity curves, strategy management, trade history, analytics, and settings panel
- **WebSocket Updates**: Live dashboard updates via WebSocket with auto-reconnect
- **JWT Authentication**: Secure API access with access/refresh token flow
- **Emergency Controls**: Kill switch via API and Telegram commands (/stop, /closeall, /resume)
- **Audit Trail**: Immutable log of all trades, config changes, and emergency actions
- **Position Reconciliation**: Automatic sync verification between local state and exchange
- **Pre-flight Checks**: Safety gates before live trading (balance, API keys, stop-loss, rate limits)
- **Exchange Rate Limiting**: Token bucket rate limiter to prevent exchange bans
- **Paper Trading**: Safe paper trading mode by default with finite balance and fee tracking
- **Validation**: Automated paper trading validation with go/no-go criteria
- **Docker**: Multi-stage build with Node.js frontend compilation and health checks

## How It Works

```
Market Data --> Data Collection --> Multi-Timeframe Candles
                                          |
                                Market Regime Detection
                                (trending/ranging/volatile)
                                          |
                               Strategy Analysis (12+ strategies)
                               Each adapts to detected regime
                                          |
                                Signal Ensemble Voting
                                (min N strategies must agree)
                                          |
                                Trend Filter (4h confirmation)
                                Rejects counter-trend signals
                                          |
                                Risk Manager Validation
                                (daily loss, drawdown, position limits)
                                          |
                                Portfolio Risk Check
                                (exposure, correlation, sector limits)
                                          |
                                ATR-Based Position Sizing
                                          |
                                Smart Order Execution
                                (limit orders, TWAP for large orders)
                                          |
                                Position Management
                                (TP1 partial -> TP2 full, trailing SL)
                                          |
                                Strategy Performance Tracking
                                (auto-disable underperformers)
```

## Architecture

```
src/bot/
├── main.py                 # TradingBot orchestrator, CLI entry point
├── config.py               # pydantic-settings configuration
├── validation.py           # Paper trading validation framework
├── models/                 # Pydantic v2 data models (OHLCV, Order, Signal, Portfolio)
├── exchanges/              # Exchange adapters (Binance, Upbit) via ccxt
│   ├── base.py             # ExchangeAdapter ABC
│   ├── factory.py          # Factory pattern for adapter creation
│   ├── rate_limiter.py     # Token bucket rate limiter (per-exchange)
│   └── resilient.py        # ResilientExchange (circuit breaker, retry, rate limit)
├── data/                   # Data layer
│   ├── collector.py        # DataCollector (multi-timeframe)
│   ├── store.py            # DataStore (SQLAlchemy + aiosqlite)
│   ├── models.py           # SQLAlchemy models (OHLCV, trades, audit log)
│   ├── websocket_feed.py   # WebSocket real-time data feed
│   └── order_book.py       # Order book analysis (imbalance detection)
├── strategies/             # Plugin-based strategies
│   ├── base.py             # BaseStrategy ABC + StrategyRegistry
│   ├── ensemble.py         # SignalEnsemble voting system
│   ├── trend_filter.py     # Multi-timeframe trend confirmation
│   ├── regime.py           # Market regime detector (ADX, ATR, BB)
│   ├── indicators.py       # Shared technical indicators (ATR)
│   ├── technical/          # MA Crossover, RSI, MACD, Bollinger, VWAP, Momentum
│   ├── arbitrage/          # Cross-exchange arbitrage
│   ├── dca/                # Dollar Cost Averaging
│   └── ml/                 # ML price prediction (GradientBoosting, CV)
├── risk/                   # Risk management
│   ├── manager.py          # RiskManager (sizing, daily loss, drawdown)
│   └── portfolio_risk.py   # PortfolioRiskManager (exposure, correlation, heat)
├── execution/              # Order execution
│   ├── engine.py           # ExecutionEngine (paper + live modes)
│   ├── paper_portfolio.py  # PaperPortfolio (finite balance, fees)
│   ├── position_manager.py # PositionManager (SL/TP1/TP2/trailing)
│   ├── smart_executor.py   # SmartExecutor (limit orders, TWAP)
│   ├── preflight.py        # PreFlightChecker (8 safety checks for live mode)
│   └── reconciler.py       # PositionReconciler (local vs exchange sync)
├── backtest/               # Backtesting engine
│   └── engine.py           # BacktestEngine with SL/TP, slippage, walk-forward
├── monitoring/             # Monitoring & alerting
│   ├── logger.py           # Structured logging (structlog)
│   ├── strategy_tracker.py # Per-strategy performance tracking & auto-disable
│   ├── audit.py            # AuditLogger (immutable event log)
│   ├── metrics.py          # MetricsCollector
│   └── telegram.py         # Telegram notifications + command handling
└── dashboard/              # FastAPI + React dashboard
    ├── app.py              # API endpoints, auth, WebSocket, SPA serving
    ├── auth.py             # JWT authentication (python-jose)
    ├── websocket.py        # WebSocket connection manager
    └── static/             # Built React frontend assets

frontend/                   # React 18 + TypeScript + Vite + Tailwind CSS
├── src/
│   ├── pages/              # Dashboard, Trades, Positions, Strategies, Analytics,
│   │                       # Settings, Login
│   ├── components/         # Reusable UI components
│   ├── hooks/              # useWebSocket, useAuth
│   └── api/                # API client (axios) + TypeScript types
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend development)
- pip

### Installation

```bash
git clone <repo-url>
cd coin-trading-bot

# Install Python dependencies
pip install -e ".[dev]"

# Install frontend dependencies
make frontend-install

# Copy and configure environment
cp .env.example .env
# Edit .env with your exchange API keys
```

### Build Everything

```bash
# Build frontend + verify Python package
make build
```

### Run in Paper Trading Mode

```bash
python -m bot.main
```

The dashboard is available at `http://localhost:8000`.

### Frontend Development

```bash
# Start Vite dev server with hot reload (proxies API to :8003)
make frontend-dev

# Build for production
make frontend-build
```

### Run Paper Trading Validation

```bash
# Run 48-hour validation with go/no-go report
python -m bot.main --validate --duration=48h

# Custom duration
python -m bot.main --validate --duration=2d
```

### Run Backtests

```bash
# Strategy comparison
python -m bot.backtest
```

### Run Tests

```bash
# Full test suite
make test

# Lint
make lint

# Both
make verify
```

### Start Dashboard

```bash
uvicorn bot.dashboard.app:app --port 8000
```

### Docker Deployment

```bash
# Build and start (includes frontend build in Docker)
docker-compose up -d

# View logs
docker-compose logs -f bot

# Health check
curl http://localhost:8000/health
```

## Live Trading Checklist

Before switching to live mode:

1. Run paper trading validation: `python -m bot.main --validate --duration=48h`
2. Verify GO recommendation in the validation report
3. Set a real dashboard password: `DASHBOARD_PASSWORD=your-secure-password`
4. Configure stop-loss: `STOP_LOSS_PCT=5.0`
5. Configure daily loss limit: `DAILY_LOSS_LIMIT_PCT=3.0`
6. Verify Binance API key permissions (spot trading only, no withdrawal)
7. Configure Telegram notifications for alerts
8. Start with conservative position sizes
9. Switch mode: `TRADING_MODE=live`
10. Monitor closely for the first 24 hours

See [Live Trading Guide](docs/live-trading.md) for detailed instructions.

## API Endpoints

All data endpoints are under `/api/*` and protected by JWT auth (when enabled).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (always accessible) |
| `/api/status` | GET | Bot status and metrics |
| `/api/trades` | GET | Trade history (paginated) |
| `/api/portfolio` | GET | Current portfolio state |
| `/api/positions` | GET | Open positions |
| `/api/strategies` | GET | Strategy stats |
| `/api/strategies/{name}/toggle` | POST | Enable/disable strategy |
| `/api/analytics` | GET | Performance analytics |
| `/api/settings` | GET/PUT | View/update bot configuration |
| `/api/audit` | GET | Audit log (filterable) |
| `/api/emergency/stop` | POST | Emergency stop (halt trading) |
| `/api/emergency/close-all` | POST | Emergency close all positions |
| `/api/emergency/resume` | POST | Resume after emergency |
| `/api/emergency` | GET | Current emergency state |
| `/api/preflight` | GET | Pre-flight check results |
| `/api/reconciliation` | GET | Position reconciliation results |
| `/api/auth/login` | POST | Login (returns JWT tokens) |
| `/api/auth/refresh` | POST | Refresh access token |
| `/api/auth/logout` | POST | Logout (invalidate refresh token) |
| `/api/ws` | WebSocket | Real-time state updates |

## Documentation

- [Strategy Guide](docs/strategies.md) - How each strategy works and when it's effective
- [Configuration Reference](docs/configuration.md) - All environment variables and config options
- [Backtesting Guide](docs/backtesting.md) - Running and interpreting backtests
- [Risk Management](docs/risk-management.md) - Risk controls and recommended settings
- [Operational Runbook](docs/operational-runbook.md) - Starting, monitoring, and handling emergencies
- [Dashboard Guide](docs/dashboard.md) - Dashboard pages and features
- [Live Trading Guide](docs/live-trading.md) - Step-by-step guide from paper to live

## Safety

- **Paper trading is the default** - live trading requires explicit `TRADING_MODE=live`
- **Pre-flight checks** block live startup without stop-loss, daily loss limit, and API connectivity
- **Emergency kill switch** via API (`/api/emergency/stop`) and Telegram (`/stop`)
- **JWT authentication** protects all API endpoints when a dashboard password is set
- **Audit trail** records every trade, config change, and emergency action
- **Position reconciliation** detects drift between local state and exchange
- **Exchange rate limiting** prevents IP bans from API overuse
- All signals pass through the RiskManager before execution
- Signal ensemble requires multiple strategies to agree before trading
- Trend filter rejects signals against the higher-timeframe trend
- Portfolio-level risk checks prevent over-concentration
- ATR-based position sizing adapts to volatility
- Multi-level take-profit with trailing stop-loss
- Strategy auto-disable removes underperforming strategies
- Circuit breaker pattern protects against exchange API failures
- Daily loss limits and max drawdown protection halt trading automatically
- Paper trading validation provides go/no-go assessment before going live
