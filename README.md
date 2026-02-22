# Coin Trading Bot

A Python-based cryptocurrency automated trading system with multiple strategies, risk management, backtesting, and a monitoring dashboard.

## Features

- **12+ Strategies**: MA Crossover (with filters), RSI (with divergence), MACD, Bollinger Squeeze, VWAP, Composite Momentum, Funding Rate, DCA, Cross-Exchange Arbitrage, ML Prediction
- **Signal Pipeline**: Ensemble voting, multi-timeframe trend filtering, market regime detection
- **Risk Management**: Position sizing (ATR-based), stop-loss/take-profit (multi-level), trailing stop, portfolio-level exposure/correlation limits, daily loss limits, max drawdown
- **Backtesting**: Realistic simulation with SL/TP enforcement, dynamic slippage, walk-forward analysis, strategy comparison CLI
- **Monitoring**: Strategy performance tracker with auto-disable, structured logging, Telegram alerts, real-time web dashboard with equity curves
- **Paper Trading**: Safe paper trading mode by default with finite balance and fee tracking
- **Validation**: Automated paper trading validation with go/no-go criteria
- **Docker**: Containerized deployment with health checks

## How It Works

```
Market Data → Data Collection → Multi-Timeframe Candles
                                        ↓
                              Market Regime Detection
                              (trending/ranging/volatile)
                                        ↓
                             Strategy Analysis (12+ strategies)
                             Each adapts to detected regime
                                        ↓
                              Signal Ensemble Voting
                              (min N strategies must agree)
                                        ↓
                              Trend Filter (4h confirmation)
                              Rejects counter-trend signals
                                        ↓
                              Risk Manager Validation
                              (daily loss, drawdown, position limits)
                                        ↓
                              Portfolio Risk Check
                              (exposure, correlation, sector limits)
                                        ↓
                              ATR-Based Position Sizing
                                        ↓
                              Smart Order Execution
                              (limit orders, TWAP for large orders)
                                        ↓
                              Position Management
                              (TP1 partial → TP2 full, trailing SL)
                                        ↓
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
│   └── resilient.py        # ResilientExchange (circuit breaker, retry)
├── data/                   # Data layer
│   ├── collector.py        # DataCollector (multi-timeframe)
│   ├── store.py            # DataStore (SQLAlchemy + aiosqlite)
│   ├── websocket_feed.py   # WebSocket real-time data feed
│   └── order_book.py       # Order book analysis (imbalance detection)
├── strategies/             # Plugin-based strategies
│   ├── base.py             # BaseStrategy ABC + StrategyRegistry
│   ├── ensemble.py         # SignalEnsemble voting system
│   ├── trend_filter.py     # Multi-timeframe trend confirmation
│   ├── regime.py           # Market regime detector (ADX, ATR, BB)
│   ├── indicators.py       # Shared technical indicators (ATR)
│   ├── technical/          # MA Crossover, RSI, MACD, Bollinger Squeeze,
│   │                       # VWAP, Composite Momentum
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
│   └── smart_executor.py   # SmartExecutor (limit orders, TWAP)
├── backtest/               # Backtesting engine
│   └── engine.py           # BacktestEngine with SL/TP, slippage, walk-forward
├── monitoring/             # Monitoring & alerting
│   ├── logger.py           # Structured logging (structlog)
│   ├── strategy_tracker.py # Per-strategy performance tracking & auto-disable
│   └── telegram.py         # Telegram notifications
└── dashboard/              # FastAPI web dashboard
    └── app.py              # Real-time dashboard with equity curves
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone <repo-url>
cd coin-trading-bot

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your exchange API keys
```

### Run in Paper Trading Mode

```bash
python -m bot.main
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
pytest tests/ -v
```

### Run Linter

```bash
ruff check src/ tests/
```

### Start Dashboard

```bash
uvicorn bot.dashboard.app:app --port 8000
```

### Docker Deployment

```bash
docker-compose up -d
docker-compose logs -f bot
```

## Documentation

- [Strategy Guide](docs/strategies.md) - How each strategy works and when it's effective
- [Configuration Reference](docs/configuration.md) - All environment variables and config options
- [Backtesting Guide](docs/backtesting.md) - Running and interpreting backtests
- [Risk Management](docs/risk-management.md) - Risk controls and recommended settings
- [Operational Runbook](docs/operational-runbook.md) - Starting, monitoring, and handling emergencies

## Safety

- **Paper trading is the default** - live trading requires explicit `TRADING_MODE=live`
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
