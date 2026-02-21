# Crypto Trading Bot

## Overview
Python-based cryptocurrency automated trading system with multiple strategies, risk management, backtesting, and a monitoring dashboard.

## Project Structure
```
src/bot/                    # Main application package
├── main.py                 # TradingBot orchestrator, CLI entry point
├── config.py               # pydantic-settings configuration
├── models/                 # Pydantic v2 data models (OHLCV, Order, Signal, Portfolio)
├── exchanges/              # Exchange adapters via ccxt (Binance, Upbit)
├── data/                   # DataCollector + DataStore (SQLAlchemy + aiosqlite)
├── strategies/             # Plugin-based strategies
│   ├── base.py             # BaseStrategy ABC + StrategyRegistry
│   ├── technical/          # MA Crossover, RSI, MACD, Bollinger
│   ├── arbitrage/          # Cross-exchange arbitrage
│   ├── dca/                # Dollar Cost Averaging
│   └── ml/                 # ML price prediction (scikit-learn)
├── risk/                   # RiskManager (position sizing, loss limits, drawdown protection)
├── execution/              # ExecutionEngine (order lifecycle, retry, paper trading)
├── backtest/               # BacktestEngine (historical simulation)
├── monitoring/             # Logging (structlog), MetricsCollector, Telegram alerts
└── dashboard/              # FastAPI web dashboard
tests/                      # pytest + pytest-asyncio test suite
scripts/ralph/              # Ralph automation (prd.json, progress.txt)
```

## Commands

```bash
# Run the bot (paper trading by default)
TRADING_MODE=paper python -m bot.main

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Run backtests
python -m bot.backtest

# Start dashboard
uvicorn bot.dashboard.app:app --port 8000
```

## Key Patterns

- **Async-first**: All I/O operations use async/await
- **Exchange Adapter**: ABC base class with Factory pattern — never use ccxt directly in business logic
- **Strategy Plugin**: BaseStrategy ABC + StrategyRegistry with decorator registration
- **Pydantic v2 Models**: All data structures are Pydantic models with validators
- **Configuration**: pydantic-settings loading from .env with optional YAML override
- **Risk-First**: All signals pass through RiskManager before execution
- **Paper Trading Default**: Bot runs in paper mode unless explicitly configured for live

## Environment Variables

See `.env.example` for all configuration options. Copy to `.env` and fill in values.

## Critical Rules

- NEVER hardcode API keys — use environment variables
- NEVER call real exchange APIs in tests — always mock ccxt
- NEVER execute live trades without explicit configuration
- ALL signals must pass through RiskManager
- Default trading mode is ALWAYS paper
