# Coin Trading Bot

A Python-based cryptocurrency automated trading system with multiple strategies, risk management, backtesting, and a monitoring dashboard.

## Features

- **Multiple Strategies**: MA Crossover, RSI, MACD, Bollinger Bands, DCA, Cross-Exchange Arbitrage, ML Prediction
- **Exchange Support**: Binance and Upbit via ccxt
- **Risk Management**: Position sizing, stop-loss, daily loss limits, max drawdown protection
- **Backtesting**: Test strategies against historical data with configurable fees/slippage
- **Monitoring**: Structured logging, performance metrics, Telegram alerts, web dashboard
- **Paper Trading**: Safe paper trading mode by default
- **Docker**: Containerized deployment with health checks

## Architecture

```
src/bot/
├── main.py                 # TradingBot orchestrator, CLI entry point
├── config.py               # pydantic-settings configuration
├── models/                 # Pydantic v2 data models
├── exchanges/              # Exchange adapters (Binance, Upbit) via ccxt
├── data/                   # DataCollector + DataStore (SQLAlchemy + aiosqlite)
├── strategies/             # Plugin-based strategies
│   ├── base.py             # BaseStrategy ABC + StrategyRegistry
│   ├── technical/          # MA Crossover, RSI, MACD, Bollinger
│   ├── arbitrage/          # Cross-exchange arbitrage
│   ├── dca/                # Dollar Cost Averaging
│   └── ml/                 # ML price prediction (scikit-learn)
├── risk/                   # RiskManager (position sizing, loss limits)
├── execution/              # ExecutionEngine, CircuitBreaker, ResilientExchange
├── backtest/               # BacktestEngine (historical simulation)
├── monitoring/             # Logging, MetricsCollector, Telegram alerts
└── dashboard/              # FastAPI web dashboard
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
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f bot
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_MODE` | `paper` | Trading mode: `paper` or `live` |
| `BINANCE_API_KEY` | | Binance API key |
| `BINANCE_SECRET_KEY` | | Binance secret key |
| `BINANCE_TESTNET` | `true` | Use Binance testnet |
| `UPBIT_API_KEY` | | Upbit API key |
| `UPBIT_SECRET_KEY` | | Upbit secret key |
| `DATABASE_URL` | `sqlite+aiosqlite:///data/trading.db` | Database connection URL |
| `TELEGRAM_BOT_TOKEN` | | Telegram bot token for alerts |
| `TELEGRAM_CHAT_ID` | | Telegram chat ID |
| `MAX_POSITION_SIZE_PCT` | `10.0` | Max position size as % of portfolio |
| `DAILY_LOSS_LIMIT_PCT` | `5.0` | Daily loss limit as % of portfolio |
| `MAX_DRAWDOWN_PCT` | `15.0` | Max drawdown % before halting |
| `STOP_LOSS_PCT` | `3.0` | Per-trade stop-loss % |
| `MAX_CONCURRENT_POSITIONS` | `5` | Max concurrent open positions |
| `LOOP_INTERVAL_SECONDS` | `60` | Trading loop interval in seconds |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `DASHBOARD_PORT` | `8000` | Dashboard web server port |
| `SYMBOLS` | `["BTC/USDT"]` | Trading symbols |

### YAML Override

Create a `config.yaml` file to override settings:

```yaml
trading_mode: paper
symbols:
  - BTC/USDT
  - ETH/USDT
loop_interval_seconds: 30
log_level: DEBUG
```

### Exchange Setup

1. **Binance**: Create an API key at https://www.binance.com/en/my/settings/api-management. For testing, enable testnet.
2. **Upbit**: Create an API key at https://upbit.com/mypage/open_api_management.

## Strategy Development

Create a custom strategy by extending `BaseStrategy`:

```python
from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy, strategy_registry


@strategy_registry.register
class MyStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "my_strategy"

    @property
    def required_history_length(self) -> int:
        return 20

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs) -> TradingSignal:
        # Your strategy logic here
        last_price = ohlcv_data[-1].close

        return TradingSignal(
            strategy_name=self.name,
            symbol=kwargs.get("symbol", "BTC/USDT"),
            action=SignalAction.BUY,  # or SELL or HOLD
            confidence=0.8,
        )
```

## Backtesting

Run backtests with the `BacktestEngine`:

```python
from bot.backtest.engine import BacktestEngine
from bot.strategies.technical.rsi import RSIStrategy

engine = BacktestEngine(
    initial_capital=10000.0,
    fee_pct=0.1,      # 0.1% trading fee
    slippage_pct=0.05, # 0.05% slippage
)

strategy = RSIStrategy()
result = await engine.run(strategy, candle_data, symbol="BTC/USDT")

print(f"Total Return: {result.metrics.total_return_pct}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio}")
print(f"Max Drawdown: {result.metrics.max_drawdown_pct}%")
print(f"Win Rate: {result.metrics.win_rate}%")
print(f"Total Trades: {result.metrics.total_trades}")
```

## Dashboard API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | HTML dashboard page |
| `/status` | GET | Bot status (running/stopped) |
| `/trades` | GET | Recent trades |
| `/metrics` | GET | Performance metrics |
| `/portfolio` | GET | Current portfolio |
| `/health` | GET | Health check for Docker |

## Deployment

### Manual Setup

```bash
# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env

# Run the bot
python -m bot.main
```

### Docker

```bash
# Build and start
docker-compose up -d

# Check health
curl http://localhost:8000/health

# Stop
docker-compose down
```

### Monitoring

- **Dashboard**: Open `http://localhost:8000` in your browser
- **Telegram**: Configure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` for real-time alerts
- **Logs**: JSON-structured logs via structlog. Use `LOG_LEVEL=DEBUG` for verbose output.

## Safety

- **Paper trading is the default** - live trading requires explicit `TRADING_MODE=live`
- All signals pass through the RiskManager before execution
- Circuit breaker pattern protects against exchange API failures
- Daily loss limits and max drawdown protection halt trading automatically
