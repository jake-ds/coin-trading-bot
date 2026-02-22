# Configuration Reference

## Loading Order

1. **Environment variables** (highest priority)
2. **`.env` file** (loaded by pydantic-settings)
3. **`config.yaml`** (optional override file)
4. **Defaults** (defined in `Settings` class)

## Environment Variables

### Core Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TRADING_MODE` | `paper`/`live` | `paper` | Trading mode. Live requires explicit configuration |
| `SYMBOLS` | JSON list | `["BTC/USDT"]` | Trading pair symbols |
| `LOOP_INTERVAL_SECONDS` | int | `60` | Main trading loop interval (seconds) |
| `LOG_LEVEL` | string | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `CONFIG_FILE` | string | `""` | Path to YAML config override file |

### Exchange Credentials

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BINANCE_API_KEY` | string | `""` | Binance API key |
| `BINANCE_SECRET_KEY` | string | `""` | Binance secret key |
| `BINANCE_TESTNET` | bool | `true` | Use Binance testnet |
| `UPBIT_API_KEY` | string | `""` | Upbit API key |
| `UPBIT_SECRET_KEY` | string | `""` | Upbit secret key |

### Database

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATABASE_URL` | string | `sqlite+aiosqlite:///data/trading.db` | SQLAlchemy database URL |

### Risk Management

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_POSITION_SIZE_PCT` | float | `10.0` | Max single position size (% of portfolio) |
| `STOP_LOSS_PCT` | float | `3.0` | Per-trade stop-loss percentage |
| `TAKE_PROFIT_PCT` | float | `5.0` | Per-trade take-profit percentage |
| `TRAILING_STOP_ENABLED` | bool | `false` | Enable trailing stop-loss |
| `TRAILING_STOP_PCT` | float | `2.0` | Trailing stop distance (%) |
| `DAILY_LOSS_LIMIT_PCT` | float | `5.0` | Max daily loss before halting (%) |
| `MAX_DRAWDOWN_PCT` | float | `15.0` | Max drawdown before halting (%) |
| `MAX_CONCURRENT_POSITIONS` | int | `5` | Maximum simultaneous open positions |

### Dynamic Position Sizing (ATR-based)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RISK_PER_TRADE_PCT` | float | `1.0` | Risk per trade as % of portfolio |
| `ATR_MULTIPLIER` | float | `2.0` | ATR multiplier for stop distance |
| `ATR_PERIOD` | int | `14` | ATR calculation period |

### Portfolio-Level Risk

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_TOTAL_EXPOSURE_PCT` | float | `60.0` | Max total portfolio exposure (%) |
| `MAX_CORRELATION` | float | `0.8` | Max allowed correlation between positions |
| `CORRELATION_WINDOW` | int | `30` | Correlation calculation window (candles) |
| `MAX_POSITIONS_PER_SECTOR` | int | `3` | Max positions per sector |
| `MAX_PORTFOLIO_HEAT` | float | `0.15` | Max portfolio heat (risk-weighted exposure) |
| `SECTOR_MAP` | JSON dict | `{}` | Symbol-to-sector mapping |

### Paper Trading

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PAPER_INITIAL_BALANCE` | float | `10000.0` | Starting balance for paper trading |
| `PAPER_FEE_PCT` | float | `0.1` | Simulated trading fee (%) |

### Signal Ensemble

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SIGNAL_MIN_AGREEMENT` | int | `2` | Min strategies that must agree |
| `STRATEGY_WEIGHTS` | JSON dict | `{}` | Per-strategy weight overrides |

### Multi-Timeframe

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TIMEFRAMES` | JSON list | `["15m","1h","4h","1d"]` | Data collection timeframes |
| `TREND_TIMEFRAME` | string | `4h` | Timeframe for trend filter |

### Smart Execution

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PREFER_LIMIT_ORDERS` | bool | `true` | Prefer limit orders over market |
| `LIMIT_ORDER_TIMEOUT_SECONDS` | float | `30.0` | Timeout before fallback to market |
| `TWAP_CHUNK_COUNT` | int | `5` | TWAP order split count |

### Strategy Auto-Disable

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `STRATEGY_MAX_CONSECUTIVE_LOSSES` | int | `5` | Consecutive losses before disable |
| `STRATEGY_MIN_WIN_RATE_PCT` | float | `40.0` | Min win rate before disable |
| `STRATEGY_MIN_TRADES_FOR_EVAL` | int | `20` | Min trades before evaluation |
| `STRATEGY_RE_ENABLE_CHECK_HOURS` | float | `24.0` | Hours before re-enable check |

### Funding Rate Strategy

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FUNDING_EXTREME_POSITIVE_RATE` | float | `0.0005` | Positive rate threshold |
| `FUNDING_EXTREME_NEGATIVE_RATE` | float | `-0.0003` | Negative rate threshold |
| `FUNDING_CONFIDENCE_SCALE` | float | `10.0` | Confidence scaling factor |
| `FUNDING_SPREAD_THRESHOLD_PCT` | float | `0.5` | Perp-spot spread threshold |
| `FUNDING_RATE_HISTORY_LIMIT` | int | `50` | Funding rate history lookback |

### Validation (Go/No-Go)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VALIDATION_MIN_WIN_RATE_PCT` | float | `45.0` | Min win rate for GO |
| `VALIDATION_MIN_SHARPE_RATIO` | float | `0.5` | Min Sharpe ratio for GO |
| `VALIDATION_MAX_DRAWDOWN_PCT` | float | `15.0` | Max drawdown for GO |
| `VALIDATION_MIN_TRADES` | int | `10` | Min trades for GO |

### WebSocket

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WEBSOCKET_ENABLED` | bool | `false` | Enable WebSocket data feed |
| `WEBSOCKET_POLL_INTERVAL` | float | `5.0` | WS poll interval (seconds) |
| `WEBSOCKET_MAX_RECONNECT_DELAY` | float | `60.0` | Max reconnect delay (seconds) |

### Telegram

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | string | `""` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | string | `""` | Telegram chat ID |

### Dashboard

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DASHBOARD_PORT` | int | `8000` | Dashboard web server port |
| `ALLOWED_ORIGINS` | JSON list | `["http://localhost","http://localhost:8000"]` | CORS allowed origins |

## YAML Override

Create a `config.yaml` file to override any settings:

```yaml
trading_mode: paper
symbols:
  - BTC/USDT
  - ETH/USDT
  - SOL/USDT
loop_interval_seconds: 30
log_level: DEBUG

# Risk settings
max_position_size_pct: 5.0
stop_loss_pct: 2.5
take_profit_pct: 7.5
trailing_stop_enabled: true
trailing_stop_pct: 1.5

# Ensemble
signal_min_agreement: 2
strategy_weights:
  ma_crossover: 1.2
  rsi: 1.0
  bollinger: 0.8

# Portfolio risk
sector_map:
  BTC/USDT: layer1
  ETH/USDT: layer1
  SOL/USDT: layer1
  DOGE/USDT: meme
```
