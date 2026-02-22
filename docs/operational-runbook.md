# Operational Runbook

## Starting Paper Trading

### 1. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your exchange API keys. For initial testing, use Binance testnet:

```
TRADING_MODE=paper
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
BINANCE_TESTNET=true
SYMBOLS=["BTC/USDT","ETH/USDT"]
```

### 2. Start the Bot

```bash
python -m bot.main
```

### 3. Monitor

- **Dashboard**: Open `http://localhost:8000`
- **Logs**: Watch stdout for structured JSON logs
- **Telegram**: Configure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` for alerts

## Running Validation

Before going live, run the automated validation:

```bash
# Run 48-hour paper trading validation
python -m bot.main --validate --duration=48h
```

The validation report will:
- Run paper trading for the specified duration
- Generate a GO/NO-GO recommendation
- Save the report to `data/validation_report_<timestamp>.json`
- Print a summary to stdout
- Send results via Telegram (if configured)

### Go/No-Go Criteria (default)

| Criterion | Threshold | Description |
|-----------|-----------|-------------|
| Min Trades | >= 10 | Enough trades for statistical significance |
| Win Rate | >= 45% | Minimum profitable trade ratio |
| Sharpe Ratio | >= 0.5 | Minimum risk-adjusted return |
| Max Drawdown | <= 15% | Maximum peak-to-trough decline |

If any criterion fails, the report shows NO-GO with details on which criteria failed and by how much.

## Going Live

### Pre-live Checklist

1. Paper trading validation returns GO
2. Review the validation report: win rate, Sharpe, drawdown
3. Set conservative risk parameters initially:
   ```
   MAX_POSITION_SIZE_PCT=3.0
   DAILY_LOSS_LIMIT_PCT=2.0
   MAX_CONCURRENT_POSITIONS=3
   ```
4. Ensure Telegram notifications are configured
5. Verify Binance API key permissions (spot trading only, no withdrawal)

### Switch to Live

```bash
TRADING_MODE=live
BINANCE_TESTNET=false
```

### Initial Live Period

- Monitor closely for the first 24 hours
- Check the dashboard frequently
- Verify orders are filling correctly
- Compare live results with paper trading expectations
- Gradually increase position sizes after confirming stability

## Handling Emergencies

### Exchange API Down

**Symptoms**: Repeated errors in logs about connection failures, circuit breaker tripped.

**Action**:
1. The `ResilientExchange` wrapper automatically retries with exponential backoff
2. Circuit breaker will trip after repeated failures, pausing API calls
3. No new trades will be placed while circuit breaker is open
4. Existing positions remain (stop-loss orders are on-exchange if using stop-limit)
5. Monitor exchange status page
6. Bot will automatically resume when the exchange is back

### Large Drawdown

**Symptoms**: Portfolio value declining significantly, daily loss approaching limit.

**Action**:
1. Check if market conditions have changed dramatically
2. Review which strategies are losing (check dashboard strategy stats)
3. If daily loss limit hits, trading auto-halts for the day
4. If max drawdown hits, trading halts until manual intervention
5. Consider reducing position sizes or disabling aggressive strategies

### Bot Crash / Restart

**Symptoms**: Bot process exited unexpectedly.

**Action**:
1. Check logs for the error: `docker-compose logs bot --tail 100`
2. Restart: `python -m bot.main` or `docker-compose restart bot`
3. The bot will re-initialize all components
4. Open positions on the exchange are not affected by bot restarts
5. Paper portfolio state is in-memory and will reset on restart

### Strategy Underperforming

**Symptoms**: Strategy auto-disabled by StrategyTracker, or consistently losing.

**Action**:
1. Check strategy stats on dashboard
2. Review if market regime has changed (ranging vs trending)
3. The StrategyTracker will auto-disable after `strategy_max_consecutive_losses`
4. Disabled strategies are re-evaluated after `strategy_re_enable_check_hours`
5. You can manually disable via the dashboard toggle endpoint

## Docker Operations

### Start

```bash
docker-compose up -d
```

### View Logs

```bash
# All logs
docker-compose logs -f bot

# Last 100 lines
docker-compose logs --tail 100 bot
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Restart

```bash
docker-compose restart bot
```

### Stop

```bash
docker-compose down
```

## Monitoring Checklist (Daily)

1. Check dashboard: overall P&L, open positions, equity curve
2. Review strategy stats: any strategies auto-disabled?
3. Check error count in logs
4. Verify data collection is working (candles being stored)
5. Confirm Telegram alerts are being received
6. Review daily P&L against expectations

## Useful API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard HTML page |
| `GET /health` | Health check (for Docker/monitoring) |
| `GET /status` | Bot status (running/stopped) |
| `GET /trades` | Recent trades list |
| `GET /metrics` | Performance metrics |
| `GET /portfolio` | Current portfolio state |
| `GET /strategies` | Strategy status and stats |
| `POST /strategies/{name}/toggle` | Enable/disable a strategy |
