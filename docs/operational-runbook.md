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
3. Set a real dashboard password: `DASHBOARD_PASSWORD=your-secure-password`
4. Configure stop-loss: `STOP_LOSS_PCT=5.0` (required for live mode)
5. Configure daily loss limit: `DAILY_LOSS_LIMIT_PCT=3.0` (required for live mode)
6. Set conservative risk parameters initially:
   ```
   MAX_POSITION_SIZE_PCT=3.0
   MAX_CONCURRENT_POSITIONS=3
   ```
7. Ensure Telegram notifications are configured
8. Verify Binance API key permissions (spot trading only, no withdrawal)

### Automated Pre-flight Checks

When `TRADING_MODE=live`, the bot runs 8 pre-flight checks before starting:

| Check | Status | Description |
|-------|--------|-------------|
| API Key Validity | FAIL if bad | Tests exchange connection |
| Sufficient Balance | FAIL if < $100 | Verifies USD balance |
| Symbol Availability | FAIL if missing | Verifies trading pairs exist |
| Rate Limit Configured | WARN if off | Checks rate limiter is enabled |
| Stop-Loss Configured | FAIL if 0 | Requires stop_loss_pct > 0 |
| Daily Loss Limit | FAIL if 0 | Requires daily_loss_limit_pct > 0 |
| Password Changed | WARN if default | Checks dashboard auth is enabled |
| Paper Validation | WARN if missing | Checks for GO validation report |

Any FAIL check prevents startup. WARN checks allow startup with alerts.

### Switch to Live

```bash
TRADING_MODE=live
BINANCE_TESTNET=false
```

### Initial Live Period

- Monitor closely for the first 24 hours
- Check the dashboard frequently
- Verify orders are filling correctly
- Check position reconciliation results (`GET /api/reconciliation`)
- Compare live results with paper trading expectations
- Gradually increase position sizes after confirming stability

## Emergency Procedures

### Emergency Stop (Kill Switch)

**Via Dashboard API:**
```bash
# Stop trading immediately (keeps positions open)
curl -X POST http://localhost:8000/api/emergency/stop \
  -H "Authorization: Bearer $TOKEN"

# Close all positions and stop
curl -X POST http://localhost:8000/api/emergency/close-all \
  -H "Authorization: Bearer $TOKEN"

# Resume trading after emergency
curl -X POST http://localhost:8000/api/emergency/resume \
  -H "Authorization: Bearer $TOKEN"

# Check current emergency state
curl http://localhost:8000/api/emergency \
  -H "Authorization: Bearer $TOKEN"
```

**Via Telegram:**
- `/stop` - Halt trading, cancel pending orders
- `/closeall` - Halt trading and close all positions at market
- `/resume` - Resume trading after emergency stop
- `/status` - Check bot status, positions, and emergency state

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

All data endpoints require JWT authentication when `DASHBOARD_PASSWORD` is set.

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check (for Docker/monitoring) |
| `GET /api/status` | Bot status (running/stopped) |
| `GET /api/trades` | Recent trades list (paginated) |
| `GET /api/portfolio` | Current portfolio state |
| `GET /api/positions` | Open positions |
| `GET /api/strategies` | Strategy status and stats |
| `POST /api/strategies/{name}/toggle` | Enable/disable a strategy |
| `GET /api/analytics` | Performance analytics |
| `GET /api/settings` | Current configuration |
| `PUT /api/settings` | Update settings (hot-reload) |
| `GET /api/audit` | Audit log (filterable) |
| `POST /api/emergency/stop` | Emergency stop |
| `POST /api/emergency/close-all` | Emergency close all |
| `POST /api/emergency/resume` | Resume after emergency |
| `GET /api/emergency` | Emergency state |
| `GET /api/preflight` | Pre-flight check results |
| `GET /api/reconciliation` | Position reconciliation results |
| `POST /api/auth/login` | Login (returns JWT) |
| `POST /api/auth/refresh` | Refresh access token |
| `WebSocket /api/ws` | Real-time updates |
