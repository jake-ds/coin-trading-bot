# Live Trading Guide

Step-by-step guide for transitioning from paper trading to live trading.

## Prerequisites

Before going live, ensure you have:

1. A validated paper trading session with a GO recommendation
2. Exchange API keys with spot trading permissions (no withdrawal)
3. Telegram notifications configured for alerts
4. Understanding of the risk parameters and their implications

## Step 1: Paper Trading Validation

Run the automated validation to confirm the bot is performing well:

```bash
python -m bot.main --validate --duration=48h
```

Review the validation report:
- **Win rate**: Should be >= 45%
- **Sharpe ratio**: Should be >= 0.5
- **Max drawdown**: Should be <= 15%
- **Minimum trades**: At least 10 trades for statistical significance

The report is saved to `data/validation_report_<timestamp>.json`.

## Step 2: Configure Safety Parameters

Set these required parameters in your `.env` file:

```bash
# Required for live mode
STOP_LOSS_PCT=5.0              # Stop-loss percentage (FAIL if 0)
DAILY_LOSS_LIMIT_PCT=3.0       # Max daily loss (FAIL if 0)

# Recommended conservative settings
MAX_POSITION_SIZE_PCT=3.0      # Max position as % of portfolio
MAX_CONCURRENT_POSITIONS=3     # Limit concurrent positions
MAX_DRAWDOWN_PCT=10.0          # Max drawdown before halt
```

## Step 3: Secure the Dashboard

Set a real password to enable JWT authentication:

```bash
DASHBOARD_PASSWORD=your-secure-password-here
DASHBOARD_USERNAME=admin                        # Optional, default 'admin'
JWT_SECRET=your-random-secret-key              # Optional, auto-generated if unset
```

## Step 4: Configure Telegram Alerts

```bash
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
```

Telegram commands available after startup:
- `/stop` - Emergency stop
- `/closeall` - Close all positions
- `/resume` - Resume trading
- `/status` - Check bot status

## Step 5: Configure Exchange

```bash
BINANCE_API_KEY=your-api-key
BINANCE_SECRET_KEY=your-secret-key
BINANCE_TESTNET=false          # Set to false for live
SYMBOLS=["BTC/USDT","ETH/USDT"]
```

Ensure your API key has:
- Spot trading enabled
- No withdrawal permissions (for safety)
- IP whitelist configured (recommended)

## Step 6: Switch to Live Mode

```bash
TRADING_MODE=live
```

Start the bot:

```bash
python -m bot.main
```

Or with Docker:

```bash
docker-compose up -d
```

## Step 7: Pre-flight Checks

When starting in live mode, the bot automatically runs 8 pre-flight checks:

| Check | Required | Description |
|-------|----------|-------------|
| API Key Validity | Yes | Tests exchange connection |
| Sufficient Balance | Yes | Verifies >= $100 USD |
| Symbol Availability | Yes | Verifies trading pairs exist |
| Rate Limit Configured | Recommended | Checks rate limiter is on |
| Stop-Loss Configured | Yes | stop_loss_pct must be > 0 |
| Daily Loss Limit | Yes | daily_loss_limit_pct must be > 0 |
| Password Changed | Recommended | Dashboard auth should be enabled |
| Paper Validation | Recommended | Should have a GO validation report |

If any required check fails, the bot will not start. Fix the issue and retry.

## Step 8: Monitor the First 24 Hours

### What to Watch

1. **Dashboard**: Open `http://localhost:8000` and monitor:
   - Portfolio value and return
   - Open positions and their PnL
   - Strategy performance
   - Emergency state

2. **Telegram**: Watch for alerts about:
   - Trade executions
   - Position reconciliation discrepancies
   - Strategy auto-disabling
   - Emergency events

3. **Logs**: Check structured logs for any errors:
   ```bash
   docker-compose logs -f bot
   ```

4. **Position Reconciliation**: The bot periodically compares local state with exchange:
   ```bash
   curl http://localhost:8000/api/reconciliation \
     -H "Authorization: Bearer $TOKEN"
   ```

5. **Audit Trail**: Review all actions:
   ```bash
   curl http://localhost:8000/api/audit \
     -H "Authorization: Bearer $TOKEN"
   ```

### Common Issues

**Rate limiting**: If you see throttled requests, the rate limiter is working correctly. This prevents exchange bans.

**Position discrepancies**: If reconciliation detects mismatches, investigate immediately. This could indicate partially filled orders or manual trades on the exchange.

**Strategy auto-disabled**: Check which strategy was disabled and why. This is a safety feature that removes underperforming strategies.

## Emergency Procedures

### Stop Trading Immediately

**Via API:**
```bash
curl -X POST http://localhost:8000/api/emergency/stop \
  -H "Authorization: Bearer $TOKEN"
```

**Via Telegram:**
```
/stop
```

### Close All Positions and Stop

**Via API:**
```bash
curl -X POST http://localhost:8000/api/emergency/close-all \
  -H "Authorization: Bearer $TOKEN"
```

**Via Telegram:**
```
/closeall
```

### Resume After Emergency

**Via API:**
```bash
curl -X POST http://localhost:8000/api/emergency/resume \
  -H "Authorization: Bearer $TOKEN"
```

**Via Telegram:**
```
/resume
```

## Scaling Up

After confirming stable live operation:

1. **Gradually increase position sizes**: Raise `MAX_POSITION_SIZE_PCT`
2. **Add more symbols**: Expand the `SYMBOLS` list
3. **Enable more strategies**: Toggle strategies via the dashboard
4. **Increase concurrent positions**: Raise `MAX_CONCURRENT_POSITIONS`
5. **Review and adjust risk parameters**: Use the Settings page for hot-reload

Always make changes incrementally and monitor results before making further adjustments.
