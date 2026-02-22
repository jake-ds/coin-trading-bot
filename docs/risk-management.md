# Risk Management

## Overview

The bot implements a multi-layered risk management system that validates every signal before execution. No trade can bypass the risk checks.

## Risk Layers

### Layer 1: Signal Ensemble Voting

Before risk checks, the ensemble ensures multiple strategies agree:

- Requires `min_agreement` strategies to agree on direction
- Conflicting signals (BUY + SELL) result in HOLD
- Higher-timeframe trend filter rejects counter-trend signals

### Layer 2: Individual Trade Risk (RiskManager)

| Control | Config | Description |
|---------|--------|-------------|
| Daily Loss Limit | `daily_loss_limit_pct=5.0` | Halts trading when daily loss exceeds threshold |
| Max Drawdown | `max_drawdown_pct=15.0` | Halts trading when peak-to-trough decline exceeds threshold |
| Max Concurrent Positions | `max_concurrent_positions=5` | Limits number of simultaneous positions |
| Max Position Size | `max_position_size_pct=10.0` | Caps single position as % of portfolio |

### Layer 3: Portfolio-Level Risk (PortfolioRiskManager)

| Control | Config | Description |
|---------|--------|-------------|
| Total Exposure | `max_total_exposure_pct=60.0` | Max portfolio allocated to positions |
| Correlation Limit | `max_correlation=0.8` | Rejects positions correlated with existing ones |
| Sector Concentration | `max_positions_per_sector=3` | Limits positions per asset sector |
| Portfolio Heat | `max_portfolio_heat=0.15` | Risk-weighted exposure limit |

### Layer 4: Position Sizing (ATR-based)

Dynamic sizing based on volatility:

```
position_size = (portfolio_value * risk_per_trade_pct) / (ATR * atr_multiplier)
```

- Higher volatility → smaller positions
- Lower volatility → larger positions (up to max_position_size_pct cap)

### Layer 5: Position Management (PositionManager)

| Exit Type | Trigger | Action |
|-----------|---------|--------|
| Stop-Loss | Price drops `stop_loss_pct` below entry | Full exit |
| Take-Profit 1 | Price rises `tp1_pct` above entry | Partial exit (50%) |
| Take-Profit 2 | Price rises `take_profit_pct` above entry | Full exit of remainder |
| Trailing Stop | Price drops `trailing_stop_pct` from highest since entry | Full exit |

### Layer 6: Strategy Auto-Disable (StrategyTracker)

Automatically disables underperforming strategies:

| Criterion | Config | Trigger |
|-----------|--------|---------|
| Consecutive Losses | `strategy_max_consecutive_losses=5` | Strategy generates 5+ consecutive losing trades |
| Win Rate | `strategy_min_win_rate_pct=40.0` | Win rate drops below threshold (after min trades) |
| Re-enable Check | `strategy_re_enable_check_hours=24.0` | Disabled strategies checked for re-enable periodically |

## Recommended Settings

### Conservative Profile

For capital preservation with lower returns:

```yaml
max_position_size_pct: 3.0
stop_loss_pct: 2.0
take_profit_pct: 4.0
daily_loss_limit_pct: 2.0
max_drawdown_pct: 8.0
max_concurrent_positions: 3
risk_per_trade_pct: 0.5
max_total_exposure_pct: 30.0
signal_min_agreement: 3
```

### Moderate Profile (Default)

Balanced risk-reward:

```yaml
max_position_size_pct: 10.0
stop_loss_pct: 3.0
take_profit_pct: 5.0
daily_loss_limit_pct: 5.0
max_drawdown_pct: 15.0
max_concurrent_positions: 5
risk_per_trade_pct: 1.0
max_total_exposure_pct: 60.0
signal_min_agreement: 2
```

### Aggressive Profile

Higher risk tolerance for experienced traders:

```yaml
max_position_size_pct: 15.0
stop_loss_pct: 5.0
take_profit_pct: 10.0
daily_loss_limit_pct: 8.0
max_drawdown_pct: 25.0
max_concurrent_positions: 8
risk_per_trade_pct: 2.0
max_total_exposure_pct: 80.0
signal_min_agreement: 2
```

## Risk Events

The bot logs and optionally sends Telegram alerts for:

- Daily loss limit reached (trading halted)
- Max drawdown reached (trading halted)
- Portfolio risk rejection (position blocked)
- Strategy auto-disabled
- Circuit breaker tripped (exchange API failure)
