# Backtesting Guide

## Overview

The backtesting engine simulates trading strategies against historical data with realistic conditions: stop-loss/take-profit enforcement, dynamic slippage, fees, and walk-forward analysis.

## Running Backtests

### CLI Strategy Comparison

```bash
# Compare all registered strategies
python -m bot.backtest

# This will:
# 1. Load historical data from the database
# 2. Run each strategy independently
# 3. Print comparison table with metrics
```

### Programmatic Usage

```python
import asyncio
from bot.backtest.engine import BacktestEngine
from bot.strategies.technical.rsi import RSIStrategy
from bot.models import OHLCV

engine = BacktestEngine(
    initial_capital=10000.0,
    fee_pct=0.1,       # 0.1% trading fee (Binance maker)
    slippage_pct=0.05,  # 0.05% slippage simulation
)

strategy = RSIStrategy()
result = await engine.run(strategy, candle_data, symbol="BTC/USDT")
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | 10000.0 | Starting portfolio value (USDT) |
| `fee_pct` | 0.1 | Trading fee per trade (%) |
| `slippage_pct` | 0.05 | Simulated slippage (%) |

## Interpreting Results

### Key Metrics

| Metric | Good | Warning | Description |
|--------|------|---------|-------------|
| Total Return | > 0% | < -5% | Net profit/loss percentage |
| Sharpe Ratio | > 1.0 | < 0.5 | Risk-adjusted return (higher = better) |
| Sortino Ratio | > 1.5 | < 0.5 | Downside risk-adjusted return |
| Max Drawdown | < 10% | > 20% | Largest peak-to-trough decline |
| Win Rate | > 50% | < 40% | Percentage of profitable trades |
| Profit Factor | > 1.5 | < 1.0 | Gross profit / gross loss |
| Total Trades | > 20 | < 5 | Statistical significance |

### Result Fields

```python
result.metrics.total_return_pct  # Net return percentage
result.metrics.sharpe_ratio      # Sharpe ratio
result.metrics.max_drawdown_pct  # Maximum drawdown
result.metrics.win_rate          # Win rate percentage
result.metrics.total_trades      # Number of trades
result.metrics.profit_factor     # Gross profit / gross loss
result.trades                    # List of all trades
result.equity_curve              # Portfolio value over time
```

## Realistic Backtesting Features

### Stop-Loss / Take-Profit Enforcement

The backtest engine enforces SL/TP exits during simulation, matching the live PositionManager behavior:

- **Stop-loss**: Closes position when price drops below SL level
- **Take-profit 1**: Partial exit (50%) at TP1 level
- **Take-profit 2**: Full exit at TP2 level

### Dynamic Slippage

Slippage varies based on:
- Order size relative to available liquidity
- Market volatility (higher volatility = more slippage)
- Base slippage rate (configurable)

### Walk-Forward Analysis

Walk-forward analysis trains on historical data and tests on out-of-sample data:

1. Split data into N folds
2. For each fold: train on previous data, test on current fold
3. Aggregate out-of-sample results
4. Prevents overfitting to historical data

## Tips

- **Minimum data**: Use at least 200 candles for reliable results
- **Fee impact**: Frequent trading strategies need higher returns to overcome fees
- **Regime awareness**: Test strategies across different market conditions
- **Statistical significance**: Require at least 20-30 trades for meaningful metrics
- **Walk-forward**: Always use walk-forward when optimizing parameters to avoid overfitting
- **Multiple timeframes**: Test on different timeframes (15m, 1h, 4h, 1d) for robustness
