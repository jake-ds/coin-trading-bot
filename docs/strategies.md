# Strategy Guide

## Overview

The bot uses a plugin-based strategy system where multiple strategies analyze the market independently, and their signals are combined through an ensemble voting system. Strategies adapt to the detected market regime.

## Signal Pipeline

1. **Market Regime Detection** - Classifies market as TRENDING_UP, TRENDING_DOWN, RANGING, or HIGH_VOLATILITY
2. **Strategy Analysis** - Each active strategy generates BUY/SELL/HOLD signals
3. **Ensemble Voting** - Requires `min_agreement` strategies to agree before acting
4. **Trend Filtering** - Rejects signals against the higher-timeframe trend

## Strategies

### MA Crossover (`ma_crossover`)

**How it works**: Detects crossovers between a short-period and long-period Simple Moving Average. A bullish crossover (short crosses above long) generates BUY; bearish crossover generates SELL.

**Confirmation filters** (optional):
- Volume confirmation: Requires above-average volume during crossover
- Momentum confirmation: Requires bullish candle for BUY, bearish for SELL
- ADX filter: Requires ADX > threshold (trending market)
- Trend strength: Requires MA distance to be expanding
- Cooldown: Minimum candles between signals

**Effective when**: Trending markets (TRENDING_UP, TRENDING_DOWN)
**Disabled in**: RANGING regime (whipsaws)

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `short_period` | 20 | Short MA period |
| `long_period` | 50 | Long MA period |
| `volume_confirmation` | false | Require volume spike |
| `adx_filter_enabled` | false | Require trend strength |
| `cooldown_candles` | 0 | Min candles between signals |

### RSI with Divergence (`rsi`)

**How it works**: Uses Relative Strength Index with oversold/overbought levels. Enhanced with divergence detection (price makes new low but RSI doesn't = bullish divergence).

**Effective when**: RANGING and HIGH_VOLATILITY (mean reversion)
**Adapted in**: Wider thresholds in TRENDING markets to avoid false signals

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `period` | 14 | RSI calculation period |
| `oversold` | 30 | Oversold threshold (BUY signal) |
| `overbought` | 70 | Overbought threshold (SELL signal) |

### MACD (`macd`)

**How it works**: Moving Average Convergence Divergence. BUY when MACD line crosses above signal line; SELL when it crosses below.

**Effective when**: Trending markets
**Parameters**: `fast_period=12`, `slow_period=26`, `signal_period=9`

### Bollinger Band Squeeze Breakout (`bollinger`)

**How it works**: Detects Bollinger Band squeeze (low bandwidth) followed by breakout. BUY on upper breakout after squeeze; SELL on lower breakout.

**Effective when**: Transitioning from RANGING to TRENDING
**Parameters**: `period=20`, `std_dev=2.0`

### VWAP Strategy (`vwap`)

**How it works**: Volume-Weighted Average Price. BUY when price crosses above VWAP with volume confirmation; SELL when crosses below.

**Effective when**: Intraday trending
**Parameters**: `period=20`

### Composite Momentum (`composite_momentum`)

**How it works**: Combines RSI, MACD, and Stochastic oscillator. Requires all three to agree on direction for a signal.

**Effective when**: Strong trends with momentum confirmation
**Parameters**: Uses default RSI/MACD/Stochastic parameters

### Funding Rate Strategy (`funding_rate`)

**How it works**: Monitors perpetual-spot price spread and funding rates. When funding is extremely positive (longs paying), generates contrarian SHORT signal. When extremely negative, generates contrarian LONG signal.

**Effective when**: Futures markets with high funding rate extremes
**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `funding_extreme_positive_rate` | 0.0005 | Positive rate threshold |
| `funding_extreme_negative_rate` | -0.0003 | Negative rate threshold |

### DCA Strategy (`dca`)

**How it works**: Dollar Cost Averaging. Generates periodic BUY signals at fixed intervals regardless of price, with optional price-drop acceleration.

**Effective when**: Long-term accumulation, any market condition

### Cross-Exchange Arbitrage (`arbitrage`)

**How it works**: Detects price differences between exchanges (Binance vs Upbit). When spread exceeds threshold, generates simultaneous BUY on cheaper and SELL on more expensive exchange.

**Effective when**: Multiple exchanges configured with sufficient liquidity

### ML Prediction (`ml_prediction`)

**How it works**: GradientBoosting classifier trained on technical features (RSI, MACD, BB position, volume ratio). Uses cross-validation for training. Generates signals based on predicted price direction.

**Effective when**: After sufficient training data (200+ candles minimum)

## Market Regime Adaptation

| Regime | MA Crossover | RSI | Bollinger | MACD |
|--------|-------------|-----|-----------|------|
| TRENDING_UP | Faster periods (10/30) | Wider thresholds | Active | Active |
| TRENDING_DOWN | Faster periods (10/30) | Wider thresholds | Active | Active |
| RANGING | **Disabled** | Default thresholds | Active (squeeze) | Active |
| HIGH_VOLATILITY | Default periods | Default thresholds | Active | Active |

## Ensemble Voting

The `SignalEnsemble` combines signals from all active strategies:

- Collects BUY/SELL/HOLD from each strategy
- Requires at least `min_agreement` strategies to agree on a direction
- If both BUY and SELL present → HOLD (conflict)
- Final confidence = weighted average of agreeing strategies
- Weights configurable per strategy via `strategy_weights`

## Creating Custom Strategies

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

    def adapt_to_regime(self, regime):
        # Optional: adapt parameters based on market regime
        pass

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs) -> TradingSignal:
        symbol = kwargs.get("symbol", "UNKNOWN")
        # Your analysis logic here
        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
        )
```
