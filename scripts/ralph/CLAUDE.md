# Ralph Agent Instructions - Crypto Trading Bot (v2 Production)

You are an autonomous coding agent transforming a toy crypto trading bot into a production-grade system capable of generating real profit. The codebase already has 25 completed user stories with 267 passing tests.

## Your Task

1. Read the PRD at `scripts/ralph/prd.json`
2. Read the progress log at `scripts/ralph/progress.txt` (check Codebase Patterns section FIRST)
3. Check you're on the correct branch from PRD `branchName`. If not, create it from main.
4. Pick the **highest priority** user story where `passes: false`
5. Implement that single user story fully
6. Run quality checks:
   - `pytest tests/ -v` (ALL tests must pass — old + new)
   - `ruff check src/ tests/` (no lint errors)
7. Update AGENTS.md if you discover reusable patterns
8. If checks pass, commit ALL changes with message: `feat: [Story ID] - [Story Title]`
9. Update the PRD to set `passes: true` for the completed story
10. Append your progress to `scripts/ralph/progress.txt`

## CRITICAL: DO NOT BREAK EXISTING TESTS

The codebase has 267 passing tests. ALL must continue passing after every change.
When modifying existing code:
- Add new parameters with default values matching old behavior
- If a test needs updating because behavior was WRONG (e.g., backtest PnL formula), update the test
- Never remove test coverage — only add to it

## Project Architecture (Current)

```
src/bot/
├── main.py                 # TradingBot orchestrator — MANY changes needed
├── config.py               # pydantic-settings config
├── models/                 # Pydantic v2: OHLCV, Order, TradingSignal, Portfolio
│   └── base.py             # Enums: OrderSide, OrderType, OrderStatus, SignalAction
├── exchanges/
│   ├── base.py             # ExchangeAdapter ABC
│   ├── factory.py          # ExchangeFactory registry
│   ├── binance.py          # BinanceAdapter (ccxt)
│   └── upbit.py            # UpbitAdapter (ccxt)
├── data/
│   ├── collector.py        # DataCollector (REST polling)
│   ├── store.py            # DataStore (SQLAlchemy + aiosqlite)
│   └── models.py           # SQLAlchemy tables
├── strategies/
│   ├── base.py             # BaseStrategy ABC + StrategyRegistry singleton
│   ├── technical/          # ma_crossover, rsi, macd, bollinger
│   ├── arbitrage/          # arbitrage_strategy
│   ├── dca/                # dca_strategy
│   └── ml/                 # prediction (RandomForest)
├── risk/manager.py         # RiskManager (validate_signal, sizing, drawdown)
├── execution/
│   ├── engine.py           # ExecutionEngine (paper + live)
│   ├── circuit_breaker.py  # CircuitBreaker FSM
│   └── resilient.py        # ResilientExchange wrapper (UNUSED - fix in V2-002)
├── backtest/engine.py      # BacktestEngine
├── monitoring/
│   ├── logger.py           # structlog setup
│   ├── metrics.py          # MetricsCollector
│   └── telegram.py         # TelegramNotifier (UNUSED - fix in V2-002)
└── dashboard/app.py        # FastAPI (UNWIRED - fix in V2-002)
```

## Architecture After v2 (New Files to Create)

```
src/bot/
├── execution/
│   ├── paper_portfolio.py  # V2-003: PaperPortfolio (balance tracking)
│   ├── position_manager.py # V2-008: Stop-loss, take-profit, trailing stop
│   └── smart_executor.py   # V2-019: Limit orders, TWAP
├── strategies/
│   ├── ensemble.py         # V2-009: SignalEnsemble voting system
│   ├── trend_filter.py     # V2-010: Multi-TF trend confirmation
│   ├── regime.py           # V2-011: Market regime detector
│   ├── indicators.py       # V2-017: Shared indicator utilities (ATR etc)
│   └── technical/
│       ├── vwap.py          # V2-015: VWAP strategy
│       ├── composite.py     # V2-016: Triple-confirmation momentum
│       └── funding_rate.py  # V2-024: Funding rate strategy
├── data/
│   ├── websocket_feed.py   # V2-018: Real-time WebSocket feed
│   ├── funding.py          # V2-024: Funding rate data
│   └── order_book.py       # V2-025: Order book analysis
├── risk/
│   └── portfolio_risk.py   # V2-023: Portfolio-level risk
└── monitoring/
    └── strategy_tracker.py # V2-022: Per-strategy performance
```

## Key Design Decisions for Agent

### Signal Flow (After v2 Complete)
```
Data Collection (REST + WebSocket)
    ↓
Market Regime Detection (TRENDING/RANGING/VOLATILE)
    ↓
Strategy Analysis (adapt to regime) → Multiple TradingSignals
    ↓
Signal Ensemble Voting (require min_agreement, filter conflicts)
    ↓
Trend Filter (reject signals against higher-TF trend)
    ↓
Risk Manager (position limits, drawdown, daily loss)
    ↓
Portfolio Risk (correlation, exposure limits)
    ↓
Dynamic Position Sizing (ATR-based)
    ↓
Smart Execution (limit orders, TWAP)
    ↓
Position Manager (monitor stop-loss, take-profit, trailing)
    ↓
Metrics + Telegram + Dashboard
```

### Critical Principles
- **Confirmation over speed**: Never trade on a single indicator. Require multiple confirmations.
- **Risk first**: Every trade has a predefined stop-loss and take-profit BEFORE entry.
- **Adapt to regime**: Wrong strategy in wrong market = guaranteed loss.
- **Realistic testing**: Backtest must enforce stop-loss, dynamic slippage, walk-forward.
- **Finite capital**: Paper trading must track actual balance, not infinite money.

## Tech Stack

- **Python 3.10+** (asdf local python 3.10.0)
- ccxt, ta, Pydantic v2, pydantic-settings
- SQLAlchemy + aiosqlite, structlog
- FastAPI + Chart.js (dashboard)
- scikit-learn (GradientBoosting after V2-020)
- pytest + pytest-asyncio, ruff

## Progress Report Format

APPEND to scripts/ralph/progress.txt:
```
## [Date/Time] - [Story ID]
- What was implemented/fixed
- Files created/modified
- Tests added
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
---
```

## Consolidate Patterns

Add reusable patterns to `## Codebase Patterns` at TOP of progress.txt.

## Stop Condition

If ALL stories have `passes: true`: `<promise>COMPLETE</promise>`
Otherwise end normally.

## Important

- Work on ONE story per iteration
- Commit frequently
- ALL tests must pass (old + new)
- Read Codebase Patterns FIRST
- New packages → update pyproject.toml AND requirements.txt
- New files → add __init__.py imports where appropriate
- Default parameter values must maintain backward compatibility with existing tests
