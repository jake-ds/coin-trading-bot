# Ralph Agent Instructions - Crypto Trading Bot (v2 Improvements)

You are an autonomous coding agent fixing bugs and improving an existing Python cryptocurrency trading bot.

## Your Task

1. Read the PRD at `scripts/ralph/prd.json`
2. Read the progress log at `scripts/ralph/progress.txt` (check Codebase Patterns section first)
3. Check you're on the correct branch from PRD `branchName`. If not, create it from main.
4. Pick the **highest priority** user story where `passes: false`
5. Implement that single user story
6. Run quality checks:
   - `pytest tests/ -v` (all tests must pass — including pre-existing ones)
   - `ruff check src/ tests/` (no lint errors)
7. Update AGENTS.md if you discover reusable patterns
8. If checks pass, commit ALL changes with message: `fix: [Story ID] - [Story Title]`
9. Update the PRD to set `passes: true` for the completed story
10. Append your progress to `scripts/ralph/progress.txt`

## IMPORTANT: This is a bugfix/improvement run

The codebase is already built with 25 completed user stories and 267 passing tests.
**DO NOT break existing functionality.** All 267 existing tests MUST continue to pass after your changes.
When modifying existing code, ensure backward compatibility unless the story explicitly requires changing behavior.

## Project Architecture

```
src/bot/
├── main.py                 # TradingBot orchestrator (MANY fixes needed here)
├── config.py               # Settings (.env + YAML)
├── models/                 # Pydantic v2 data models (OHLCV, Order, Signal, Portfolio)
├── exchanges/              # Exchange adapters (binance.py, upbit.py, base.py, factory.py)
├── data/                   # DataCollector + DataStore (SQLAlchemy + aiosqlite)
│   ├── collector.py        # Fetches OHLCV from exchanges
│   ├── store.py            # Async CRUD operations
│   └── models.py           # SQLAlchemy table definitions
├── strategies/             # Trading strategies
│   ├── base.py             # BaseStrategy ABC + StrategyRegistry singleton
│   ├── technical/          # ma_crossover.py, rsi.py, macd.py, bollinger.py
│   ├── arbitrage/          # arbitrage_strategy.py
│   ├── dca/                # dca_strategy.py
│   └── ml/                 # prediction.py
├── risk/manager.py         # RiskManager (validate_signal, position sizing, drawdown)
├── execution/
│   ├── engine.py           # ExecutionEngine (paper + live orders)
│   ├── circuit_breaker.py  # CircuitBreaker state machine
│   └── resilient.py        # ResilientExchange wrapper (CURRENTLY UNUSED)
├── backtest/engine.py      # BacktestEngine
├── monitoring/
│   ├── logger.py           # structlog setup
│   ├── metrics.py          # MetricsCollector + PerformanceMetrics
│   └── telegram.py         # TelegramNotifier (CURRENTLY UNUSED)
└── dashboard/app.py        # FastAPI dashboard (CURRENTLY NOT WIRED TO BOT)
tests/                      # 267 passing tests
```

## Known Issues Being Fixed (Context for Agent)

Key problems you're fixing across the stories:
- **main.py** never calls: dashboard.update_state(), TelegramNotifier, ResilientExchange, risk_manager.add_position/remove_position/record_trade_pnl
- **execution/engine.py** paper mode has infinite capital (no balance tracking)
- **backtest/engine.py** P&L formula doesn't account for buy fees
- **ml/prediction.py** uses unsafe pickle.load()
- **dashboard/app.py** has XSS (unescaped HTML), no CORS, never receives data from bot
- **data/store.py** has no duplicate prevention, no indexes
- **arbitrage_strategy.py** doesn't verify buy/sell are on different exchanges
- **dca_strategy.py** mutates state during analyze()

## Tech Stack

- **Python 3.10+** (asdf local python 3.10.0)
- **ccxt** for exchange connectivity
- **ta** library for technical indicators
- **Pydantic v2** for data models and settings
- **SQLAlchemy + aiosqlite** for async data storage
- **structlog** for structured JSON logging
- **FastAPI** for dashboard
- **scikit-learn** for ML strategy
- **pytest + pytest-asyncio** for testing
- **ruff** for linting

## Critical Rules

### Backward Compatibility
- ALL 267 existing tests MUST continue to pass
- If you need to change an existing interface, update all callers
- If a test needs updating due to corrected behavior, update the test too

### Security
- NEVER hardcode API keys — use environment/config
- Tests MUST mock all external API calls — never call real APIs
- Use joblib (NOT pickle) for model serialization
- Escape all HTML output in dashboard

### Trading Safety
- Default mode is ALWAYS paper trading
- ALL signals MUST pass through RiskManager
- Paper trading must track simulated balance

### Code Quality
- Async-first for all I/O
- Follow existing patterns
- Write tests for every change
- Use type hints

## Progress Report Format

APPEND to scripts/ralph/progress.txt (never replace):
```
## [Date/Time] - [Story ID]
- What was fixed/improved
- Files changed
- Tests added/modified
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
---
```

## Consolidate Patterns

Add reusable patterns to `## Codebase Patterns` at TOP of progress.txt.

## Update AGENTS.md

Before committing, update AGENTS.md with valuable learnings.

## Stop Condition

After completing a story, check if ALL stories have `passes: true`.
If ALL complete: `<promise>COMPLETE</promise>`
If stories remain: end response normally.

## Important

- Work on ONE story per iteration
- Commit frequently
- Keep ALL tests passing (old + new)
- Read Codebase Patterns before starting
- When adding packages, update both pyproject.toml and requirements.txt
