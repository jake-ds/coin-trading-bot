# Ralph Agent Instructions - Crypto Trading Bot

You are an autonomous coding agent building a Python cryptocurrency trading bot.

## Your Task

1. Read the PRD at `scripts/ralph/prd.json`
2. Read the progress log at `scripts/ralph/progress.txt` (check Codebase Patterns section first)
3. Check you're on the correct branch from PRD `branchName`. If not, check it out or create from main.
4. Pick the **highest priority** user story where `passes: false`
5. Implement that single user story
6. Run quality checks:
   - `pytest tests/ -v` (all tests must pass)
   - `ruff check src/ tests/` (no lint errors)
7. Update AGENTS.md if you discover reusable patterns
8. If checks pass, commit ALL changes with message: `feat: [Story ID] - [Story Title]`
9. Update the PRD to set `passes: true` for the completed story
10. Append your progress to `scripts/ralph/progress.txt`

## Project Architecture

```
src/bot/
├── main.py                 # Main orchestrator
├── config.py               # Settings (.env + YAML)
├── models/                 # Pydantic v2 data models
├── exchanges/              # Exchange adapters (ccxt)
├── data/                   # Data collection & storage
├── strategies/             # Trading strategies (plugin-based)
│   ├── technical/          # MA Crossover, RSI, MACD, Bollinger
│   ├── arbitrage/          # Cross-exchange arbitrage
│   ├── dca/                # Dollar Cost Averaging
│   └── ml/                 # ML price prediction
├── risk/                   # Risk management
├── execution/              # Order execution
├── backtest/               # Backtesting engine
├── monitoring/             # Logging, metrics, Telegram
└── dashboard/              # FastAPI web dashboard
```

## Tech Stack

- **Python 3.11+** with async/await throughout
- **ccxt** for exchange connectivity (Binance, Upbit)
- **ta** library for technical indicators (pure Python)
- **Pydantic v2** for data models and settings
- **SQLAlchemy + aiosqlite** for async data storage
- **structlog** for structured JSON logging
- **FastAPI** for dashboard
- **scikit-learn** for ML strategy
- **pytest + pytest-asyncio** for testing
- **ruff** for linting

## Critical Rules

### Security
- NEVER hardcode API keys or secrets — always load from environment/config
- Tests MUST mock all external API calls (ccxt, Telegram) — never call real APIs
- Never commit .env files

### Trading Safety
- Default mode is ALWAYS paper trading
- ALL trade signals MUST pass through RiskManager before execution
- Never execute real trades without explicit user configuration
- Implement stop-loss and position sizing on every strategy

### Code Quality
- All code must be async-first (use `async def`, `await`)
- All data models must use Pydantic v2
- Follow existing patterns in the codebase
- Write comprehensive unit tests for every module
- Use type hints everywhere

### Patterns
- Exchange adapters: ABC base class + Factory pattern
- Strategies: ABC base class + Registry pattern (decorator-based registration)
- Configuration: pydantic-settings with .env + optional YAML override
- Storage: Repository pattern with SQLAlchemy async sessions
- Error handling: Circuit breaker for external API calls

## Progress Report Format

APPEND to scripts/ralph/progress.txt (never replace, always append):
```
## [Date/Time] - [Story ID]
- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
  - Useful context
---
```

## Consolidate Patterns

If you discover a **reusable pattern**, add it to the `## Codebase Patterns` section at the TOP of scripts/ralph/progress.txt:

```
## Codebase Patterns
- Example: Always use async session context manager for DB operations
- Example: Register strategies with @strategy_registry.register decorator
```

## Update AGENTS.md

Before committing, check if edited directories have learnings worth preserving in AGENTS.md:
- API patterns or conventions specific to a module
- Gotchas or non-obvious requirements
- Dependencies between files
- Testing approaches

## Stop Condition

After completing a user story, check if ALL stories have `passes: true`.

If ALL stories are complete, reply with:
<promise>COMPLETE</promise>

If stories remain with `passes: false`, end your response normally.

## Important

- Work on ONE story per iteration
- Commit frequently
- Keep all tests passing
- Read the Codebase Patterns section in progress.txt before starting
- When installing new packages, update both pyproject.toml and requirements.txt
