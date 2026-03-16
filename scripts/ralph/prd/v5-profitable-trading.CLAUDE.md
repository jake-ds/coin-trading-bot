# Ralph Agent Instructions - Crypto Trading Bot (V5 Profitable Trading)

You are an autonomous coding agent upgrading a crypto trading bot to pursue realistic profitability. The codebase has V2 (production hardening), V3 (quant trading), and V4 (live-ready dashboard) completed with 1865+ passing tests and a React frontend.

## Your Task

1. Read the PRD at `scripts/ralph/prd.json`
2. Read the progress log at `scripts/ralph/progress.txt` (check Codebase Patterns section FIRST)
3. Check you're on the correct branch from PRD `branchName`. If not, create it from main.
4. Pick the **highest priority** user story where `passes: false`
5. Implement that single user story fully
6. Run quality checks:
   - `pytest tests/ -v` (ALL tests must pass — old + new)
   - `ruff check src/ tests/` (no lint errors)
   - If frontend files changed: `cd frontend && npm run build` (must succeed)
7. If checks pass, commit ALL changes with message: `feat: [Story ID] - [Story Title]`
8. Update the PRD to set `passes: true` for the completed story
9. Append your progress to `scripts/ralph/progress.txt`

## CRITICAL: DO NOT BREAK EXISTING TESTS

The codebase has 1865+ passing tests. ALL must continue passing after every change.
When modifying existing code:
- Add new parameters with default values matching old behavior
- Never remove test coverage — only add to it
- New features use defaults that preserve existing behavior
- If a test checks PnL values and you add fee deduction, update the expected value — don't remove the test

## V5 Architecture Additions

### New Modules
```
src/bot/engines/
├── cost_model.py          # V5-001: CostModel (fee/slippage calculator)
├── tracker.py             # V5-005: EngineTracker (performance metrics)
├── tuner.py               # V5-008: ParameterTuner (auto-adjustment)
├── (existing engines modified for cost-awareness)

src/bot/research/          # V5-010: Research framework (NEW directory)
├── __init__.py
├── base.py                # ResearchTask ABC
├── report.py              # ResearchReport dataclass
├── backtest_runner.py     # SimpleBacktestRunner
└── experiments/
    ├── __init__.py
    ├── volatility_regime.py
    ├── cointegration.py
    ├── optimal_grid.py
    └── funding_prediction.py
```

### Modified Files
```
src/bot/engines/funding_arb.py         # CostModel integration
src/bot/engines/grid_trading.py        # CostModel integration
src/bot/engines/cross_exchange_arb.py  # CostModel integration
src/bot/engines/stat_arb.py            # CostModel integration
src/bot/engines/manager.py             # Tracker, tuner, rebalance, research loops
src/bot/engines/portfolio_manager.py   # Sharpe-weighted rebalance
src/bot/config.py                      # Multi-pair symbols, tuner/research config
src/bot/dashboard/app.py               # Performance, research API endpoints
frontend/src/pages/Engines.tsx         # Role descriptions, symbols, params
frontend/src/pages/Performance.tsx     # NEW: Performance dashboard
frontend/src/pages/Research.tsx        # NEW: Research experiments dashboard
frontend/src/api/types.ts              # New TypeScript types
frontend/src/App.tsx                   # New routes
```

## Engine System Overview

The bot runs 4 independent engines managed by `EngineManager`:

| Engine | What It Does | Config Prefix |
|--------|-------------|---------------|
| `funding_rate_arb` | Delta-neutral funding rate arbitrage (spot + perp hedge) | `funding_arb_*` |
| `grid_trading` | Grid trading (buy/sell orders at fixed intervals) | `grid_*` |
| `cross_exchange_arb` | Cross-exchange spot price arbitrage | `cross_arb_*` |
| `stat_arb` | Statistical pairs arbitrage (z-score mean reversion) | `stat_arb_*` |

### Key Classes
- `BaseEngine` (ABC): Abstract engine with `_execute_cycle()`, lifecycle methods, DecisionStep tracking
- `DecisionStep`: label (Korean), observation, threshold, result, category (evaluate/decide/execute/skip)
- `EngineCycleResult`: Per-cycle results with decisions list, actions, PnL
- `EngineManager`: Registers and manages engine lifecycle as asyncio tasks
- `PortfolioManager`: Capital allocation (30%/25%/20%/15%), drawdown protection

### CostModel Integration Pattern (V5-001/002)

```python
# In engine __init__:
from bot.engines.cost_model import CostModel
self._cost_model = CostModel()  # Uses default Binance VIP0 fees

# In _execute_cycle:
cost = self._cost_model.round_trip_cost(notional, legs=4)
net = self._cost_model.net_profit(gross_pnl, notional, legs=4)
decisions.append(DecisionStep(
    label="비용 분석",
    observation=f"총비용=${cost:.2f}, 순수익=${net:.2f}",
    threshold="순수익 > 0",
    result="수익" if net > 0 else "손실",
    category="evaluate"
))
```

### EngineTracker Pattern (V5-005/006)

```python
# In EngineManager:
self._tracker = EngineTracker()

# After each engine cycle:
self._tracker.record_cycle(engine_name, cycle_result)
if cycle_result.pnl_update:
    trade = TradeRecord(...)
    self._tracker.record_trade(engine_name, trade)

# API:
metrics = self._tracker.get_metrics("funding_rate_arb", window_hours=24)
```

### ParameterTuner Pattern (V5-008/009)

```python
# Tuner modifies Settings, engines read on next cycle:
changes = tuner.evaluate_and_adjust("grid_trading", metrics, current_params)
tuner.apply_changes(changes, settings)  # calls settings.reload()
```

## Frontend Patterns

- React 18 + TypeScript + Tailwind CSS 3 + Vite
- Dark theme by default (bg-gray-900 page, bg-gray-800 cards)
- recharts for charts (Line, Area, Bar, Pie)
- Custom hooks in frontend/src/hooks/
- Components in frontend/src/components/common/
- Types in frontend/src/api/types.ts
- API calls via axios instance in frontend/src/api/client.ts
- Routes in frontend/src/App.tsx

## Config Pattern

All new config fields go in `Settings` class in `config.py`:
```python
# Field with default
tuner_enabled: bool = True

# Add to SETTINGS_METADATA
"tuner_enabled": {
    "section": "Engine Tuning",
    "description": "Enable automatic parameter tuning",
    "type": "bool",
    "requires_restart": False,
},
```

## Testing Pattern

```python
# tests/engines/test_cost_model.py
import pytest
from bot.engines.cost_model import CostModel

class TestCostModel:
    def test_round_trip_cost_default(self):
        cm = CostModel()
        # 2 legs, $10000 notional
        # cost = 10000 * (0.04 + 0.01) * 2 / 100 = $10.00
        assert cm.round_trip_cost(10000, legs=2) == pytest.approx(10.0)
```

## Progress Report Format

APPEND to scripts/ralph/progress.txt:
```
## [Date/Time] - [Story ID]
- What was implemented
- Files created/modified
- Tests added (count)
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
- ALWAYS commit after completing a story
- ALL Python tests must pass (old + new)
- `npm run build` must succeed if frontend files changed
- Read Codebase Patterns FIRST
- New packages → update pyproject.toml AND requirements.txt
- New files → add __init__.py imports where appropriate
- Default parameter values must maintain backward compatibility
- CostModel fees as percentages: 0.02 means 0.02%, NOT 0.0002
- DecisionStep labels in Korean for dashboard display
- NEVER expose API keys in API responses
