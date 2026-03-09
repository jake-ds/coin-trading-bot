# Ralph Agent Instructions - Crypto Trading Bot (V6 Advanced Trading)

You are an autonomous coding agent building production-grade advanced trading capabilities. The codebase has V1-V5 completed (foundation → production → quant → dashboard → profitable trading) plus a Token Scanner Engine, with 2202+ passing tests and a React frontend.

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

The codebase has 2202+ passing tests. ALL must continue passing after every change.
When modifying existing code:
- Add new parameters with default values matching old behavior
- Never remove test coverage — only add to it
- New features use defaults that preserve existing behavior
- If a test checks PnL values and you add fee deduction, update the expected value — don't remove the test

## V6 Architecture Overview

V6 adds 6 major capabilities in dependency order:

```
Theme 1: Real Data Research (V6-001 → V6-004)
├── HistoricalDataProvider ← DataStore (existing)
├── DataCollector batch backfill ← OpportunityRegistry (V5)
├── 4 experiments upgraded to real data
└── ResearchDeployer (auto-apply + rollback)

Theme 2: Advanced Risk Models (V6-005 → V6-008)
├── VolatilityService ← GARCHModel (existing quant/)
├── VaR/CVaR enforcement ← risk_metrics.py (existing quant/)
├── DynamicPositionSizer ← VolatilityService
└── CorrelationRiskController ← PortfolioRiskManager

Theme 3: Frontend Analytics (V6-009 → V6-011)
├── TradeExplorer page
├── Heatmaps page
└── RiskDashboard page

Theme 4: Metrics Persistence (V6-012 → V6-013)
├── MetricsPersistence layer ← DataStore + EngineTracker
└── Historical analysis API + frontend

Theme 5: Market Regime Detection (V6-014 → V6-015)
├── MarketRegimeDetector ← VolatilityService
└── Engine adaptive behavior + circuit breaker

Theme 6: Docker (V6-016 → V6-017)
├── Dockerfile + docker-compose
└── Graceful shutdown + final verification
```

## Existing Architecture (V5 + Scanner)

### Engine System
```
src/bot/engines/
├── base.py                # BaseEngine ABC, DecisionStep, EngineCycleResult
├── manager.py             # EngineManager (lifecycle, tracker, tuner, rebalance, research loops)
├── portfolio_manager.py   # PortfolioManager (capital allocation, drawdown, rebalance)
├── tracker.py             # EngineTracker (in-memory TradeRecord, EngineMetrics)
├── tuner.py               # ParameterTuner (auto-adjust by Sharpe, TUNER_CONFIG bounds)
├── cost_model.py          # CostModel (fee + slippage calculator)
├── opportunity_registry.py # OpportunityRegistry (shared scanner → engine data)
├── scanner.py             # TokenScannerEngine (batch API scan, public ccxt fallback)
├── funding_arb.py         # FundingRateArbEngine (delta-neutral)
├── grid_trading.py        # GridTradingEngine
├── cross_exchange_arb.py  # CrossExchangeArbEngine
└── stat_arb.py            # StatisticalArbEngine (pairs z-score)
```

### Data Layer
```
src/bot/data/
├── models.py    # SQLAlchemy: OHLCVRecord, TradeRecord, FundingRateRecord,
│                #             PortfolioSnapshot, AuditLogRecord
├── store.py     # DataStore (async CRUD: candles, trades, funding, portfolio, audit)
└── collector.py # DataCollector (multi-exchange OHLCV collection, validation, backfill)
```

### Quant Modules (already implemented, use these!)
```
src/bot/quant/
├── risk_metrics.py   # parametric_var(), historical_var(), cornish_fisher_var(),
│                     # cvar(), sortino_ratio(), calmar_ratio(), information_ratio()
├── volatility.py     # GARCHModel (fit, forecast, dynamic_stop_loss),
│                     # VolatilityRegime enum, classify_volatility_regime()
├── statistics.py     # Statistical analysis utilities
├── microstructure.py # Bid-ask, order book, liquidity metrics
└── portfolio.py      # Covariance, efficient frontier
```

### Risk Management
```
src/bot/risk/
├── manager.py         # RiskManager (daily loss, drawdown, position sizing, halt)
└── portfolio_risk.py  # PortfolioRiskManager (exposure, correlation, sector, heat, VaR)
```

### Research Framework
```
src/bot/research/
├── base.py               # ResearchTask ABC (target_engine, run_experiment, apply_findings)
├── report.py             # ResearchReport dataclass
├── backtest_runner.py    # SimpleBacktestRunner
└── experiments/
    ├── volatility_regime.py   # ATR-based grid spacing optimization
    ├── cointegration.py       # Engle-Granger pairs test
    ├── optimal_grid.py        # Grid parameter optimization
    └── funding_prediction.py  # Funding rate pattern analysis
```

## Key Patterns

### Engine Registry Integration (V5 pattern — follow this for V6)
```python
# In engine __init__:
self._registry: OpportunityRegistry | None = None
self._dynamic_sizer: DynamicPositionSizer | None = None    # V6 NEW
self._regime_detector: MarketRegimeDetector | None = None   # V6 NEW

def set_registry(self, registry): self._registry = registry
def set_sizer(self, sizer): self._dynamic_sizer = sizer     # V6 NEW
def set_regime_detector(self, det): self._regime_detector = det  # V6 NEW

# In _run_cycle() — always check for None:
if self._registry:
    discovered = self._registry.get_symbols(...)
if self._dynamic_sizer:
    size = self._dynamic_sizer.calculate_size(...)
if self._regime_detector:
    adjustments = self._get_regime_adjustments()
```

### DecisionStep (Korean labels for dashboard)
```python
decisions.append(DecisionStep(
    label="시장 레짐",                    # Korean label
    observation="현재: HIGH, BTC 변동성 2.1σ",
    threshold="HIGH: threshold×1.3, size×0.7",
    result="보수적 모드 적용",
    category="evaluate",  # evaluate|decide|execute|skip
))
```

### CostModel Integration
```python
from bot.engines.cost_model import CostModel
self._cost_model = CostModel()
cost = self._cost_model.round_trip_cost(notional, legs=4)
net = self._cost_model.net_profit(gross_pnl, notional, legs=4)
```

### EngineTracker Pattern
```python
# In EngineManager callback:
self._tracker.record_cycle(engine_name, cycle_result)
if cycle_result.pnl_update:
    trade = TradeRecord(...)
    self._tracker.record_trade(engine_name, trade)
metrics = self._tracker.get_metrics("funding_rate_arb", window_hours=24)
```

### Background Loop Pattern (EngineManager)
```python
async def _my_loop(self) -> None:
    """Background task pattern used by tuner, rebalance, research."""
    await asyncio.sleep(3600)  # Initial delay (vary by loop)
    while True:
        try:
            # ... do work ...
            logger.info("my_loop_completed", ...)
        except Exception as e:
            logger.error("my_loop_error", error=str(e))
        await asyncio.sleep(self._settings.my_interval_hours * 3600)
```

### DataStore Pattern
```python
# All DB operations are async:
async with self._session_factory() as session:
    stmt = select(Model).where(Model.field == value)
    result = await session.execute(stmt)
    records = result.scalars().all()
```

### Config Pattern
```python
# In config.py Settings class:
my_new_field: bool = True  # Default preserves existing behavior

# In SETTINGS_METADATA:
"my_new_field": {
    "section": "Section Name",
    "description": "What this does",
    "type": "bool",
    "requires_restart": False,
},
```

### Dashboard API Pattern
```python
# In dashboard/app.py:
@router.get("/api/my/endpoint")
async def my_endpoint(
    token: str = Depends(validate_access_token),  # Auth required
):
    manager = _get_engine_manager()
    if not manager:
        return {"error": "not_available"}
    return {"data": manager.some_method()}

# For health endpoint: NO auth dependency
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", ...}
```

### Frontend Patterns
- React 18 + TypeScript + Tailwind CSS 3 + Vite
- Dark theme: `bg-gray-900` page, `bg-gray-800` cards, `text-white`
- Charts: recharts (Line, Area, Bar, Pie, RadialBar)
- Custom SVG/div grids for heatmaps (recharts lacks native heatmap)
- Types in `frontend/src/api/types.ts`
- API via axios: `frontend/src/api/client.ts`
- Routes in `frontend/src/App.tsx`
- CSV download: `new Blob([csv], {type: 'text/csv'})` + `URL.createObjectURL()`

### Testing Pattern
```python
import pytest
from unittest.mock import AsyncMock, MagicMock

class TestMyModule:
    def test_basic(self):
        # Sync test
        result = MyClass().method()
        assert result == expected

    @pytest.mark.asyncio
    async def test_async(self):
        # Async test
        mock_store = MagicMock()
        mock_store.get_candles = AsyncMock(return_value=[...])
        provider = HistoricalDataProvider(data_store=mock_store)
        prices = await provider.get_prices("BTC/USDT")
        assert len(prices) > 0
```

## Dependency Map (Build Order)

```
V6-001 HistoricalDataProvider     (independent — extends DataStore)
V6-002 DataCollector backfill     (depends on V6-001 for DataProvider concept)
V6-003 Experiments real data      (depends on V6-001)
V6-004 Research deployer          (depends on V6-003)
V6-005 VolatilityService          (depends on V6-001 for data_provider)
V6-006 VaR/CVaR enforcement       (depends on V6-005 optionally)
V6-007 Dynamic position sizing    (depends on V6-005)
V6-008 Correlation controller     (independent of V6-005/006/007)
V6-009 Trade Explorer page        (independent — uses existing tracker data)
V6-010 Heatmaps page              (independent — uses existing tracker data)
V6-011 Risk Dashboard page        (depends on V6-006, V6-008 for API data)
V6-012 Metrics persistence        (independent — extends DataStore + Tracker)
V6-013 Historical analysis        (depends on V6-012)
V6-014 Regime detector            (depends on V6-005)
V6-015 Engine adaptation          (depends on V6-014)
V6-016 Docker                     (independent — infra)
V6-017 Final verification         (depends on ALL above)
```

## Important Rules

- Work on ONE story per iteration
- ALWAYS commit after completing a story
- ALL Python tests must pass (old + new) — baseline: 2202+
- `npm run build` must succeed if frontend files changed
- Read Codebase Patterns FIRST from progress.txt
- New parameters MUST have defaults that preserve existing behavior
- New packages → update pyproject.toml AND requirements.txt
- New files → add `__init__.py` imports where appropriate
- DecisionStep labels in Korean for dashboard display
- NEVER expose API keys in API responses
- async methods in DataStore/DataProvider — use AsyncMock in tests
- GARCHModel.fit() can fail — ALWAYS wrap in try/except
- VolatilityService/RegimeDetector/DynamicSizer = None means no-op (backward compat)
- Dashboard endpoints need `Depends(validate_access_token)` except /api/health
- Frontend: dark theme only (bg-gray-900), responsive, use Tailwind

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
