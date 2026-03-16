# Ralph Agent Instructions - Crypto Trading Bot (V4 Live-Ready)

You are an autonomous coding agent building a production-ready live trading system with a modern React dashboard. The codebase has V2 (production hardening, 30 stories) and V3 (quant trading, 20 stories) completed with 1400+ passing tests.

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

The codebase has 1400+ passing tests. ALL must continue passing after every change.
When modifying existing code:
- Add new parameters with default values matching old behavior
- Never remove test coverage — only add to it
- New features are disabled by default (feature flags in config.py)
- API prefix change (/api/*) must keep backward compat in tests

## V4 Architecture

### Frontend (NEW)
```
frontend/                        # React SPA
├── src/
│   ├── api/
│   │   ├── client.ts            # axios instance with auth interceptor
│   │   └── types.ts             # TypeScript types matching Python models
│   ├── components/
│   │   ├── layout/              # Header, Sidebar, Footer
│   │   ├── common/              # StatusBadge, MetricCard, LoadingSkeleton
│   │   └── charts/              # EquityCurve, DrawdownChart, Heatmap
│   ├── hooks/
│   │   ├── useWebSocket.ts      # Real-time updates hook
│   │   └── useAuth.ts           # JWT auth hook
│   ├── pages/
│   │   ├── Dashboard.tsx        # Portfolio overview
│   │   ├── Trades.tsx           # Trade history + open positions
│   │   ├── Strategies.tsx       # Strategy management
│   │   ├── Analytics.tsx        # Performance charts
│   │   ├── Settings.tsx         # Configuration panel
│   │   ├── AuditLog.tsx         # Audit trail viewer
│   │   └── Login.tsx            # Authentication
│   ├── App.tsx                  # Router + layout
│   └── main.tsx                 # Entry point
├── package.json
├── vite.config.ts               # Proxy to FastAPI in dev
├── tsconfig.json
└── tailwind.config.js
```

### Backend Extensions
```
src/bot/
├── dashboard/
│   ├── app.py                   # FastAPI — /api/* prefix, serves React static
│   ├── auth.py                  # V4-007: JWT auth
│   └── ws.py                    # V4-003: WebSocket endpoint
├── exchanges/
│   └── rate_limiter.py          # V4-009: Token bucket rate limiter
├── execution/
│   ├── reconciler.py            # V4-010: Position reconciliation
│   └── preflight.py             # V4-011: Pre-flight safety checks
└── monitoring/
    └── audit.py                 # V4-013: Audit trail logger
```

## Frontend Development Guide

### Setup
```bash
cd frontend
npm install
npm run dev    # Dev server with proxy to FastAPI
npm run build  # Production build to src/bot/dashboard/static/
```

### Key Patterns
- Use TypeScript strictly (no `any` types)
- Tailwind CSS for styling, dark theme by default
- recharts for charts (or react-chartjs-2)
- axios for API calls with interceptor for JWT
- React Router v6 for routing
- Custom hooks for shared logic (useWebSocket, useAuth)

### Testing Frontend
- Frontend tests are NOT required (Ralph focuses on Python tests)
- But `npm run build` MUST succeed (TypeScript compilation check)

## API Prefix Migration

V4-001 moves all API endpoints under `/api/*`:
```
/status         → /api/status
/trades         → /api/trades
/metrics        → /api/metrics
/portfolio      → /api/portfolio
/strategies     → /api/strategies
/equity-curve   → /api/equity-curve
/open-positions → /api/open-positions
/regime         → /api/regime
/health         → /api/health (stays at /health too for Docker healthcheck)
```

Existing Python tests that call these endpoints need updating. Use a helper or constant for the prefix.

## Tech Stack

### Backend (existing)
- Python 3.10+, FastAPI, ccxt, Pydantic v2, pydantic-settings
- SQLAlchemy + aiosqlite, structlog
- scikit-learn, statsmodels, scipy, arch
- pytest + pytest-asyncio, ruff

### Frontend (new in V4)
- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS 3 (styling)
- recharts (charts)
- axios (HTTP client)
- react-router-dom v6 (routing)

### New Backend Dependencies
- python-jose[cryptography] (JWT)

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
- All new features disabled by default in config.py
- NEVER expose API keys in API responses
- JWT secret: auto-generate with secrets.token_hex(32) if not configured
