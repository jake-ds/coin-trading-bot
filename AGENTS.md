# Codebase Patterns & Agent Knowledge

This file is maintained by Ralph agents to share knowledge across iterations.

## Architecture

- **Async-first**: Use `async def` for all I/O-bound operations
- **Pydantic v2**: All data models inherit from `pydantic.BaseModel`, use `model_validate()` not `parse_obj()`
- **Exchange Adapter Pattern**: `ExchangeAdapter` ABC in `exchanges/base.py`, concrete adapters in separate files, `ExchangeFactory` for instantiation
- **Strategy Registry Pattern**: `BaseStrategy` ABC in `strategies/base.py`, use `@strategy_registry.register` decorator to register strategies
- **Repository Pattern**: `DataStore` in `data/store.py` handles all DB operations via SQLAlchemy async sessions

## Testing

- All tests in `tests/` directory mirroring `src/bot/` structure
- Use `pytest-asyncio` for async tests
- Mock all external calls (ccxt, Telegram API)
- Use in-memory SQLite for database tests

## Dependencies

- Exchange adapters depend on core models
- Strategies depend on models and ta library
- Risk manager depends on models and config
- Execution engine depends on exchange adapters, risk manager, and data store
- Orchestrator (main.py) depends on everything

## Conventions

- Import from `bot.models` for data types
- Import from `bot.config` for Settings
- Use `structlog.get_logger()` for logging
- Decimal precision: use Python float for calculations (not Decimal) for compatibility with ta and sklearn
