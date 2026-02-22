"""Engine lifecycle manager — starts, stops, and monitors all trading engines."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from bot.engines.base import BaseEngine, EngineCycleResult
    from bot.engines.portfolio_manager import PortfolioManager

logger = structlog.get_logger(__name__)


class EngineManager:
    """Manages the lifecycle of multiple trading engines.

    Registers engines, starts/stops them as asyncio tasks, and provides
    aggregated status reporting.
    """

    def __init__(self, portfolio_manager: PortfolioManager):
        self._portfolio_manager = portfolio_manager
        self._engines: dict[str, BaseEngine] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, engine: BaseEngine) -> None:
        """Register an engine for management."""
        if engine.name in self._engines:
            raise ValueError(f"Engine '{engine.name}' is already registered")
        self._engines[engine.name] = engine
        logger.info("engine_registered", engine=engine.name)

    def get_engine(self, name: str) -> BaseEngine | None:
        return self._engines.get(name)

    @property
    def engines(self) -> dict[str, BaseEngine]:
        return dict(self._engines)

    # ------------------------------------------------------------------
    # Lifecycle — all engines
    # ------------------------------------------------------------------

    async def start_all(self) -> None:
        """Start all registered engines as background asyncio tasks."""
        for name, engine in self._engines.items():
            if name not in self._tasks or self._tasks[name].done():
                self._tasks[name] = asyncio.create_task(
                    engine.run(), name=f"engine-{name}"
                )
                logger.info("engine_task_created", engine=name)

    async def stop_all(self) -> None:
        """Stop all running engines gracefully."""
        for name, engine in self._engines.items():
            engine._running = False
        # Wait for all tasks to finish
        tasks = [t for t in self._tasks.values() if not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("all_engines_stopped")

    # ------------------------------------------------------------------
    # Lifecycle — single engine
    # ------------------------------------------------------------------

    async def start_engine(self, name: str) -> bool:
        """Start a single engine by name. Returns True if started."""
        engine = self._engines.get(name)
        if engine is None:
            return False
        if name in self._tasks and not self._tasks[name].done():
            return False  # already running
        self._tasks[name] = asyncio.create_task(
            engine.run(), name=f"engine-{name}"
        )
        return True

    async def stop_engine(self, name: str) -> bool:
        """Stop a single engine by name."""
        engine = self._engines.get(name)
        if engine is None:
            return False
        engine._running = False
        task = self._tasks.get(name)
        if task and not task.done():
            await asyncio.wait_for(task, timeout=10.0)
        self._tasks.pop(name, None)
        return True

    async def pause_engine(self, name: str) -> bool:
        engine = self._engines.get(name)
        if engine is None:
            return False
        await engine.pause()
        return True

    async def resume_engine(self, name: str) -> bool:
        engine = self._engines.get(name)
        if engine is None:
            return False
        await engine.resume()
        return True

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, dict]:
        """Return status dict for all engines."""
        return {
            name: engine.get_status_dict()
            for name, engine in self._engines.items()
        }

    def get_all_cycle_logs(self) -> list[dict]:
        """Aggregate recent cycle results across all engines."""
        all_results: list[EngineCycleResult] = []
        for engine in self._engines.values():
            all_results.extend(engine.cycle_history)
        # Sort by timestamp and return as dicts
        all_results.sort(key=lambda r: r.timestamp)
        return [
            {
                "engine_name": r.engine_name,
                "cycle_num": r.cycle_num,
                "timestamp": r.timestamp,
                "duration_ms": r.duration_ms,
                "actions_taken": r.actions_taken,
                "positions": r.positions,
                "signals": r.signals,
                "pnl_update": r.pnl_update,
                "metadata": r.metadata,
            }
            for r in all_results[-100:]  # last 100 across all engines
        ]

    def get_engine_cycle_log(self, name: str) -> list[dict]:
        """Get cycle log for a specific engine."""
        engine = self._engines.get(name)
        if engine is None:
            return []
        return [
            {
                "engine_name": r.engine_name,
                "cycle_num": r.cycle_num,
                "timestamp": r.timestamp,
                "duration_ms": r.duration_ms,
                "actions_taken": r.actions_taken,
                "positions": r.positions,
                "signals": r.signals,
                "pnl_update": r.pnl_update,
                "metadata": r.metadata,
            }
            for r in engine.cycle_history
        ]

    def get_engine_positions(self, name: str) -> list[dict]:
        """Get current positions for a specific engine."""
        engine = self._engines.get(name)
        if engine is None:
            return []
        return list(engine.positions.values())
