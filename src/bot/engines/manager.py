"""Engine lifecycle manager — starts, stops, and monitors all trading engines."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from bot.engines.tracker import EngineTracker, TradeRecord
from bot.engines.tuner import ParameterTuner

if TYPE_CHECKING:
    from bot.config import Settings
    from bot.engines.base import BaseEngine, EngineCycleResult
    from bot.engines.portfolio_manager import PortfolioManager

logger = structlog.get_logger(__name__)


class EngineManager:
    """Manages the lifecycle of multiple trading engines.

    Registers engines, starts/stops them as asyncio tasks, and provides
    aggregated status reporting.
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        settings: Settings | None = None,
    ):
        self._portfolio_manager = portfolio_manager
        self._settings = settings
        self._engines: dict[str, BaseEngine] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self.tracker = EngineTracker()
        self.tuner = ParameterTuner()
        self._tuner_task: asyncio.Task | None = None
        self._rebalance_task: asyncio.Task | None = None
        self._rebalance_history: list[dict] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, engine: BaseEngine) -> None:
        """Register an engine for management."""
        if engine.name in self._engines:
            raise ValueError(f"Engine '{engine.name}' is already registered")
        self._engines[engine.name] = engine
        # Wire up tracker to record every cycle
        existing_cb = engine._on_cycle_complete

        def _on_cycle(result: EngineCycleResult) -> None:
            self._record_cycle_to_tracker(engine.name, result)
            if existing_cb is not None:
                return existing_cb(result)

        engine.set_on_cycle_complete(_on_cycle)
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
    # Tracker integration
    # ------------------------------------------------------------------

    def _record_cycle_to_tracker(
        self, engine_name: str, result: EngineCycleResult
    ) -> None:
        """Record cycle and extract trade records from cycle result."""
        self.tracker.record_cycle(engine_name, result)

        # Extract trades from actions_taken (PnL-bearing actions)
        if result.pnl_update != 0 and result.actions_taken:
            for action in result.actions_taken:
                pnl = action.get("pnl", action.get("profit", 0))
                if pnl == 0:
                    continue
                cost = action.get("cost", 0)
                gross = action.get("gross_pnl", action.get("gross_profit", pnl + cost))
                symbol = action.get("symbol", action.get("pair", "unknown"))

                trade = TradeRecord(
                    engine_name=engine_name,
                    symbol=symbol,
                    side=action.get("side", action.get("action", "unknown")),
                    entry_price=action.get("entry_price", action.get("price_a", 0)),
                    exit_price=action.get("exit_price", action.get("exit_zscore", 0)),
                    quantity=action.get("quantity", action.get("qty", 0)),
                    pnl=gross,
                    cost=cost,
                    net_pnl=pnl,
                    entry_time=action.get("entry_time", result.timestamp),
                    exit_time=action.get("exit_time", datetime.now(timezone.utc).isoformat()),
                    hold_time_seconds=action.get("hold_time_seconds", 0),
                )
                self.tracker.record_trade(engine_name, trade)

    # ------------------------------------------------------------------
    # Tuner and rebalance loops
    # ------------------------------------------------------------------

    async def start_background_loops(self) -> None:
        """Start tuner and rebalance background loops."""
        s = self._settings
        if s and getattr(s, "tuner_enabled", False):
            self._tuner_task = asyncio.create_task(
                self._tuner_loop(), name="tuner-loop"
            )
        if s and getattr(s, "engine_rebalance_enabled", False):
            self._rebalance_task = asyncio.create_task(
                self._rebalance_loop(), name="rebalance-loop"
            )

    async def _tuner_loop(self) -> None:
        """Periodically evaluate and adjust engine parameters."""
        s = self._settings
        initial_delay = 3600  # 1 hour before first run
        interval = (
            getattr(s, "tuner_interval_hours", 24) * 3600 if s else 86400
        )

        await asyncio.sleep(initial_delay)

        while True:
            try:
                if s and not getattr(s, "tuner_enabled", False):
                    await asyncio.sleep(interval)
                    continue

                all_metrics = self.tracker.get_all_metrics(window_hours=24)
                for engine_name in self._engines:
                    metrics = all_metrics.get(engine_name)
                    if metrics is None or metrics.total_trades < 2:
                        continue

                    # Get current params from settings
                    current_params = self._get_engine_params(engine_name)
                    changes = self.tuner.evaluate_and_adjust(
                        engine_name, metrics, current_params
                    )
                    if changes and s:
                        self.tuner.apply_changes(changes, s)
                        logger.info(
                            "tuner_adjusted",
                            engine=engine_name,
                            changes=[c.to_dict() for c in changes],
                        )
            except Exception:
                logger.exception("tuner_loop_error")

            await asyncio.sleep(interval)

    async def _rebalance_loop(self) -> None:
        """Periodically rebalance capital allocation across engines."""
        s = self._settings
        initial_delay = 7200  # 2 hours before first run (offset from tuner)
        interval = (
            getattr(s, "engine_rebalance_interval_hours", 24) * 3600
            if s
            else 86400
        )

        await asyncio.sleep(initial_delay)

        while True:
            try:
                if s and not getattr(s, "engine_rebalance_enabled", False):
                    await asyncio.sleep(interval)
                    continue

                all_metrics = self.tracker.get_all_metrics(window_hours=24)
                new_allocs = self._portfolio_manager.rebalance_allocations(
                    all_metrics
                )
                if new_allocs:
                    self._rebalance_history.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "allocations": {
                            k: round(v, 4) for k, v in new_allocs.items()
                        },
                    })
                    # Keep last 30 entries
                    self._rebalance_history = self._rebalance_history[-30:]
                    logger.info("rebalance_complete", allocations=new_allocs)
            except Exception:
                logger.exception("rebalance_loop_error")

            await asyncio.sleep(interval)

    def _get_engine_params(self, engine_name: str) -> dict:
        """Read current engine-specific params from settings."""
        from bot.engines.tuner import TUNER_CONFIG

        config = TUNER_CONFIG.get(engine_name, {})
        params: dict = {}
        if self._settings:
            for param_name in config:
                if hasattr(self._settings, param_name):
                    params[param_name] = getattr(self._settings, param_name)
        return params

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
        return [r.to_dict() for r in all_results[-100:]]

    def get_engine_cycle_log(self, name: str) -> list[dict]:
        """Get cycle log for a specific engine."""
        engine = self._engines.get(name)
        if engine is None:
            return []
        return [r.to_dict() for r in engine.cycle_history]

    def get_engine_positions(self, name: str) -> list[dict]:
        """Get current positions for a specific engine."""
        engine = self._engines.get(name)
        if engine is None:
            return []
        return list(engine.positions.values())
