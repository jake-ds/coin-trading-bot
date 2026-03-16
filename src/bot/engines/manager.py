"""Engine lifecycle manager — starts, stops, and monitors all trading engines."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from bot.engines.tracker import EngineTracker, TradeRecord

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
        self._collector = None
        self._metrics_persistence = None
        self._snapshot_task: asyncio.Task | None = None
        self._regime_detector = None
        self._regime_task: asyncio.Task | None = None
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_paused_engines: set[str] = set()
        self._circuit_breaker_task: asyncio.Task | None = None

    def set_collector(self, collector) -> None:
        """Set the DataCollector for backfill background loop."""
        self._collector = collector

    def set_metrics_persistence(self, persistence) -> None:
        """Set the MetricsPersistence for saving trades/metrics to DB."""
        self._metrics_persistence = persistence

    def set_regime_detector(self, detector) -> None:
        """Set the MarketRegimeDetector for real-time regime detection."""
        self._regime_detector = detector

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
                if not engine.supports_live and not engine._paper_mode:
                    logger.warning(
                        "engine_paper_only_forced",
                        engine=name,
                    )
                    engine._paper_mode = True
                self._tasks[name] = asyncio.create_task(
                    engine.run(), name=f"engine-{name}"
                )
                logger.info("engine_task_created", engine=name)

    async def stop_all(self) -> None:
        """Stop all running engines gracefully."""
        for name, engine in self._engines.items():
            engine._running = False
        tasks = [t for t in self._tasks.values() if not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("all_engines_stopped")

    # ------------------------------------------------------------------
    # Lifecycle — single engine
    # ------------------------------------------------------------------

    async def start_engine(self, name: str) -> bool:
        engine = self._engines.get(name)
        if engine is None:
            return False
        if name in self._tasks and not self._tasks[name].done():
            return False
        self._tasks[name] = asyncio.create_task(
            engine.run(), name=f"engine-{name}"
        )
        return True

    async def stop_engine(self, name: str) -> bool:
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

        if result.pnl_update != 0 and result.actions_taken:
            for action in result.actions_taken:
                pnl = action.get("pnl", action.get("profit", 0))
                if pnl == 0:
                    continue
                cost = action.get("cost", 0)
                gross = action.get("gross_pnl", action.get("gross_profit", pnl + cost))
                symbol = action.get("symbol", "unknown")

                engine_obj = self._engines.get(engine_name)
                trade_mode = "paper"
                if engine_obj and not engine_obj._paper_mode:
                    trade_mode = "live"

                trade = TradeRecord(
                    engine_name=engine_name,
                    symbol=symbol,
                    side=action.get("side", action.get("action", "unknown")),
                    entry_price=action.get("entry_price", action.get("price", 0)),
                    exit_price=action.get("exit_price", 0),
                    quantity=action.get("quantity", 0),
                    pnl=gross,
                    cost=cost,
                    net_pnl=pnl,
                    entry_time=action.get("entry_time", result.timestamp),
                    exit_time=action.get("exit_time", datetime.now(timezone.utc).isoformat()),
                    hold_time_seconds=action.get("hold_time_seconds", 0),
                    mode=trade_mode,
                )
                self.tracker.record_trade(engine_name, trade)
                if self._metrics_persistence is not None:
                    asyncio.ensure_future(
                        self._metrics_persistence.save_trade(
                            engine_name, trade,
                        )
                    )

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def start_background_loops(self) -> None:
        """Start metric snapshot and regime detection background loops."""
        s = self._settings

        if (
            self._metrics_persistence is not None
            and getattr(s, "metrics_persistence_enabled", True)
        ):
            interval = getattr(s, "metrics_snapshot_interval_minutes", 5.0)
            self._snapshot_task = asyncio.create_task(
                self._metrics_persistence._snapshot_loop(interval),
                name="metrics-snapshot-loop",
            )

        if (
            self._regime_detector is not None
            and getattr(s, "regime_detection_enabled", True)
        ):
            interval_s = getattr(s, "regime_detection_interval_seconds", 300.0)
            self._regime_task = asyncio.create_task(
                self._regime_detector._detection_loop(interval_s),
                name="regime-detection-loop",
            )

        if (
            self._regime_detector is not None
            and getattr(s, "regime_adaptation_enabled", True)
        ):
            self._circuit_breaker_task = asyncio.create_task(
                self._circuit_breaker_loop(),
                name="circuit-breaker-loop",
            )

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    async def _circuit_breaker_check(self) -> None:
        """Check if CRISIS regime warrants pausing all engines."""
        if self._regime_detector is None:
            return
        s = self._settings
        if s and not getattr(s, "regime_adaptation_enabled", True):
            return

        threshold_min = (
            getattr(s, "crisis_circuit_breaker_minutes", 30.0)
            if s else 30.0
        )

        if self._regime_detector.is_crisis():
            duration = self._regime_detector.get_regime_duration()
            if duration >= threshold_min and not self._circuit_breaker_active:
                self._circuit_breaker_active = True
                self._circuit_breaker_paused_engines.clear()
                for name, engine in self._engines.items():
                    if engine.status.value == "running":
                        await engine.pause()
                        self._circuit_breaker_paused_engines.add(name)
                logger.warning(
                    "circuit_breaker_triggered",
                    regime="CRISIS",
                    duration_minutes=round(duration, 1),
                    paused_engines=sorted(self._circuit_breaker_paused_engines),
                )
        else:
            if self._circuit_breaker_active:
                resumed = []
                for name in list(self._circuit_breaker_paused_engines):
                    engine = self._engines.get(name)
                    if engine and engine.status.value == "paused":
                        await engine.resume()
                        resumed.append(name)
                logger.info(
                    "circuit_breaker_released",
                    resumed_engines=sorted(resumed),
                )
                self._circuit_breaker_active = False
                self._circuit_breaker_paused_engines.clear()

    async def _circuit_breaker_loop(self) -> None:
        await asyncio.sleep(60)
        while True:
            try:
                await self._circuit_breaker_check()
            except Exception as e:
                logger.error("circuit_breaker_loop_error", error=str(e))
            await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, dict]:
        return {
            name: engine.get_status_dict()
            for name, engine in self._engines.items()
        }

    def get_all_cycle_logs(self) -> list[dict]:
        all_results: list[EngineCycleResult] = []
        for engine in self._engines.values():
            all_results.extend(engine.cycle_history)
        all_results.sort(key=lambda r: r.timestamp)
        return [r.to_dict() for r in all_results[-100:]]

    def get_engine_cycle_log(self, name: str) -> list[dict]:
        engine = self._engines.get(name)
        if engine is None:
            return []
        return [r.to_dict() for r in engine.cycle_history]

    def get_engine_positions(self, name: str) -> list[dict]:
        engine = self._engines.get(name)
        if engine is None:
            return []
        return list(engine.positions.values())

    async def get_live_positions(self, name: str) -> list[dict]:
        engine = self._engines.get(name)
        if engine is None:
            return []
        return list(engine.positions.values())
