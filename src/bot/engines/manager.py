"""Engine lifecycle manager — starts, stops, and monitors all trading engines."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from bot.engines.tracker import EngineTracker, TradeRecord
from bot.engines.tuner import ParameterTuner
from bot.research.base import ResearchTask

if TYPE_CHECKING:
    from bot.config import Settings
    from bot.engines.base import BaseEngine, EngineCycleResult
    from bot.engines.portfolio_manager import PortfolioManager
    from bot.research.deployer import ResearchDeployer

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
        self._research_task: asyncio.Task | None = None
        self._rebalance_history: list[dict] = []
        self._research_experiments: list[ResearchTask] = []
        self._research_reports: list[dict] = []
        self._deployer: ResearchDeployer | None = None
        self._regression_task: asyncio.Task | None = None
        self._collector = None
        self._backfill_task: asyncio.Task | None = None
        self._correlation_controller = None

    def set_collector(self, collector) -> None:
        """Set the DataCollector for backfill background loop."""
        self._collector = collector

    def set_deployer(self, deployer: ResearchDeployer) -> None:
        """Set the ResearchDeployer for auto-deploying research findings."""
        self._deployer = deployer

    def set_correlation_controller(self, controller) -> None:
        """Set the CorrelationRiskController for cross-engine risk monitoring."""
        self._correlation_controller = controller

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

        # Update correlation controller with latest positions
        if self._correlation_controller is not None:
            self._sync_correlation_positions()

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

    def register_experiment(self, experiment: ResearchTask) -> None:
        """Register a research experiment for the research loop."""
        self._research_experiments.append(experiment)
        logger.info(
            "experiment_registered",
            name=experiment.__class__.__name__,
            target=experiment.target_engine,
        )

    async def start_background_loops(self) -> None:
        """Start tuner, rebalance, research, and backfill background loops."""
        s = self._settings
        if s and getattr(s, "tuner_enabled", False):
            self._tuner_task = asyncio.create_task(
                self._tuner_loop(), name="tuner-loop"
            )
        if s and getattr(s, "engine_rebalance_enabled", False):
            self._rebalance_task = asyncio.create_task(
                self._rebalance_loop(), name="rebalance-loop"
            )
        if s and getattr(s, "research_enabled", False):
            self._research_task = asyncio.create_task(
                self._research_loop(), name="research-loop"
            )
        if self._collector and getattr(s, "data_backfill_enabled", True):
            self._backfill_task = asyncio.create_task(
                self._collector._backfill_loop(
                    registry=self.opportunity_registry,
                    settings=s,
                ),
                name="backfill-loop",
            )
        if self._deployer and getattr(s, "research_auto_deploy", True):
            self._regression_task = asyncio.create_task(
                self._regression_check_loop(), name="regression-check-loop"
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

                # Factor in concentration risk when rebalancing
                if self._correlation_controller is not None:
                    self._sync_correlation_positions()
                    report = (
                        self._correlation_controller.get_concentration_report()
                    )
                    if report.get("alerts"):
                        logger.warning(
                            "rebalance_concentration_alerts",
                            alerts=report["alerts"],
                        )

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

    async def _research_loop(self) -> None:
        """Periodically run research experiments."""
        s = self._settings
        initial_delay = 10800  # 3 hours before first run
        interval = (
            getattr(s, "research_interval_hours", 24) * 3600
            if s
            else 86400
        )

        await asyncio.sleep(initial_delay)

        while True:
            try:
                if s and not getattr(s, "research_enabled", False):
                    await asyncio.sleep(interval)
                    continue

                for experiment in self._research_experiments:
                    try:
                        report = experiment.run_experiment()
                        self._research_reports.append(report.to_dict())
                        # Keep last 50 reports
                        self._research_reports = (
                            self._research_reports[-50:]
                        )

                        # Use deployer if available, otherwise fallback
                        if (
                            self._deployer
                            and s
                            and getattr(s, "research_auto_deploy", True)
                        ):
                            result = self._deployer.deploy(report)
                            if result.success:
                                logger.info(
                                    "research_deployed",
                                    experiment=report.experiment_name,
                                    snapshot_id=result.snapshot_id,
                                    changes=[
                                        c.to_dict()
                                        for c in result.deployed_changes
                                    ],
                                )
                        elif report.improvement_significant:
                            changes = experiment.apply_findings()
                            if changes and s:
                                self.tuner.apply_changes(changes, s)
                                logger.info(
                                    "research_applied",
                                    experiment=report.experiment_name,
                                    changes=[
                                        c.to_dict() for c in changes
                                    ],
                                )
                        logger.info(
                            "research_complete",
                            experiment=report.experiment_name,
                            significant=report.improvement_significant,
                        )
                    except Exception:
                        logger.exception(
                            "research_experiment_error",
                            experiment=experiment.__class__.__name__,
                        )
            except Exception:
                logger.exception("research_loop_error")

            await asyncio.sleep(interval)

    async def _regression_check_loop(self) -> None:
        """Periodically check for performance regression after deployments."""
        s = self._settings
        initial_delay = 21600  # 6 hours before first check
        interval = (
            getattr(s, "research_regression_check_hours", 6.0) * 3600
            if s
            else 21600
        )

        await asyncio.sleep(initial_delay)

        while True:
            try:
                if not self._deployer:
                    await asyncio.sleep(interval)
                    continue

                # Check regression for each engine that has a recent deploy
                history = self._deployer.get_deploy_history()
                for record in reversed(history):
                    if record.get("rolled_back"):
                        continue
                    # Check each engine that was affected
                    engines_checked: set[str] = set()
                    for change in record.get("changes", []):
                        engine_name = change.get("engine_name", "")
                        if engine_name and engine_name not in engines_checked:
                            engines_checked.add(engine_name)
                            if self._deployer.check_regression(engine_name):
                                snapshot_id = record.get("snapshot_id", "")
                                if snapshot_id:
                                    logger.warning(
                                        "auto_rollback_triggered",
                                        engine=engine_name,
                                        snapshot_id=snapshot_id,
                                    )
                                    self._deployer.rollback(snapshot_id)
                    # Only check the most recent non-rolled-back deployment
                    break

            except Exception:
                logger.exception("regression_check_loop_error")

            await asyncio.sleep(interval)

    def _sync_correlation_positions(self) -> None:
        """Gather positions from all engines and push to correlation controller."""
        if self._correlation_controller is None:
            return
        engine_positions: dict[str, list[dict]] = {}
        for name, engine in self._engines.items():
            positions = []
            for sym, pos in engine.positions.items():
                notional = pos.get("notional", 0)
                if notional == 0:
                    # Estimate notional from quantity * entry_price
                    qty = pos.get("quantity", 0)
                    price = pos.get("entry_price", 0)
                    notional = abs(qty * price)
                positions.append({
                    "symbol": sym,
                    "side": pos.get("side", "long"),
                    "notional": abs(notional),
                })
            engine_positions[name] = positions
        self._correlation_controller.update_positions(engine_positions)

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
    # Opportunity registry
    # ------------------------------------------------------------------

    @property
    def opportunity_registry(self):
        """Return the OpportunityRegistry from the scanner engine, if any."""
        scanner = self._engines.get("token_scanner")
        if scanner and hasattr(scanner, "registry"):
            return scanner.registry
        return None

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
