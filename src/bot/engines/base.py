"""Base engine abstract class for the multi-engine trading system."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from bot.engines.portfolio_manager import PortfolioManager
    from bot.exchanges.base import ExchangeAdapter

logger = structlog.get_logger(__name__)


class EngineStatus(str, Enum):
    """Lifecycle status of a trading engine."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class EngineCycleResult:
    """Result of a single engine cycle."""

    engine_name: str
    cycle_num: int
    timestamp: str
    duration_ms: float
    actions_taken: list[dict] = field(default_factory=list)
    positions: list[dict] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)
    pnl_update: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseEngine(ABC):
    """Abstract base class for an independently-operating trading engine.

    Each engine has its own loop interval, position tracking, and capital
    allocation from the PortfolioManager.
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        exchanges: list[ExchangeAdapter] | None = None,
        loop_interval: float = 60.0,
        max_positions: int = 5,
        paper_mode: bool = True,
    ):
        self._portfolio_manager = portfolio_manager
        self._exchanges = exchanges or []
        self._loop_interval = loop_interval
        self._max_positions = max_positions
        self._paper_mode = paper_mode

        self._status = EngineStatus.STOPPED
        self._running = False
        self._paused = False
        self._cycle_count = 0
        self._total_pnl = 0.0
        self._allocated_capital = 0.0
        self._positions: dict[str, dict[str, Any]] = {}
        self._cycle_history: list[EngineCycleResult] = []
        self._error_message: str | None = None

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique engine identifier."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the engine's strategy."""

    @abstractmethod
    async def _run_cycle(self) -> EngineCycleResult:
        """Execute one trading cycle. Must return an EngineCycleResult."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def status(self) -> EngineStatus:
        return self._status

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def total_pnl(self) -> float:
        return self._total_pnl

    @property
    def allocated_capital(self) -> float:
        return self._allocated_capital

    @property
    def positions(self) -> dict[str, dict[str, Any]]:
        return dict(self._positions)

    @property
    def position_count(self) -> int:
        return len(self._positions)

    @property
    def loop_interval(self) -> float:
        return self._loop_interval

    @property
    def max_positions(self) -> int:
        return self._max_positions

    @property
    def cycle_history(self) -> list[EngineCycleResult]:
        return list(self._cycle_history)

    @property
    def error_message(self) -> str | None:
        return self._error_message

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Request capital and mark as running."""
        if self._status == EngineStatus.RUNNING:
            return
        self._running = True
        self._paused = False
        self._status = EngineStatus.RUNNING
        self._error_message = None
        # Request initial capital allocation
        self._allocated_capital = self._portfolio_manager.request_capital(
            self.name, self._portfolio_manager.get_max_allocation(self.name)
        )
        logger.info(
            "engine_started",
            engine=self.name,
            allocated_capital=self._allocated_capital,
        )

    async def stop(self) -> None:
        """Stop the engine and release capital."""
        self._running = False
        self._paused = False
        self._status = EngineStatus.STOPPED
        if self._allocated_capital > 0:
            self._portfolio_manager.release_capital(
                self.name, self._allocated_capital
            )
            self._allocated_capital = 0.0
        logger.info("engine_stopped", engine=self.name)

    async def pause(self) -> None:
        """Pause the engine — keeps capital allocated but skips cycles."""
        if self._status != EngineStatus.RUNNING:
            return
        self._paused = True
        self._status = EngineStatus.PAUSED
        logger.info("engine_paused", engine=self.name)

    async def resume(self) -> None:
        """Resume a paused engine."""
        if self._status != EngineStatus.PAUSED:
            return
        self._paused = False
        self._status = EngineStatus.RUNNING
        logger.info("engine_resumed", engine=self.name)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main engine loop — called by EngineManager as an asyncio task."""
        await self.start()
        try:
            while self._running:
                if self._paused:
                    await asyncio.sleep(1.0)
                    continue

                try:
                    start_time = time.monotonic()
                    result = await self._run_cycle()
                    duration_ms = (time.monotonic() - start_time) * 1000
                    result.duration_ms = duration_ms
                    self._cycle_count += 1
                    self._total_pnl += result.pnl_update

                    # Report PnL to portfolio manager
                    if result.pnl_update != 0:
                        self._portfolio_manager.report_pnl(
                            self.name, result.pnl_update
                        )

                    # Keep last 50 cycle results
                    self._cycle_history.append(result)
                    if len(self._cycle_history) > 50:
                        self._cycle_history = self._cycle_history[-50:]

                    logger.debug(
                        "engine_cycle_complete",
                        engine=self.name,
                        cycle=self._cycle_count,
                        duration_ms=round(duration_ms, 1),
                        actions=len(result.actions_taken),
                        pnl_update=result.pnl_update,
                    )
                except Exception as e:
                    self._status = EngineStatus.ERROR
                    self._error_message = str(e)
                    logger.error(
                        "engine_cycle_error",
                        engine=self.name,
                        error=str(e),
                        exc_info=True,
                    )
                    # Wait before retrying on error
                    await asyncio.sleep(self._loop_interval)
                    # Reset to running to allow retry
                    if self._running:
                        self._status = EngineStatus.RUNNING

                await asyncio.sleep(self._loop_interval)
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def _add_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        **extra: Any,
    ) -> None:
        """Track a new position internally."""
        self._positions[symbol] = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "entry_price": entry_price,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        }

    def _remove_position(self, symbol: str) -> dict | None:
        """Remove and return a tracked position."""
        return self._positions.pop(symbol, None)

    def _has_capacity(self) -> bool:
        """Check if the engine can open more positions."""
        return len(self._positions) < self._max_positions

    def get_status_dict(self) -> dict[str, Any]:
        """Build a status summary for the dashboard."""
        return {
            "name": self.name,
            "description": self.description,
            "status": self._status.value,
            "cycle_count": self._cycle_count,
            "total_pnl": round(self._total_pnl, 2),
            "allocated_capital": round(self._allocated_capital, 2),
            "position_count": len(self._positions),
            "max_positions": self._max_positions,
            "loop_interval": self._loop_interval,
            "error": self._error_message,
        }
