"""Base class for research experiments."""

from __future__ import annotations

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Coroutine, TypeVar

from bot.engines.tuner import ParamChange
from bot.research.report import ResearchReport

if TYPE_CHECKING:
    from bot.research.data_provider import HistoricalDataProvider

T = TypeVar("T")


class ResearchTask(ABC):
    """Abstract base for a financial engineering research experiment."""

    def __init__(self, data_provider: HistoricalDataProvider | None = None) -> None:
        self.data_provider = data_provider

    @property
    @abstractmethod
    def target_engine(self) -> str:
        """Engine name this experiment targets."""

    @abstractmethod
    def run_experiment(self, **kwargs: object) -> ResearchReport:
        """Run the experiment and return a report."""

    @abstractmethod
    def apply_findings(self) -> list[ParamChange]:
        """Return recommended parameter changes from the last experiment run."""

    def _run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run an async coroutine from sync context.

        Works whether called from inside or outside an existing event loop.
        """
        try:
            asyncio.get_running_loop()
            # Inside an async context — run in a separate thread with its own loop
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            # No running loop
            return asyncio.run(coro)
