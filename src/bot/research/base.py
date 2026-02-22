"""Base class for research experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod

from bot.engines.tuner import ParamChange
from bot.research.report import ResearchReport


class ResearchTask(ABC):
    """Abstract base for a financial engineering research experiment."""

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
