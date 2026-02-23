"""Research framework for automated financial engineering experiments."""

__all__ = [
    "HistoricalDataProvider",
    "ResearchDeployer",
    "ResearchReport",
    "ResearchTask",
]

# Lazy imports to avoid circular dependency:
# research.base → engines.tuner → engines.__init__ → engines.manager → research.base


def __getattr__(name: str):  # type: ignore[misc]
    if name == "ResearchTask":
        from bot.research.base import ResearchTask

        return ResearchTask
    if name == "ResearchReport":
        from bot.research.report import ResearchReport

        return ResearchReport
    if name == "HistoricalDataProvider":
        from bot.research.data_provider import HistoricalDataProvider

        return HistoricalDataProvider
    if name == "ResearchDeployer":
        from bot.research.deployer import ResearchDeployer

        return ResearchDeployer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
