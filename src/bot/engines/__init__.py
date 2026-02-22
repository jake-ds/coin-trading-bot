"""Multi-engine autonomous trading system.

Provides independent trading engines that each run their own strategy loop,
manage their own positions, and receive capital allocation from a central
PortfolioManager.
"""

from bot.engines.base import BaseEngine, EngineCycleResult, EngineStatus
from bot.engines.manager import EngineManager
from bot.engines.portfolio_manager import PortfolioManager

__all__ = [
    "BaseEngine",
    "EngineCycleResult",
    "EngineManager",
    "EngineStatus",
    "PortfolioManager",
]
