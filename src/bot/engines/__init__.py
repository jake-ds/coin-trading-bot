"""Multi-engine autonomous trading system.

Provides independent trading engines that each run their own strategy loop,
manage their own positions, and receive capital allocation from a central
PortfolioManager.
"""

from bot.engines.base import BaseEngine, EngineCycleResult, EngineStatus
from bot.engines.cost_model import CostModel
from bot.engines.manager import EngineManager
from bot.engines.portfolio_manager import PortfolioManager
from bot.engines.tracker import EngineTracker

__all__ = [
    "BaseEngine",
    "CostModel",
    "EngineCycleResult",
    "EngineManager",
    "EngineStatus",
    "EngineTracker",
    "PortfolioManager",
]
