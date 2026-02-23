"""Multi-engine autonomous trading system.

Provides independent trading engines that each run their own strategy loop,
manage their own positions, and receive capital allocation from a central
PortfolioManager.
"""

from bot.engines.base import BaseEngine, EngineCycleResult, EngineStatus
from bot.engines.cost_model import CostModel
from bot.engines.manager import EngineManager
from bot.engines.opportunity_registry import (
    Opportunity,
    OpportunityRegistry,
    OpportunityType,
)
from bot.engines.portfolio_manager import PortfolioManager
from bot.engines.scanner import TokenScannerEngine
from bot.engines.tracker import EngineTracker
from bot.engines.tuner import ParameterTuner

__all__ = [
    "BaseEngine",
    "CostModel",
    "EngineCycleResult",
    "EngineManager",
    "EngineStatus",
    "EngineTracker",
    "Opportunity",
    "OpportunityRegistry",
    "OpportunityType",
    "ParameterTuner",
    "PortfolioManager",
    "TokenScannerEngine",
]
