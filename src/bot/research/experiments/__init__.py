"""Research experiments for automated strategy improvement."""

from bot.research.experiments.cointegration import CointegrationExperiment
from bot.research.experiments.funding_prediction import FundingPredictionExperiment
from bot.research.experiments.optimal_grid import OptimalGridExperiment
from bot.research.experiments.volatility_regime import VolatilityRegimeExperiment

__all__ = [
    "CointegrationExperiment",
    "FundingPredictionExperiment",
    "OptimalGridExperiment",
    "VolatilityRegimeExperiment",
]
