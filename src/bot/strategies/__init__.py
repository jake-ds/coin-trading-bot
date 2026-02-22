"""Trading strategies package."""

from bot.strategies.base import BaseStrategy, StrategyRegistry, strategy_registry
from bot.strategies.ensemble import SignalEnsemble
from bot.strategies.indicators import calculate_atr, calculate_atr_series
from bot.strategies.regime import MarketRegime, MarketRegimeDetector
from bot.strategies.trend_filter import TrendDirection, TrendFilter

__all__ = [
    "BaseStrategy",
    "MarketRegime",
    "MarketRegimeDetector",
    "SignalEnsemble",
    "StrategyRegistry",
    "TrendDirection",
    "TrendFilter",
    "calculate_atr",
    "calculate_atr_series",
    "strategy_registry",
]
