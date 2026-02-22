"""Base strategy interface and strategy registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from bot.models import OHLCV, TradingSignal

if TYPE_CHECKING:
    from bot.strategies.regime import MarketRegime


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""

    @property
    @abstractmethod
    def required_history_length(self) -> int:
        """Return the minimum number of candles required for analysis."""

    @abstractmethod
    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        """Analyze OHLCV data and generate a trading signal.

        Args:
            ohlcv_data: List of OHLCV candles (oldest first)
            **kwargs: Additional context (e.g., symbol)

        Returns:
            TradingSignal with the recommended action
        """

    def adapt_to_regime(self, regime: MarketRegime) -> None:
        """Adjust strategy parameters based on detected market regime.

        Override in subclasses to adapt behavior. Default is a no-op.

        Args:
            regime: The detected market regime.
        """


class StrategyRegistry:
    """Singleton registry for managing trading strategies."""

    _instance: "StrategyRegistry | None" = None
    _strategies: dict[str, BaseStrategy]
    _active: set[str]

    def __new__(cls) -> "StrategyRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._strategies = {}
            cls._instance._active = set()
        return cls._instance

    def register(self, strategy: BaseStrategy | type[BaseStrategy]) -> BaseStrategy:
        """Register a strategy instance or class.

        Can be used as a decorator for strategy classes:

            @strategy_registry.register
            class MyStrategy(BaseStrategy):
                ...

        Or called directly with an instance:

            strategy_registry.register(my_strategy)
        """
        if isinstance(strategy, type):
            # Used as class decorator - instantiate and register
            instance = strategy()
            self._strategies[instance.name] = instance
            self._active.add(instance.name)
            return instance
        else:
            # Register an existing instance
            self._strategies[strategy.name] = strategy
            self._active.add(strategy.name)
            return strategy

    def unregister(self, name: str) -> None:
        """Remove a strategy from the registry."""
        self._strategies.pop(name, None)
        self._active.discard(name)

    def get(self, name: str) -> BaseStrategy | None:
        """Get a strategy by name."""
        return self._strategies.get(name)

    def get_active(self) -> list[BaseStrategy]:
        """Get all active strategies."""
        return [s for name, s in self._strategies.items() if name in self._active]

    def get_all(self) -> list[BaseStrategy]:
        """Get all registered strategies."""
        return list(self._strategies.values())

    def enable(self, name: str) -> bool:
        """Enable a strategy. Returns True if the strategy exists."""
        if name in self._strategies:
            self._active.add(name)
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a strategy. Returns True if the strategy was active."""
        if name in self._active:
            self._active.discard(name)
            return True
        return False

    def is_active(self, name: str) -> bool:
        """Check if a strategy is active."""
        return name in self._active

    def clear(self) -> None:
        """Clear all registered strategies."""
        self._strategies.clear()
        self._active.clear()


# Global singleton
strategy_registry = StrategyRegistry()
