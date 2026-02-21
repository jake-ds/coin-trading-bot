"""Exchange factory for creating adapter instances."""

from typing import Any

from bot.exchanges.base import ExchangeAdapter

# Registry of exchange adapter classes
_adapter_registry: dict[str, type[ExchangeAdapter]] = {}


def register_adapter(name: str, adapter_class: type[ExchangeAdapter]) -> None:
    """Register an exchange adapter class."""
    _adapter_registry[name.lower()] = adapter_class


def get_registered_adapters() -> list[str]:
    """Return list of registered adapter names."""
    return list(_adapter_registry.keys())


class ExchangeFactory:
    """Factory for creating exchange adapter instances."""

    @staticmethod
    def create(name: str, **kwargs: Any) -> ExchangeAdapter:
        """Create an exchange adapter by name.

        Args:
            name: Exchange name (e.g., 'binance', 'upbit')
            **kwargs: Configuration passed to the adapter constructor

        Returns:
            ExchangeAdapter instance

        Raises:
            ValueError: If the exchange name is not registered
        """
        adapter_class = _adapter_registry.get(name.lower())
        if adapter_class is None:
            available = ", ".join(get_registered_adapters()) or "none"
            raise ValueError(
                f"Unknown exchange: '{name}'. Available: {available}"
            )
        return adapter_class(**kwargs)

    @staticmethod
    def available() -> list[str]:
        """Return list of available exchange adapters."""
        return get_registered_adapters()
