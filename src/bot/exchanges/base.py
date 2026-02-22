"""Abstract exchange adapter interface."""

from abc import ABC, abstractmethod

from bot.models import OHLCV, Order, OrderSide, OrderType


class ExchangeAdapter(ABC):
    """Abstract base class for exchange adapters.

    All methods are async to support non-blocking I/O.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the exchange name."""

    @abstractmethod
    async def get_ticker(self, symbol: str) -> dict:
        """Get current ticker data for a symbol.

        Returns dict with at least: bid, ask, last, volume.
        """

    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> list[OHLCV]:
        """Fetch OHLCV candlestick data."""

    @abstractmethod
    async def get_balance(self) -> dict[str, float]:
        """Get account balances. Returns {currency: available_amount}."""

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
    ) -> Order:
        """Create a new order on the exchange."""

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an existing order. Returns True if cancelled successfully."""

    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        """Get the current status of an order."""

    @abstractmethod
    async def get_order_book(
        self, symbol: str, limit: int = 20
    ) -> dict:
        """Get order book for a symbol.

        Returns dict with 'bids' and 'asks' lists of [price, quantity] pairs.
        """

    async def close(self) -> None:
        """Clean up resources. Override if needed."""
