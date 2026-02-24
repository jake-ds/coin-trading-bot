"""Binance USDM Futures exchange adapter using ccxt."""

from datetime import datetime, timezone

import ccxt.async_support as ccxt

from bot.exchanges.base import ExchangeAdapter
from bot.exchanges.factory import register_adapter
from bot.models import OHLCV, Order, OrderSide, OrderStatus, OrderType


class BinanceFuturesAdapter(ExchangeAdapter):
    """Binance USDM perpetual futures adapter via ccxt.

    Supports leverage, funding rates, and futures-specific position management.
    """

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        testnet: bool = True,
        **kwargs,
    ):
        self._exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": secret_key,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
        if testnet:
            self._exchange.enable_demo_trading(True)

    @property
    def name(self) -> str:
        return "binance_futures"

    # ------------------------------------------------------------------
    # ExchangeAdapter ABC implementation
    # ------------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> dict:
        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            return {
                "bid": ticker.get("bid", 0.0),
                "ask": ticker.get("ask", 0.0),
                "last": ticker.get("last", 0.0),
                "volume": ticker.get("baseVolume", 0.0),
            }
        except ccxt.BadSymbol as e:
            raise ValueError(f"Invalid symbol: {symbol}") from e
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error fetching ticker: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error: {e}") from e

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> list[OHLCV]:
        try:
            candles = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return [
                OHLCV(
                    timestamp=datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc),
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=float(c[5]),
                    symbol=symbol,
                    timeframe=timeframe,
                )
                for c in candles
            ]
        except ccxt.BadSymbol as e:
            raise ValueError(f"Invalid symbol: {symbol}") from e
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error fetching OHLCV: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error: {e}") from e

    async def get_balance(self) -> dict[str, float]:
        try:
            balance = await self._exchange.fetch_balance()
            return {
                currency: float(amount)
                for currency, amount in balance.get("free", {}).items()
                if float(amount) > 0
            }
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error fetching balance: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error: {e}") from e

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
    ) -> Order:
        try:
            result = await self._exchange.create_order(
                symbol=symbol,
                type=order_type.value.lower(),
                side=side.value.lower(),
                amount=quantity,
                price=price,
            )
            return self._parse_order(result)
        except ccxt.BadSymbol as e:
            raise ValueError(f"Invalid symbol: {symbol}") from e
        except ccxt.InsufficientFunds as e:
            raise ValueError(f"Insufficient funds: {e}") from e
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error creating order: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error: {e}") from e

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except ccxt.OrderNotFound:
            return False
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error cancelling order: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error: {e}") from e

    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        try:
            result = await self._exchange.fetch_order(order_id, symbol)
            return self._parse_order(result)
        except ccxt.OrderNotFound as e:
            raise ValueError(f"Order not found: {order_id}") from e
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error fetching order: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error: {e}") from e

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        try:
            book = await self._exchange.fetch_order_book(symbol, limit)
            return {
                "bids": book.get("bids", []),
                "asks": book.get("asks", []),
            }
        except ccxt.BadSymbol as e:
            raise ValueError(f"Invalid symbol: {symbol}") from e
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error fetching order book: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error: {e}") from e

    async def close(self) -> None:
        await self._exchange.close()

    # ------------------------------------------------------------------
    # Futures-specific methods
    # ------------------------------------------------------------------

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol."""
        try:
            await self._exchange.set_leverage(leverage, symbol)
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Error setting leverage: {e}") from e

    async def get_funding_rate(self, symbol: str) -> dict:
        """Fetch current funding rate for a perpetual contract."""
        try:
            data = await self._exchange.fetch_funding_rate(symbol)
            return {
                "symbol": symbol,
                "funding_rate": data.get("fundingRate", 0.0),
                "mark_price": float(data.get("markPrice") or 0),
                "index_price": float(data.get("indexPrice") or 0),
                "next_funding_time": data.get("fundingTimestamp"),
            }
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Error fetching funding rate: {e}") from e

    async def get_positions(self) -> list[dict]:
        """Get all open futures positions."""
        try:
            positions = await self._exchange.fetch_positions()
            result = []
            for pos in positions:
                contracts = float(pos.get("contracts", 0))
                if contracts == 0:
                    continue
                result.append({
                    "symbol": pos.get("symbol", ""),
                    "side": pos.get("side", ""),
                    "contracts": contracts,
                    "entry_price": float(pos.get("entryPrice", 0)),
                    "mark_price": float(pos.get("markPrice", 0)),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    "leverage": int(pos.get("leverage", 1)),
                    "margin_mode": pos.get("marginMode", "cross"),
                })
            return result
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Error fetching positions: {e}") from e

    async def set_margin_mode(self, symbol: str, mode: str) -> None:
        """Set margin mode ('cross' or 'isolated') for a symbol."""
        try:
            await self._exchange.set_margin_mode(mode, symbol)
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Error setting margin mode: {e}") from e

    # ------------------------------------------------------------------
    # Order parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_order(data: dict) -> Order:
        """Convert ccxt order response to Order model."""
        status_map = {
            "open": OrderStatus.SUBMITTED,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.CANCELLED,
            "rejected": OrderStatus.FAILED,
        }
        raw_status = data.get("status", "open")
        status = status_map.get(raw_status, OrderStatus.PENDING)

        order_type = OrderType.LIMIT if data.get("type") == "limit" else OrderType.MARKET
        # Market orders must have price=0 per Order model validator;
        # ccxt may return the fill price in the 'price' field for market orders.
        price = float(data.get("price") or 0) if order_type == OrderType.LIMIT else 0

        filled_at = None
        if status == OrderStatus.FILLED and data.get("lastTradeTimestamp"):
            filled_at = datetime.fromtimestamp(
                data["lastTradeTimestamp"] / 1000, tz=timezone.utc
            )

        return Order(
            id=str(data.get("id", "")),
            exchange="binance_futures",
            symbol=data.get("symbol", ""),
            side=OrderSide.BUY if data.get("side") == "buy" else OrderSide.SELL,
            type=order_type,
            price=price,
            quantity=float(data.get("amount", 0)),
            status=status,
            filled_at=filled_at,
            filled_price=float(data["average"]) if data.get("average") else None,
            filled_quantity=float(data["filled"]) if data.get("filled") else None,
            fee=float(data.get("fee", {}).get("cost", 0)) if data.get("fee") else 0.0,
        )


register_adapter("binance_futures", BinanceFuturesAdapter)
