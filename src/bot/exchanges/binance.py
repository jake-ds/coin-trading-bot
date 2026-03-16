"""Binance exchange adapter using ccxt."""

from datetime import datetime, timezone

import ccxt.async_support as ccxt

from bot.exchanges.base import ExchangeAdapter
from bot.exchanges.factory import register_adapter
from bot.models import OHLCV, Order, OrderSide, OrderStatus, OrderType


class BinanceAdapter(ExchangeAdapter):
    """Binance exchange adapter implemented via ccxt."""

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        testnet: bool = True,
        **kwargs,
    ):
        options = {}
        if testnet:
            options["defaultType"] = "spot"
        self._exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": secret_key,
                "sandbox": testnet,
                "enableRateLimit": True,
                "timeout": 30000,  # 30s timeout (P2-11)
                "options": options,
            }
        )

    @property
    def name(self) -> str:
        return "binance"

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

    # ------------------------------------------------------------------
    # Margin trading methods
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_amount(amount: float) -> str:
        """Format amount as a fixed-decimal string (no scientific notation)."""
        return f"{amount:.8f}".rstrip("0").rstrip(".")

    async def margin_borrow(self, coin: str, amount: float) -> str:
        """Borrow an asset from the cross-margin account.

        Returns the transaction ID on success.
        """
        try:
            result = await self._exchange.sapi_post_margin_loan(
                {"asset": coin, "amount": self._fmt_amount(amount)}
            )
            tran_id = str(result.get("tranId", ""))
            return tran_id
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error borrowing {coin}: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Margin borrow failed for {coin}: {e}") from e

    async def margin_repay(self, coin: str, amount: float) -> None:
        """Repay a borrowed asset to the cross-margin account."""
        try:
            await self._exchange.sapi_post_margin_repay(
                {"asset": coin, "amount": self._fmt_amount(amount)}
            )
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error repaying {coin}: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Margin repay failed for {coin}: {e}") from e

    async def create_margin_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
    ) -> Order:
        """Place an order on the cross-margin account."""
        try:
            result = await self._exchange.create_order(
                symbol=symbol,
                type=order_type.value.lower(),
                side=side.value.lower(),
                amount=quantity,
                price=price,
                params={"type": "margin"},
            )
            return self._parse_order(result)
        except ccxt.InsufficientFunds as e:
            raise ValueError(f"Insufficient margin funds: {e}") from e
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error placing margin order: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Margin order failed: {e}") from e

    async def get_margin_balance(self) -> dict[str, float]:
        """Fetch free balances from the cross-margin account."""
        try:
            balance = await self._exchange.fetch_balance({"type": "margin"})
            return {
                currency: float(amount)
                for currency, amount in balance.get("free", {}).items()
                if float(amount) > 0
            }
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error fetching margin balance: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Margin balance fetch failed: {e}") from e

    async def transfer_to_margin(self, coin: str, amount: float) -> str:
        """Transfer asset from spot wallet to cross-margin account.

        type=1 means SPOT → CROSS_MARGIN in Binance SAPI.
        Returns the transaction ID.
        """
        try:
            result = await self._exchange.sapi_post_margin_transfer(
                {"asset": coin, "amount": self._fmt_amount(amount), "type": "1"}
            )
            return str(result.get("tranId", ""))
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error transferring {coin}: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Transfer to margin failed for {coin}: {e}") from e

    async def close(self) -> None:
        await self._exchange.close()

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
        price = float(data.get("price") or 0)

        filled_at = None
        if status == OrderStatus.FILLED and data.get("lastTradeTimestamp"):
            filled_at = datetime.fromtimestamp(
                data["lastTradeTimestamp"] / 1000, tz=timezone.utc
            )

        return Order(
            id=str(data.get("id", "")),
            exchange="binance",
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


register_adapter("binance", BinanceAdapter)
