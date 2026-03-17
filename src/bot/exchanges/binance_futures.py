"""Binance USDT-M Futures exchange adapter using ccxt."""

from datetime import datetime, timezone

import ccxt.async_support as ccxt
import structlog

from bot.exchanges.base import ExchangeAdapter
from bot.exchanges.factory import register_adapter
from bot.models import OHLCV, Order, OrderSide, OrderStatus, OrderType

logger = structlog.get_logger(__name__)


class BinanceFuturesAdapter(ExchangeAdapter):
    """Binance USDT-M Futures adapter via ccxt.

    Key differences from spot:
    - defaultType='future' for USDT-margined linear futures
    - Symbol format: BTC/USDT:USDT (ccxt unified)
    - Supports set_leverage() and set_margin_mode()
    - Supports short selling (side='sell' to open, side='buy' to close)
    - reduceOnly parameter for closing positions
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
                "sandbox": testnet,
                "enableRateLimit": True,
                "timeout": 30000,
                "options": {
                    "defaultType": "future",
                },
            }
        )
        self._leverage_set: dict[str, int] = {}
        self._margin_mode_set: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "binance_futures"

    @staticmethod
    def to_futures_symbol(symbol: str) -> str:
        """Convert spot symbol (BTC/USDT) to futures format (BTC/USDT:USDT)."""
        if ":USDT" in symbol:
            return symbol
        return f"{symbol}:USDT"

    @staticmethod
    def to_spot_symbol(symbol: str) -> str:
        """Convert futures symbol (BTC/USDT:USDT) to spot format (BTC/USDT)."""
        return symbol.replace(":USDT", "")

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol."""
        futures_sym = self.to_futures_symbol(symbol)
        try:
            await self._exchange.set_leverage(leverage, futures_sym)
            self._leverage_set[symbol] = leverage
            logger.info(
                "leverage_set",
                symbol=symbol,
                leverage=leverage,
            )
        except ccxt.ExchangeError as e:
            # Some exchanges return error if leverage is already set
            if "No need to change" in str(e):
                self._leverage_set[symbol] = leverage
            else:
                raise RuntimeError(f"Set leverage failed: {e}") from e

    async def set_margin_mode(self, symbol: str, mode: str = "isolated") -> None:
        """Set margin mode (isolated or cross) for a symbol."""
        futures_sym = self.to_futures_symbol(symbol)
        try:
            await self._exchange.set_margin_mode(mode, futures_sym)
            self._margin_mode_set[symbol] = mode
            logger.info(
                "margin_mode_set",
                symbol=symbol,
                mode=mode,
            )
        except ccxt.ExchangeError as e:
            if "No need to change" in str(e):
                self._margin_mode_set[symbol] = mode
            else:
                raise RuntimeError(f"Set margin mode failed: {e}") from e

    async def ensure_leverage_and_margin(
        self, symbol: str, leverage: int, margin_mode: str = "isolated"
    ) -> None:
        """Ensure leverage and margin mode are set for a symbol (idempotent)."""
        if symbol not in self._margin_mode_set:
            try:
                await self.set_margin_mode(symbol, margin_mode)
            except Exception as e:
                logger.warning(
                    "margin_mode_set_error",
                    symbol=symbol,
                    mode=margin_mode,
                    error=str(e),
                )
        if self._leverage_set.get(symbol) != leverage:
            try:
                await self.set_leverage(symbol, leverage)
            except Exception as e:
                logger.warning(
                    "leverage_set_error",
                    symbol=symbol,
                    leverage=leverage,
                    error=str(e),
                )

    async def get_ticker(self, symbol: str) -> dict:
        futures_sym = self.to_futures_symbol(symbol)
        try:
            ticker = await self._exchange.fetch_ticker(futures_sym)
            return {
                "bid": ticker.get("bid", 0.0),
                "ask": ticker.get("ask", 0.0),
                "last": ticker.get("last", 0.0),
                "volume": ticker.get("baseVolume", 0.0),
            }
        except ccxt.BadSymbol as e:
            raise ValueError(f"Invalid futures symbol: {symbol}") from e
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error: {e}") from e

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> list[OHLCV]:
        futures_sym = self.to_futures_symbol(symbol)
        try:
            candles = await self._exchange.fetch_ohlcv(
                futures_sym, timeframe, limit=limit
            )
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
            raise ValueError(f"Invalid futures symbol: {symbol}") from e
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error: {e}") from e
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
            raise ConnectionError(f"Network error: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error: {e}") from e

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
        reduce_only: bool = False,
    ) -> Order:
        """Create a futures order.

        For SHORT: side=SELL to open, side=BUY with reduce_only=True to close.
        For LONG: side=BUY to open, side=SELL with reduce_only=True to close.
        """
        futures_sym = self.to_futures_symbol(symbol)
        params = {}
        if reduce_only:
            params["reduceOnly"] = True
        try:
            result = await self._exchange.create_order(
                symbol=futures_sym,
                type=order_type.value.lower(),
                side=side.value.lower(),
                amount=quantity,
                price=price,
                params=params,
            )
            return self._parse_order(result, symbol)
        except ccxt.BadSymbol as e:
            raise ValueError(f"Invalid futures symbol: {symbol}") from e
        except ccxt.InsufficientFunds as e:
            raise ValueError(f"Insufficient futures margin: {e}") from e
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Futures order failed: {e}") from e

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        futures_sym = self.to_futures_symbol(symbol)
        try:
            await self._exchange.cancel_order(order_id, futures_sym)
            return True
        except ccxt.OrderNotFound:
            return False
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error: {e}") from e

    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        futures_sym = self.to_futures_symbol(symbol)
        try:
            result = await self._exchange.fetch_order(order_id, futures_sym)
            return self._parse_order(result, symbol)
        except ccxt.OrderNotFound as e:
            raise ValueError(f"Order not found: {order_id}") from e
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error: {e}") from e

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        futures_sym = self.to_futures_symbol(symbol)
        try:
            book = await self._exchange.fetch_order_book(futures_sym, limit)
            return {
                "bids": book.get("bids", []),
                "asks": book.get("asks", []),
            }
        except ccxt.BadSymbol as e:
            raise ValueError(f"Invalid futures symbol: {symbol}") from e
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error: {e}") from e

    async def fetch_futures_positions(self) -> list[dict]:
        """Fetch all open futures positions from exchange."""
        try:
            positions = await self._exchange.fetch_positions()
            return [
                {
                    "symbol": self.to_spot_symbol(p.get("symbol", "")),
                    "side": p.get("side", ""),
                    "contracts": float(p.get("contracts", 0)),
                    "notional": float(p.get("notional", 0)),
                    "unrealized_pnl": float(p.get("unrealizedPnl", 0)),
                    "entry_price": float(p.get("entryPrice", 0)),
                    "mark_price": float(p.get("markPrice", 0)),
                    "leverage": int(p.get("leverage", 1)),
                    "margin_mode": p.get("marginMode", ""),
                    "liquidation_price": float(p.get("liquidationPrice", 0) or 0),
                }
                for p in positions
                if float(p.get("contracts", 0)) > 0
            ]
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error: {e}") from e
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Fetch positions failed: {e}") from e

    async def fetch_funding_rate(self, symbol: str) -> dict:
        """Fetch current funding rate for a symbol."""
        futures_sym = self.to_futures_symbol(symbol)
        try:
            rate = await self._exchange.fetch_funding_rate(futures_sym)
            return {
                "symbol": symbol,
                "funding_rate": float(rate.get("fundingRate", 0)),
                "next_funding_time": rate.get("fundingDatetime", ""),
            }
        except Exception as e:
            logger.warning("funding_rate_error", symbol=symbol, error=str(e))
            return {"symbol": symbol, "funding_rate": 0.0, "next_funding_time": ""}

    async def close(self) -> None:
        await self._exchange.close()

    @staticmethod
    def _parse_order(data: dict, original_symbol: str = "") -> Order:
        """Convert ccxt futures order response to Order model."""
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

        order_type = (
            OrderType.LIMIT if data.get("type") == "limit" else OrderType.MARKET
        )
        price = float(data.get("price") or 0)

        filled_at = None
        if status == OrderStatus.FILLED and data.get("lastTradeTimestamp"):
            filled_at = datetime.fromtimestamp(
                data["lastTradeTimestamp"] / 1000, tz=timezone.utc
            )

        # Use original symbol (spot format) if provided
        symbol = original_symbol or data.get("symbol", "")

        return Order(
            id=str(data.get("id", "")),
            exchange="binance_futures",
            symbol=symbol,
            side=OrderSide.BUY if data.get("side") == "buy" else OrderSide.SELL,
            type=order_type,
            price=price,
            quantity=float(data.get("amount", 0)),
            status=status,
            filled_at=filled_at,
            filled_price=float(data["average"]) if data.get("average") else None,
            filled_quantity=float(data["filled"]) if data.get("filled") else None,
            fee=(
                float(data.get("fee", {}).get("cost", 0))
                if data.get("fee")
                else 0.0
            ),
        )


register_adapter("binance_futures", BinanceFuturesAdapter)
