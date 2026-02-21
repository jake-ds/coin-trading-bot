"""Order execution engine with paper trading and retry logic."""

import asyncio
import uuid
from datetime import datetime, timezone

import structlog

from bot.data.store import DataStore
from bot.exchanges.base import ExchangeAdapter
from bot.models import Order, OrderSide, OrderStatus, OrderType, TradingSignal

logger = structlog.get_logger()


class ExecutionEngine:
    """Executes orders via exchange adapters with retry logic and paper trading."""

    def __init__(
        self,
        exchange: ExchangeAdapter,
        store: DataStore,
        paper_trading: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self._exchange = exchange
        self._store = store
        self._paper_trading = paper_trading
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._pending_orders: dict[str, Order] = {}

    async def execute_signal(
        self,
        signal: TradingSignal,
        quantity: float,
        price: float | None = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> Order | None:
        """Execute a trading signal by creating an order.

        Returns the resulting Order or None if skipped.
        """
        from bot.models import SignalAction

        if signal.action == SignalAction.HOLD:
            return None

        side = OrderSide.BUY if signal.action == SignalAction.BUY else OrderSide.SELL

        if self._paper_trading:
            order = await self._paper_execute(signal.symbol, side, order_type, quantity, price)
        else:
            order = await self._live_execute(signal.symbol, side, order_type, quantity, price)

        if order:
            await self._store.save_trade(order)
            logger.info(
                "order_executed",
                order_id=order.id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                status=order.status.value,
                paper=self._paper_trading,
            )

        return order

    async def _paper_execute(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None,
    ) -> Order:
        """Simulate order execution in paper trading mode."""
        if price is None:
            try:
                ticker = await self._exchange.get_ticker(symbol)
                price = ticker.get("last", 0)
            except (ValueError, ConnectionError, RuntimeError):
                price = 0

        order_id = f"paper-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        return Order(
            id=order_id,
            exchange=self._exchange.name,
            symbol=symbol,
            side=side,
            type=order_type,
            price=0 if order_type == OrderType.MARKET else (price or 0),
            quantity=quantity,
            status=OrderStatus.FILLED,
            created_at=now,
            filled_at=now,
            filled_price=price,
            filled_quantity=quantity,
            fee=0.0,
        )

    async def _live_execute(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None,
    ) -> Order | None:
        """Execute order on real exchange with retry logic."""
        for attempt in range(self._max_retries):
            try:
                order = await self._exchange.create_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                )
                self._pending_orders[order.id] = order
                return order
            except ConnectionError as e:
                logger.warning(
                    "order_retry",
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                    error=str(e),
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
            except (ValueError, RuntimeError) as e:
                logger.error("order_failed", error=str(e))
                return None

        logger.error("order_failed_max_retries", symbol=symbol)
        return None

    async def check_order_status(self, order_id: str, symbol: str) -> Order | None:
        """Check the status of a pending order."""
        try:
            order = await self._exchange.get_order_status(order_id, symbol)
            self._pending_orders[order_id] = order
            return order
        except (ValueError, ConnectionError, RuntimeError) as e:
            logger.error("order_status_check_failed", order_id=order_id, error=str(e))
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a pending order."""
        success = await self._exchange.cancel_order(order_id, symbol)
        if success:
            self._pending_orders.pop(order_id, None)
        return success

    @property
    def pending_orders(self) -> dict[str, Order]:
        return dict(self._pending_orders)
