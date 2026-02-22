"""Smart order execution: limit orders, TWAP splitting, and fee optimization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from bot.exchanges.base import ExchangeAdapter
from bot.models import Order, OrderSide, OrderStatus, OrderType

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


@dataclass
class FillMetrics:
    """Tracks maker vs taker fill statistics for fee optimization."""

    maker_fills: int = 0
    taker_fills: int = 0
    total_maker_volume: float = 0.0
    total_taker_volume: float = 0.0
    total_maker_fees: float = 0.0
    total_taker_fees: float = 0.0

    @property
    def maker_ratio(self) -> float:
        """Ratio of maker fills to total fills."""
        total = self.maker_fills + self.taker_fills
        return self.maker_fills / total if total > 0 else 0.0

    @property
    def total_fees(self) -> float:
        return self.total_maker_fees + self.total_taker_fees

    def record_maker(self, volume: float, fee: float) -> None:
        self.maker_fills += 1
        self.total_maker_volume += volume
        self.total_maker_fees += fee

    def record_taker(self, volume: float, fee: float) -> None:
        self.taker_fills += 1
        self.total_taker_volume += volume
        self.total_taker_fees += fee

    def to_dict(self) -> dict:
        return {
            "maker_fills": self.maker_fills,
            "taker_fills": self.taker_fills,
            "total_maker_volume": round(self.total_maker_volume, 8),
            "total_taker_volume": round(self.total_taker_volume, 8),
            "total_maker_fees": round(self.total_maker_fees, 8),
            "total_taker_fees": round(self.total_taker_fees, 8),
            "maker_ratio": round(self.maker_ratio, 4),
            "total_fees": round(self.total_fees, 8),
        }


@dataclass
class TWAPPlan:
    """Plan for Time-Weighted Average Price execution."""

    total_quantity: float
    chunk_count: int
    chunk_interval_seconds: float
    chunks: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.chunks:
            chunk_size = self.total_quantity / self.chunk_count
            remainder = self.total_quantity - (chunk_size * self.chunk_count)
            self.chunks = [chunk_size] * self.chunk_count
            # Add remainder to last chunk
            if remainder > 1e-10:
                self.chunks[-1] += remainder


class SmartExecutor:
    """Intelligent order execution with limit orders, TWAP, and fee optimization.

    Features:
    - Default to limit orders at best bid/ask for maker fee savings
    - Timeout and fallback to market orders if limit not filled
    - TWAP splitting for large orders
    - Maker vs taker fill tracking
    """

    def __init__(
        self,
        exchange: ExchangeAdapter,
        prefer_limit_orders: bool = True,
        limit_order_timeout_seconds: float = 30.0,
        twap_chunk_count: int = 5,
        twap_chunk_interval_seconds: float = 10.0,
        twap_volume_threshold_pct: float = 5.0,
        maker_fee_pct: float = 0.02,
        taker_fee_pct: float = 0.1,
    ):
        self._exchange = exchange
        self._prefer_limit_orders = prefer_limit_orders
        self._limit_order_timeout_seconds = limit_order_timeout_seconds
        self._twap_chunk_count = twap_chunk_count
        self._twap_chunk_interval_seconds = twap_chunk_interval_seconds
        self._twap_volume_threshold_pct = twap_volume_threshold_pct
        self._maker_fee_pct = maker_fee_pct
        self._taker_fee_pct = taker_fee_pct
        self._fill_metrics = FillMetrics()

    @property
    def fill_metrics(self) -> FillMetrics:
        return self._fill_metrics

    def should_use_twap(
        self,
        quantity: float,
        price: float,
        avg_daily_volume: float,
    ) -> bool:
        """Determine if order is large enough to warrant TWAP execution.

        Returns True if order value exceeds twap_volume_threshold_pct of avg daily volume.
        """
        if avg_daily_volume <= 0:
            return False
        order_value = quantity * price
        volume_ratio = (order_value / avg_daily_volume) * 100
        return volume_ratio > self._twap_volume_threshold_pct

    def create_twap_plan(
        self,
        quantity: float,
        chunk_count: int | None = None,
        chunk_interval: float | None = None,
    ) -> TWAPPlan:
        """Create a TWAP execution plan splitting the order into chunks."""
        count = chunk_count or self._twap_chunk_count
        interval = chunk_interval or self._twap_chunk_interval_seconds
        return TWAPPlan(
            total_quantity=quantity,
            chunk_count=count,
            chunk_interval_seconds=interval,
        )

    async def execute_limit_with_fallback(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        limit_price: float,
        timeout: float | None = None,
    ) -> Order | None:
        """Place a limit order with timeout, falling back to market order if not filled.

        Args:
            symbol: Trading pair symbol.
            side: BUY or SELL.
            quantity: Order quantity.
            limit_price: Limit price (best bid for BUY, best ask for SELL).
            timeout: Seconds to wait for fill before falling back. Uses default if None.

        Returns:
            Filled Order or None if both limit and market failed.
        """
        timeout = timeout if timeout is not None else self._limit_order_timeout_seconds

        # Place limit order
        try:
            limit_order = await self._exchange.create_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=limit_price,
            )
        except (ConnectionError, ValueError, RuntimeError) as e:
            logger.warning(
                "limit_order_failed_placing",
                symbol=symbol,
                side=side.value,
                error=str(e),
            )
            # Fall through to market order
            return await self._execute_market_order(symbol, side, quantity)

        # Wait for fill with polling
        filled_order = await self._wait_for_fill(
            limit_order, symbol, timeout
        )

        if filled_order and filled_order.status == OrderStatus.FILLED:
            # Maker fill
            volume = quantity * (filled_order.filled_price or limit_price)
            fee = volume * self._maker_fee_pct / 100
            self._fill_metrics.record_maker(volume, fee)
            logger.info(
                "limit_order_filled",
                symbol=symbol,
                side=side.value,
                price=filled_order.filled_price,
                fill_type="maker",
            )
            return filled_order

        # Limit order not filled — cancel and use market order
        if limit_order:
            try:
                await self._exchange.cancel_order(limit_order.id, symbol)
                logger.info(
                    "limit_order_cancelled_timeout",
                    symbol=symbol,
                    order_id=limit_order.id,
                    timeout=timeout,
                )
            except (ConnectionError, ValueError, RuntimeError) as e:
                logger.warning(
                    "limit_order_cancel_failed",
                    order_id=limit_order.id,
                    error=str(e),
                )

        # Fallback to market order
        return await self._execute_market_order(symbol, side, quantity)

    async def _wait_for_fill(
        self,
        order: Order,
        symbol: str,
        timeout: float,
    ) -> Order | None:
        """Poll order status until filled or timeout."""
        if order.status == OrderStatus.FILLED:
            return order

        elapsed = 0.0
        poll_interval = min(2.0, timeout / 3)

        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            try:
                updated = await self._exchange.get_order_status(
                    order.id, symbol
                )
                if updated.status == OrderStatus.FILLED:
                    return updated
                if updated.status in (
                    OrderStatus.CANCELLED,
                    OrderStatus.FAILED,
                ):
                    return updated
            except (ConnectionError, ValueError, RuntimeError) as e:
                logger.warning(
                    "order_status_check_error",
                    order_id=order.id,
                    error=str(e),
                )

        return None

    async def _execute_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
    ) -> Order | None:
        """Execute a market order as fallback."""
        try:
            order = await self._exchange.create_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
            )
            if order and order.status == OrderStatus.FILLED:
                volume = quantity * (order.filled_price or 0)
                fee = volume * self._taker_fee_pct / 100
                self._fill_metrics.record_taker(volume, fee)
                logger.info(
                    "market_order_filled",
                    symbol=symbol,
                    side=side.value,
                    price=order.filled_price,
                    fill_type="taker",
                )
            return order
        except (ConnectionError, ValueError, RuntimeError) as e:
            logger.error(
                "market_order_failed",
                symbol=symbol,
                side=side.value,
                error=str(e),
            )
            return None

    async def execute_twap(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        plan: TWAPPlan | None = None,
    ) -> list[Order]:
        """Execute an order using TWAP — splitting into smaller chunks over time.

        Args:
            symbol: Trading pair symbol.
            side: BUY or SELL.
            quantity: Total order quantity.
            price: Reference price for limit orders.
            plan: Optional TWAPPlan. Created with defaults if None.

        Returns:
            List of executed orders (some may be None if failed).
        """
        if plan is None:
            plan = self.create_twap_plan(quantity)

        executed_orders: list[Order] = []

        for i, chunk_qty in enumerate(plan.chunks):
            if chunk_qty <= 0:
                continue

            logger.info(
                "twap_chunk_executing",
                symbol=symbol,
                chunk=i + 1,
                total_chunks=len(plan.chunks),
                chunk_qty=chunk_qty,
            )

            # Get fresh price for each chunk
            try:
                ticker = await self._exchange.get_ticker(symbol)
                if side == OrderSide.BUY:
                    chunk_price = ticker.get("bid", price)
                else:
                    chunk_price = ticker.get("ask", price)
                if not chunk_price or chunk_price <= 0:
                    chunk_price = price
            except (ConnectionError, ValueError, RuntimeError):
                chunk_price = price

            if self._prefer_limit_orders:
                order = await self.execute_limit_with_fallback(
                    symbol=symbol,
                    side=side,
                    quantity=chunk_qty,
                    limit_price=chunk_price,
                )
            else:
                order = await self._execute_market_order(
                    symbol, side, chunk_qty
                )

            if order:
                executed_orders.append(order)

            # Wait between chunks (except after the last one)
            if i < len(plan.chunks) - 1:
                await asyncio.sleep(plan.chunk_interval_seconds)

        logger.info(
            "twap_execution_complete",
            symbol=symbol,
            total_chunks=len(plan.chunks),
            filled_chunks=len(executed_orders),
        )

        return executed_orders

    async def execute_smart(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        avg_daily_volume: float = 0.0,
    ) -> Order | list[Order] | None:
        """Execute an order using the best available strategy.

        - If order is large relative to volume, use TWAP
        - If prefer_limit_orders, try limit first with timeout fallback
        - Otherwise, use market order

        Args:
            symbol: Trading pair symbol.
            side: BUY or SELL.
            quantity: Order quantity.
            price: Current price (used for limit orders and TWAP threshold).
            avg_daily_volume: Average daily volume in quote currency. 0 = unknown.

        Returns:
            Single Order, list of Orders (TWAP), or None if failed.
        """
        # Check if TWAP is needed
        if avg_daily_volume > 0 and self.should_use_twap(
            quantity, price, avg_daily_volume
        ):
            logger.info(
                "smart_executor_using_twap",
                symbol=symbol,
                quantity=quantity,
                avg_daily_volume=avg_daily_volume,
            )
            return await self.execute_twap(symbol, side, quantity, price)

        # Try limit order with fallback
        if self._prefer_limit_orders:
            # Determine limit price from order book
            try:
                ticker = await self._exchange.get_ticker(symbol)
                if side == OrderSide.BUY:
                    limit_price = ticker.get("bid", price)
                else:
                    limit_price = ticker.get("ask", price)
                if not limit_price or limit_price <= 0:
                    limit_price = price
            except (ConnectionError, ValueError, RuntimeError):
                limit_price = price

            return await self.execute_limit_with_fallback(
                symbol=symbol,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
            )

        # Simple market order
        return await self._execute_market_order(symbol, side, quantity)
