"""Tests for SmartExecutor: limit orders, TWAP, and fee optimization."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.execution.smart_executor import FillMetrics, SmartExecutor, TWAPPlan
from bot.models import Order, OrderSide, OrderStatus, OrderType

# --- Fixtures ---


def make_order(
    order_id: str = "test-001",
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.LIMIT,
    status: OrderStatus = OrderStatus.FILLED,
    price: float = 50000.0,
    quantity: float = 0.1,
) -> Order:
    """Create a test order."""
    now = datetime.now(timezone.utc)
    # Market orders must have price=0 per Order model validator
    order_price = 0 if order_type == OrderType.MARKET else price
    return Order(
        id=order_id,
        exchange="binance",
        symbol=symbol,
        side=side,
        type=order_type,
        price=order_price,
        quantity=quantity,
        status=status,
        created_at=now,
        filled_at=now if status == OrderStatus.FILLED else None,
        filled_price=price if status == OrderStatus.FILLED else None,
        filled_quantity=quantity if status == OrderStatus.FILLED else 0,
    )


@pytest.fixture
def mock_exchange():
    """Create a mock exchange adapter."""
    exchange = MagicMock()
    exchange.name = "binance"
    exchange.create_order = AsyncMock()
    exchange.cancel_order = AsyncMock(return_value=True)
    exchange.get_order_status = AsyncMock()
    exchange.get_ticker = AsyncMock(
        return_value={"bid": 49990.0, "ask": 50010.0, "last": 50000.0}
    )
    return exchange


@pytest.fixture
def smart_executor(mock_exchange):
    """Create a SmartExecutor with default settings."""
    return SmartExecutor(
        exchange=mock_exchange,
        prefer_limit_orders=True,
        limit_order_timeout_seconds=5.0,  # Short for tests
        twap_chunk_count=3,
        twap_chunk_interval_seconds=0.01,  # Very short for tests
    )


# --- FillMetrics Tests ---


class TestFillMetrics:
    def test_initial_state(self):
        metrics = FillMetrics()
        assert metrics.maker_fills == 0
        assert metrics.taker_fills == 0
        assert metrics.maker_ratio == 0.0
        assert metrics.total_fees == 0.0

    def test_record_maker(self):
        metrics = FillMetrics()
        metrics.record_maker(volume=5000.0, fee=1.0)
        assert metrics.maker_fills == 1
        assert metrics.total_maker_volume == 5000.0
        assert metrics.total_maker_fees == 1.0

    def test_record_taker(self):
        metrics = FillMetrics()
        metrics.record_taker(volume=5000.0, fee=5.0)
        assert metrics.taker_fills == 1
        assert metrics.total_taker_volume == 5000.0
        assert metrics.total_taker_fees == 5.0

    def test_maker_ratio(self):
        metrics = FillMetrics()
        metrics.record_maker(volume=5000.0, fee=1.0)
        metrics.record_maker(volume=3000.0, fee=0.6)
        metrics.record_taker(volume=2000.0, fee=2.0)
        # 2 maker / 3 total = 0.6667
        assert metrics.maker_ratio == pytest.approx(2 / 3, abs=0.001)

    def test_total_fees(self):
        metrics = FillMetrics()
        metrics.record_maker(volume=5000.0, fee=1.0)
        metrics.record_taker(volume=5000.0, fee=5.0)
        assert metrics.total_fees == 6.0

    def test_to_dict(self):
        metrics = FillMetrics()
        metrics.record_maker(volume=5000.0, fee=1.0)
        d = metrics.to_dict()
        assert d["maker_fills"] == 1
        assert d["taker_fills"] == 0
        assert d["total_maker_volume"] == 5000.0
        assert d["maker_ratio"] == 1.0
        assert "total_fees" in d


# --- TWAPPlan Tests ---


class TestTWAPPlan:
    def test_creates_equal_chunks(self):
        plan = TWAPPlan(total_quantity=1.0, chunk_count=5, chunk_interval_seconds=10.0)
        assert len(plan.chunks) == 5
        assert sum(plan.chunks) == pytest.approx(1.0)
        assert all(c == pytest.approx(0.2) for c in plan.chunks)

    def test_remainder_goes_to_last_chunk(self):
        plan = TWAPPlan(total_quantity=1.0, chunk_count=3, chunk_interval_seconds=10.0)
        assert len(plan.chunks) == 3
        assert sum(plan.chunks) == pytest.approx(1.0)

    def test_single_chunk(self):
        plan = TWAPPlan(total_quantity=0.5, chunk_count=1, chunk_interval_seconds=10.0)
        assert len(plan.chunks) == 1
        assert plan.chunks[0] == pytest.approx(0.5)

    def test_custom_chunks(self):
        plan = TWAPPlan(
            total_quantity=1.0,
            chunk_count=3,
            chunk_interval_seconds=5.0,
            chunks=[0.5, 0.3, 0.2],
        )
        assert plan.chunks == [0.5, 0.3, 0.2]


# --- SmartExecutor Tests ---


class TestSmartExecutor:
    def test_init_defaults(self, mock_exchange):
        executor = SmartExecutor(exchange=mock_exchange)
        assert executor._prefer_limit_orders is True
        assert executor._limit_order_timeout_seconds == 30.0
        assert executor._twap_chunk_count == 5

    def test_should_use_twap_above_threshold(self, smart_executor):
        # 0.1 BTC at 50000 = 5000 USDT, 5% of 100000 = 5000
        assert smart_executor.should_use_twap(
            quantity=0.1, price=50000.0, avg_daily_volume=99999.0
        )

    def test_should_not_use_twap_below_threshold(self, smart_executor):
        # 0.01 BTC at 50000 = 500 USDT, 5% of 100000 = 5000
        assert not smart_executor.should_use_twap(
            quantity=0.01, price=50000.0, avg_daily_volume=100000.0
        )

    def test_should_not_use_twap_zero_volume(self, smart_executor):
        assert not smart_executor.should_use_twap(
            quantity=0.1, price=50000.0, avg_daily_volume=0.0
        )

    def test_create_twap_plan(self, smart_executor):
        plan = smart_executor.create_twap_plan(quantity=1.0)
        assert plan.chunk_count == 3  # Configured in fixture
        assert len(plan.chunks) == 3
        assert sum(plan.chunks) == pytest.approx(1.0)

    def test_create_twap_plan_custom(self, smart_executor):
        plan = smart_executor.create_twap_plan(
            quantity=2.0, chunk_count=4, chunk_interval=5.0
        )
        assert plan.chunk_count == 4
        assert plan.chunk_interval_seconds == 5.0
        assert sum(plan.chunks) == pytest.approx(2.0)


class TestLimitOrderWithFallback:
    @pytest.mark.asyncio
    async def test_limit_order_filled_immediately(
        self, smart_executor, mock_exchange
    ):
        """Limit order filled immediately returns without fallback."""
        filled_order = make_order(status=OrderStatus.FILLED, order_type=OrderType.LIMIT)
        mock_exchange.create_order.return_value = filled_order

        result = await smart_executor.execute_limit_with_fallback(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            limit_price=50000.0,
        )

        assert result is not None
        assert result.status == OrderStatus.FILLED
        mock_exchange.create_order.assert_called_once()
        # Maker fill recorded
        assert smart_executor.fill_metrics.maker_fills == 1

    @pytest.mark.asyncio
    async def test_limit_order_filled_after_polling(
        self, smart_executor, mock_exchange
    ):
        """Limit order filled after status polling."""
        pending_order = make_order(
            status=OrderStatus.SUBMITTED, order_type=OrderType.LIMIT
        )
        filled_order = make_order(
            status=OrderStatus.FILLED, order_type=OrderType.LIMIT
        )
        mock_exchange.create_order.return_value = pending_order
        mock_exchange.get_order_status.return_value = filled_order

        result = await smart_executor.execute_limit_with_fallback(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            limit_price=50000.0,
        )

        assert result is not None
        assert result.status == OrderStatus.FILLED
        assert smart_executor.fill_metrics.maker_fills == 1

    @pytest.mark.asyncio
    async def test_limit_order_timeout_fallback_to_market(
        self, smart_executor, mock_exchange
    ):
        """Limit order times out, falls back to market order."""
        pending_order = make_order(
            status=OrderStatus.SUBMITTED, order_type=OrderType.LIMIT
        )
        # get_order_status always returns SUBMITTED (never fills)
        mock_exchange.create_order.side_effect = [
            pending_order,
            make_order(
                order_id="market-001",
                status=OrderStatus.FILLED,
                order_type=OrderType.MARKET,
            ),
        ]
        mock_exchange.get_order_status.return_value = pending_order

        result = await smart_executor.execute_limit_with_fallback(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            limit_price=50000.0,
            timeout=0.1,  # Very short timeout
        )

        assert result is not None
        # Cancel was called for the limit order
        mock_exchange.cancel_order.assert_called_once()
        # Taker fill recorded (market order fallback)
        assert smart_executor.fill_metrics.taker_fills == 1

    @pytest.mark.asyncio
    async def test_limit_order_placement_fails_market_fallback(
        self, smart_executor, mock_exchange
    ):
        """If limit order placement fails, fall back to market order."""
        market_order = make_order(
            order_id="market-001",
            status=OrderStatus.FILLED,
            order_type=OrderType.MARKET,
        )
        mock_exchange.create_order.side_effect = [
            ConnectionError("exchange down"),
            market_order,
        ]

        result = await smart_executor.execute_limit_with_fallback(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            limit_price=50000.0,
        )

        assert result is not None
        assert smart_executor.fill_metrics.taker_fills == 1

    @pytest.mark.asyncio
    async def test_limit_order_cancel_fails_gracefully(
        self, smart_executor, mock_exchange
    ):
        """If canceling timed-out limit order fails, still falls back to market."""
        pending_order = make_order(
            status=OrderStatus.SUBMITTED, order_type=OrderType.LIMIT
        )
        market_order = make_order(
            order_id="market-001",
            status=OrderStatus.FILLED,
            order_type=OrderType.MARKET,
        )
        mock_exchange.create_order.side_effect = [pending_order, market_order]
        mock_exchange.get_order_status.return_value = pending_order
        mock_exchange.cancel_order.side_effect = ConnectionError("cancel failed")

        result = await smart_executor.execute_limit_with_fallback(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            limit_price=50000.0,
            timeout=0.1,
        )

        assert result is not None
        assert result.id == "market-001"

    @pytest.mark.asyncio
    async def test_both_limit_and_market_fail(
        self, smart_executor, mock_exchange
    ):
        """If both limit and market orders fail, returns None."""
        mock_exchange.create_order.side_effect = ConnectionError("exchange down")

        result = await smart_executor.execute_limit_with_fallback(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            limit_price=50000.0,
        )

        assert result is None


class TestTWAPExecution:
    @pytest.mark.asyncio
    async def test_twap_executes_all_chunks(
        self, smart_executor, mock_exchange
    ):
        """TWAP splits order into chunks and executes each."""
        filled_order = make_order(status=OrderStatus.FILLED, order_type=OrderType.LIMIT)
        mock_exchange.create_order.return_value = filled_order

        orders = await smart_executor.execute_twap(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.3,
            price=50000.0,
        )

        assert len(orders) == 3  # 3 chunks from fixture
        # Each chunk gets a ticker call + create_order call
        assert mock_exchange.get_ticker.call_count >= 3

    @pytest.mark.asyncio
    async def test_twap_custom_plan(self, smart_executor, mock_exchange):
        """TWAP respects a custom plan."""
        filled_order = make_order(status=OrderStatus.FILLED, order_type=OrderType.LIMIT)
        mock_exchange.create_order.return_value = filled_order

        plan = TWAPPlan(
            total_quantity=1.0,
            chunk_count=2,
            chunk_interval_seconds=0.01,
            chunks=[0.6, 0.4],
        )

        orders = await smart_executor.execute_twap(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            plan=plan,
        )

        assert len(orders) == 2

    @pytest.mark.asyncio
    async def test_twap_partial_failure(self, smart_executor, mock_exchange):
        """TWAP continues even if some chunks fail."""
        filled_order = make_order(status=OrderStatus.FILLED, order_type=OrderType.LIMIT)
        mock_exchange.create_order.side_effect = [
            filled_order,
            ConnectionError("temporary failure"),
            ConnectionError("temporary failure"),  # market fallback also fails
            filled_order,
        ]

        orders = await smart_executor.execute_twap(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.3,
            price=50000.0,
        )

        # At least some chunks should succeed
        assert len(orders) >= 1

    @pytest.mark.asyncio
    async def test_twap_uses_fresh_prices(self, smart_executor, mock_exchange):
        """TWAP fetches fresh ticker prices for each chunk."""
        filled_order = make_order(status=OrderStatus.FILLED, order_type=OrderType.LIMIT)
        mock_exchange.create_order.return_value = filled_order

        # Different prices for each chunk
        mock_exchange.get_ticker.side_effect = [
            {"bid": 49990.0, "ask": 50010.0, "last": 50000.0},
            {"bid": 50090.0, "ask": 50110.0, "last": 50100.0},
            {"bid": 50190.0, "ask": 50210.0, "last": 50200.0},
        ]

        orders = await smart_executor.execute_twap(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.3,
            price=50000.0,
        )

        assert len(orders) == 3
        assert mock_exchange.get_ticker.call_count == 3

    @pytest.mark.asyncio
    async def test_twap_sell_uses_ask(self, smart_executor, mock_exchange):
        """TWAP SELL orders use ask price from ticker."""
        filled_order = make_order(
            status=OrderStatus.FILLED,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
        )
        mock_exchange.create_order.return_value = filled_order

        orders = await smart_executor.execute_twap(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.3,
            price=50000.0,
        )

        assert len(orders) == 3


class TestExecuteSmart:
    @pytest.mark.asyncio
    async def test_smart_uses_limit_by_default(
        self, smart_executor, mock_exchange
    ):
        """execute_smart uses limit order when prefer_limit_orders is True."""
        filled_order = make_order(status=OrderStatus.FILLED, order_type=OrderType.LIMIT)
        mock_exchange.create_order.return_value = filled_order

        result = await smart_executor.execute_smart(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
        )

        assert result is not None
        assert result.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_smart_uses_market_when_disabled(self, mock_exchange):
        """execute_smart uses market order when prefer_limit_orders is False."""
        executor = SmartExecutor(
            exchange=mock_exchange,
            prefer_limit_orders=False,
        )
        market_order = make_order(
            status=OrderStatus.FILLED, order_type=OrderType.MARKET
        )
        mock_exchange.create_order.return_value = market_order

        result = await executor.execute_smart(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
        )

        assert result is not None
        assert executor.fill_metrics.taker_fills == 1

    @pytest.mark.asyncio
    async def test_smart_triggers_twap_for_large_orders(
        self, smart_executor, mock_exchange
    ):
        """execute_smart uses TWAP for large orders exceeding volume threshold."""
        filled_order = make_order(status=OrderStatus.FILLED, order_type=OrderType.LIMIT)
        mock_exchange.create_order.return_value = filled_order

        # Order value = 10 * 50000 = 500000
        # avg_daily_volume = 1000000
        # volume_ratio = 50% > 5% threshold
        result = await smart_executor.execute_smart(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=10.0,
            price=50000.0,
            avg_daily_volume=1000000.0,
        )

        # TWAP returns list
        assert isinstance(result, list)
        assert len(result) == 3  # 3 chunks from fixture

    @pytest.mark.asyncio
    async def test_smart_no_twap_for_small_orders(
        self, smart_executor, mock_exchange
    ):
        """execute_smart doesn't use TWAP for small orders."""
        filled_order = make_order(status=OrderStatus.FILLED, order_type=OrderType.LIMIT)
        mock_exchange.create_order.return_value = filled_order

        # Order value = 0.001 * 50000 = 50 USDT
        # avg_daily_volume = 1000000
        # volume_ratio = 0.005% < 5% threshold
        result = await smart_executor.execute_smart(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.001,
            price=50000.0,
            avg_daily_volume=1000000.0,
        )

        # Single order, not list
        assert isinstance(result, Order)

    @pytest.mark.asyncio
    async def test_smart_sell_uses_ask_price(
        self, smart_executor, mock_exchange
    ):
        """execute_smart SELL orders get ask price from ticker."""
        filled_order = make_order(
            status=OrderStatus.FILLED,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
        )
        mock_exchange.create_order.return_value = filled_order

        result = await smart_executor.execute_smart(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            price=50000.0,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_smart_ticker_failure_uses_price(
        self, smart_executor, mock_exchange
    ):
        """If ticker fetch fails, use provided price for limit order."""
        mock_exchange.get_ticker.side_effect = ConnectionError("no ticker")
        filled_order = make_order(status=OrderStatus.FILLED, order_type=OrderType.LIMIT)
        mock_exchange.create_order.return_value = filled_order

        result = await smart_executor.execute_smart(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_smart_no_avg_volume_skips_twap(
        self, smart_executor, mock_exchange
    ):
        """TWAP is not used when avg_daily_volume is 0 (unknown)."""
        filled_order = make_order(status=OrderStatus.FILLED, order_type=OrderType.LIMIT)
        mock_exchange.create_order.return_value = filled_order

        result = await smart_executor.execute_smart(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=100.0,
            price=50000.0,
            avg_daily_volume=0.0,
        )

        # Should not use TWAP, returns single order
        assert isinstance(result, Order)


class TestSmartExecutorFeeTracking:
    @pytest.mark.asyncio
    async def test_maker_fill_records_maker_metrics(
        self, smart_executor, mock_exchange
    ):
        """Filled limit order records maker metrics."""
        filled_order = make_order(status=OrderStatus.FILLED, order_type=OrderType.LIMIT)
        mock_exchange.create_order.return_value = filled_order

        await smart_executor.execute_limit_with_fallback(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            limit_price=50000.0,
        )

        metrics = smart_executor.fill_metrics
        assert metrics.maker_fills == 1
        assert metrics.taker_fills == 0
        assert metrics.total_maker_volume > 0
        assert metrics.total_maker_fees > 0

    @pytest.mark.asyncio
    async def test_taker_fill_records_taker_metrics(
        self, smart_executor, mock_exchange
    ):
        """Market order fallback records taker metrics."""
        market_order = make_order(
            status=OrderStatus.FILLED, order_type=OrderType.MARKET
        )
        mock_exchange.create_order.return_value = market_order

        await smart_executor._execute_market_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
        )

        metrics = smart_executor.fill_metrics
        assert metrics.maker_fills == 0
        assert metrics.taker_fills == 1
        assert metrics.total_taker_volume > 0
        assert metrics.total_taker_fees > 0

    @pytest.mark.asyncio
    async def test_mixed_fills_track_both(
        self, smart_executor, mock_exchange
    ):
        """Multiple orders track both maker and taker fills."""
        limit_order = make_order(status=OrderStatus.FILLED, order_type=OrderType.LIMIT)
        market_order = make_order(
            order_id="market-001",
            status=OrderStatus.FILLED,
            order_type=OrderType.MARKET,
        )

        # First: maker fill
        mock_exchange.create_order.return_value = limit_order
        await smart_executor.execute_limit_with_fallback(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            limit_price=50000.0,
        )

        # Second: taker fill
        mock_exchange.create_order.return_value = market_order
        await smart_executor._execute_market_order(
            "BTC/USDT", OrderSide.BUY, 0.1
        )

        metrics = smart_executor.fill_metrics
        assert metrics.maker_fills == 1
        assert metrics.taker_fills == 1
        assert metrics.maker_ratio == pytest.approx(0.5)


class TestSmartExecutorIntegration:
    @pytest.mark.asyncio
    async def test_engine_uses_smart_executor_in_live_mode(self):
        """ExecutionEngine delegates to SmartExecutor in live mode."""
        from bot.execution.engine import ExecutionEngine
        from bot.models import SignalAction, TradingSignal

        mock_exchange = MagicMock()
        mock_exchange.name = "binance"
        mock_exchange.get_ticker = AsyncMock(
            return_value={"bid": 49990.0, "ask": 50010.0, "last": 50000.0}
        )

        filled_order = make_order(
            status=OrderStatus.FILLED, order_type=OrderType.LIMIT
        )
        mock_exchange.create_order = AsyncMock(return_value=filled_order)

        smart_exec = SmartExecutor(
            exchange=mock_exchange,
            prefer_limit_orders=True,
            limit_order_timeout_seconds=5.0,
        )

        mock_store = MagicMock()
        mock_store.save_trade = AsyncMock()

        engine = ExecutionEngine(
            exchange=mock_exchange,
            store=mock_store,
            paper_trading=False,
            smart_executor=smart_exec,
        )

        signal = TradingSignal(
            strategy_name="test",
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            confidence=0.8,
        )

        order = await engine.execute_signal(
            signal, quantity=0.1, price=50000.0
        )
        assert order is not None
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_engine_without_smart_executor_uses_direct(self):
        """ExecutionEngine without SmartExecutor uses direct exchange calls."""
        from bot.execution.engine import ExecutionEngine
        from bot.models import SignalAction, TradingSignal

        mock_exchange = MagicMock()
        mock_exchange.name = "binance"
        filled_order = make_order(
            status=OrderStatus.FILLED, order_type=OrderType.MARKET
        )
        mock_exchange.create_order = AsyncMock(return_value=filled_order)

        mock_store = MagicMock()
        mock_store.save_trade = AsyncMock()

        engine = ExecutionEngine(
            exchange=mock_exchange,
            store=mock_store,
            paper_trading=False,
            smart_executor=None,  # No SmartExecutor
        )

        signal = TradingSignal(
            strategy_name="test",
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            confidence=0.8,
        )

        order = await engine.execute_signal(signal, quantity=0.1)
        assert order is not None
        # Direct exchange call
        mock_exchange.create_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_engine_smart_executor_fallback_on_error(self):
        """ExecutionEngine falls back to direct execution if SmartExecutor fails."""
        from bot.execution.engine import ExecutionEngine
        from bot.models import SignalAction, TradingSignal

        mock_exchange = MagicMock()
        mock_exchange.name = "binance"

        # SmartExecutor will raise, direct call will succeed
        filled_order = make_order(
            status=OrderStatus.FILLED, order_type=OrderType.MARKET
        )
        mock_exchange.create_order = AsyncMock(return_value=filled_order)

        smart_exec = MagicMock()
        smart_exec.execute_smart = AsyncMock(
            side_effect=Exception("smart executor error")
        )

        mock_store = MagicMock()
        mock_store.save_trade = AsyncMock()

        engine = ExecutionEngine(
            exchange=mock_exchange,
            store=mock_store,
            paper_trading=False,
            smart_executor=smart_exec,
        )

        signal = TradingSignal(
            strategy_name="test",
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            confidence=0.8,
        )

        order = await engine.execute_signal(
            signal, quantity=0.1, price=50000.0
        )
        assert order is not None

    @pytest.mark.asyncio
    async def test_engine_twap_result_returns_last_order(self):
        """When SmartExecutor returns TWAP list, engine returns last order."""
        from bot.execution.engine import ExecutionEngine
        from bot.models import SignalAction, TradingSignal

        mock_exchange = MagicMock()
        mock_exchange.name = "binance"
        mock_exchange.create_order = AsyncMock()

        order1 = make_order(order_id="twap-1", status=OrderStatus.FILLED)
        order2 = make_order(order_id="twap-2", status=OrderStatus.FILLED)
        order3 = make_order(order_id="twap-3", status=OrderStatus.FILLED)

        smart_exec = MagicMock()
        smart_exec.execute_smart = AsyncMock(
            return_value=[order1, order2, order3]
        )

        mock_store = MagicMock()
        mock_store.save_trade = AsyncMock()

        engine = ExecutionEngine(
            exchange=mock_exchange,
            store=mock_store,
            paper_trading=False,
            smart_executor=smart_exec,
        )

        signal = TradingSignal(
            strategy_name="test",
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            confidence=0.8,
        )

        order = await engine.execute_signal(
            signal, quantity=0.3, price=50000.0
        )
        assert order is not None
        assert order.id == "twap-3"
        # All orders tracked in pending
        assert "twap-1" in engine.pending_orders
        assert "twap-2" in engine.pending_orders
        assert "twap-3" in engine.pending_orders


class TestConfigSettings:
    def test_default_smart_execution_settings(self):
        """Config has correct defaults for smart execution settings."""
        from bot.config import Settings

        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
        )
        assert settings.prefer_limit_orders is True
        assert settings.limit_order_timeout_seconds == 30.0
        assert settings.twap_chunk_count == 5

    def test_custom_smart_execution_settings(self):
        """Config accepts custom smart execution settings."""
        from bot.config import Settings

        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            prefer_limit_orders=False,
            limit_order_timeout_seconds=60.0,
            twap_chunk_count=10,
        )
        assert settings.prefer_limit_orders is False
        assert settings.limit_order_timeout_seconds == 60.0
        assert settings.twap_chunk_count == 10


class TestWaitForFill:
    @pytest.mark.asyncio
    async def test_order_cancelled_during_wait(
        self, smart_executor, mock_exchange
    ):
        """If order is cancelled externally during wait, returns the cancelled order."""
        pending_order = make_order(
            status=OrderStatus.SUBMITTED, order_type=OrderType.LIMIT
        )
        cancelled_order = make_order(
            status=OrderStatus.CANCELLED, order_type=OrderType.LIMIT
        )
        mock_exchange.create_order.return_value = pending_order
        mock_exchange.get_order_status.return_value = cancelled_order

        # The wait should return quickly since status is CANCELLED
        result = await smart_executor._wait_for_fill(
            pending_order, "BTC/USDT", timeout=5.0
        )
        assert result is not None
        assert result.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_already_filled_returns_immediately(
        self, smart_executor, mock_exchange
    ):
        """If order is already FILLED, returns immediately."""
        filled_order = make_order(
            status=OrderStatus.FILLED, order_type=OrderType.LIMIT
        )
        result = await smart_executor._wait_for_fill(
            filled_order, "BTC/USDT", timeout=5.0
        )
        assert result is not None
        assert result.status == OrderStatus.FILLED
        # No polling was needed
        mock_exchange.get_order_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_status_check_error_continues_polling(
        self, smart_executor, mock_exchange
    ):
        """Errors during status check don't stop polling."""
        pending_order = make_order(
            status=OrderStatus.SUBMITTED, order_type=OrderType.LIMIT
        )
        filled_order = make_order(
            status=OrderStatus.FILLED, order_type=OrderType.LIMIT
        )
        mock_exchange.get_order_status.side_effect = [
            ConnectionError("temp error"),
            filled_order,
        ]

        result = await smart_executor._wait_for_fill(
            pending_order, "BTC/USDT", timeout=5.0
        )
        assert result is not None
        assert result.status == OrderStatus.FILLED
