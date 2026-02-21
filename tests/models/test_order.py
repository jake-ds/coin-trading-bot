"""Tests for Order model."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from bot.models import Order, OrderSide, OrderStatus, OrderType


class TestOrder:
    def test_valid_limit_order(self):
        order = Order(
            id="order-001",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            price=50000.0,
            quantity=0.1,
        )
        assert order.id == "order-001"
        assert order.exchange == "binance"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.LIMIT
        assert order.price == 50000.0
        assert order.quantity == 0.1
        assert order.status == OrderStatus.PENDING

    def test_valid_market_order(self):
        order = Order(
            id="order-002",
            exchange="upbit",
            symbol="BTC/KRW",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            price=0,
            quantity=1.0,
        )
        assert order.type == OrderType.MARKET
        assert order.price == 0

    def test_market_order_with_nonzero_price_rejected(self):
        with pytest.raises(ValidationError):
            Order(
                id="order-003",
                exchange="binance",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                price=50000.0,
                quantity=0.1,
            )

    def test_limit_order_with_zero_price_rejected(self):
        with pytest.raises(ValidationError):
            Order(
                id="order-004",
                exchange="binance",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                price=0,
                quantity=0.1,
            )

    def test_zero_quantity_rejected(self):
        with pytest.raises(ValidationError):
            Order(
                id="order-005",
                exchange="binance",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                price=100.0,
                quantity=0,
            )

    def test_order_statuses(self):
        for status in OrderStatus:
            order = Order(
                id="test",
                exchange="binance",
                symbol="ETH/USDT",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                price=0,
                quantity=1.0,
                status=status,
            )
            assert order.status == status

    def test_default_timestamps(self):
        order = Order(
            id="test",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            price=0,
            quantity=1.0,
        )
        assert isinstance(order.created_at, datetime)
        assert order.updated_at is None
        assert order.filled_at is None

    def test_frozen(self):
        order = Order(
            id="test",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            price=0,
            quantity=1.0,
        )
        with pytest.raises(ValidationError):
            order.status = OrderStatus.FILLED
