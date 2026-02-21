"""Tests for exchange factory pattern."""

import pytest

from bot.exchanges.base import ExchangeAdapter
from bot.exchanges.factory import (
    ExchangeFactory,
    _adapter_registry,
    register_adapter,
)
from bot.models import OHLCV, Order, OrderSide, OrderType


class MockExchangeAdapter(ExchangeAdapter):
    """Concrete mock adapter for testing."""

    def __init__(self, **kwargs):
        self.config = kwargs

    @property
    def name(self) -> str:
        return "mock"

    async def get_ticker(self, symbol: str) -> dict:
        return {"bid": 100.0, "ask": 101.0, "last": 100.5, "volume": 1000.0}

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> list[OHLCV]:
        return []

    async def get_balance(self) -> dict[str, float]:
        return {"USDT": 10000.0}

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
    ) -> Order:
        return Order(
            id="mock-001",
            exchange="mock",
            symbol=symbol,
            side=side,
            type=order_type,
            price=price or 0,
            quantity=quantity,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        return True

    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        return Order(
            id=order_id,
            exchange="mock",
            symbol=symbol,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            price=0,
            quantity=1.0,
        )

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        return {"bids": [], "asks": []}


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean the adapter registry before and after each test."""
    original = dict(_adapter_registry)
    _adapter_registry.clear()
    yield
    _adapter_registry.clear()
    _adapter_registry.update(original)


class TestRegisterAdapter:
    def test_register_adapter(self):
        register_adapter("mock", MockExchangeAdapter)
        assert "mock" in ExchangeFactory.available()

    def test_register_case_insensitive(self):
        register_adapter("MOCK", MockExchangeAdapter)
        assert "mock" in ExchangeFactory.available()


class TestExchangeFactory:
    def test_create_registered_adapter(self):
        register_adapter("mock", MockExchangeAdapter)
        adapter = ExchangeFactory.create("mock")
        assert isinstance(adapter, ExchangeAdapter)
        assert adapter.name == "mock"

    def test_create_case_insensitive(self):
        register_adapter("mock", MockExchangeAdapter)
        adapter = ExchangeFactory.create("MOCK")
        assert adapter.name == "mock"

    def test_create_with_kwargs(self):
        register_adapter("mock", MockExchangeAdapter)
        adapter = ExchangeFactory.create("mock", api_key="test-key")
        assert adapter.config["api_key"] == "test-key"

    def test_create_unknown_exchange_raises(self):
        with pytest.raises(ValueError, match="Unknown exchange"):
            ExchangeFactory.create("nonexistent")

    def test_available_empty(self):
        assert ExchangeFactory.available() == []

    def test_available_with_registered(self):
        register_adapter("mock", MockExchangeAdapter)
        register_adapter("other", MockExchangeAdapter)
        assert sorted(ExchangeFactory.available()) == ["mock", "other"]


class TestExchangeAdapterABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ExchangeAdapter()

    @pytest.mark.asyncio
    async def test_close_default_implementation(self):
        register_adapter("mock", MockExchangeAdapter)
        adapter = ExchangeFactory.create("mock")
        await adapter.close()  # Should not raise
