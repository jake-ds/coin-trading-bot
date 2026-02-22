"""Tests for strategy interface and registry."""

from datetime import datetime
from typing import Any

import pytest

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy, StrategyRegistry, strategy_registry


class DummyStrategy(BaseStrategy):
    """Concrete test strategy."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def required_history_length(self) -> int:
        return 10

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        symbol = kwargs.get("symbol", "BTC/USDT")
        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.5,
        )


class AnotherStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "another"

    @property
    def required_history_length(self) -> int:
        return 20

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        return TradingSignal(
            strategy_name=self.name,
            symbol=kwargs.get("symbol", "BTC/USDT"),
            action=SignalAction.BUY,
            confidence=0.8,
        )


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset the registry before/after each test."""
    strategy_registry.clear()
    yield
    strategy_registry.clear()


class TestBaseStrategy:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseStrategy()

    @pytest.mark.asyncio
    async def test_concrete_strategy(self):
        s = DummyStrategy()
        assert s.name == "dummy"
        assert s.required_history_length == 10

        candle = OHLCV(
            timestamp=datetime(2024, 1, 1),
            open=100.0, high=110.0, low=95.0, close=105.0, volume=1000.0,
        )
        signal = await s.analyze([candle], symbol="ETH/USDT")
        assert signal.action == SignalAction.HOLD
        assert signal.symbol == "ETH/USDT"


class TestStrategyRegistry:
    def test_register_instance(self):
        s = DummyStrategy()
        strategy_registry.register(s)
        assert strategy_registry.get("dummy") is s

    def test_register_class_decorator(self):
        @strategy_registry.register
        class DecoratedStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "decorated"

            @property
            def required_history_length(self) -> int:
                return 5

            async def analyze(self, ohlcv_data, **kwargs):
                return TradingSignal(
                    strategy_name=self.name,
                    symbol="BTC/USDT",
                    action=SignalAction.HOLD,
                    confidence=0.5,
                )

        assert strategy_registry.get("decorated") is not None

    def test_get_nonexistent(self):
        assert strategy_registry.get("nonexistent") is None

    def test_get_active(self):
        s1 = DummyStrategy()
        s2 = AnotherStrategy()
        strategy_registry.register(s1)
        strategy_registry.register(s2)
        active = strategy_registry.get_active()
        assert len(active) == 2

    def test_get_all(self):
        s1 = DummyStrategy()
        s2 = AnotherStrategy()
        strategy_registry.register(s1)
        strategy_registry.register(s2)
        assert len(strategy_registry.get_all()) == 2

    def test_enable_disable(self):
        s = DummyStrategy()
        strategy_registry.register(s)
        assert strategy_registry.is_active("dummy") is True

        strategy_registry.disable("dummy")
        assert strategy_registry.is_active("dummy") is False
        assert len(strategy_registry.get_active()) == 0

        strategy_registry.enable("dummy")
        assert strategy_registry.is_active("dummy") is True

    def test_disable_nonexistent(self):
        assert strategy_registry.disable("nonexistent") is False

    def test_enable_nonexistent(self):
        assert strategy_registry.enable("nonexistent") is False

    def test_unregister(self):
        s = DummyStrategy()
        strategy_registry.register(s)
        strategy_registry.unregister("dummy")
        assert strategy_registry.get("dummy") is None

    def test_clear(self):
        strategy_registry.register(DummyStrategy())
        strategy_registry.register(AnotherStrategy())
        strategy_registry.clear()
        assert strategy_registry.get_all() == []

    def test_singleton(self):
        r1 = StrategyRegistry()
        r2 = StrategyRegistry()
        assert r1 is r2
