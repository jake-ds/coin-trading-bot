"""Tests for EngineManager."""

import asyncio

import pytest

from bot.engines.base import BaseEngine, EngineCycleResult, EngineStatus
from bot.engines.manager import EngineManager
from bot.engines.portfolio_manager import PortfolioManager


class StubEngine(BaseEngine):
    """Minimal engine for manager tests."""

    def __init__(self, engine_name, pm, **kwargs):
        super().__init__(pm, **kwargs)
        self._engine_name = engine_name

    @property
    def name(self):
        return self._engine_name

    @property
    def description(self):
        return f"Stub engine: {self._engine_name}"

    async def _run_cycle(self):
        return EngineCycleResult(
            engine_name=self._engine_name,
            cycle_num=self._cycle_count + 1,
            timestamp="2026-01-01T00:00:00Z",
            duration_ms=1.0,
            pnl_update=1.0,
        )


@pytest.fixture
def pm():
    return PortfolioManager(
        total_capital=10000.0,
        engine_allocations={"eng_a": 0.30, "eng_b": 0.20},
    )


@pytest.fixture
def manager(pm):
    return EngineManager(pm)


class TestRegistration:
    def test_register_engine(self, manager, pm):
        engine = StubEngine("eng_a", pm)
        manager.register(engine)
        assert "eng_a" in manager.engines
        assert manager.get_engine("eng_a") is engine

    def test_register_duplicate_raises(self, manager, pm):
        engine1 = StubEngine("eng_a", pm)
        engine2 = StubEngine("eng_a", pm)
        manager.register(engine1)
        with pytest.raises(ValueError, match="already registered"):
            manager.register(engine2)

    def test_get_nonexistent_engine(self, manager):
        assert manager.get_engine("nope") is None


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_all_and_stop_all(self, manager, pm):
        eng_a = StubEngine("eng_a", pm, loop_interval=0.01)
        eng_b = StubEngine("eng_b", pm, loop_interval=0.01)
        manager.register(eng_a)
        manager.register(eng_b)

        await manager.start_all()
        await asyncio.sleep(0.05)

        status = manager.get_status()
        # Engines should be running (or at least started)
        assert "eng_a" in status
        assert "eng_b" in status

        await manager.stop_all()
        assert eng_a.status == EngineStatus.STOPPED
        assert eng_b.status == EngineStatus.STOPPED

    @pytest.mark.asyncio
    async def test_start_stop_single_engine(self, manager, pm):
        engine = StubEngine("eng_a", pm, loop_interval=0.01)
        manager.register(engine)

        started = await manager.start_engine("eng_a")
        assert started
        await asyncio.sleep(0.05)

        stopped = await manager.stop_engine("eng_a")
        assert stopped
        assert engine.status == EngineStatus.STOPPED

    @pytest.mark.asyncio
    async def test_start_nonexistent_returns_false(self, manager):
        assert await manager.start_engine("nope") is False

    @pytest.mark.asyncio
    async def test_stop_nonexistent_returns_false(self, manager):
        assert await manager.stop_engine("nope") is False

    @pytest.mark.asyncio
    async def test_pause_resume_engine(self, manager, pm):
        engine = StubEngine("eng_a", pm, loop_interval=0.01)
        manager.register(engine)

        await manager.start_engine("eng_a")
        await asyncio.sleep(0.03)

        paused = await manager.pause_engine("eng_a")
        assert paused
        assert engine.status == EngineStatus.PAUSED

        resumed = await manager.resume_engine("eng_a")
        assert resumed
        assert engine.status == EngineStatus.RUNNING

        await manager.stop_engine("eng_a")


class TestStatus:
    def test_get_status_empty(self, manager):
        assert manager.get_status() == {}

    def test_get_status_with_engines(self, manager, pm):
        engine = StubEngine("eng_a", pm)
        manager.register(engine)
        status = manager.get_status()
        assert "eng_a" in status
        assert status["eng_a"]["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_get_all_cycle_logs(self, manager, pm):
        engine = StubEngine("eng_a", pm, loop_interval=0.01)
        manager.register(engine)

        await manager.start_engine("eng_a")
        await asyncio.sleep(0.05)
        await manager.stop_engine("eng_a")

        logs = manager.get_all_cycle_logs()
        assert len(logs) >= 1
        assert logs[0]["engine_name"] == "eng_a"

    @pytest.mark.asyncio
    async def test_get_engine_cycle_log(self, manager, pm):
        engine = StubEngine("eng_a", pm, loop_interval=0.01)
        manager.register(engine)

        await manager.start_engine("eng_a")
        await asyncio.sleep(0.05)
        await manager.stop_engine("eng_a")

        logs = manager.get_engine_cycle_log("eng_a")
        assert len(logs) >= 1

        assert manager.get_engine_cycle_log("nonexistent") == []

    def test_get_engine_positions(self, manager, pm):
        engine = StubEngine("eng_a", pm)
        engine._add_position("BTC/USDT", "long", 0.1, 50000.0)
        manager.register(engine)

        positions = manager.get_engine_positions("eng_a")
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTC/USDT"

        assert manager.get_engine_positions("nonexistent") == []
