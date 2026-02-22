"""Tests for BaseEngine ABC."""

import asyncio

import pytest

from bot.engines.base import BaseEngine, EngineCycleResult, EngineStatus
from bot.engines.portfolio_manager import PortfolioManager


class DummyEngine(BaseEngine):
    """Concrete implementation for testing."""

    def __init__(self, pm, cycle_results=None, **kwargs):
        super().__init__(pm, **kwargs)
        self._cycle_results = cycle_results or []
        self._cycle_index = 0

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "A dummy engine for testing"

    async def _run_cycle(self) -> EngineCycleResult:
        if self._cycle_index < len(self._cycle_results):
            result = self._cycle_results[self._cycle_index]
            self._cycle_index += 1
            return result
        return EngineCycleResult(
            engine_name="dummy",
            cycle_num=self._cycle_count + 1,
            timestamp="2026-01-01T00:00:00Z",
            duration_ms=0.0,
        )


@pytest.fixture
def pm():
    return PortfolioManager(
        total_capital=10000.0,
        engine_allocations={"dummy": 0.5},
    )


class TestEngineStatus:
    def test_enum_values(self):
        assert EngineStatus.STOPPED == "stopped"
        assert EngineStatus.RUNNING == "running"
        assert EngineStatus.PAUSED == "paused"
        assert EngineStatus.ERROR == "error"


class TestEngineCycleResult:
    def test_defaults(self):
        r = EngineCycleResult(
            engine_name="test",
            cycle_num=1,
            timestamp="2026-01-01T00:00:00Z",
            duration_ms=100.0,
        )
        assert r.engine_name == "test"
        assert r.actions_taken == []
        assert r.positions == []
        assert r.signals == []
        assert r.pnl_update == 0.0
        assert r.metadata == {}


class TestBaseEngine:
    def test_initial_state(self, pm):
        engine = DummyEngine(pm)
        assert engine.status == EngineStatus.STOPPED
        assert engine.cycle_count == 0
        assert engine.total_pnl == 0.0
        assert engine.allocated_capital == 0.0
        assert engine.positions == {}
        assert engine.position_count == 0

    @pytest.mark.asyncio
    async def test_start_allocates_capital(self, pm):
        engine = DummyEngine(pm)
        await engine.start()
        assert engine.status == EngineStatus.RUNNING
        assert engine.allocated_capital == 5000.0  # 50% of 10000

    @pytest.mark.asyncio
    async def test_stop_releases_capital(self, pm):
        engine = DummyEngine(pm)
        await engine.start()
        assert pm.total_allocated == 5000.0
        await engine.stop()
        assert engine.status == EngineStatus.STOPPED
        assert engine.allocated_capital == 0.0
        assert pm.total_allocated == 0.0

    @pytest.mark.asyncio
    async def test_pause_resume(self, pm):
        engine = DummyEngine(pm)
        await engine.start()
        await engine.pause()
        assert engine.status == EngineStatus.PAUSED
        await engine.resume()
        assert engine.status == EngineStatus.RUNNING

    @pytest.mark.asyncio
    async def test_pause_only_when_running(self, pm):
        engine = DummyEngine(pm)
        await engine.pause()  # Not running — should be no-op
        assert engine.status == EngineStatus.STOPPED

    @pytest.mark.asyncio
    async def test_resume_only_when_paused(self, pm):
        engine = DummyEngine(pm)
        await engine.start()
        await engine.resume()  # Already running — no-op
        assert engine.status == EngineStatus.RUNNING

    @pytest.mark.asyncio
    async def test_run_executes_cycles(self, pm):
        results = [
            EngineCycleResult(
                engine_name="dummy",
                cycle_num=i,
                timestamp="2026-01-01T00:00:00Z",
                duration_ms=10.0,
                pnl_update=5.0,
            )
            for i in range(3)
        ]
        engine = DummyEngine(pm, cycle_results=results, loop_interval=0.01)

        async def stop_after():
            await asyncio.sleep(0.05)
            engine._running = False

        asyncio.create_task(stop_after())
        await engine.run()

        assert engine.cycle_count >= 1
        assert engine.status == EngineStatus.STOPPED  # run() calls stop()

    def test_add_remove_position(self, pm):
        engine = DummyEngine(pm)
        engine._add_position("BTC/USDT", "long", 0.1, 50000.0)
        assert engine.position_count == 1
        assert "BTC/USDT" in engine.positions

        removed = engine._remove_position("BTC/USDT")
        assert removed is not None
        assert removed["symbol"] == "BTC/USDT"
        assert engine.position_count == 0

    def test_remove_nonexistent_position(self, pm):
        engine = DummyEngine(pm)
        assert engine._remove_position("ETH/USDT") is None

    def test_has_capacity(self, pm):
        engine = DummyEngine(pm, max_positions=2)
        assert engine._has_capacity()
        engine._add_position("BTC/USDT", "long", 0.1, 50000.0)
        assert engine._has_capacity()
        engine._add_position("ETH/USDT", "long", 1.0, 3000.0)
        assert not engine._has_capacity()

    def test_get_status_dict(self, pm):
        engine = DummyEngine(pm)
        status = engine.get_status_dict()
        assert status["name"] == "dummy"
        assert status["description"] == "A dummy engine for testing"
        assert status["status"] == "stopped"
        assert status["cycle_count"] == 0
        assert status["total_pnl"] == 0.0

    @pytest.mark.asyncio
    async def test_cycle_error_sets_error_status(self, pm):
        class ErrorEngine(BaseEngine):
            @property
            def name(self):
                return "error_engine"

            @property
            def description(self):
                return "Raises on cycle"

            async def _run_cycle(self):
                raise ValueError("test error")

        engine = ErrorEngine(pm, loop_interval=0.01)

        async def stop_after():
            await asyncio.sleep(0.05)
            engine._running = False

        asyncio.create_task(stop_after())
        await engine.run()
        # Engine should have recorded the error message
        assert engine.error_message == "test error"
