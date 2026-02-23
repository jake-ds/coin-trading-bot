"""Tests for circuit breaker — CRISIS regime auto-pause/resume of engines."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCircuitBreakerCheck:
    """Tests for EngineManager._circuit_breaker_check()."""

    def _make_manager(self, settings=None):
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        pm.request_capital.return_value = 1000.0
        pm.get_max_allocation.return_value = 1000.0
        mgr = EngineManager(pm, settings=settings)
        return mgr

    def _make_engine(self, name, status="running"):
        engine = MagicMock()
        engine.name = name
        engine.status = MagicMock()
        engine.status.value = status
        engine.pause = AsyncMock()
        engine.resume = AsyncMock()
        engine.positions = {}
        return engine

    def _make_detector(self, is_crisis=False, duration=0.0):
        detector = MagicMock()
        detector.is_crisis.return_value = is_crisis
        detector.get_regime_duration.return_value = duration
        return detector

    @pytest.mark.asyncio
    async def test_no_detector_noop(self):
        """No regime detector → circuit breaker does nothing."""
        mgr = self._make_manager()
        mgr._regime_detector = None
        await mgr._circuit_breaker_check()
        assert mgr._circuit_breaker_active is False

    @pytest.mark.asyncio
    async def test_adaptation_disabled_noop(self):
        """regime_adaptation_enabled=False → circuit breaker does nothing."""
        settings = MagicMock()
        settings.regime_adaptation_enabled = False
        mgr = self._make_manager(settings=settings)
        detector = self._make_detector(is_crisis=True, duration=60.0)
        mgr._regime_detector = detector
        await mgr._circuit_breaker_check()
        assert mgr._circuit_breaker_active is False

    @pytest.mark.asyncio
    async def test_crisis_below_threshold_no_trigger(self):
        """CRISIS active but duration < threshold → no trigger."""
        settings = MagicMock()
        settings.regime_adaptation_enabled = True
        settings.crisis_circuit_breaker_minutes = 30.0
        mgr = self._make_manager(settings=settings)
        detector = self._make_detector(is_crisis=True, duration=20.0)
        mgr._regime_detector = detector

        e1 = self._make_engine("funding_rate_arb")
        mgr._engines = {"funding_rate_arb": e1}

        await mgr._circuit_breaker_check()
        assert mgr._circuit_breaker_active is False
        e1.pause.assert_not_called()

    @pytest.mark.asyncio
    async def test_crisis_at_threshold_triggers(self):
        """CRISIS duration >= threshold → pauses all running engines."""
        settings = MagicMock()
        settings.regime_adaptation_enabled = True
        settings.crisis_circuit_breaker_minutes = 30.0
        mgr = self._make_manager(settings=settings)
        detector = self._make_detector(is_crisis=True, duration=30.0)
        mgr._regime_detector = detector

        e1 = self._make_engine("funding_rate_arb", "running")
        e2 = self._make_engine("grid_trading", "running")
        e3 = self._make_engine("stat_arb", "paused")
        mgr._engines = {
            "funding_rate_arb": e1,
            "grid_trading": e2,
            "stat_arb": e3,
        }

        await mgr._circuit_breaker_check()

        assert mgr._circuit_breaker_active is True
        e1.pause.assert_called_once()
        e2.pause.assert_called_once()
        # Already paused engine should NOT be re-paused
        e3.pause.assert_not_called()
        assert "funding_rate_arb" in mgr._circuit_breaker_paused_engines
        assert "grid_trading" in mgr._circuit_breaker_paused_engines
        assert "stat_arb" not in mgr._circuit_breaker_paused_engines

    @pytest.mark.asyncio
    async def test_crisis_above_threshold_triggers(self):
        """CRISIS duration well above threshold → triggers."""
        settings = MagicMock()
        settings.regime_adaptation_enabled = True
        settings.crisis_circuit_breaker_minutes = 30.0
        mgr = self._make_manager(settings=settings)
        detector = self._make_detector(is_crisis=True, duration=60.0)
        mgr._regime_detector = detector

        e1 = self._make_engine("funding_rate_arb")
        mgr._engines = {"funding_rate_arb": e1}

        await mgr._circuit_breaker_check()
        assert mgr._circuit_breaker_active is True
        e1.pause.assert_called_once()

    @pytest.mark.asyncio
    async def test_already_active_no_double_trigger(self):
        """If breaker already active, don't pause again."""
        settings = MagicMock()
        settings.regime_adaptation_enabled = True
        settings.crisis_circuit_breaker_minutes = 30.0
        mgr = self._make_manager(settings=settings)
        detector = self._make_detector(is_crisis=True, duration=60.0)
        mgr._regime_detector = detector
        mgr._circuit_breaker_active = True  # Already active

        e1 = self._make_engine("funding_rate_arb")
        mgr._engines = {"funding_rate_arb": e1}

        await mgr._circuit_breaker_check()
        # Should NOT pause again
        e1.pause.assert_not_called()

    @pytest.mark.asyncio
    async def test_crisis_resolved_resumes_engines(self):
        """When CRISIS clears and breaker was active, resume paused engines."""
        settings = MagicMock()
        settings.regime_adaptation_enabled = True
        settings.crisis_circuit_breaker_minutes = 30.0
        mgr = self._make_manager(settings=settings)
        detector = self._make_detector(is_crisis=False)
        mgr._regime_detector = detector
        mgr._circuit_breaker_active = True
        mgr._circuit_breaker_paused_engines = {
            "funding_rate_arb",
            "grid_trading",
        }

        e1 = self._make_engine("funding_rate_arb", "paused")
        e2 = self._make_engine("grid_trading", "paused")
        mgr._engines = {"funding_rate_arb": e1, "grid_trading": e2}

        await mgr._circuit_breaker_check()

        assert mgr._circuit_breaker_active is False
        assert len(mgr._circuit_breaker_paused_engines) == 0
        e1.resume.assert_called_once()
        e2.resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_crisis_resolved_only_resumes_paused(self):
        """Only resume engines that are actually paused (not stopped/error)."""
        settings = MagicMock()
        settings.regime_adaptation_enabled = True
        settings.crisis_circuit_breaker_minutes = 30.0
        mgr = self._make_manager(settings=settings)
        detector = self._make_detector(is_crisis=False)
        mgr._regime_detector = detector
        mgr._circuit_breaker_active = True
        mgr._circuit_breaker_paused_engines = {
            "funding_rate_arb",
            "grid_trading",
        }

        e1 = self._make_engine("funding_rate_arb", "paused")
        e2 = self._make_engine("grid_trading", "stopped")
        mgr._engines = {"funding_rate_arb": e1, "grid_trading": e2}

        await mgr._circuit_breaker_check()

        e1.resume.assert_called_once()
        e2.resume.assert_not_called()

    @pytest.mark.asyncio
    async def test_not_crisis_no_breaker_noop(self):
        """Not CRISIS and breaker not active → no-op."""
        settings = MagicMock()
        settings.regime_adaptation_enabled = True
        mgr = self._make_manager(settings=settings)
        detector = self._make_detector(is_crisis=False)
        mgr._regime_detector = detector

        e1 = self._make_engine("funding_rate_arb")
        mgr._engines = {"funding_rate_arb": e1}

        await mgr._circuit_breaker_check()
        assert mgr._circuit_breaker_active is False
        e1.pause.assert_not_called()
        e1.resume.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_cycle_trigger_then_release(self):
        """Full cycle: normal → CRISIS 30m → trigger → CRISIS resolves → release."""
        settings = MagicMock()
        settings.regime_adaptation_enabled = True
        settings.crisis_circuit_breaker_minutes = 30.0
        mgr = self._make_manager(settings=settings)

        e1 = self._make_engine("funding_rate_arb", "running")
        mgr._engines = {"funding_rate_arb": e1}

        # Phase 1: CRISIS just started (5 min)
        detector = self._make_detector(is_crisis=True, duration=5.0)
        mgr._regime_detector = detector
        await mgr._circuit_breaker_check()
        assert mgr._circuit_breaker_active is False
        e1.pause.assert_not_called()

        # Phase 2: CRISIS at 30 min
        detector.get_regime_duration.return_value = 30.0
        await mgr._circuit_breaker_check()
        assert mgr._circuit_breaker_active is True
        e1.pause.assert_called_once()

        # Phase 3: CRISIS resolves
        detector.is_crisis.return_value = False
        e1.status.value = "paused"
        await mgr._circuit_breaker_check()
        assert mgr._circuit_breaker_active is False
        e1.resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_settings_uses_defaults(self):
        """No settings → uses default 30.0 minutes threshold."""
        mgr = self._make_manager(settings=None)
        detector = self._make_detector(is_crisis=True, duration=30.0)
        mgr._regime_detector = detector

        e1 = self._make_engine("funding_rate_arb")
        mgr._engines = {"funding_rate_arb": e1}

        await mgr._circuit_breaker_check()
        assert mgr._circuit_breaker_active is True


class TestCircuitBreakerLoop:
    """Tests for EngineManager._circuit_breaker_loop()."""

    @pytest.mark.asyncio
    async def test_loop_calls_check(self):
        """Loop calls _circuit_breaker_check periodically."""
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        pm.request_capital.return_value = 1000.0
        pm.get_max_allocation.return_value = 1000.0
        mgr = EngineManager(pm)
        mgr._regime_detector = MagicMock()
        mgr._regime_detector.is_crisis.return_value = False

        call_count = 0

        async def mock_check():
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise asyncio.CancelledError()

        mgr._circuit_breaker_check = mock_check

        with pytest.raises(asyncio.CancelledError):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await mgr._circuit_breaker_loop()

        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_loop_handles_exception(self):
        """Loop continues after exception in _circuit_breaker_check."""
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        pm.request_capital.return_value = 1000.0
        pm.get_max_allocation.return_value = 1000.0
        mgr = EngineManager(pm)

        call_count = 0

        async def mock_check():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("test error")
            raise asyncio.CancelledError()

        mgr._circuit_breaker_check = mock_check

        with pytest.raises(asyncio.CancelledError):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await mgr._circuit_breaker_loop()

        assert call_count >= 2


class TestStartBackgroundLoopsCircuitBreaker:
    """Tests for circuit breaker task in start_background_loops."""

    @pytest.mark.asyncio
    async def test_starts_circuit_breaker_when_enabled(self):
        """Circuit breaker loop starts when detector + adaptation enabled."""
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        pm.request_capital.return_value = 1000.0
        pm.get_max_allocation.return_value = 1000.0
        settings = MagicMock()
        settings.regime_adaptation_enabled = True
        settings.regime_detection_enabled = True
        settings.regime_detection_interval_seconds = 300.0
        settings.tuner_enabled = False
        settings.engine_rebalance_enabled = False
        settings.research_enabled = False
        settings.data_backfill_enabled = False
        settings.research_auto_deploy = False
        settings.metrics_persistence_enabled = False
        mgr = EngineManager(pm, settings=settings)
        mgr._regime_detector = MagicMock()
        mgr._regime_detector._detection_loop = AsyncMock()

        await mgr.start_background_loops()

        assert mgr._circuit_breaker_task is not None
        assert not mgr._circuit_breaker_task.done()
        # Clean up
        mgr._circuit_breaker_task.cancel()
        if mgr._regime_task:
            mgr._regime_task.cancel()
        try:
            await mgr._circuit_breaker_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_no_circuit_breaker_when_disabled(self):
        """Circuit breaker loop not started when adaptation disabled."""
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        pm.request_capital.return_value = 1000.0
        pm.get_max_allocation.return_value = 1000.0
        settings = MagicMock()
        settings.regime_adaptation_enabled = False
        settings.regime_detection_enabled = True
        settings.regime_detection_interval_seconds = 300.0
        settings.tuner_enabled = False
        settings.engine_rebalance_enabled = False
        settings.research_enabled = False
        settings.data_backfill_enabled = False
        settings.research_auto_deploy = False
        settings.metrics_persistence_enabled = False
        mgr = EngineManager(pm, settings=settings)
        mgr._regime_detector = MagicMock()
        mgr._regime_detector._detection_loop = AsyncMock()

        await mgr.start_background_loops()

        assert mgr._circuit_breaker_task is None
        # Clean up
        if mgr._regime_task:
            mgr._regime_task.cancel()

    @pytest.mark.asyncio
    async def test_no_circuit_breaker_without_detector(self):
        """Circuit breaker loop not started when no detector."""
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        pm.request_capital.return_value = 1000.0
        pm.get_max_allocation.return_value = 1000.0
        settings = MagicMock()
        settings.regime_adaptation_enabled = True
        settings.tuner_enabled = False
        settings.engine_rebalance_enabled = False
        settings.research_enabled = False
        settings.data_backfill_enabled = False
        settings.research_auto_deploy = False
        settings.metrics_persistence_enabled = False
        mgr = EngineManager(pm, settings=settings)
        # No detector set

        await mgr.start_background_loops()
        assert mgr._circuit_breaker_task is None


class TestConfigSettings:
    """Tests for regime_adaptation_enabled and crisis_circuit_breaker_minutes."""

    def test_defaults(self):
        from bot.config import Settings

        s = Settings()
        assert s.regime_adaptation_enabled is True
        assert s.crisis_circuit_breaker_minutes == 30.0

    def test_metadata_entries(self):
        from bot.config import SETTINGS_METADATA

        assert "regime_adaptation_enabled" in SETTINGS_METADATA
        meta_adapt = SETTINGS_METADATA["regime_adaptation_enabled"]
        assert meta_adapt["section"] == "Market Regime"
        assert meta_adapt["type"] == "bool"
        assert meta_adapt["requires_restart"] is False

        assert "crisis_circuit_breaker_minutes" in SETTINGS_METADATA
        meta_cb = SETTINGS_METADATA["crisis_circuit_breaker_minutes"]
        assert meta_cb["section"] == "Market Regime"
        assert meta_cb["type"] == "float"
        assert meta_cb["requires_restart"] is False


class TestMainWiring:
    """Tests for main.py wiring of regime_detector to engines."""

    def test_regime_detector_wired_to_engines(self):
        """Verify that regime_detector is set on individual engines."""
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        pm.request_capital.return_value = 1000.0
        pm.get_max_allocation.return_value = 1000.0
        mgr = EngineManager(pm)

        # Create mock engines
        engines = {}
        for name in ["funding_rate_arb", "grid_trading", "token_scanner"]:
            e = MagicMock()
            e.name = name
            e.set_regime_detector = MagicMock()
            engines[name] = e

        # Simulate main.py wiring pattern
        detector = MagicMock()
        mgr.set_regime_detector(detector)
        for engine in engines.values():
            if (
                hasattr(engine, "set_regime_detector")
                and engine.name != "token_scanner"
            ):
                engine.set_regime_detector(detector)

        # Verify
        engines["funding_rate_arb"].set_regime_detector.assert_called_once_with(
            detector,
        )
        engines["grid_trading"].set_regime_detector.assert_called_once_with(
            detector,
        )
        engines["token_scanner"].set_regime_detector.assert_not_called()
        assert mgr._regime_detector is detector
