"""Tests for MarketRegimeDetector — real-time regime classification + CRISIS."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bot.quant.volatility import VolatilityRegime
from bot.risk.regime_detector import MarketRegime, MarketRegimeDetector

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _mock_vol_service(
    regime: VolatilityRegime = VolatilityRegime.NORMAL,
    forecast: float | None = 0.02,
    cond_vol: np.ndarray | None = None,
) -> MagicMock:
    """Build mock VolatilityService."""
    svc = MagicMock()
    svc.get_market_regime.return_value = regime
    svc.get_forecast.return_value = forecast

    model = MagicMock()
    if cond_vol is not None:
        model.conditional_volatility = cond_vol
    else:
        model.conditional_volatility = np.array([0.01, 0.02, 0.015, 0.018])
    svc.get_model.return_value = model

    return svc


# ──────────────────────────────────────────────────────────────
# MarketRegime enum
# ──────────────────────────────────────────────────────────────


class TestMarketRegime:
    def test_enum_values(self):
        assert MarketRegime.LOW == "LOW"
        assert MarketRegime.NORMAL == "NORMAL"
        assert MarketRegime.HIGH == "HIGH"
        assert MarketRegime.CRISIS == "CRISIS"

    def test_enum_count(self):
        assert len(MarketRegime) == 4


# ──────────────────────────────────────────────────────────────
# detect_regime
# ──────────────────────────────────────────────────────────────


class TestDetectRegime:
    def test_no_vol_service(self):
        detector = MarketRegimeDetector(volatility_service=None)
        regime = detector.detect_regime()
        assert regime == MarketRegime.NORMAL

    def test_low_volatility(self):
        svc = _mock_vol_service(regime=VolatilityRegime.LOW)
        detector = MarketRegimeDetector(volatility_service=svc)
        regime = detector.detect_regime()
        assert regime == MarketRegime.LOW

    def test_normal_volatility(self):
        svc = _mock_vol_service(regime=VolatilityRegime.NORMAL)
        detector = MarketRegimeDetector(volatility_service=svc)
        regime = detector.detect_regime()
        assert regime == MarketRegime.NORMAL

    def test_high_volatility(self):
        svc = _mock_vol_service(regime=VolatilityRegime.HIGH)
        detector = MarketRegimeDetector(volatility_service=svc)
        regime = detector.detect_regime()
        assert regime == MarketRegime.HIGH

    def test_crisis_via_btc_24h_change(self):
        svc = _mock_vol_service(regime=VolatilityRegime.NORMAL)
        detector = MarketRegimeDetector(volatility_service=svc)
        detector.set_btc_24h_change(-12.0)
        regime = detector.detect_regime()
        assert regime == MarketRegime.CRISIS

    def test_crisis_via_volatility_spike(self):
        # forecast = 0.05, median_cond_vol = 0.0175, ratio ≈ 2.86 > 2.5
        svc = _mock_vol_service(
            regime=VolatilityRegime.HIGH,
            forecast=0.05,
            cond_vol=np.array([0.01, 0.02, 0.015, 0.02]),
        )
        detector = MarketRegimeDetector(
            volatility_service=svc, crisis_threshold=2.5
        )
        regime = detector.detect_regime()
        assert regime == MarketRegime.CRISIS

    def test_no_crisis_below_threshold(self):
        # forecast = 0.03, median_cond_vol = 0.0175, ratio ≈ 1.71 < 2.5
        svc = _mock_vol_service(
            regime=VolatilityRegime.HIGH,
            forecast=0.03,
            cond_vol=np.array([0.01, 0.02, 0.015, 0.02]),
        )
        detector = MarketRegimeDetector(
            volatility_service=svc, crisis_threshold=2.5
        )
        regime = detector.detect_regime()
        assert regime == MarketRegime.HIGH  # Not CRISIS

    def test_crisis_btc_exactly_minus_10(self):
        svc = _mock_vol_service(regime=VolatilityRegime.NORMAL)
        detector = MarketRegimeDetector(volatility_service=svc)
        detector.set_btc_24h_change(-10.0)
        regime = detector.detect_regime()
        assert regime == MarketRegime.CRISIS

    def test_no_crisis_btc_minus_9(self):
        svc = _mock_vol_service(regime=VolatilityRegime.NORMAL)
        detector = MarketRegimeDetector(volatility_service=svc)
        detector.set_btc_24h_change(-9.0)
        regime = detector.detect_regime()
        assert regime == MarketRegime.NORMAL

    def test_crisis_no_model(self):
        """With no model, only BTC change can trigger CRISIS."""
        svc = _mock_vol_service(regime=VolatilityRegime.HIGH)
        svc.get_model.return_value = None
        detector = MarketRegimeDetector(volatility_service=svc)
        regime = detector.detect_regime()
        assert regime == MarketRegime.HIGH  # No model = no vol crisis

    def test_crisis_no_forecast(self):
        svc = _mock_vol_service(regime=VolatilityRegime.NORMAL, forecast=None)
        detector = MarketRegimeDetector(volatility_service=svc)
        regime = detector.detect_regime()
        assert regime == MarketRegime.NORMAL


# ──────────────────────────────────────────────────────────────
# Regime history
# ──────────────────────────────────────────────────────────────


class TestRegimeHistory:
    def test_no_transition_no_history(self):
        svc = _mock_vol_service(regime=VolatilityRegime.NORMAL)
        detector = MarketRegimeDetector(volatility_service=svc)
        detector.detect_regime()
        assert detector.get_regime_history() == []

    def test_transition_records_history(self):
        svc = _mock_vol_service(regime=VolatilityRegime.NORMAL)
        detector = MarketRegimeDetector(volatility_service=svc)
        detector.detect_regime()  # stays NORMAL

        svc.get_market_regime.return_value = VolatilityRegime.HIGH
        detector.detect_regime()  # NORMAL → HIGH

        history = detector.get_regime_history()
        assert len(history) == 1
        assert history[0]["from_regime"] == "NORMAL"
        assert history[0]["to_regime"] == "HIGH"
        assert "timestamp" in history[0]
        assert "duration_minutes" in history[0]
        assert "trigger" in history[0]

    def test_multiple_transitions(self):
        svc = _mock_vol_service(regime=VolatilityRegime.NORMAL)
        detector = MarketRegimeDetector(volatility_service=svc)
        detector.detect_regime()

        svc.get_market_regime.return_value = VolatilityRegime.HIGH
        detector.detect_regime()

        svc.get_market_regime.return_value = VolatilityRegime.LOW
        detector.detect_regime()

        history = detector.get_regime_history()
        assert len(history) == 2
        assert history[0]["to_regime"] == "HIGH"
        assert history[1]["to_regime"] == "LOW"

    def test_history_capped_at_100(self):
        svc = _mock_vol_service(regime=VolatilityRegime.NORMAL)
        detector = MarketRegimeDetector(volatility_service=svc)

        regimes = [VolatilityRegime.LOW, VolatilityRegime.HIGH]
        for i in range(120):
            svc.get_market_regime.return_value = regimes[i % 2]
            detector.detect_regime()

        assert len(detector.get_regime_history()) <= 100

    def test_history_returns_copy(self):
        svc = _mock_vol_service(regime=VolatilityRegime.NORMAL)
        detector = MarketRegimeDetector(volatility_service=svc)
        h1 = detector.get_regime_history()
        h2 = detector.get_regime_history()
        assert h1 is not h2


# ──────────────────────────────────────────────────────────────
# get_current_regime & get_regime_duration & is_crisis
# ──────────────────────────────────────────────────────────────


class TestAccessors:
    def test_get_current_regime_default(self):
        detector = MarketRegimeDetector()
        assert detector.get_current_regime() == MarketRegime.NORMAL

    def test_get_current_regime_after_detect(self):
        svc = _mock_vol_service(regime=VolatilityRegime.HIGH)
        detector = MarketRegimeDetector(volatility_service=svc)
        detector.detect_regime()
        assert detector.get_current_regime() == MarketRegime.HIGH

    def test_get_regime_duration(self):
        detector = MarketRegimeDetector()
        # Just created, duration should be >= 0 and small
        assert detector.get_regime_duration() >= 0.0

    def test_is_crisis_false(self):
        detector = MarketRegimeDetector()
        assert detector.is_crisis() is False

    def test_is_crisis_true(self):
        svc = _mock_vol_service(regime=VolatilityRegime.NORMAL)
        detector = MarketRegimeDetector(volatility_service=svc)
        detector.set_btc_24h_change(-15.0)
        detector.detect_regime()
        assert detector.is_crisis() is True

    def test_set_btc_24h_change(self):
        detector = MarketRegimeDetector()
        detector.set_btc_24h_change(-5.0)
        assert detector._btc_24h_change == -5.0


# ──────────────────────────────────────────────────────────────
# _detection_loop
# ──────────────────────────────────────────────────────────────


class TestDetectionLoop:
    @pytest.mark.asyncio
    async def test_loop_calls_detect(self):
        svc = _mock_vol_service(regime=VolatilityRegime.NORMAL)
        detector = MarketRegimeDetector(volatility_service=svc)

        call_count = 0
        original_detect = detector.detect_regime

        def counting_detect():
            nonlocal call_count
            call_count += 1
            return original_detect()

        detector.detect_regime = counting_detect

        with patch("bot.risk.regime_detector.asyncio") as mock_asyncio:
            sleep_calls = 0

            async def mock_sleep(seconds):
                nonlocal sleep_calls
                sleep_calls += 1
                if sleep_calls > 1:
                    raise asyncio.CancelledError()

            mock_asyncio.sleep = mock_sleep
            mock_asyncio.CancelledError = asyncio.CancelledError

            with pytest.raises(asyncio.CancelledError):
                await detector._detection_loop(interval_seconds=60)

            assert call_count >= 1

    @pytest.mark.asyncio
    async def test_loop_handles_exception(self):
        detector = MarketRegimeDetector()

        def raising_detect():
            raise RuntimeError("detection failed")

        detector.detect_regime = raising_detect

        with patch("bot.risk.regime_detector.asyncio") as mock_asyncio:
            sleep_calls = 0

            async def mock_sleep(seconds):
                nonlocal sleep_calls
                sleep_calls += 1
                if sleep_calls > 1:
                    raise asyncio.CancelledError()

            mock_asyncio.sleep = mock_sleep
            mock_asyncio.CancelledError = asyncio.CancelledError

            with pytest.raises(asyncio.CancelledError):
                await detector._detection_loop(interval_seconds=60)


# ──────────────────────────────────────────────────────────────
# Trigger reason
# ──────────────────────────────────────────────────────────────


class TestTriggerReason:
    def test_crisis_trigger_btc(self):
        svc = _mock_vol_service(regime=VolatilityRegime.NORMAL)
        detector = MarketRegimeDetector(volatility_service=svc)
        detector.set_btc_24h_change(-12.0)
        detector.detect_regime()
        history = detector.get_regime_history()
        assert len(history) == 1
        assert "BTC 24h change" in history[0]["trigger"]

    def test_non_crisis_trigger(self):
        svc = _mock_vol_service(regime=VolatilityRegime.HIGH)
        detector = MarketRegimeDetector(volatility_service=svc)
        detector.detect_regime()
        history = detector.get_regime_history()
        assert len(history) == 1
        assert "volatility regime" in history[0]["trigger"]


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────


class TestConfig:
    def test_config_defaults(self):
        from bot.config import Settings

        s = Settings(
            _env_file=None,
            exchange="binance",
            api_key="test",
            api_secret="test",
        )
        assert s.regime_detection_enabled is True
        assert s.regime_crisis_threshold == 2.5
        assert s.regime_detection_interval_seconds == 300.0

    def test_settings_metadata(self):
        from bot.config import SETTINGS_METADATA

        assert "regime_detection_enabled" in SETTINGS_METADATA
        assert "regime_crisis_threshold" in SETTINGS_METADATA
        assert "regime_detection_interval_seconds" in SETTINGS_METADATA
        for key in [
            "regime_detection_enabled",
            "regime_crisis_threshold",
            "regime_detection_interval_seconds",
        ]:
            assert SETTINGS_METADATA[key]["section"] == "Market Regime"


# ──────────────────────────────────────────────────────────────
# EngineManager integration
# ──────────────────────────────────────────────────────────────


class TestEngineManagerIntegration:
    def test_set_regime_detector(self):
        from bot.engines.manager import EngineManager
        from bot.engines.portfolio_manager import PortfolioManager

        pm = PortfolioManager(total_capital=10000)
        mgr = EngineManager(pm)

        detector = MarketRegimeDetector()
        mgr.set_regime_detector(detector)
        assert mgr._regime_detector is detector

    def test_default_none(self):
        from bot.engines.manager import EngineManager
        from bot.engines.portfolio_manager import PortfolioManager

        pm = PortfolioManager(total_capital=10000)
        mgr = EngineManager(pm)
        assert mgr._regime_detector is None


# ──────────────────────────────────────────────────────────────
# Dashboard endpoint
# ──────────────────────────────────────────────────────────────


class TestDashboardEndpoint:
    @pytest.mark.asyncio
    async def test_no_engine_manager(self):
        from bot.dashboard.app import get_market_regime, set_engine_manager

        set_engine_manager(None)
        result = await get_market_regime()
        assert result["current"] == "NORMAL"
        assert result["history"] == []

    @pytest.mark.asyncio
    async def test_no_detector(self):
        from bot.dashboard.app import get_market_regime, set_engine_manager

        mgr = MagicMock(spec=[])
        set_engine_manager(mgr)
        result = await get_market_regime()
        assert result["current"] == "NORMAL"

    @pytest.mark.asyncio
    async def test_with_detector(self):
        from bot.dashboard.app import get_market_regime, set_engine_manager

        mgr = MagicMock()
        svc = _mock_vol_service(regime=VolatilityRegime.HIGH)
        detector = MarketRegimeDetector(volatility_service=svc)
        detector.detect_regime()
        mgr._regime_detector = detector
        set_engine_manager(mgr)

        result = await get_market_regime()
        assert result["current"] == "HIGH"
        assert "since" in result
        assert "duration_minutes" in result
        assert isinstance(result["history"], list)


# ──────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────


class TestImports:
    def test_import_from_risk_package(self):
        from bot.risk import MarketRegime, MarketRegimeDetector

        assert MarketRegime is not None
        assert MarketRegimeDetector is not None
