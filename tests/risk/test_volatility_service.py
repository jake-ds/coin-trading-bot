"""Tests for VolatilityService — GARCH-based volatility forecasting."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from bot.quant.volatility import VolatilityRegime
from bot.risk.volatility_service import VolatilityService


def _make_data_provider(
    returns: list[float] | None = None,
    prices: list[float] | None = None,
) -> MagicMock:
    """Create a mock HistoricalDataProvider."""
    provider = MagicMock()
    if returns is not None:
        provider.get_returns = AsyncMock(return_value=returns)
    else:
        provider.get_returns = AsyncMock(return_value=[])
    if prices is not None:
        provider.get_prices = AsyncMock(return_value=prices)
    else:
        provider.get_prices = AsyncMock(return_value=[])
    return provider


def _generate_returns(n: int = 200, seed: int = 42) -> list[float]:
    """Generate synthetic return series for testing."""
    rng = np.random.RandomState(seed)
    return list(rng.normal(0.0, 0.02, n))


class TestVolatilityServiceInit:
    """Test construction and defaults."""

    def test_init_default(self):
        svc = VolatilityService()
        assert svc._data_provider is None
        assert svc._models == {}
        assert svc._forecasts == {}
        assert svc._regimes == {}
        assert svc._last_fit == {}

    def test_init_with_provider(self):
        provider = MagicMock()
        svc = VolatilityService(data_provider=provider)
        assert svc._data_provider is provider


class TestFitSymbol:
    """Test fit_symbol method."""

    @pytest.mark.asyncio
    async def test_fit_symbol_success(self):
        returns = _generate_returns(200)
        provider = _make_data_provider(returns=returns)
        svc = VolatilityService(data_provider=provider)

        success = await svc.fit_symbol("BTC/USDT")
        assert success is True
        assert "BTC/USDT" in svc._models
        assert "BTC/USDT" in svc._forecasts
        assert "BTC/USDT" in svc._regimes
        assert "BTC/USDT" in svc._last_fit
        assert isinstance(svc._forecasts["BTC/USDT"], float)
        assert isinstance(svc._regimes["BTC/USDT"], VolatilityRegime)

        # Verify data_provider was called with correct args
        provider.get_returns.assert_awaited_once_with(
            "BTC/USDT", timeframe="1h", lookback_days=60
        )

    @pytest.mark.asyncio
    async def test_fit_symbol_custom_lookback(self):
        returns = _generate_returns(100)
        provider = _make_data_provider(returns=returns)
        svc = VolatilityService(data_provider=provider)

        await svc.fit_symbol("ETH/USDT", lookback_days=30)
        provider.get_returns.assert_awaited_once_with(
            "ETH/USDT", timeframe="1h", lookback_days=30
        )

    @pytest.mark.asyncio
    async def test_fit_symbol_no_provider(self):
        svc = VolatilityService(data_provider=None)
        success = await svc.fit_symbol("BTC/USDT")
        assert success is False
        assert "BTC/USDT" not in svc._models

    @pytest.mark.asyncio
    async def test_fit_symbol_insufficient_data(self):
        provider = _make_data_provider(returns=[0.01, 0.02])
        svc = VolatilityService(data_provider=provider)

        success = await svc.fit_symbol("BTC/USDT")
        assert success is False
        assert "BTC/USDT" not in svc._models

    @pytest.mark.asyncio
    async def test_fit_symbol_empty_returns(self):
        provider = _make_data_provider(returns=[])
        svc = VolatilityService(data_provider=provider)

        success = await svc.fit_symbol("BTC/USDT")
        assert success is False

    @pytest.mark.asyncio
    async def test_fit_symbol_garch_failure(self):
        """When GARCHModel.fit() returns success=False."""
        returns = _generate_returns(200)
        provider = _make_data_provider(returns=returns)
        svc = VolatilityService(data_provider=provider)

        with patch("bot.risk.volatility_service.GARCHModel") as MockGARCH:
            mock_model = MagicMock()
            mock_model.fit.return_value = {"success": False}
            MockGARCH.return_value = mock_model

            success = await svc.fit_symbol("BTC/USDT")
            assert success is False

    @pytest.mark.asyncio
    async def test_fit_symbol_exception(self):
        """When data_provider raises an exception."""
        provider = MagicMock()
        provider.get_returns = AsyncMock(side_effect=RuntimeError("DB error"))
        svc = VolatilityService(data_provider=provider)

        success = await svc.fit_symbol("BTC/USDT")
        assert success is False

    @pytest.mark.asyncio
    async def test_fit_symbol_updates_last_fit_time(self):
        returns = _generate_returns(200)
        provider = _make_data_provider(returns=returns)
        svc = VolatilityService(data_provider=provider)

        before = datetime.now(timezone.utc)
        await svc.fit_symbol("BTC/USDT")
        after = datetime.now(timezone.utc)

        fit_time = svc._last_fit["BTC/USDT"]
        assert before <= fit_time <= after


class TestFitAll:
    """Test fit_all method."""

    @pytest.mark.asyncio
    async def test_fit_all_success(self):
        returns = _generate_returns(200)
        provider = _make_data_provider(returns=returns)
        svc = VolatilityService(data_provider=provider)

        results = await svc.fit_all(["BTC/USDT", "ETH/USDT"])
        assert results["BTC/USDT"] is True
        assert results["ETH/USDT"] is True
        assert len(svc._models) == 2

    @pytest.mark.asyncio
    async def test_fit_all_partial_failure(self):
        """Some symbols succeed, some fail."""
        call_count = 0

        async def mock_get_returns(symbol, **kwargs):
            nonlocal call_count
            call_count += 1
            if symbol == "FAIL/USDT":
                return [0.01]  # Too few returns
            return _generate_returns(200)

        provider = MagicMock()
        provider.get_returns = AsyncMock(side_effect=mock_get_returns)
        svc = VolatilityService(data_provider=provider)

        results = await svc.fit_all(["BTC/USDT", "FAIL/USDT", "ETH/USDT"])
        assert results["BTC/USDT"] is True
        assert results["FAIL/USDT"] is False
        assert results["ETH/USDT"] is True
        assert len(svc._models) == 2

    @pytest.mark.asyncio
    async def test_fit_all_empty_list(self):
        svc = VolatilityService(data_provider=MagicMock())
        results = await svc.fit_all([])
        assert results == {}


class TestGetForecast:
    """Test get_forecast method."""

    @pytest.mark.asyncio
    async def test_get_forecast_after_fit(self):
        returns = _generate_returns(200)
        provider = _make_data_provider(returns=returns)
        svc = VolatilityService(data_provider=provider)

        await svc.fit_symbol("BTC/USDT")
        forecast = svc.get_forecast("BTC/USDT")
        assert forecast is not None
        assert isinstance(forecast, float)
        assert forecast > 0

    def test_get_forecast_not_fitted(self):
        svc = VolatilityService()
        assert svc.get_forecast("BTC/USDT") is None

    def test_get_forecast_unknown_symbol(self):
        svc = VolatilityService()
        assert svc.get_forecast("UNKNOWN/USDT") is None

    @pytest.mark.asyncio
    async def test_get_forecast_horizon_greater_than_1(self):
        returns = _generate_returns(200)
        provider = _make_data_provider(returns=returns)
        svc = VolatilityService(data_provider=provider)

        await svc.fit_symbol("BTC/USDT")
        forecast_h5 = svc.get_forecast("BTC/USDT", horizon=5)
        assert forecast_h5 is not None
        assert isinstance(forecast_h5, float)

    def test_get_forecast_horizon_no_model(self):
        svc = VolatilityService()
        assert svc.get_forecast("BTC/USDT", horizon=5) is None


class TestGetRegime:
    """Test get_regime and related methods."""

    @pytest.mark.asyncio
    async def test_get_regime_after_fit(self):
        returns = _generate_returns(200)
        provider = _make_data_provider(returns=returns)
        svc = VolatilityService(data_provider=provider)

        await svc.fit_symbol("BTC/USDT")
        regime = svc.get_regime("BTC/USDT")
        assert isinstance(regime, VolatilityRegime)

    def test_get_regime_not_fitted(self):
        svc = VolatilityService()
        assert svc.get_regime("BTC/USDT") == VolatilityRegime.NORMAL

    def test_get_regime_unknown_symbol(self):
        svc = VolatilityService()
        assert svc.get_regime("UNKNOWN") == VolatilityRegime.NORMAL

    @pytest.mark.asyncio
    async def test_get_all_regimes(self):
        returns = _generate_returns(200)
        provider = _make_data_provider(returns=returns)
        svc = VolatilityService(data_provider=provider)

        await svc.fit_all(["BTC/USDT", "ETH/USDT"])
        regimes = svc.get_all_regimes()
        assert isinstance(regimes, dict)
        assert "BTC/USDT" in regimes
        assert "ETH/USDT" in regimes
        # Should be a copy, not the internal dict
        regimes["NEW"] = VolatilityRegime.HIGH
        assert "NEW" not in svc._regimes

    def test_get_all_regimes_empty(self):
        svc = VolatilityService()
        assert svc.get_all_regimes() == {}


class TestGetMarketRegime:
    """Test get_market_regime method."""

    @pytest.mark.asyncio
    async def test_market_regime_with_btc(self):
        returns = _generate_returns(200)
        provider = _make_data_provider(returns=returns)
        svc = VolatilityService(data_provider=provider)

        await svc.fit_symbol("BTC/USDT")
        regime = svc.get_market_regime()
        assert isinstance(regime, VolatilityRegime)

    def test_market_regime_no_btc(self):
        svc = VolatilityService()
        assert svc.get_market_regime() == VolatilityRegime.NORMAL

    @pytest.mark.asyncio
    async def test_market_regime_only_eth(self):
        """When only ETH is fitted, BTC regime is still NORMAL."""
        returns = _generate_returns(200)
        provider = _make_data_provider(returns=returns)
        svc = VolatilityService(data_provider=provider)

        await svc.fit_symbol("ETH/USDT")
        assert svc.get_market_regime() == VolatilityRegime.NORMAL


class TestNeedsRefit:
    """Test needs_refit method."""

    def test_needs_refit_never_fitted(self):
        svc = VolatilityService()
        assert svc.needs_refit("BTC/USDT") is True

    @pytest.mark.asyncio
    async def test_needs_refit_just_fitted(self):
        returns = _generate_returns(200)
        provider = _make_data_provider(returns=returns)
        svc = VolatilityService(data_provider=provider)

        await svc.fit_symbol("BTC/USDT")
        assert svc.needs_refit("BTC/USDT") is False

    def test_needs_refit_expired(self):
        svc = VolatilityService()
        svc._last_fit["BTC/USDT"] = datetime.now(timezone.utc) - timedelta(hours=7)
        assert svc.needs_refit("BTC/USDT", max_age_hours=6.0) is True

    def test_needs_refit_not_expired(self):
        svc = VolatilityService()
        svc._last_fit["BTC/USDT"] = datetime.now(timezone.utc) - timedelta(hours=2)
        assert svc.needs_refit("BTC/USDT", max_age_hours=6.0) is False

    def test_needs_refit_custom_max_age(self):
        svc = VolatilityService()
        svc._last_fit["BTC/USDT"] = datetime.now(timezone.utc) - timedelta(hours=1.5)
        assert svc.needs_refit("BTC/USDT", max_age_hours=1.0) is True
        assert svc.needs_refit("BTC/USDT", max_age_hours=2.0) is False


class TestGetModel:
    """Test get_model method."""

    @pytest.mark.asyncio
    async def test_get_model_after_fit(self):
        returns = _generate_returns(200)
        provider = _make_data_provider(returns=returns)
        svc = VolatilityService(data_provider=provider)

        await svc.fit_symbol("BTC/USDT")
        model = svc.get_model("BTC/USDT")
        assert model is not None
        assert model.is_fitted

    def test_get_model_not_fitted(self):
        svc = VolatilityService()
        assert svc.get_model("BTC/USDT") is None


class TestGracefulDegradation:
    """Test behavior when data_provider is None."""

    def test_no_provider_forecast_none(self):
        svc = VolatilityService(data_provider=None)
        assert svc.get_forecast("BTC/USDT") is None

    def test_no_provider_regime_normal(self):
        svc = VolatilityService(data_provider=None)
        assert svc.get_regime("BTC/USDT") == VolatilityRegime.NORMAL

    def test_no_provider_market_regime_normal(self):
        svc = VolatilityService(data_provider=None)
        assert svc.get_market_regime() == VolatilityRegime.NORMAL

    def test_no_provider_all_regimes_empty(self):
        svc = VolatilityService(data_provider=None)
        assert svc.get_all_regimes() == {}

    @pytest.mark.asyncio
    async def test_no_provider_fit_symbol_fails(self):
        svc = VolatilityService(data_provider=None)
        success = await svc.fit_symbol("BTC/USDT")
        assert success is False

    @pytest.mark.asyncio
    async def test_no_provider_fit_all_fails(self):
        svc = VolatilityService(data_provider=None)
        results = await svc.fit_all(["BTC/USDT", "ETH/USDT"])
        assert results == {"BTC/USDT": False, "ETH/USDT": False}


class TestFitLoop:
    """Test _fit_loop background task."""

    @pytest.mark.asyncio
    async def test_fit_loop_executes_fit_all(self):
        returns = _generate_returns(200)
        provider = _make_data_provider(returns=returns)
        svc = VolatilityService(data_provider=provider)

        # Patch asyncio.sleep to avoid actual waiting
        sleep_path = "bot.risk.volatility_service.asyncio.sleep"
        with patch(sleep_path, new_callable=AsyncMock) as mock_sleep:
            # Make sleep raise after first iteration to break the loop
            call_count = 0

            async def limited_sleep(seconds):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    raise asyncio.CancelledError()

            mock_sleep.side_effect = limited_sleep

            with pytest.raises(asyncio.CancelledError):
                await svc._fit_loop(["BTC/USDT"], interval_hours=1.0)

            # First sleep is 60s (initial delay), second would be interval
            assert mock_sleep.await_count == 2
            assert "BTC/USDT" in svc._models

    @pytest.mark.asyncio
    async def test_fit_loop_handles_exceptions(self):
        """Loop should continue even if fit_all fails."""
        provider = MagicMock()
        provider.get_returns = AsyncMock(side_effect=RuntimeError("DB down"))
        svc = VolatilityService(data_provider=provider)

        sleep_path = "bot.risk.volatility_service.asyncio.sleep"
        with patch(sleep_path, new_callable=AsyncMock) as mock_sleep:
            call_count = 0

            async def limited_sleep(seconds):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    raise asyncio.CancelledError()

            mock_sleep.side_effect = limited_sleep

            with pytest.raises(asyncio.CancelledError):
                await svc._fit_loop(["BTC/USDT"], interval_hours=1.0)

            # Should have attempted despite error
            assert mock_sleep.await_count == 2


class TestImport:
    """Test that VolatilityService is properly exported."""

    def test_import_from_risk_package(self):
        from bot.risk import VolatilityService as VS
        assert VS is VolatilityService

    def test_import_volatility_regime(self):
        from bot.quant.volatility import VolatilityRegime
        assert VolatilityRegime.LOW.value == "LOW"
        assert VolatilityRegime.NORMAL.value == "NORMAL"
        assert VolatilityRegime.HIGH.value == "HIGH"
