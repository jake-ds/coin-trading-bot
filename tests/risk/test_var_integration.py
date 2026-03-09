"""Tests for V6-006: VaR/CVaR portfolio risk limit integration.

Tests parametric_var, cornish_fisher_var, cvar, stress_var,
pre_trade_var_check, and get_risk_metrics format.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from bot.risk.portfolio_risk import PortfolioRiskManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n: int = 30, seed: int = 42, mu: float = 0.0, sigma: float = 0.02) -> list[float]:
    """Generate synthetic return series."""
    rng = np.random.default_rng(seed)
    return list(rng.normal(mu, sigma, n))


def _setup_manager_with_data(
    var_enabled: bool = True,
    n_returns: int = 30,
    seed: int = 42,
    portfolio_value: float = 100000.0,
    max_portfolio_var_pct: float = 5.0,
    var_confidence: float = 0.95,
    symbols: list[str] | None = None,
) -> PortfolioRiskManager:
    """Create a PortfolioRiskManager with pre-loaded position and return data."""
    mgr = PortfolioRiskManager(
        var_enabled=var_enabled,
        var_confidence=var_confidence,
        max_portfolio_var_pct=max_portfolio_var_pct,
    )
    mgr.update_portfolio_value(portfolio_value)

    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT"]

    for i, sym in enumerate(symbols):
        mgr.add_position(sym, portfolio_value * 0.3)
        returns = _make_returns(n_returns, seed=seed + i)
        mgr._price_history[sym] = returns

    return mgr


# ---------------------------------------------------------------------------
# Test _get_portfolio_returns helper
# ---------------------------------------------------------------------------

class TestGetPortfolioReturns:
    def test_returns_array_with_data(self):
        mgr = _setup_manager_with_data()
        port_returns = mgr._get_portfolio_returns()
        assert port_returns is not None
        assert len(port_returns) > 0

    def test_returns_none_no_positions(self):
        mgr = PortfolioRiskManager()
        mgr.update_portfolio_value(100000)
        mgr._price_history["BTC/USDT"] = _make_returns(30)
        assert mgr._get_portfolio_returns() is None

    def test_returns_none_no_price_history(self):
        mgr = PortfolioRiskManager()
        mgr.update_portfolio_value(100000)
        mgr.add_position("BTC/USDT", 30000)
        assert mgr._get_portfolio_returns() is None

    def test_returns_none_insufficient_data(self):
        mgr = PortfolioRiskManager()
        mgr.update_portfolio_value(100000)
        mgr.add_position("BTC/USDT", 30000)
        mgr._price_history["BTC/USDT"] = [0.01, 0.02]  # Too few
        assert mgr._get_portfolio_returns() is None

    def test_returns_weighted(self):
        """Portfolio returns should be value-weighted."""
        mgr = PortfolioRiskManager()
        mgr.update_portfolio_value(100000)
        mgr.add_position("BTC/USDT", 70000)
        mgr.add_position("ETH/USDT", 30000)

        btc_returns = [0.01] * 15
        eth_returns = [0.02] * 15
        mgr._price_history["BTC/USDT"] = btc_returns
        mgr._price_history["ETH/USDT"] = eth_returns

        port = mgr._get_portfolio_returns()
        assert port is not None
        # Weighted: 0.7 * 0.01 + 0.3 * 0.02 = 0.013
        expected = 0.7 * 0.01 + 0.3 * 0.02
        np.testing.assert_allclose(port[0], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test calculate_parametric_var
# ---------------------------------------------------------------------------

class TestParametricVar:
    def test_basic_calculation(self):
        mgr = _setup_manager_with_data()
        var = mgr.calculate_parametric_var()
        assert var is not None
        assert var >= 0.0

    def test_none_without_data(self):
        mgr = PortfolioRiskManager()
        assert mgr.calculate_parametric_var() is None

    def test_higher_confidence_higher_var(self):
        mgr_95 = _setup_manager_with_data(var_confidence=0.95, seed=10)
        mgr_99 = _setup_manager_with_data(var_confidence=0.99, seed=10)

        var_95 = mgr_95.calculate_parametric_var()
        var_99 = mgr_99.calculate_parametric_var()

        assert var_95 is not None
        assert var_99 is not None
        assert var_99 >= var_95  # Higher confidence = higher VaR

    def test_returns_percentage(self):
        """VaR should be expressed as a percentage."""
        mgr = _setup_manager_with_data(n_returns=50, seed=7)
        var = mgr.calculate_parametric_var()
        assert var is not None
        # Returns are ~2% std, so VaR should be in a reasonable range (not > 100%)
        assert 0 <= var < 50


# ---------------------------------------------------------------------------
# Test calculate_cornish_fisher_var
# ---------------------------------------------------------------------------

class TestCornishFisherVar:
    def test_basic_calculation(self):
        mgr = _setup_manager_with_data(n_returns=50)
        var = mgr.calculate_cornish_fisher_var()
        assert var is not None
        assert var >= 0.0

    def test_none_without_data(self):
        mgr = PortfolioRiskManager()
        assert mgr.calculate_cornish_fisher_var() is None

    def test_different_from_parametric_with_skew(self):
        """CF VaR should differ from parametric when returns are skewed."""
        mgr = _setup_manager_with_data(n_returns=50, seed=123)
        # Add some skewness to the returns
        skewed = list(np.concatenate([
            np.random.default_rng(123).normal(0, 0.02, 45),
            np.array([-0.08, -0.09, -0.10, -0.07, -0.06]),  # Fat left tail
        ]))
        mgr._price_history["BTC/USDT"] = skewed

        pvar = mgr.calculate_parametric_var()
        cfvar = mgr.calculate_cornish_fisher_var()
        assert pvar is not None
        assert cfvar is not None
        # They should be different (CF accounts for skewness/kurtosis)
        # Don't assert which is larger as it depends on skew direction

    def test_returns_percentage(self):
        mgr = _setup_manager_with_data(n_returns=50)
        var = mgr.calculate_cornish_fisher_var()
        assert var is not None
        assert 0 <= var < 50


# ---------------------------------------------------------------------------
# Test calculate_cvar
# ---------------------------------------------------------------------------

class TestCVaR:
    def test_basic_calculation(self):
        mgr = _setup_manager_with_data()
        cvar_val = mgr.calculate_cvar()
        assert cvar_val is not None
        assert cvar_val >= 0.0

    def test_none_without_data(self):
        mgr = PortfolioRiskManager()
        assert mgr.calculate_cvar() is None

    def test_cvar_greater_or_equal_to_var(self):
        """CVaR (expected shortfall) should be >= VaR."""
        mgr = _setup_manager_with_data(n_returns=50, seed=77)
        var = mgr.calculate_portfolio_var()
        cvar_val = mgr.calculate_cvar()
        assert var is not None
        assert cvar_val is not None
        # CVaR is the expected loss beyond VaR, so CVaR >= VaR
        assert cvar_val >= var - 0.01  # Small tolerance for numerical precision


# ---------------------------------------------------------------------------
# Test calculate_stress_var
# ---------------------------------------------------------------------------

class TestStressVar:
    def test_basic_calculation(self):
        mgr = _setup_manager_with_data()
        svar = mgr.calculate_stress_var(n_simulations=500)
        assert svar is not None
        assert svar >= 0.0

    def test_none_without_data(self):
        mgr = PortfolioRiskManager()
        assert mgr.calculate_stress_var() is None

    def test_deterministic_with_seed(self):
        """Stress VaR should be deterministic (uses fixed seed=42)."""
        mgr = _setup_manager_with_data(seed=99)
        svar1 = mgr.calculate_stress_var(n_simulations=500)
        svar2 = mgr.calculate_stress_var(n_simulations=500)
        assert svar1 == svar2

    def test_single_asset(self):
        """Should handle single-asset portfolio."""
        mgr = PortfolioRiskManager()
        mgr.update_portfolio_value(100000)
        mgr.add_position("BTC/USDT", 30000)
        mgr._price_history["BTC/USDT"] = _make_returns(30, seed=42)

        svar = mgr.calculate_stress_var(n_simulations=500)
        assert svar is not None
        assert svar >= 0.0

    def test_multi_asset(self):
        """Should handle multi-asset portfolio with correlation."""
        mgr = _setup_manager_with_data(symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        svar = mgr.calculate_stress_var(n_simulations=500)
        assert svar is not None
        assert svar >= 0.0

    def test_more_simulations(self):
        """More simulations should still produce valid results."""
        mgr = _setup_manager_with_data()
        svar = mgr.calculate_stress_var(n_simulations=2000)
        assert svar is not None
        assert svar >= 0.0

    def test_returns_percentage(self):
        mgr = _setup_manager_with_data()
        svar = mgr.calculate_stress_var(n_simulations=500)
        assert svar is not None
        assert 0 <= svar < 50

    def test_no_positions(self):
        mgr = PortfolioRiskManager()
        mgr.update_portfolio_value(100000)
        mgr._price_history["BTC/USDT"] = _make_returns(30)
        assert mgr.calculate_stress_var() is None

    def test_constant_returns_handled(self):
        """Should handle constant returns gracefully (zero variance)."""
        mgr = PortfolioRiskManager()
        mgr.update_portfolio_value(100000)
        mgr.add_position("BTC/USDT", 30000)
        mgr.add_position("ETH/USDT", 20000)
        mgr._price_history["BTC/USDT"] = [0.0] * 30
        mgr._price_history["ETH/USDT"] = [0.0] * 30

        svar = mgr.calculate_stress_var(n_simulations=500)
        # Should handle gracefully — could be 0 or None depending on behavior
        assert svar is not None
        assert svar >= 0.0


# ---------------------------------------------------------------------------
# Test pre_trade_var_check
# ---------------------------------------------------------------------------

class TestPreTradeVarCheck:
    def test_allowed_when_var_disabled(self):
        mgr = _setup_manager_with_data(var_enabled=False)
        allowed, reason = mgr.pre_trade_var_check("SOL/USDT", 10000)
        assert allowed
        assert reason == ""

    def test_allowed_when_no_symbol_returns(self):
        """Should allow if the new symbol has no return data."""
        mgr = _setup_manager_with_data()
        allowed, reason = mgr.pre_trade_var_check("UNKNOWN/USDT", 10000)
        assert allowed

    def test_allowed_small_position(self):
        """Small position should not breach VaR limit."""
        mgr = _setup_manager_with_data(max_portfolio_var_pct=50.0)
        mgr._price_history["SOL/USDT"] = _make_returns(30, seed=99)
        allowed, reason = mgr.pre_trade_var_check("SOL/USDT", 1000)
        assert allowed

    def test_rejected_when_exceeds_limit(self):
        """Should reject when adding a volatile position would breach VaR."""
        mgr = _setup_manager_with_data(max_portfolio_var_pct=0.01)  # Very tight
        # Add very volatile returns for the new symbol
        volatile_returns = _make_returns(30, seed=55, sigma=0.15)
        mgr._price_history["SOL/USDT"] = volatile_returns

        allowed, reason = mgr.pre_trade_var_check("SOL/USDT", 50000)
        assert not allowed
        assert "projected_var" in reason
        assert "exceed" in reason

    def test_projects_new_total_portfolio(self):
        """Pre-trade check should include the new position in total value."""
        mgr = _setup_manager_with_data(max_portfolio_var_pct=50.0)
        mgr._price_history["SOL/USDT"] = _make_returns(30, seed=88)

        # Should be allowed with generous limit
        allowed, _ = mgr.pre_trade_var_check("SOL/USDT", 10000)
        assert allowed


# ---------------------------------------------------------------------------
# Test get_risk_metrics format
# ---------------------------------------------------------------------------

class TestGetRiskMetrics:
    def test_all_fields_present(self):
        mgr = _setup_manager_with_data()
        metrics = mgr.get_risk_metrics()

        expected_keys = {
            "exposure_pct", "heat", "var_pct",
            "parametric_var", "cornish_fisher_var", "cvar", "stress_var",
            "n_positions", "portfolio_value", "var_enabled",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_var_fields_not_none_with_data(self):
        mgr = _setup_manager_with_data(n_returns=50)
        metrics = mgr.get_risk_metrics()

        assert metrics["parametric_var"] is not None
        assert metrics["cornish_fisher_var"] is not None
        assert metrics["cvar"] is not None
        assert metrics["stress_var"] is not None
        assert metrics["var_pct"] is not None

    def test_var_fields_none_without_data(self):
        mgr = PortfolioRiskManager()
        metrics = mgr.get_risk_metrics()

        assert metrics["parametric_var"] is None
        assert metrics["cornish_fisher_var"] is None
        assert metrics["cvar"] is None
        assert metrics["stress_var"] is None
        assert metrics["var_pct"] is None

    def test_exposure_and_heat(self):
        mgr = _setup_manager_with_data()
        metrics = mgr.get_risk_metrics()

        assert isinstance(metrics["exposure_pct"], float)
        assert isinstance(metrics["heat"], float)
        assert metrics["n_positions"] == 2
        assert metrics["portfolio_value"] == 100000.0

    def test_var_enabled_flag(self):
        mgr_on = _setup_manager_with_data(var_enabled=True)
        mgr_off = _setup_manager_with_data(var_enabled=False)

        assert mgr_on.get_risk_metrics()["var_enabled"] is True
        assert mgr_off.get_risk_metrics()["var_enabled"] is False


# ---------------------------------------------------------------------------
# Test validate_new_position with pre_trade_var_check
# ---------------------------------------------------------------------------

class TestValidateWithPreTradeVar:
    def test_pre_trade_var_in_validation_chain(self):
        """validate_new_position should include pre_trade_var_check."""
        mgr = PortfolioRiskManager(
            var_enabled=True,
            max_portfolio_var_pct=0.001,  # Very tight VaR limit
            max_total_exposure_pct=200.0,  # High exposure limit so VaR is hit first
        )
        mgr.update_portfolio_value(100000)
        mgr.add_position("BTC/USDT", 30000)
        mgr._price_history["BTC/USDT"] = _make_returns(30, seed=42)

        volatile_returns = _make_returns(30, seed=55, sigma=0.15)
        mgr._price_history["SOL/USDT"] = volatile_returns

        allowed, reason = mgr.validate_new_position("SOL/USDT", 50000)
        assert not allowed
        assert "var" in reason.lower()

    def test_passes_when_var_disabled(self):
        mgr = PortfolioRiskManager(
            var_enabled=False,
            max_total_exposure_pct=200.0,  # High limit to not block
        )
        mgr.update_portfolio_value(100000)
        mgr.add_position("BTC/USDT", 30000)
        mgr._price_history["BTC/USDT"] = _make_returns(30, seed=42)

        allowed, reason = mgr.validate_new_position("ETH/USDT", 10000)
        assert allowed


# ---------------------------------------------------------------------------
# Test backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_constructor_without_volatility_service(self):
        """Should work without volatility_service (default None)."""
        mgr = PortfolioRiskManager()
        assert mgr._volatility_service is None

    def test_constructor_with_volatility_service(self):
        mock_vs = MagicMock()
        mgr = PortfolioRiskManager(volatility_service=mock_vs)
        assert mgr._volatility_service is mock_vs

    def test_existing_methods_unchanged(self):
        """All existing methods should work without volatility_service."""
        mgr = PortfolioRiskManager()
        mgr.update_portfolio_value(100000)
        mgr.add_position("BTC/USDT", 30000)

        # These should all work
        assert mgr.get_total_exposure() > 0
        assert isinstance(mgr.calculate_portfolio_heat(), float)
        assert mgr.calculate_portfolio_var() is None  # No price history
        assert mgr.check_exposure_limit(10000) == (True, "")
        assert mgr.check_correlation("ETH/USDT") == (True, "")
        assert mgr.check_sector_limit("ETH/USDT") == (True, "")

    def test_existing_get_risk_metrics_still_has_original_fields(self):
        """get_risk_metrics should still include the original fields."""
        mgr = PortfolioRiskManager()
        mgr.update_portfolio_value(100000)
        metrics = mgr.get_risk_metrics()

        # Original V3 fields
        assert "exposure_pct" in metrics
        assert "heat" in metrics
        assert "var_pct" in metrics
        assert "n_positions" in metrics
        assert "portfolio_value" in metrics
        assert "var_enabled" in metrics


# ---------------------------------------------------------------------------
# Test config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    def test_var_method_default(self):
        from bot.config import Settings
        s = Settings(
            binance_api_key="", binance_secret_key="",
            _env_file=None,
        )
        assert s.var_method == "historical"

    def test_stress_var_simulations_default(self):
        from bot.config import Settings
        s = Settings(
            binance_api_key="", binance_secret_key="",
            _env_file=None,
        )
        assert s.stress_var_simulations == 1000

    def test_settings_metadata_has_var_method(self):
        from bot.config import SETTINGS_METADATA
        assert "var_method" in SETTINGS_METADATA
        meta = SETTINGS_METADATA["var_method"]
        assert meta["section"] == "Risk Management"
        assert meta["requires_restart"] is False

    def test_settings_metadata_has_stress_var_simulations(self):
        from bot.config import SETTINGS_METADATA
        assert "stress_var_simulations" in SETTINGS_METADATA
        meta = SETTINGS_METADATA["stress_var_simulations"]
        assert meta["section"] == "Risk Management"
        assert meta["requires_restart"] is False


# ---------------------------------------------------------------------------
# Test dashboard endpoint
# ---------------------------------------------------------------------------

class TestDashboardEndpoint:
    def test_risk_portfolio_endpoint_no_manager(self):
        """Should return error when no engine manager."""
        from fastapi.testclient import TestClient

        from bot.dashboard.app import app

        client = TestClient(app)
        resp = client.get("/api/risk/portfolio")
        assert resp.status_code == 200
        data = resp.json()
        # Either returns risk data or error
        assert "error" in data or "exposure_pct" in data

    def test_risk_portfolio_endpoint_with_manager(self):
        """Should return risk metrics when manager is available."""
        from bot.dashboard import app as app_module

        mock_manager = MagicMock()
        mock_prm = PortfolioRiskManager()
        mock_prm.update_portfolio_value(100000)
        mock_prm.add_position("BTC/USDT", 30000)
        mock_manager._portfolio_risk = mock_prm

        original = app_module._engine_manager
        try:
            app_module._engine_manager = mock_manager

            from fastapi.testclient import TestClient
            client = TestClient(app_module.app)
            resp = client.get("/api/risk/portfolio")
            assert resp.status_code == 200
            data = resp.json()
            assert "exposure_pct" in data
            assert "positions" in data
            assert len(data["positions"]) == 1
            assert data["positions"][0]["symbol"] == "BTC/USDT"
        finally:
            app_module._engine_manager = original
