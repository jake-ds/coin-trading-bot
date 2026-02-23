"""Tests for DynamicPositionSizer — GARCH + ATR + fixed position sizing."""

from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from bot.risk.dynamic_sizer import DynamicPositionSizer, PositionSize


class TestPositionSizeDataclass:
    """Test the PositionSize dataclass."""

    def test_creation(self):
        ps = PositionSize(
            quantity=1.0,
            notional_value=50000.0,
            risk_amount=100.0,
            vol_multiplier=1.0,
            method="fixed",
        )
        assert ps.quantity == 1.0
        assert ps.notional_value == 50000.0
        assert ps.risk_amount == 100.0
        assert ps.vol_multiplier == 1.0
        assert ps.method == "fixed"


class TestDynamicPositionSizerInit:
    """Test initialization."""

    def test_default_init(self):
        sizer = DynamicPositionSizer()
        assert sizer._volatility_service is None
        assert sizer._portfolio_risk is None
        assert sizer._base_risk_pct == 1.0
        assert sizer._vol_scale_factor == 1.0

    def test_custom_init(self):
        mock_vs = MagicMock()
        mock_pr = MagicMock()
        sizer = DynamicPositionSizer(
            volatility_service=mock_vs,
            portfolio_risk=mock_pr,
            base_risk_pct=2.0,
            vol_scale_factor=1.5,
        )
        assert sizer._volatility_service is mock_vs
        assert sizer._portfolio_risk is mock_pr
        assert sizer._base_risk_pct == 2.0
        assert sizer._vol_scale_factor == 1.5


class TestFixedSizing:
    """Test fixed sizing fallback (no GARCH, no ATR)."""

    def test_fixed_sizing_basic(self):
        sizer = DynamicPositionSizer(base_risk_pct=1.0, vol_scale_factor=1.0)
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
        )
        assert ps.method == "fixed"
        assert ps.vol_multiplier == 1.0
        # risk_amount = 100000 * 0.01 * 1.0 = 1000
        assert ps.risk_amount == 1000.0
        # quantity = 1000 / (50000 * 1.0) = 0.02
        assert ps.quantity == pytest.approx(0.02, rel=1e-6)
        assert ps.notional_value == pytest.approx(1000.0, rel=1e-6)

    def test_fixed_sizing_different_risk_pct(self):
        sizer = DynamicPositionSizer(base_risk_pct=2.5, vol_scale_factor=1.0)
        ps = sizer.calculate_size(
            symbol="ETH/USDT",
            price=3000.0,
            portfolio_value=50000.0,
        )
        assert ps.method == "fixed"
        # risk_amount = 50000 * 0.025 * 1.0 = 1250
        assert ps.risk_amount == 1250.0
        # quantity = 1250 / 3000 = 0.4167
        assert ps.quantity == pytest.approx(1250.0 / 3000.0, rel=1e-6)

    def test_fixed_sizing_vol_scale_factor(self):
        sizer = DynamicPositionSizer(base_risk_pct=1.0, vol_scale_factor=2.0)
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
        )
        # risk_amount = 100000 * 0.01 * 1.0 = 1000
        # quantity = 1000 / (50000 * 2.0) = 0.01
        assert ps.quantity == pytest.approx(0.01, rel=1e-6)

    def test_fixed_sizing_zero_price(self):
        sizer = DynamicPositionSizer()
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=0.0,
            portfolio_value=100000.0,
        )
        assert ps.quantity == 0.0

    def test_fixed_sizing_zero_portfolio(self):
        sizer = DynamicPositionSizer()
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=0.0,
        )
        assert ps.quantity == 0.0
        assert ps.risk_amount == 0.0


class TestGARCHSizing:
    """Test GARCH-based dynamic sizing."""

    def _make_volatility_service(self, forecast, cond_vol_array):
        """Helper to create mock VolatilityService with GARCH data."""
        mock_vs = MagicMock()
        mock_vs.get_forecast.return_value = forecast

        mock_model = MagicMock()
        type(mock_model).conditional_volatility = PropertyMock(
            return_value=np.array(cond_vol_array) if cond_vol_array is not None else None
        )
        mock_vs.get_model.return_value = mock_model

        return mock_vs

    def test_garch_high_vol_smaller_position(self):
        """When current volatility (forecast) is higher than median, reduce position."""
        # median_vol = 0.02, forecast = 0.04 → vol_multiplier = 0.02/0.04 = 0.5
        mock_vs = self._make_volatility_service(
            forecast=0.04,
            cond_vol_array=[0.01, 0.02, 0.02, 0.03, 0.02],
        )
        sizer = DynamicPositionSizer(
            volatility_service=mock_vs,
            base_risk_pct=1.0,
            vol_scale_factor=1.0,
        )
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
        )
        assert ps.method == "garch"
        assert ps.vol_multiplier == pytest.approx(0.5, rel=1e-6)
        # risk_amount = 100000 * 0.01 * 0.5 = 500
        assert ps.risk_amount == pytest.approx(500.0, rel=1e-6)
        assert ps.quantity == pytest.approx(0.01, rel=1e-6)

    def test_garch_low_vol_larger_position(self):
        """When current volatility (forecast) is lower than median, increase position."""
        # median_vol = 0.04, forecast = 0.02 → vol_multiplier = 0.04/0.02 = 2.0
        mock_vs = self._make_volatility_service(
            forecast=0.02,
            cond_vol_array=[0.03, 0.04, 0.04, 0.05, 0.04],
        )
        sizer = DynamicPositionSizer(
            volatility_service=mock_vs,
            base_risk_pct=1.0,
            vol_scale_factor=1.0,
        )
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
        )
        assert ps.method == "garch"
        assert ps.vol_multiplier == pytest.approx(2.0, rel=1e-6)
        # risk_amount = 100000 * 0.01 * 2.0 = 2000
        assert ps.risk_amount == pytest.approx(2000.0, rel=1e-6)

    def test_garch_vol_multiplier_clamped_lower(self):
        """vol_multiplier should be clamped to minimum 0.25."""
        # median_vol = 0.01, forecast = 0.10 → raw multiplier = 0.1 → clamped to 0.25
        mock_vs = self._make_volatility_service(
            forecast=0.10,
            cond_vol_array=[0.01, 0.01, 0.01],
        )
        sizer = DynamicPositionSizer(volatility_service=mock_vs)
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
        )
        assert ps.vol_multiplier == 0.25
        assert ps.method == "garch"

    def test_garch_vol_multiplier_clamped_upper(self):
        """vol_multiplier should be clamped to maximum 2.0."""
        # median_vol = 0.10, forecast = 0.01 → raw multiplier = 10.0 → clamped to 2.0
        mock_vs = self._make_volatility_service(
            forecast=0.01,
            cond_vol_array=[0.10, 0.10, 0.10],
        )
        sizer = DynamicPositionSizer(volatility_service=mock_vs)
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
        )
        assert ps.vol_multiplier == 2.0

    def test_garch_zero_forecast_falls_back_to_fixed(self):
        """If GARCH forecast is 0, fall back to fixed."""
        mock_vs = self._make_volatility_service(
            forecast=0.0,
            cond_vol_array=[0.02, 0.02],
        )
        sizer = DynamicPositionSizer(volatility_service=mock_vs)
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
        )
        assert ps.method == "fixed"
        assert ps.vol_multiplier == 1.0

    def test_garch_none_forecast_falls_back_to_fixed(self):
        """If GARCH forecast is None (no fit), fall back to fixed."""
        mock_vs = MagicMock()
        mock_vs.get_forecast.return_value = None
        sizer = DynamicPositionSizer(volatility_service=mock_vs)
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
        )
        assert ps.method == "fixed"

    def test_garch_no_model_falls_back_to_fixed(self):
        """If no GARCH model (get_model returns None), fall back to fixed."""
        mock_vs = MagicMock()
        mock_vs.get_forecast.return_value = 0.02
        mock_vs.get_model.return_value = None
        sizer = DynamicPositionSizer(volatility_service=mock_vs)
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
        )
        assert ps.method == "fixed"

    def test_garch_no_cond_vol_falls_back_to_fixed(self):
        """If model has no conditional_volatility, fall back to fixed."""
        mock_vs = MagicMock()
        mock_vs.get_forecast.return_value = 0.02
        mock_model = MagicMock()
        type(mock_model).conditional_volatility = PropertyMock(return_value=None)
        mock_vs.get_model.return_value = mock_model
        sizer = DynamicPositionSizer(volatility_service=mock_vs)
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
        )
        assert ps.method == "fixed"

    def test_garch_zero_median_vol_falls_back_to_fixed(self):
        """If median conditional vol is 0, fall back to fixed."""
        mock_vs = self._make_volatility_service(
            forecast=0.02,
            cond_vol_array=[0.0, 0.0, 0.0],
        )
        sizer = DynamicPositionSizer(volatility_service=mock_vs)
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
        )
        assert ps.method == "fixed"


class TestATRSizing:
    """Test ATR-based sizing fallback."""

    def test_atr_high_vol_smaller_position(self):
        """Higher ATR → smaller position."""
        sizer = DynamicPositionSizer(base_risk_pct=1.0, vol_scale_factor=1.0)
        # price=50000, atr=2500 → atr_pct=0.05
        # target_atr = 0.01 → vol_multiplier = 0.01 / 0.05 = 0.2 → clamped to 0.25
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
            atr=2500.0,
        )
        assert ps.method == "atr"
        assert ps.vol_multiplier == 0.25  # Clamped from 0.2

    def test_atr_low_vol_larger_position(self):
        """Lower ATR → larger position."""
        sizer = DynamicPositionSizer(base_risk_pct=1.0, vol_scale_factor=1.0)
        # price=50000, atr=250 → atr_pct=0.005
        # target_atr = 0.01 → vol_multiplier = 0.01 / 0.005 = 2.0
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
            atr=250.0,
        )
        assert ps.method == "atr"
        assert ps.vol_multiplier == pytest.approx(2.0, rel=1e-6)

    def test_atr_moderate_vol(self):
        """Moderate ATR → moderate multiplier."""
        sizer = DynamicPositionSizer(base_risk_pct=1.0, vol_scale_factor=1.0)
        # price=50000, atr=500 → atr_pct=0.01
        # target_atr = 0.01 → vol_multiplier = 0.01 / 0.01 = 1.0
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
            atr=500.0,
        )
        assert ps.method == "atr"
        assert ps.vol_multiplier == pytest.approx(1.0, rel=1e-6)

    def test_atr_zero_falls_back_to_fixed(self):
        """ATR=0 should fall back to fixed."""
        sizer = DynamicPositionSizer()
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
            atr=0.0,
        )
        assert ps.method == "fixed"

    def test_atr_negative_falls_back_to_fixed(self):
        """Negative ATR should fall back to fixed."""
        sizer = DynamicPositionSizer()
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
            atr=-100.0,
        )
        assert ps.method == "fixed"

    def test_garch_takes_precedence_over_atr(self):
        """GARCH should be used when both GARCH and ATR are available."""
        mock_vs = MagicMock()
        mock_vs.get_forecast.return_value = 0.02
        mock_model = MagicMock()
        type(mock_model).conditional_volatility = PropertyMock(
            return_value=np.array([0.02, 0.02, 0.02])
        )
        mock_vs.get_model.return_value = mock_model

        sizer = DynamicPositionSizer(volatility_service=mock_vs)
        ps = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
            atr=1000.0,  # Should be ignored since GARCH works
        )
        assert ps.method == "garch"


class TestValidateSize:
    """Test position size validation/clipping."""

    def test_no_clipping_within_limit(self):
        sizer = DynamicPositionSizer()
        ps = PositionSize(
            quantity=0.02,
            notional_value=1000.0,
            risk_amount=100.0,
            vol_multiplier=1.0,
            method="fixed",
        )
        result = sizer.validate_size(ps, portfolio_value=100000.0, max_pct=10.0)
        assert result.quantity == 0.02
        assert result.notional_value == 1000.0

    def test_clipping_exceeds_limit(self):
        sizer = DynamicPositionSizer()
        ps = PositionSize(
            quantity=1.0,
            notional_value=50000.0,
            risk_amount=1000.0,
            vol_multiplier=1.5,
            method="garch",
        )
        # max = 100000 * 5% = 5000
        result = sizer.validate_size(ps, portfolio_value=100000.0, max_pct=5.0)
        assert result.notional_value == pytest.approx(5000.0, rel=1e-6)
        assert result.quantity == pytest.approx(0.1, rel=1e-6)
        assert result.risk_amount == pytest.approx(100.0, rel=1e-6)
        assert result.vol_multiplier == 1.5  # Unchanged
        assert result.method == "garch"  # Unchanged

    def test_clipping_zero_portfolio(self):
        sizer = DynamicPositionSizer()
        ps = PositionSize(
            quantity=1.0,
            notional_value=50000.0,
            risk_amount=1000.0,
            vol_multiplier=1.0,
            method="fixed",
        )
        # No clipping when portfolio_value is 0
        result = sizer.validate_size(ps, portfolio_value=0.0, max_pct=10.0)
        assert result.quantity == 1.0

    def test_clipping_zero_max_pct(self):
        sizer = DynamicPositionSizer()
        ps = PositionSize(
            quantity=1.0,
            notional_value=50000.0,
            risk_amount=1000.0,
            vol_multiplier=1.0,
            method="fixed",
        )
        # No clipping when max_pct is 0
        result = sizer.validate_size(ps, portfolio_value=100000.0, max_pct=0.0)
        assert result.quantity == 1.0

    def test_clipping_at_exact_boundary(self):
        sizer = DynamicPositionSizer()
        ps = PositionSize(
            quantity=0.1,
            notional_value=5000.0,
            risk_amount=100.0,
            vol_multiplier=1.0,
            method="fixed",
        )
        # max = 100000 * 5% = 5000 → exactly at boundary, no clipping
        result = sizer.validate_size(ps, portfolio_value=100000.0, max_pct=5.0)
        assert result.quantity == 0.1

    def test_clipping_zero_notional(self):
        sizer = DynamicPositionSizer()
        ps = PositionSize(
            quantity=0.0,
            notional_value=0.0,
            risk_amount=0.0,
            vol_multiplier=1.0,
            method="fixed",
        )
        result = sizer.validate_size(ps, portfolio_value=100000.0, max_pct=5.0)
        assert result.quantity == 0.0


class TestVolMultiplierBounds:
    """Test that vol_multiplier is always within [0.25, 2.0]."""

    def test_lower_bound(self):
        """Extremely high volatility → clamped to 0.25."""
        sizer = DynamicPositionSizer(base_risk_pct=1.0)
        # price=100, atr=50 → atr_pct=0.5
        # target_atr = 0.01 → raw mult = 0.02 → clamped to 0.25
        ps = sizer.calculate_size(
            symbol="X/USDT",
            price=100.0,
            portfolio_value=10000.0,
            atr=50.0,
        )
        assert ps.vol_multiplier == 0.25

    def test_upper_bound(self):
        """Extremely low volatility → clamped to 2.0."""
        sizer = DynamicPositionSizer(base_risk_pct=1.0)
        # price=100, atr=0.001 → atr_pct=0.00001
        # target_atr = 0.01 → raw mult = 1000 → clamped to 2.0
        ps = sizer.calculate_size(
            symbol="X/USDT",
            price=100.0,
            portfolio_value=10000.0,
            atr=0.001,
        )
        assert ps.vol_multiplier == 2.0


class TestDecisionStepCreation:
    """Test that DecisionStep can be created from PositionSize data."""

    def test_decision_step_fields(self):
        from bot.engines.base import DecisionStep

        ps = PositionSize(
            quantity=0.02,
            notional_value=1000.0,
            risk_amount=100.0,
            vol_multiplier=0.5,
            method="garch",
        )
        step = DecisionStep(
            label="포지션 사이징",
            observation=(
                f"방법: {ps.method}, 변동성 배수: {ps.vol_multiplier:.2f}, "
                f"수량: {ps.quantity:.6f}"
            ),
            threshold="변동성 배수 범위: [0.25, 2.0]",
            result=f"사이즈 결정: ${ps.notional_value:.2f}",
            category="evaluate",
        )
        assert step.label == "포지션 사이징"
        assert "garch" in step.observation
        assert "0.50" in step.observation
        assert step.category == "evaluate"


class TestBaseEngineIntegration:
    """Test BaseEngine _dynamic_sizer field and set_sizer method."""

    def test_base_engine_has_dynamic_sizer_field(self):
        from unittest.mock import MagicMock

        from bot.engines.base import BaseEngine

        # Create a concrete subclass for testing
        class ConcreteEngine(BaseEngine):
            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return "test engine"

            async def _run_cycle(self):
                pass

        pm = MagicMock()
        engine = ConcreteEngine(portfolio_manager=pm)
        assert engine._dynamic_sizer is None

    def test_set_sizer(self):
        from unittest.mock import MagicMock

        from bot.engines.base import BaseEngine

        class ConcreteEngine(BaseEngine):
            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return "test engine"

            async def _run_cycle(self):
                pass

        pm = MagicMock()
        engine = ConcreteEngine(portfolio_manager=pm)
        sizer = DynamicPositionSizer()
        engine.set_sizer(sizer)
        assert engine._dynamic_sizer is sizer


class TestConfigIntegration:
    """Test config settings for dynamic sizing."""

    def test_config_defaults(self):
        from bot.config import Settings

        s = Settings(
            binance_api_key="test",
            binance_secret_key="test",
            _env_file=None,
        )
        assert s.dynamic_sizing_enabled is True
        assert s.vol_scale_factor == 1.0
        assert s.max_position_scale == 2.0

    def test_config_custom_values(self):
        from bot.config import Settings

        s = Settings(
            binance_api_key="test",
            binance_secret_key="test",
            dynamic_sizing_enabled=False,
            vol_scale_factor=1.5,
            max_position_scale=3.0,
            _env_file=None,
        )
        assert s.dynamic_sizing_enabled is False
        assert s.vol_scale_factor == 1.5
        assert s.max_position_scale == 3.0

    def test_settings_metadata_entries(self):
        from bot.config import SETTINGS_METADATA

        assert "dynamic_sizing_enabled" in SETTINGS_METADATA
        assert "vol_scale_factor" in SETTINGS_METADATA
        assert "max_position_scale" in SETTINGS_METADATA

        meta = SETTINGS_METADATA["dynamic_sizing_enabled"]
        assert meta["section"] == "Risk Management"
        assert meta["type"] == "bool"
        assert meta["requires_restart"] is False


class TestImportPaths:
    """Test that DynamicPositionSizer is importable from expected paths."""

    def test_import_from_module(self):
        from bot.risk.dynamic_sizer import DynamicPositionSizer, PositionSize

        assert DynamicPositionSizer is not None
        assert PositionSize is not None

    def test_import_from_package(self):
        from bot.risk import DynamicPositionSizer, PositionSize

        assert DynamicPositionSizer is not None
        assert PositionSize is not None
