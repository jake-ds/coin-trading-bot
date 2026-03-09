"""V6 Integration Tests — verify all V6 modules import, config defaults, and instantiation."""

from __future__ import annotations

from unittest.mock import MagicMock

from bot.config import SETTINGS_METADATA, Settings

# ──────────────────────────────────────────────────────────────
# V6 Module Imports
# ──────────────────────────────────────────────────────────────


class TestV6ModuleImports:
    """Verify that all V6 modules can be imported without errors."""

    # Theme 1: Real Data Research
    def test_import_historical_data_provider(self):
        from bot.research.data_provider import HistoricalDataProvider
        assert HistoricalDataProvider is not None

    def test_import_research_deployer(self):
        from bot.research.deployer import (
            DeployDecision,
            DeployResult,
            ResearchDeployer,
        )
        assert ResearchDeployer is not None
        assert DeployDecision is not None
        assert DeployResult is not None

    def test_import_experiments(self):
        from bot.research.experiments.cointegration import (
            CointegrationExperiment,
        )
        from bot.research.experiments.funding_prediction import (
            FundingPredictionExperiment,
        )
        from bot.research.experiments.optimal_grid import (
            OptimalGridExperiment,
        )
        from bot.research.experiments.volatility_regime import (
            VolatilityRegimeExperiment,
        )
        assert VolatilityRegimeExperiment is not None
        assert CointegrationExperiment is not None
        assert OptimalGridExperiment is not None
        assert FundingPredictionExperiment is not None

    # Theme 2: Advanced Risk Models
    def test_import_volatility_service(self):
        from bot.risk.volatility_service import VolatilityService
        assert VolatilityService is not None

    def test_import_dynamic_sizer(self):
        from bot.risk.dynamic_sizer import DynamicPositionSizer, PositionSize
        assert DynamicPositionSizer is not None
        assert PositionSize is not None

    def test_import_correlation_controller(self):
        from bot.risk.correlation_controller import (
            CorrelationRiskController,
        )
        assert CorrelationRiskController is not None

    def test_import_portfolio_risk_var(self):
        from bot.risk.portfolio_risk import PortfolioRiskManager
        assert hasattr(PortfolioRiskManager, "calculate_parametric_var")
        assert hasattr(PortfolioRiskManager, "calculate_cornish_fisher_var")
        assert hasattr(PortfolioRiskManager, "calculate_cvar")
        assert hasattr(PortfolioRiskManager, "calculate_stress_var")
        assert hasattr(PortfolioRiskManager, "pre_trade_var_check")

    # Theme 4: Metrics Persistence
    def test_import_metrics_persistence(self):
        from bot.engines.metrics_persistence import MetricsPersistence
        assert MetricsPersistence is not None

    def test_import_db_models(self):
        from bot.data.models import EngineMetricSnapshot, EngineTradeRecord
        assert EngineTradeRecord is not None
        assert EngineMetricSnapshot is not None

    # Theme 5: Market Regime Detection
    def test_import_regime_detector(self):
        from bot.risk.regime_detector import (
            MarketRegime,
            MarketRegimeDetector,
        )
        assert MarketRegimeDetector is not None
        assert MarketRegime is not None

    # Engine base extensions
    def test_base_engine_has_v6_fields(self):
        from bot.engines.base import BaseEngine
        # V6-007: dynamic sizer
        assert hasattr(BaseEngine, "set_sizer")
        # V6-008: correlation controller
        assert hasattr(BaseEngine, "set_correlation_controller")
        # V6-015: regime detector
        assert hasattr(BaseEngine, "set_regime_detector")
        assert hasattr(BaseEngine, "_get_regime_adjustments")

    # Engine manager extensions
    def test_engine_manager_has_v6_methods(self):
        from bot.engines.manager import EngineManager
        assert hasattr(EngineManager, "set_collector")
        assert hasattr(EngineManager, "set_deployer")
        assert hasattr(EngineManager, "set_correlation_controller")
        assert hasattr(EngineManager, "set_metrics_persistence")
        assert hasattr(EngineManager, "set_regime_detector")
        assert hasattr(EngineManager, "_circuit_breaker_check")

    # Risk module __init__ exports
    def test_risk_module_exports(self):
        from bot.risk import (
            CorrelationRiskController,
            DynamicPositionSizer,
            PositionSize,
            VolatilityService,
        )
        assert VolatilityService is not None
        assert DynamicPositionSizer is not None
        assert PositionSize is not None
        assert CorrelationRiskController is not None

    # DataStore new methods
    def test_datastore_has_v6_methods(self):
        from bot.data.store import DataStore
        assert hasattr(DataStore, "get_available_symbols")
        assert hasattr(DataStore, "get_engine_metric_snapshots")
        assert hasattr(DataStore, "get_engine_trades")

    # Dashboard endpoints
    def test_dashboard_has_v6_endpoints(self):
        from bot.dashboard.app import (
            get_heatmap,
            get_market_regime,
            get_risk_correlation,
            get_risk_drawdown,
            get_risk_portfolio,
            get_trade_detail,
            health_check,
            set_store_ref,
        )
        assert health_check is not None
        assert set_store_ref is not None
        assert get_trade_detail is not None
        assert get_heatmap is not None
        assert get_risk_portfolio is not None
        assert get_risk_correlation is not None
        assert get_risk_drawdown is not None
        assert get_market_regime is not None

    # EngineTracker bulk_load
    def test_tracker_has_bulk_load(self):
        from bot.engines.tracker import EngineTracker
        assert hasattr(EngineTracker, "bulk_load_trades")


# ──────────────────────────────────────────────────────────────
# V6 Config Defaults
# ──────────────────────────────────────────────────────────────


class TestV6ConfigDefaults:
    """Verify all V6 config fields have proper defaults."""

    def test_data_backfill_defaults(self):
        s = Settings()
        assert s.data_backfill_enabled is True
        assert s.data_backfill_interval_hours == 6.0
        assert s.data_backfill_days == 30

    def test_research_deploy_defaults(self):
        s = Settings()
        assert s.research_auto_deploy is True
        assert s.research_regression_check_hours == 6.0

    def test_var_defaults(self):
        s = Settings()
        assert s.var_method == "historical"
        assert s.stress_var_simulations == 1000

    def test_dynamic_sizing_defaults(self):
        s = Settings()
        assert s.dynamic_sizing_enabled is True
        assert s.vol_scale_factor == 1.0
        assert s.max_position_scale == 2.0

    def test_correlation_defaults(self):
        s = Settings()
        assert s.cross_engine_correlation_enabled is True
        assert s.max_symbol_concentration_pct == 40.0

    def test_metrics_persistence_defaults(self):
        s = Settings()
        assert s.metrics_persistence_enabled is True
        assert s.metrics_snapshot_interval_minutes == 5.0
        assert s.metrics_retention_days == 90

    def test_regime_detection_defaults(self):
        s = Settings()
        assert s.regime_detection_enabled is True
        assert s.regime_crisis_threshold == 2.5
        assert s.regime_detection_interval_seconds == 300.0

    def test_regime_adaptation_defaults(self):
        s = Settings()
        assert s.regime_adaptation_enabled is True
        assert s.crisis_circuit_breaker_minutes == 30.0

    def test_shutdown_defaults(self):
        s = Settings()
        assert s.shutdown_timeout_seconds == 30.0

    def test_all_v6_fields_in_metadata(self):
        """Every V6 config field should be present in SETTINGS_METADATA."""
        v6_fields = [
            "data_backfill_enabled",
            "data_backfill_interval_hours",
            "data_backfill_days",
            "research_auto_deploy",
            "research_regression_check_hours",
            "var_method",
            "stress_var_simulations",
            "dynamic_sizing_enabled",
            "vol_scale_factor",
            "max_position_scale",
            "cross_engine_correlation_enabled",
            "max_symbol_concentration_pct",
            "metrics_persistence_enabled",
            "metrics_snapshot_interval_minutes",
            "metrics_retention_days",
            "regime_detection_enabled",
            "regime_crisis_threshold",
            "regime_detection_interval_seconds",
            "regime_adaptation_enabled",
            "crisis_circuit_breaker_minutes",
            "shutdown_timeout_seconds",
        ]
        for field in v6_fields:
            assert field in SETTINGS_METADATA, (
                f"Config field '{field}' missing from SETTINGS_METADATA"
            )


# ──────────────────────────────────────────────────────────────
# V6 Class Instantiation
# ──────────────────────────────────────────────────────────────


class TestV6ClassInstantiation:
    """Verify key V6 classes can be instantiated with defaults/mocks."""

    def test_historical_data_provider(self):
        from bot.research.data_provider import HistoricalDataProvider

        mock_store = MagicMock()
        provider = HistoricalDataProvider(data_store=mock_store)
        assert provider is not None

    def test_volatility_service_no_provider(self):
        from bot.risk.volatility_service import VolatilityService

        svc = VolatilityService()
        assert svc.get_forecast("BTC/USDT") is None
        assert svc.get_regime("BTC/USDT").value == "NORMAL"

    def test_dynamic_position_sizer_no_deps(self):
        from bot.risk.dynamic_sizer import DynamicPositionSizer

        sizer = DynamicPositionSizer()
        result = sizer.calculate_size(
            symbol="BTC/USDT",
            price=50000.0,
            portfolio_value=100000.0,
        )
        assert result.method == "fixed"
        assert result.vol_multiplier == 1.0

    def test_correlation_controller_no_prm(self):
        from bot.risk.correlation_controller import (
            CorrelationRiskController,
        )

        ctrl = CorrelationRiskController()
        report = ctrl.get_concentration_report()
        assert "per_symbol" in report
        assert "alerts" in report

    def test_regime_detector_no_vol_service(self):
        from bot.risk.regime_detector import MarketRegimeDetector

        detector = MarketRegimeDetector(volatility_service=None)
        regime = detector.get_current_regime()
        assert regime.value == "NORMAL"

    def test_research_deployer(self):
        from bot.engines.tracker import EngineTracker
        from bot.engines.tuner import ParameterTuner
        from bot.research.deployer import ResearchDeployer

        deployer = ResearchDeployer(
            tuner=ParameterTuner(),
            settings=Settings(),
            tracker=EngineTracker(),
        )
        assert deployer.get_deploy_history() == []

    def test_metrics_persistence(self):
        from bot.engines.metrics_persistence import MetricsPersistence
        from bot.engines.tracker import EngineTracker

        mock_store = MagicMock()
        tracker = EngineTracker()
        persistence = MetricsPersistence(
            data_store=mock_store, tracker=tracker,
        )
        assert persistence is not None

    def test_market_regime_enum(self):
        from bot.risk.regime_detector import MarketRegime

        assert MarketRegime.LOW.value == "LOW"
        assert MarketRegime.NORMAL.value == "NORMAL"
        assert MarketRegime.HIGH.value == "HIGH"
        assert MarketRegime.CRISIS.value == "CRISIS"

    def test_engine_manager_with_portfolio_manager(self):
        from bot.engines.manager import EngineManager
        from bot.engines.portfolio_manager import PortfolioManager

        pm = PortfolioManager(total_capital=10000.0)
        mgr = EngineManager(pm)
        assert mgr is not None
        assert mgr.tracker is not None
        assert mgr.tuner is not None
