"""Tests for V5-011: Research loop integration."""

from bot.engines.manager import EngineManager
from bot.engines.portfolio_manager import PortfolioManager
from bot.research.experiments.cointegration import CointegrationExperiment
from bot.research.experiments.funding_prediction import (
    FundingPredictionExperiment,
)
from bot.research.experiments.optimal_grid import OptimalGridExperiment
from bot.research.experiments.volatility_regime import (
    VolatilityRegimeExperiment,
)

# ------------------------------------------------------------------ #
# EngineManager research integration
# ------------------------------------------------------------------ #


class TestManagerResearchIntegration:
    def test_manager_has_research_fields(self):
        pm = PortfolioManager(total_capital=10000)
        mgr = EngineManager(portfolio_manager=pm)
        assert mgr._research_experiments == []
        assert mgr._research_reports == []

    def test_register_experiment(self):
        pm = PortfolioManager(total_capital=10000)
        mgr = EngineManager(portfolio_manager=pm)
        exp = VolatilityRegimeExperiment()
        mgr.register_experiment(exp)
        assert len(mgr._research_experiments) == 1
        assert mgr._research_experiments[0] is exp

    def test_register_multiple_experiments(self):
        pm = PortfolioManager(total_capital=10000)
        mgr = EngineManager(portfolio_manager=pm)
        mgr.register_experiment(VolatilityRegimeExperiment())
        mgr.register_experiment(CointegrationExperiment())
        mgr.register_experiment(OptimalGridExperiment())
        mgr.register_experiment(FundingPredictionExperiment())
        assert len(mgr._research_experiments) == 4


# ------------------------------------------------------------------ #
# Config fields
# ------------------------------------------------------------------ #


class TestResearchConfig:
    def test_research_enabled_default(self):
        from bot.config import load_settings

        s = load_settings()
        assert s.research_enabled is True
        assert s.research_interval_hours == 24

    def test_settings_metadata_exists(self):
        from bot.config import SETTINGS_METADATA

        assert "research_enabled" in SETTINGS_METADATA
        assert "research_interval_hours" in SETTINGS_METADATA


# ------------------------------------------------------------------ #
# Research experiments run correctly
# ------------------------------------------------------------------ #


class TestExperimentsRunCorrectly:
    def test_volatility_regime(self):
        exp = VolatilityRegimeExperiment()
        report = exp.run_experiment()
        d = report.to_dict()
        assert d["experiment_name"] == "volatility_regime"
        assert "results" in d
        assert isinstance(d["timestamp"], str)

    def test_cointegration(self):
        exp = CointegrationExperiment()
        report = exp.run_experiment()
        d = report.to_dict()
        assert d["experiment_name"] == "cointegration"

    def test_optimal_grid(self):
        exp = OptimalGridExperiment()
        report = exp.run_experiment()
        d = report.to_dict()
        assert d["experiment_name"] == "optimal_grid"

    def test_funding_prediction(self):
        exp = FundingPredictionExperiment()
        report = exp.run_experiment()
        d = report.to_dict()
        assert d["experiment_name"] == "funding_prediction"

    def test_report_serialization(self):
        """All experiments produce serializable reports."""
        experiments = [
            VolatilityRegimeExperiment(),
            CointegrationExperiment(),
            OptimalGridExperiment(),
            FundingPredictionExperiment(),
        ]
        for exp in experiments:
            report = exp.run_experiment()
            d = report.to_dict()
            assert isinstance(d, dict)
            assert "recommended_changes" in d
            assert isinstance(d["recommended_changes"], list)
