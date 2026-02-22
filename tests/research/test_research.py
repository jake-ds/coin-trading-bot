"""Tests for V5-010: Research framework — base module, experiments, backtest runner."""

import numpy as np

from bot.engines.tuner import ParamChange
from bot.research.backtest_runner import BacktestResult, SimpleBacktestRunner
from bot.research.base import ResearchTask
from bot.research.experiments.cointegration import CointegrationExperiment
from bot.research.experiments.funding_prediction import FundingPredictionExperiment
from bot.research.experiments.optimal_grid import OptimalGridExperiment
from bot.research.experiments.volatility_regime import VolatilityRegimeExperiment
from bot.research.report import ResearchReport

# ------------------------------------------------------------------ #
# ResearchReport
# ------------------------------------------------------------------ #


class TestResearchReport:
    def test_fields(self):
        r = ResearchReport(
            experiment_name="test",
            hypothesis="H",
            methodology="M",
            data_period="1d",
        )
        assert r.experiment_name == "test"
        assert r.timestamp  # auto-filled

    def test_to_dict(self):
        r = ResearchReport(
            experiment_name="test",
            hypothesis="H",
            methodology="M",
            data_period="1d",
            results={"key": 42},
            conclusion="C",
        )
        d = r.to_dict()
        assert d["experiment_name"] == "test"
        assert d["results"]["key"] == 42
        assert d["recommended_changes"] == []

    def test_to_dict_with_changes(self):
        r = ResearchReport(
            experiment_name="test",
            hypothesis="H",
            methodology="M",
            data_period="1d",
            recommended_changes=[
                ParamChange("e", "p", 1, 2, "reason"),
            ],
        )
        d = r.to_dict()
        assert len(d["recommended_changes"]) == 1
        assert d["recommended_changes"][0]["old_value"] == 1


# ------------------------------------------------------------------ #
# ResearchTask ABC
# ------------------------------------------------------------------ #


class TestResearchTaskABC:
    def test_cannot_instantiate(self):
        import pytest

        with pytest.raises(TypeError):
            ResearchTask()  # type: ignore[abstract]

    def test_subclass_interface(self):
        class MyTask(ResearchTask):
            @property
            def target_engine(self) -> str:
                return "test_engine"

            def run_experiment(self, **kwargs: object) -> ResearchReport:
                return ResearchReport("t", "h", "m", "d")

            def apply_findings(self) -> list[ParamChange]:
                return []

        task = MyTask()
        assert task.target_engine == "test_engine"
        report = task.run_experiment()
        assert isinstance(report, ResearchReport)
        assert task.apply_findings() == []


# ------------------------------------------------------------------ #
# SimpleBacktestRunner
# ------------------------------------------------------------------ #


class TestBacktestRunner:
    def test_empty_prices(self):
        runner = SimpleBacktestRunner()
        result = runner.run([], lambda p, i: 0.0)
        assert isinstance(result, BacktestResult)
        assert result.sharpe == 0.0

    def test_single_price(self):
        runner = SimpleBacktestRunner()
        result = runner.run([100.0], lambda p, i: 1.0)
        assert result.num_trades == 0

    def test_buy_and_hold(self):
        prices = [100.0, 110.0, 105.0, 115.0]
        runner = SimpleBacktestRunner()
        result = runner.run(prices, lambda p, i: 1.0)
        assert result.total_return > 0
        assert result.num_trades == 1  # one position change: 0 -> 1

    def test_no_position(self):
        prices = [100.0, 110.0, 120.0]
        runner = SimpleBacktestRunner()
        result = runner.run(prices, lambda p, i: 0.0)
        assert result.total_return == 0.0
        assert result.num_trades == 0

    def test_trades_counted(self):
        prices = [100.0, 105.0, 95.0, 110.0, 100.0]
        runner = SimpleBacktestRunner()
        # Alternate between buy and sell
        result = runner.run(
            prices,
            lambda p, i: 1.0 if i % 2 == 0 else -1.0,
        )
        assert result.num_trades >= 2

    def test_sharpe_positive_trend(self):
        # Monotonically increasing prices -> buy & hold should have positive Sharpe
        prices = [100 + i * 0.5 for i in range(100)]
        runner = SimpleBacktestRunner()
        result = runner.run(prices, lambda p, i: 1.0)
        assert result.sharpe > 0

    def test_max_drawdown_computed(self):
        # Prices go up then down
        prices = [100.0, 110.0, 120.0, 100.0, 90.0]
        runner = SimpleBacktestRunner()
        result = runner.run(prices, lambda p, i: 1.0)
        assert result.max_drawdown > 0


# ------------------------------------------------------------------ #
# VolatilityRegimeExperiment
# ------------------------------------------------------------------ #


class TestVolatilityRegime:
    def test_runs_with_defaults(self):
        exp = VolatilityRegimeExperiment()
        report = exp.run_experiment()
        assert report.experiment_name == "volatility_regime"
        assert "fixed_sharpe" in report.results
        assert "dynamic_sharpe" in report.results

    def test_target_engine(self):
        assert VolatilityRegimeExperiment().target_engine == "grid_trading"

    def test_runs_with_custom_prices(self):
        prices = [100.0 + i * 0.1 for i in range(100)]
        exp = VolatilityRegimeExperiment()
        report = exp.run_experiment(prices=prices)
        assert "median_atr" in report.results

    def test_insufficient_data(self):
        exp = VolatilityRegimeExperiment()
        report = exp.run_experiment(prices=[100.0, 101.0])
        assert "error" in report.results

    def test_apply_findings_empty_when_no_run(self):
        exp = VolatilityRegimeExperiment()
        assert exp.apply_findings() == []


# ------------------------------------------------------------------ #
# CointegrationExperiment
# ------------------------------------------------------------------ #


class TestCointegration:
    def test_runs_with_defaults(self):
        exp = CointegrationExperiment()
        report = exp.run_experiment()
        assert report.experiment_name == "cointegration"
        assert isinstance(report.results, dict)

    def test_target_engine(self):
        assert CointegrationExperiment().target_engine == "stat_arb"

    def test_detects_cointegrated_pair(self):
        rng = np.random.default_rng(42)
        n = 200
        common = np.cumsum(rng.normal(0, 1, n))
        a = common + rng.normal(0, 0.1, n)
        b = 0.9 * common + rng.normal(0, 0.1, n) + 5

        exp = CointegrationExperiment()
        report = exp.run_experiment(
            pairs={"test_pair": (a.tolist(), b.tolist())}
        )
        pair_result = report.results["test_pair"]
        assert pair_result["cointegrated"] is True

    def test_non_cointegrated_pair(self):
        rng = np.random.default_rng(42)
        n = 200
        a = np.cumsum(rng.normal(0, 1, n)) + 100
        b = np.cumsum(rng.normal(0, 1, n)) + 50

        exp = CointegrationExperiment()
        report = exp.run_experiment(
            pairs={"test_pair": (a.tolist(), b.tolist())}
        )
        pair_result = report.results["test_pair"]
        assert pair_result["cointegrated"] is False

    def test_apply_findings(self):
        exp = CointegrationExperiment()
        exp.run_experiment()
        changes = exp.apply_findings()
        assert isinstance(changes, list)


# ------------------------------------------------------------------ #
# OptimalGridExperiment
# ------------------------------------------------------------------ #


class TestOptimalGrid:
    def test_runs_with_defaults(self):
        exp = OptimalGridExperiment()
        report = exp.run_experiment()
        assert report.experiment_name == "optimal_grid"
        assert len(report.results) > 0

    def test_target_engine(self):
        assert OptimalGridExperiment().target_engine == "grid_trading"

    def test_has_spacing_results(self):
        exp = OptimalGridExperiment()
        report = exp.run_experiment()
        # Should have results for each spacing option
        assert any("spacing" in k for k in report.results)

    def test_apply_findings(self):
        exp = OptimalGridExperiment()
        exp.run_experiment()
        changes = exp.apply_findings()
        assert isinstance(changes, list)


# ------------------------------------------------------------------ #
# FundingPredictionExperiment
# ------------------------------------------------------------------ #


class TestFundingPrediction:
    def test_runs_with_defaults(self):
        exp = FundingPredictionExperiment()
        report = exp.run_experiment()
        assert report.experiment_name == "funding_prediction"
        assert "mean_rate" in report.results

    def test_target_engine(self):
        assert FundingPredictionExperiment().target_engine == "funding_rate_arb"

    def test_insufficient_data(self):
        exp = FundingPredictionExperiment()
        report = exp.run_experiment(funding_rates=[0.0001] * 10)
        assert "error" in report.results

    def test_positive_bias_detected(self):
        # All positive rates -> should detect positive bias
        rates = [0.0005] * 200
        exp = FundingPredictionExperiment()
        report = exp.run_experiment(funding_rates=rates)
        assert report.results["positive_pct"] == 1.0

    def test_apply_findings(self):
        exp = FundingPredictionExperiment()
        exp.run_experiment()
        changes = exp.apply_findings()
        assert isinstance(changes, list)
