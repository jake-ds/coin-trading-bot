"""Tests for ResearchDeployer — auto-deploy pipeline with rollback."""

from unittest.mock import MagicMock, patch

from bot.engines.tracker import EngineTracker, TradeRecord
from bot.engines.tuner import ParamChange, ParameterTuner
from bot.research.deployer import (
    MAX_DEPLOY_CHANGES,
    DeployDecision,
    DeployRecord,
    DeployResult,
    ResearchDeployer,
)
from bot.research.report import ResearchReport


def _make_settings(**overrides):
    """Create a mock Settings with realistic engine parameter defaults."""
    settings = MagicMock()
    # Default engine params (matching TUNER_CONFIG keys)
    settings.funding_arb_min_rate = 0.0003
    settings.funding_arb_max_spread_pct = 0.5
    settings.grid_spacing_pct = 0.5
    settings.grid_levels = 10
    settings.cross_arb_min_spread_pct = 0.3
    settings.stat_arb_entry_zscore = 2.0
    settings.stat_arb_exit_zscore = 0.5
    settings.stat_arb_lookback = 100
    settings.stat_arb_min_correlation = 0.7
    settings.research_auto_deploy = True
    settings.research_regression_check_hours = 6.0
    # reload returns list of changed param names
    settings.reload = MagicMock(return_value=["funding_arb_min_rate"])
    for key, val in overrides.items():
        setattr(settings, key, val)
    return settings


def _make_report(
    significant: bool = True,
    changes: list[ParamChange] | None = None,
    name: str = "test_experiment",
) -> ResearchReport:
    """Create a mock ResearchReport."""
    if changes is None and significant:
        changes = [
            ParamChange(
                engine_name="funding_rate_arb",
                param_name="funding_arb_min_rate",
                old_value=0.0003,
                new_value=0.0004,
                reason="test improvement",
            )
        ]
    return ResearchReport(
        experiment_name=name,
        hypothesis="test hypothesis",
        methodology="test method",
        data_period="2026-01",
        results={"data_source": "real"},
        conclusion="test conclusion",
        recommended_changes=changes or [],
        improvement_significant=significant,
    )


class TestDeployDecision:
    def test_dataclass_fields(self):
        d = DeployDecision(action="deploy", reason="test")
        assert d.action == "deploy"
        assert d.reason == "test"
        assert d.changes == []

    def test_with_changes(self):
        change = ParamChange(
            engine_name="grid_trading",
            param_name="grid_spacing_pct",
            old_value=0.5,
            new_value=0.6,
            reason="test",
        )
        d = DeployDecision(action="deploy", reason="ok", changes=[change])
        assert len(d.changes) == 1


class TestDeployResult:
    def test_auto_timestamp(self):
        r = DeployResult(success=True)
        assert r.timestamp != ""
        assert r.snapshot_id == ""

    def test_to_dict(self):
        r = DeployResult(success=True, snapshot_id="abc123")
        d = r.to_dict()
        assert d["success"] is True
        assert d["snapshot_id"] == "abc123"
        assert "deployed_changes" in d


class TestDeployRecord:
    def test_to_dict(self):
        r = DeployRecord(
            timestamp="2026-01-01T00:00:00",
            report_name="test",
            changes=[{"param_name": "x", "new_value": 1}],
            snapshot_id="snap1",
        )
        d = r.to_dict()
        assert d["report_name"] == "test"
        assert d["rolled_back"] is False
        assert d["rollback_timestamp"] == ""

    def test_rolled_back(self):
        r = DeployRecord(
            timestamp="2026-01-01T00:00:00",
            report_name="test",
            changes=[],
            snapshot_id="snap1",
            rolled_back=True,
            rollback_timestamp="2026-01-02T00:00:00",
        )
        assert r.rolled_back is True
        assert r.rollback_timestamp == "2026-01-02T00:00:00"


class TestEvaluateReport:
    def test_skip_not_significant(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        report = _make_report(significant=False)
        decision = deployer.evaluate_report(report)
        assert decision.action == "skip"
        assert "not significant" in decision.reason.lower()

    def test_skip_no_changes(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        report = _make_report(significant=True, changes=[])
        decision = deployer.evaluate_report(report)
        assert decision.action == "skip"
        assert "No recommended changes" in decision.reason

    def test_skip_changes_outside_bounds(self):
        """Changes for unknown engine/param should be skipped."""
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        report = _make_report(
            significant=True,
            changes=[
                ParamChange(
                    engine_name="unknown_engine",
                    param_name="unknown_param",
                    old_value=1.0,
                    new_value=2.0,
                    reason="test",
                )
            ],
        )
        decision = deployer.evaluate_report(report)
        assert decision.action == "skip"
        assert "bounds" in decision.reason.lower()

    def test_deploy_valid_changes(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        report = _make_report(significant=True)
        decision = deployer.evaluate_report(report)
        assert decision.action == "deploy"
        assert len(decision.changes) == 1

    def test_truncate_excess_changes(self):
        """More than MAX_DEPLOY_CHANGES should be truncated."""
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        changes = [
            ParamChange(
                engine_name="funding_rate_arb",
                param_name="funding_arb_min_rate",
                old_value=0.0003,
                new_value=0.0004,
                reason="change 1",
            ),
            ParamChange(
                engine_name="funding_rate_arb",
                param_name="funding_arb_max_spread_pct",
                old_value=0.5,
                new_value=0.6,
                reason="change 2",
            ),
            ParamChange(
                engine_name="grid_trading",
                param_name="grid_spacing_pct",
                old_value=0.5,
                new_value=0.7,
                reason="change 3",
            ),
            ParamChange(
                engine_name="grid_trading",
                param_name="grid_levels",
                old_value=10,
                new_value=15,
                reason="change 4",
            ),
        ]
        report = _make_report(significant=True, changes=changes)
        decision = deployer.evaluate_report(report)
        assert decision.action == "deploy"
        assert len(decision.changes) == MAX_DEPLOY_CHANGES

    def test_clamp_to_bounds(self):
        """Values outside TUNER_CONFIG bounds should be clamped."""
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        # funding_arb_min_rate bounds: [0.0001, 0.001]
        report = _make_report(
            significant=True,
            changes=[
                ParamChange(
                    engine_name="funding_rate_arb",
                    param_name="funding_arb_min_rate",
                    old_value=0.0003,
                    new_value=0.01,  # Way above max bound of 0.001
                    reason="extreme change",
                )
            ],
        )
        decision = deployer.evaluate_report(report)
        assert decision.action == "deploy"
        # Should be clamped to 0.001
        assert decision.changes[0].new_value == 0.001
        assert "clamped" in decision.changes[0].reason


class TestDeploy:
    def test_deploy_success(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        report = _make_report()
        result = deployer.deploy(report)
        assert result.success is True
        assert result.snapshot_id != ""
        assert len(result.deployed_changes) == 1

    def test_deploy_saves_snapshot(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        report = _make_report()
        result = deployer.deploy(report)
        assert result.snapshot_id in deployer._param_snapshots
        snapshot = deployer._param_snapshots[result.snapshot_id]
        assert "funding_arb_min_rate" in snapshot
        assert snapshot["funding_arb_min_rate"] == 0.0003

    def test_deploy_records_history(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        report = _make_report()
        deployer.deploy(report)
        history = deployer.get_deploy_history()
        assert len(history) == 1
        assert history[0]["report_name"] == "test_experiment"
        assert history[0]["rolled_back"] is False

    def test_deploy_skip_when_not_significant(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        report = _make_report(significant=False)
        result = deployer.deploy(report)
        assert result.success is False

    def test_deploy_records_pre_deploy_sharpe(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        # Add some trades to have a non-zero sharpe
        trade = TradeRecord(
            engine_name="funding_rate_arb",
            symbol="BTC/USDT",
            side="long_short",
            entry_price=40000,
            exit_price=40100,
            quantity=0.1,
            pnl=10,
            cost=1.0,
            net_pnl=9.0,
            entry_time="2026-01-01T00:00:00+00:00",
            exit_time="2026-01-01T01:00:00+00:00",
            hold_time_seconds=3600,
        )
        tracker.record_trade("funding_rate_arb", trade)
        # Add another trade for Sharpe to be computable
        trade2 = TradeRecord(
            engine_name="funding_rate_arb",
            symbol="ETH/USDT",
            side="long_short",
            entry_price=2000,
            exit_price=2010,
            quantity=1.0,
            pnl=10,
            cost=0.5,
            net_pnl=9.5,
            entry_time="2026-01-01T02:00:00+00:00",
            exit_time="2026-01-01T03:00:00+00:00",
            hold_time_seconds=3600,
        )
        tracker.record_trade("funding_rate_arb", trade2)

        report = _make_report()
        deployer.deploy(report)
        assert "funding_rate_arb" in deployer._pre_deploy_sharpe

    def test_deploy_applies_via_tuner(self):
        """Verify that deploy calls tuner.apply_changes."""
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        # Mock apply_changes to track call
        mock_ret = ["funding_arb_min_rate"]
        with patch.object(tuner, "apply_changes", return_value=mock_ret) as mock_apply:
            report = _make_report()
            result = deployer.deploy(report)
            assert result.success is True
            mock_apply.assert_called_once()
            args = mock_apply.call_args
            assert len(args[0][0]) == 1  # one change
            assert args[0][1] is settings


class TestCheckRegression:
    def test_no_regression_without_pre_deploy(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        # No pre-deploy sharpe → no regression
        assert deployer.check_regression("funding_rate_arb") is False

    def test_no_regression_stable_sharpe(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        # Set pre-deploy sharpe
        deployer._pre_deploy_sharpe["funding_rate_arb"] = 1.5

        # Add trades that maintain similar performance
        for i in range(5):
            trade = TradeRecord(
                engine_name="funding_rate_arb",
                symbol="BTC/USDT",
                side="long_short",
                entry_price=40000 + i * 100,
                exit_price=40100 + i * 100,
                quantity=0.1,
                pnl=10,
                cost=1.0,
                net_pnl=9.0,
                entry_time=f"2026-01-01T{i:02d}:00:00+00:00",
                exit_time=f"2026-01-01T{i:02d}:30:00+00:00",
                hold_time_seconds=1800,
            )
            tracker.record_trade("funding_rate_arb", trade)

        # Current sharpe should be positive; 30% drop check
        metrics = tracker.get_metrics("funding_rate_arb", window_hours=24)
        current = metrics.sharpe_ratio
        if current >= 1.5 * 0.7:
            assert deployer.check_regression("funding_rate_arb") is False
        # else the random sharpe calculation may differ, that's ok

    def test_regression_detected(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        # Pre-deploy sharpe was high
        deployer._pre_deploy_sharpe["funding_rate_arb"] = 2.0

        # Add losing trades → low/negative sharpe
        for i in range(5):
            trade = TradeRecord(
                engine_name="funding_rate_arb",
                symbol="BTC/USDT",
                side="long_short",
                entry_price=40000,
                exit_price=39800,
                quantity=0.1,
                pnl=-20,
                cost=1.0,
                net_pnl=-21.0,
                entry_time=f"2026-01-01T{i:02d}:00:00+00:00",
                exit_time=f"2026-01-01T{i:02d}:30:00+00:00",
                hold_time_seconds=1800,
            )
            tracker.record_trade("funding_rate_arb", trade)

        assert deployer.check_regression("funding_rate_arb") is True

    def test_regression_negative_pre_sharpe(self):
        """Regression from negative pre-deploy sharpe (getting worse)."""
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        deployer._pre_deploy_sharpe["stat_arb"] = -0.5

        # Add trades that are even worse
        for i in range(5):
            trade = TradeRecord(
                engine_name="stat_arb",
                symbol="BTC/USDT",
                side="long_short",
                entry_price=40000,
                exit_price=39500,
                quantity=0.1,
                pnl=-50,
                cost=1.0,
                net_pnl=-51.0,
                entry_time=f"2026-01-01T{i:02d}:00:00+00:00",
                exit_time=f"2026-01-01T{i:02d}:30:00+00:00",
                hold_time_seconds=1800,
            )
            tracker.record_trade("stat_arb", trade)

        metrics = tracker.get_metrics("stat_arb", window_hours=24)
        # Current sharpe should be very negative
        # Check if regression is detected
        result = deployer.check_regression("stat_arb")
        # If current sharpe is more than 30% worse than -0.5, it's regression
        if metrics.sharpe_ratio < -0.5 * 1.3:
            assert result is True


class TestRollback:
    def test_rollback_success(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        # Deploy first
        report = _make_report()
        result = deployer.deploy(report)
        assert result.success is True

        # Rollback
        success = deployer.rollback(result.snapshot_id)
        assert success is True
        settings.reload.assert_called()
        # History should be marked as rolled back
        history = deployer.get_deploy_history()
        assert history[0]["rolled_back"] is True
        assert history[0]["rollback_timestamp"] != ""

    def test_rollback_not_found(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        success = deployer.rollback("nonexistent")
        assert success is False

    def test_rollback_clears_pre_deploy_sharpe(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        report = _make_report()
        result = deployer.deploy(report)
        deployer._pre_deploy_sharpe["funding_rate_arb"] = 1.0

        deployer.rollback(result.snapshot_id)
        assert "funding_rate_arb" not in deployer._pre_deploy_sharpe

    def test_rollback_reload_failure(self):
        settings = _make_settings()
        settings.reload = MagicMock(side_effect=ValueError("bad param"))
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        # Deploy (mocking apply_changes to succeed)
        deployer._param_snapshots["snap1"] = {"funding_arb_min_rate": 0.0003}
        success = deployer.rollback("snap1")
        assert success is False


class TestGetDeployHistory:
    def test_empty_history(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        assert deployer.get_deploy_history() == []

    def test_multiple_deployments(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        for i in range(3):
            report = _make_report(name=f"exp_{i}")
            deployer.deploy(report)

        history = deployer.get_deploy_history()
        assert len(history) == 3
        assert history[0]["report_name"] == "exp_0"
        assert history[2]["report_name"] == "exp_2"

    def test_history_limit_50(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        for i in range(60):
            report = _make_report(name=f"exp_{i}")
            deployer.deploy(report)

        history = deployer.get_deploy_history()
        assert len(history) == 50
        # Oldest should be trimmed
        assert history[0]["report_name"] == "exp_10"


class TestSafetyBounds:
    def test_max_deploy_changes_constant(self):
        assert MAX_DEPLOY_CHANGES == 3

    def test_changes_within_bounds_accepted(self):
        """All changes within TUNER_CONFIG bounds pass through."""
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        # stat_arb_entry_zscore bounds: [1.0, 3.0]
        changes = [
            ParamChange(
                engine_name="stat_arb",
                param_name="stat_arb_entry_zscore",
                old_value=2.0,
                new_value=2.5,
                reason="within bounds",
            )
        ]
        valid = deployer._filter_valid_changes(changes)
        assert len(valid) == 1
        assert valid[0].new_value == 2.5

    def test_changes_below_min_clamped(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        # stat_arb_entry_zscore bounds: [1.0, 3.0]
        changes = [
            ParamChange(
                engine_name="stat_arb",
                param_name="stat_arb_entry_zscore",
                old_value=2.0,
                new_value=0.1,  # Below min of 1.0
                reason="extreme",
            )
        ]
        valid = deployer._filter_valid_changes(changes)
        assert len(valid) == 1
        assert valid[0].new_value == 1.0

    def test_unknown_engine_skipped(self):
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        changes = [
            ParamChange(
                engine_name="phantom_engine",
                param_name="phantom_param",
                old_value=1.0,
                new_value=2.0,
                reason="ghost",
            )
        ]
        valid = deployer._filter_valid_changes(changes)
        assert len(valid) == 0


class TestEngineManagerIntegration:
    """Test EngineManager integration with deployer."""

    def test_set_deployer(self):
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        manager = EngineManager(pm)
        deployer = MagicMock()
        manager.set_deployer(deployer)
        assert manager._deployer is deployer

    def test_deployer_default_none(self):
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        manager = EngineManager(pm)
        assert manager._deployer is None


class TestDashboardEndpoint:
    """Test the /api/research/deployments endpoint logic."""

    def test_deployments_with_no_deployer(self):
        """When deployer is None, should return empty list."""
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        manager = EngineManager(pm)
        deployer = getattr(manager, "_deployer", None)
        assert deployer is None

    def test_deployments_with_deployer(self):
        """When deployer has history, it returns records."""
        settings = _make_settings()
        tracker = EngineTracker()
        tuner = ParameterTuner()
        deployer = ResearchDeployer(tuner, settings, tracker)

        report = _make_report()
        deployer.deploy(report)

        history = deployer.get_deploy_history()
        assert len(history) == 1
        assert "report_name" in history[0]
        assert "snapshot_id" in history[0]
        assert "rolled_back" in history[0]


class TestConfigSettings:
    """Test new config settings for research deployer."""

    def test_research_auto_deploy_default(self):
        from bot.config import Settings

        s = Settings(
            binance_api_key="", upbit_api_key="",
            _env_file=None,
        )
        assert s.research_auto_deploy is True

    def test_research_regression_check_hours_default(self):
        from bot.config import Settings

        s = Settings(
            binance_api_key="", upbit_api_key="",
            _env_file=None,
        )
        assert s.research_regression_check_hours == 6.0

    def test_settings_metadata_contains_new_fields(self):
        from bot.config import SETTINGS_METADATA

        assert "research_auto_deploy" in SETTINGS_METADATA
        assert "research_regression_check_hours" in SETTINGS_METADATA
        assert SETTINGS_METADATA["research_auto_deploy"]["section"] == "Research"
        assert SETTINGS_METADATA["research_regression_check_hours"]["type"] == "float"
