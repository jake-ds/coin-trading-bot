"""Tests for ParameterTuner — V5-008 automatic strategy parameter adjustment."""


from bot.engines.tracker import EngineMetrics
from bot.engines.tuner import (
    MAX_ADJUSTMENT_PCT,
    TUNER_CONFIG,
    ParamChange,
    ParameterTuner,
)

# ------------------------------------------------------------------ #
# ParamChange dataclass
# ------------------------------------------------------------------ #


class TestParamChange:
    def test_fields(self):
        c = ParamChange(
            engine_name="grid_trading",
            param_name="grid_spacing_pct",
            old_value=1.0,
            new_value=1.15,
            reason="test",
        )
        assert c.engine_name == "grid_trading"
        assert c.timestamp  # auto-filled

    def test_to_dict(self):
        c = ParamChange(
            engine_name="e", param_name="p",
            old_value=1, new_value=2, reason="r",
        )
        d = c.to_dict()
        assert d["old_value"] == 1
        assert d["new_value"] == 2


# ------------------------------------------------------------------ #
# TUNER_CONFIG
# ------------------------------------------------------------------ #


class TestTunerConfig:
    def test_all_engines_have_config(self):
        expected = ["funding_rate_arb", "grid_trading", "cross_exchange_arb", "stat_arb"]
        for name in expected:
            assert name in TUNER_CONFIG

    def test_bounds_are_valid(self):
        for engine, params in TUNER_CONFIG.items():
            for param, bounds in params.items():
                assert bounds.min_val < bounds.max_val, (
                    f"{engine}.{param}: min >= max"
                )


# ------------------------------------------------------------------ #
# ParameterTuner — Sharpe < 0 → conservative
# ------------------------------------------------------------------ #


class TestConservativeAdjustment:
    """Sharpe < 0 should raise thresholds (conservative)."""

    def test_raises_min_rate(self):
        tuner = ParameterTuner()
        metrics = EngineMetrics(sharpe_ratio=-0.5, total_trades=5)
        params = {"funding_arb_min_rate": 0.0003}

        changes = tuner.evaluate_and_adjust("funding_rate_arb", metrics, params)
        rate_changes = [c for c in changes if c.param_name == "funding_arb_min_rate"]
        assert len(rate_changes) == 1
        assert rate_changes[0].new_value > 0.0003  # Raised

    def test_raises_entry_zscore(self):
        tuner = ParameterTuner()
        metrics = EngineMetrics(sharpe_ratio=-1.0, total_trades=5)
        params = {"stat_arb_entry_zscore": 2.0, "stat_arb_exit_zscore": 0.5,
                  "stat_arb_lookback": 100, "stat_arb_min_correlation": 0.7}

        changes = tuner.evaluate_and_adjust("stat_arb", metrics, params)
        zscore_change = next(
            (c for c in changes if c.param_name == "stat_arb_entry_zscore"), None
        )
        assert zscore_change is not None
        assert zscore_change.new_value > 2.0

    def test_widens_grid_spacing(self):
        tuner = ParameterTuner()
        metrics = EngineMetrics(sharpe_ratio=-0.3, total_trades=5)
        params = {"grid_spacing_pct": 1.0, "grid_levels": 10}

        changes = tuner.evaluate_and_adjust("grid_trading", metrics, params)
        spacing_change = next(
            (c for c in changes if c.param_name == "grid_spacing_pct"), None
        )
        assert spacing_change is not None
        assert spacing_change.new_value > 1.0

    def test_reduces_grid_levels(self):
        tuner = ParameterTuner()
        metrics = EngineMetrics(sharpe_ratio=-0.3, total_trades=5)
        params = {"grid_spacing_pct": 1.0, "grid_levels": 10}

        changes = tuner.evaluate_and_adjust("grid_trading", metrics, params)
        levels_change = next(
            (c for c in changes if c.param_name == "grid_levels"), None
        )
        assert levels_change is not None
        assert levels_change.new_value < 10

    def test_adjustment_max_bounded(self):
        """Conservative adjustment should not exceed MAX_ADJUSTMENT_PCT."""
        tuner = ParameterTuner()
        metrics = EngineMetrics(sharpe_ratio=-5.0, total_trades=5)
        params = {"cross_arb_min_spread_pct": 0.3}

        changes = tuner.evaluate_and_adjust("cross_exchange_arb", metrics, params)
        if changes:
            c = changes[0]
            max_new = 0.3 * (1 + MAX_ADJUSTMENT_PCT)
            assert c.new_value <= max_new + 1e-10


# ------------------------------------------------------------------ #
# ParameterTuner — Sharpe >= 1.0 → aggressive
# ------------------------------------------------------------------ #


class TestAggressiveAdjustment:
    """Sharpe >= 1.0 should lower thresholds (more aggressive)."""

    def test_lowers_min_rate(self):
        tuner = ParameterTuner()
        metrics = EngineMetrics(sharpe_ratio=1.5, total_trades=10)
        params = {"funding_arb_min_rate": 0.0005}

        changes = tuner.evaluate_and_adjust("funding_rate_arb", metrics, params)
        rate_change = next(
            (c for c in changes if c.param_name == "funding_arb_min_rate"), None
        )
        assert rate_change is not None
        assert rate_change.new_value < 0.0005

    def test_lowers_entry_zscore(self):
        tuner = ParameterTuner()
        metrics = EngineMetrics(sharpe_ratio=2.0, total_trades=10)
        params = {"stat_arb_entry_zscore": 2.0, "stat_arb_exit_zscore": 0.5,
                  "stat_arb_lookback": 100, "stat_arb_min_correlation": 0.7}

        changes = tuner.evaluate_and_adjust("stat_arb", metrics, params)
        zscore_change = next(
            (c for c in changes if c.param_name == "stat_arb_entry_zscore"), None
        )
        assert zscore_change is not None
        assert zscore_change.new_value < 2.0


# ------------------------------------------------------------------ #
# Steady state (0.5 <= sharpe < 1.0) — no changes
# ------------------------------------------------------------------ #


class TestSteadyState:
    def test_no_changes_when_steady(self):
        tuner = ParameterTuner()
        metrics = EngineMetrics(sharpe_ratio=0.7, total_trades=10)
        params = {"funding_arb_min_rate": 0.0003}

        changes = tuner.evaluate_and_adjust("funding_rate_arb", metrics, params)
        assert len(changes) == 0


# ------------------------------------------------------------------ #
# Bounds enforcement
# ------------------------------------------------------------------ #


class TestBoundsEnforcement:
    def test_clamps_to_max(self):
        tuner = ParameterTuner()
        metrics = EngineMetrics(sharpe_ratio=-2.0, total_trades=5)
        # Value already near max — should clamp
        bounds = TUNER_CONFIG["funding_rate_arb"]["funding_arb_min_rate"]
        params = {"funding_arb_min_rate": bounds.max_val * 0.95}

        changes = tuner.evaluate_and_adjust("funding_rate_arb", metrics, params)
        if changes:
            assert changes[0].new_value <= bounds.max_val

    def test_clamps_to_min(self):
        tuner = ParameterTuner()
        metrics = EngineMetrics(sharpe_ratio=2.0, total_trades=10)
        # Value already near min
        bounds = TUNER_CONFIG["cross_exchange_arb"]["cross_arb_min_spread_pct"]
        params = {"cross_arb_min_spread_pct": bounds.min_val * 1.05}

        changes = tuner.evaluate_and_adjust("cross_exchange_arb", metrics, params)
        if changes:
            assert changes[0].new_value >= bounds.min_val

    def test_integer_params_rounded(self):
        tuner = ParameterTuner()
        metrics = EngineMetrics(sharpe_ratio=-1.0, total_trades=5)
        params = {"grid_spacing_pct": 1.0, "grid_levels": 15}

        changes = tuner.evaluate_and_adjust("grid_trading", metrics, params)
        levels_change = next(
            (c for c in changes if c.param_name == "grid_levels"), None
        )
        if levels_change:
            assert isinstance(levels_change.new_value, int)


# ------------------------------------------------------------------ #
# History tracking
# ------------------------------------------------------------------ #


class TestHistory:
    def test_empty_history(self):
        tuner = ParameterTuner()
        assert tuner.get_history("funding_rate_arb") == []

    def test_history_records_changes(self):
        tuner = ParameterTuner()
        metrics = EngineMetrics(sharpe_ratio=-1.0, total_trades=5)
        params = {"funding_arb_min_rate": 0.0003}
        tuner.evaluate_and_adjust("funding_rate_arb", metrics, params)

        history = tuner.get_history("funding_rate_arb")
        assert len(history) >= 1
        assert history[0].engine_name == "funding_rate_arb"

    def test_history_accumulates(self):
        tuner = ParameterTuner()
        metrics = EngineMetrics(sharpe_ratio=-1.0, total_trades=5)
        params = {"funding_arb_min_rate": 0.0003}

        tuner.evaluate_and_adjust("funding_rate_arb", metrics, params)
        tuner.evaluate_and_adjust("funding_rate_arb", metrics, params)

        history = tuner.get_history("funding_rate_arb")
        assert len(history) >= 2


# ------------------------------------------------------------------ #
# apply_changes
# ------------------------------------------------------------------ #


class TestApplyChanges:
    def test_apply_calls_reload(self):
        tuner = ParameterTuner()

        class MockSettings:
            def reload(self, updates):
                self.last_updates = updates
                return list(updates.keys())

        settings = MockSettings()
        changes = [
            ParamChange(
                engine_name="e", param_name="grid_spacing_pct",
                old_value=1.0, new_value=1.15, reason="test",
            ),
        ]
        result = tuner.apply_changes(changes, settings)
        assert result == ["grid_spacing_pct"]
        assert settings.last_updates == {"grid_spacing_pct": 1.15}

    def test_apply_empty_changes(self):
        tuner = ParameterTuner()

        class MockSettings:
            def reload(self, updates):
                return []

        result = tuner.apply_changes([], MockSettings())
        assert result == []

    def test_apply_handles_error(self):
        tuner = ParameterTuner()

        class MockSettings:
            def reload(self, updates):
                raise ValueError("Bad param")

        result = tuner.apply_changes(
            [ParamChange("e", "bad", 1, 2, "test")],
            MockSettings(),
        )
        assert result == []


# ------------------------------------------------------------------ #
# DecisionSteps
# ------------------------------------------------------------------ #


class TestDecisionSteps:
    def test_get_decisions(self):
        tuner = ParameterTuner()
        changes = [
            ParamChange("e", "grid_spacing_pct", 1.0, 1.15, "test reason"),
        ]
        decisions = tuner.get_decisions(changes)
        assert len(decisions) == 1
        assert "파라미터 조정" in decisions[0].label
        assert decisions[0].category == "decide"


# ------------------------------------------------------------------ #
# Unknown engine
# ------------------------------------------------------------------ #


class TestUnknownEngine:
    def test_unknown_engine_returns_empty(self):
        tuner = ParameterTuner()
        metrics = EngineMetrics(sharpe_ratio=-1.0, total_trades=5)
        changes = tuner.evaluate_and_adjust("nonexistent", metrics, {})
        assert changes == []
