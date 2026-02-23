"""Tests for Risk Drawdown endpoint — /api/risk/drawdown."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bot.engines.portfolio_manager import PortfolioManager

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _make_pm_with_history() -> PortfolioManager:
    """Build a PortfolioManager with drawdown history from PnL reports."""
    pm = PortfolioManager(
        total_capital=10000.0,
        engine_allocations={"engine_a": 0.5, "engine_b": 0.5},
        max_drawdown_pct=15.0,
    )
    # Simulate PnL reports that generate drawdown history
    pm.report_pnl("engine_a", 100.0)   # equity goes up → drawdown 0%
    pm.report_pnl("engine_a", -200.0)  # equity drops → drawdown > 0%
    pm.report_pnl("engine_b", 50.0)    # partial recovery
    return pm


def _mock_engine_manager_with_pm() -> MagicMock:
    mgr = MagicMock()
    mgr._portfolio_manager = _make_pm_with_history()
    return mgr


# ──────────────────────────────────────────────────────────────
# /api/risk/drawdown
# ──────────────────────────────────────────────────────────────


class TestDrawdownEndpoint:
    @pytest.mark.asyncio
    async def test_no_engine_manager(self):
        from bot.dashboard.app import get_risk_drawdown, set_engine_manager

        set_engine_manager(None)
        result = await get_risk_drawdown()
        assert result["history"] == []

    @pytest.mark.asyncio
    async def test_no_portfolio_manager(self):
        from bot.dashboard.app import get_risk_drawdown, set_engine_manager

        mgr = MagicMock(spec=[])  # no attributes
        set_engine_manager(mgr)
        result = await get_risk_drawdown()
        assert result["history"] == []

    @pytest.mark.asyncio
    async def test_with_history(self):
        from bot.dashboard.app import get_risk_drawdown, set_engine_manager

        mgr = _mock_engine_manager_with_pm()
        set_engine_manager(mgr)
        result = await get_risk_drawdown()
        assert len(result["history"]) == 3
        # Each entry should have timestamp, drawdown_pct, equity
        for entry in result["history"]:
            assert "timestamp" in entry
            assert "drawdown_pct" in entry
            assert "equity" in entry

    @pytest.mark.asyncio
    async def test_history_values(self):
        from bot.dashboard.app import get_risk_drawdown, set_engine_manager

        mgr = _mock_engine_manager_with_pm()
        set_engine_manager(mgr)
        result = await get_risk_drawdown()
        history = result["history"]
        # First report: +100, equity=10100, peak=10100, drawdown=0
        assert history[0]["drawdown_pct"] == 0.0
        assert history[0]["equity"] == 10100.0
        # Second report: -200, equity=9900, peak=10100, drawdown>0
        assert history[1]["drawdown_pct"] > 0
        assert history[1]["equity"] == 9900.0
        # Third report: +50, equity=9950, peak=10100, still in drawdown
        assert history[2]["drawdown_pct"] > 0
        assert history[2]["equity"] == 9950.0


# ──────────────────────────────────────────────────────────────
# PortfolioManager._drawdown_history
# ──────────────────────────────────────────────────────────────


class TestPortfolioManagerDrawdownHistory:
    def test_initial_empty(self):
        pm = PortfolioManager(total_capital=10000.0)
        assert pm._drawdown_history == []

    def test_report_pnl_adds_entry(self):
        pm = PortfolioManager(total_capital=10000.0)
        pm.report_pnl("engine_a", 50.0)
        assert len(pm._drawdown_history) == 1
        entry = pm._drawdown_history[0]
        assert "timestamp" in entry
        assert "drawdown_pct" in entry
        assert "equity" in entry

    def test_multiple_reports(self):
        pm = PortfolioManager(total_capital=10000.0)
        for _ in range(5):
            pm.report_pnl("engine_a", -10.0)
        assert len(pm._drawdown_history) == 5
        # Drawdown should increase
        assert pm._drawdown_history[-1]["drawdown_pct"] > 0

    def test_history_capped_at_1000(self):
        pm = PortfolioManager(total_capital=10000.0)
        for i in range(1100):
            pm.report_pnl("engine_a", 0.01)
        assert len(pm._drawdown_history) == 1000

    def test_drawdown_zero_on_new_high(self):
        pm = PortfolioManager(total_capital=10000.0)
        pm.report_pnl("engine_a", 100.0)
        assert pm._drawdown_history[-1]["drawdown_pct"] == 0.0

    def test_drawdown_nonzero_after_loss(self):
        pm = PortfolioManager(total_capital=10000.0)
        pm.report_pnl("engine_a", 100.0)  # peak=10100
        pm.report_pnl("engine_a", -200.0)  # equity=9900
        entry = pm._drawdown_history[-1]
        expected_dd = round((10100 - 9900) / 10100 * 100, 2)
        assert entry["drawdown_pct"] == expected_dd
