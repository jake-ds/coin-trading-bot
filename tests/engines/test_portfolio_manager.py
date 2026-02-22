"""Tests for PortfolioManager."""

import pytest

from bot.engines.portfolio_manager import PortfolioManager


@pytest.fixture
def pm():
    return PortfolioManager(
        total_capital=10000.0,
        engine_allocations={
            "engine_a": 0.30,
            "engine_b": 0.25,
            "engine_c": 0.20,
        },
        max_drawdown_pct=15.0,
    )


class TestPortfolioManagerInit:
    def test_initial_state(self, pm):
        assert pm.total_capital == 10000.0
        assert pm.available_capital == 10000.0
        assert pm.total_allocated == 0.0
        assert pm.total_pnl == 0.0

    def test_max_allocation(self, pm):
        assert pm.get_max_allocation("engine_a") == 3000.0
        assert pm.get_max_allocation("engine_b") == 2500.0
        assert pm.get_max_allocation("engine_c") == 2000.0
        assert pm.get_max_allocation("unknown") == 0.0


class TestCapitalAllocation:
    def test_request_within_limit(self, pm):
        allocated = pm.request_capital("engine_a", 2000.0)
        assert allocated == 2000.0
        assert pm.total_allocated == 2000.0
        assert pm.available_capital == 8000.0

    def test_request_capped_by_engine_limit(self, pm):
        allocated = pm.request_capital("engine_a", 5000.0)
        assert allocated == 3000.0  # engine_a limited to 30%

    def test_request_capped_by_available(self, pm):
        pm.request_capital("engine_a", 3000.0)
        pm.request_capital("engine_b", 2500.0)
        pm.request_capital("engine_c", 2000.0)
        # Only 2500 left available, but "engine_a" already at max
        allocated = pm.request_capital("engine_a", 1000.0)
        assert allocated == 0.0  # already at max

    def test_request_zero_for_unregistered(self, pm):
        allocated = pm.request_capital("unknown_engine", 1000.0)
        assert allocated == 0.0

    def test_release_capital(self, pm):
        pm.request_capital("engine_a", 2000.0)
        pm.release_capital("engine_a", 1000.0)
        assert pm.total_allocated == 1000.0
        assert pm.available_capital == 9000.0

    def test_release_all_capital(self, pm):
        pm.request_capital("engine_a", 2000.0)
        pm.release_capital("engine_a", 2000.0)
        assert pm.total_allocated == 0.0
        assert "engine_a" not in pm._allocated

    def test_release_more_than_allocated(self, pm):
        pm.request_capital("engine_a", 1000.0)
        pm.release_capital("engine_a", 5000.0)  # Can't release more than allocated
        assert pm.total_allocated == 0.0

    def test_incremental_allocation(self, pm):
        first = pm.request_capital("engine_a", 1000.0)
        assert first == 1000.0
        second = pm.request_capital("engine_a", 1500.0)
        assert second == 1500.0
        third = pm.request_capital("engine_a", 1000.0)
        assert third == 500.0  # Only 500 left under 3000 cap


class TestPnLTracking:
    def test_report_pnl(self, pm):
        pm.report_pnl("engine_a", 100.0)
        assert pm.get_engine_pnl("engine_a") == 100.0
        assert pm.total_pnl == 100.0

    def test_cumulative_pnl(self, pm):
        pm.report_pnl("engine_a", 50.0)
        pm.report_pnl("engine_a", -20.0)
        assert pm.get_engine_pnl("engine_a") == 30.0

    def test_multi_engine_pnl(self, pm):
        pm.report_pnl("engine_a", 100.0)
        pm.report_pnl("engine_b", -50.0)
        assert pm.total_pnl == 50.0


class TestDrawdownTracking:
    def test_no_drawdown_initially(self, pm):
        assert pm.get_global_drawdown() == 0.0
        assert not pm.is_drawdown_breached()

    def test_drawdown_from_loss(self, pm):
        pm.report_pnl("engine_a", 1000.0)  # peak = 11000
        pm.report_pnl("engine_a", -2000.0)  # current = 9000
        # drawdown = (11000 - 9000) / 11000 * 100 ≈ 18.18%
        dd = pm.get_global_drawdown()
        assert 18.0 < dd < 19.0
        assert pm.is_drawdown_breached()  # > 15%

    def test_drawdown_not_breached(self, pm):
        pm.report_pnl("engine_a", 100.0)
        pm.report_pnl("engine_a", -50.0)
        assert not pm.is_drawdown_breached()


class TestEngineAllocationStatus:
    def test_get_engine_allocation(self, pm):
        pm.request_capital("engine_a", 2000.0)
        pm.report_pnl("engine_a", 100.0)
        info = pm.get_engine_allocation("engine_a")
        assert info["engine"] == "engine_a"
        assert info["allocated"] == 2000.0
        assert info["max_allowed"] == 3000.0
        assert info["pnl"] == 100.0

    def test_get_summary(self, pm):
        pm.request_capital("engine_a", 1000.0)
        pm.request_capital("engine_b", 500.0)
        pm.report_pnl("engine_a", 50.0)
        summary = pm.get_summary()
        assert summary["total_capital"] == 10000.0
        assert summary["total_allocated"] == 1500.0
        assert summary["available_capital"] == 8500.0
        assert summary["total_pnl"] == 50.0
        assert "engine_a" in summary["engine_allocations"]
        assert "engine_b" in summary["engine_allocations"]
