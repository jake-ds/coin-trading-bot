"""Tests for CorrelationRiskController — cross-engine symbol concentration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bot.risk.correlation_controller import CorrelationRiskController

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _mock_portfolio_risk(portfolio_value: float = 100_000.0):
    """Create a mock PortfolioRiskManager with a given portfolio value."""
    prm = MagicMock()
    prm.portfolio_value = portfolio_value
    return prm


def _make_positions(positions: list[tuple[str, str, float]]) -> list[dict]:
    """Create position dicts from (symbol, side, notional) tuples."""
    return [
        {"symbol": sym, "side": side, "notional": notional}
        for sym, side, notional in positions
    ]


# ──────────────────────────────────────────────────────────────
# Init
# ──────────────────────────────────────────────────────────────


class TestInit:
    def test_default_params(self):
        ctrl = CorrelationRiskController()
        assert ctrl._max_cross_engine_correlation == 0.85
        assert ctrl._max_symbol_concentration == 0.4
        assert ctrl._portfolio_risk is None
        assert ctrl._engine_positions == {}

    def test_custom_params(self):
        prm = _mock_portfolio_risk()
        ctrl = CorrelationRiskController(
            portfolio_risk=prm,
            max_cross_engine_correlation=0.7,
            max_symbol_concentration=0.3,
        )
        assert ctrl._max_cross_engine_correlation == 0.7
        assert ctrl._max_symbol_concentration == 0.3
        assert ctrl._portfolio_risk is prm


# ──────────────────────────────────────────────────────────────
# update_positions
# ──────────────────────────────────────────────────────────────


class TestUpdatePositions:
    def test_updates_positions(self):
        ctrl = CorrelationRiskController()
        positions = {
            "funding_arb": _make_positions([("BTC/USDT", "long", 10000)]),
            "grid_trading": _make_positions([("ETH/USDT", "long", 5000)]),
        }
        ctrl.update_positions(positions)
        assert "funding_arb" in ctrl._engine_positions
        assert len(ctrl._engine_positions["funding_arb"]) == 1

    def test_replaces_old_positions(self):
        ctrl = CorrelationRiskController()
        ctrl.update_positions({"eng_a": [{"symbol": "BTC", "side": "long", "notional": 100}]})
        ctrl.update_positions({"eng_b": [{"symbol": "ETH", "side": "long", "notional": 200}]})
        assert "eng_a" not in ctrl._engine_positions
        assert "eng_b" in ctrl._engine_positions

    def test_empty_positions(self):
        ctrl = CorrelationRiskController()
        ctrl.update_positions({})
        assert ctrl._engine_positions == {}


# ──────────────────────────────────────────────────────────────
# calculate_cross_engine_correlation
# ──────────────────────────────────────────────────────────────


class TestCrossEngineCorrelation:
    def test_no_engines(self):
        ctrl = CorrelationRiskController()
        result = ctrl.calculate_cross_engine_correlation()
        assert result == {}

    def test_single_engine(self):
        ctrl = CorrelationRiskController()
        ctrl.update_positions({
            "funding_arb": _make_positions([("BTC/USDT", "long", 10000)]),
        })
        result = ctrl.calculate_cross_engine_correlation()
        assert result == {}  # No pairs to compare

    def test_no_overlap(self):
        ctrl = CorrelationRiskController()
        ctrl.update_positions({
            "funding_arb": _make_positions([("BTC/USDT", "long", 10000)]),
            "grid_trading": _make_positions([("ETH/USDT", "long", 5000)]),
        })
        result = ctrl.calculate_cross_engine_correlation()
        key = "funding_arb|grid_trading"
        assert key in result
        assert result[key]["overlap_symbols"] == []
        assert result[key]["overlap_pct"] == 0.0
        assert result[key]["concentration_score"] == 0.0

    def test_full_overlap(self):
        ctrl = CorrelationRiskController()
        ctrl.update_positions({
            "eng_a": _make_positions([("BTC/USDT", "long", 10000)]),
            "eng_b": _make_positions([("BTC/USDT", "long", 5000)]),
        })
        result = ctrl.calculate_cross_engine_correlation()
        key = "eng_a|eng_b"
        assert result[key]["overlap_symbols"] == ["BTC/USDT"]
        assert result[key]["overlap_pct"] == 1.0
        assert result[key]["concentration_score"] == 1.0

    def test_partial_overlap(self):
        ctrl = CorrelationRiskController()
        ctrl.update_positions({
            "eng_a": _make_positions([
                ("BTC/USDT", "long", 10000),
                ("ETH/USDT", "long", 5000),
            ]),
            "eng_b": _make_positions([
                ("BTC/USDT", "long", 8000),
                ("SOL/USDT", "long", 3000),
            ]),
        })
        result = ctrl.calculate_cross_engine_correlation()
        key = "eng_a|eng_b"
        assert result[key]["overlap_symbols"] == ["BTC/USDT"]
        # 3 unique symbols, 1 overlap → 1/3
        assert result[key]["overlap_pct"] == pytest.approx(1 / 3, abs=0.01)
        # overlap notional = 10000 + 8000 = 18000, total = 26000
        assert result[key]["concentration_score"] == pytest.approx(18000 / 26000, abs=0.01)

    def test_three_engines(self):
        ctrl = CorrelationRiskController()
        ctrl.update_positions({
            "eng_a": _make_positions([("BTC/USDT", "long", 10000)]),
            "eng_b": _make_positions([("BTC/USDT", "long", 5000)]),
            "eng_c": _make_positions([("ETH/USDT", "long", 3000)]),
        })
        result = ctrl.calculate_cross_engine_correlation()
        # 3 pairs: eng_a|eng_b, eng_a|eng_c, eng_b|eng_c
        assert len(result) == 3
        assert result["eng_a|eng_b"]["overlap_symbols"] == ["BTC/USDT"]
        assert result["eng_a|eng_c"]["overlap_symbols"] == []
        assert result["eng_b|eng_c"]["overlap_symbols"] == []

    def test_empty_positions_in_engine(self):
        ctrl = CorrelationRiskController()
        ctrl.update_positions({
            "eng_a": [],
            "eng_b": _make_positions([("BTC/USDT", "long", 10000)]),
        })
        result = ctrl.calculate_cross_engine_correlation()
        key = "eng_a|eng_b"
        assert result[key]["overlap_pct"] == 0.0


# ──────────────────────────────────────────────────────────────
# check_symbol_concentration
# ──────────────────────────────────────────────────────────────


class TestCheckSymbolConcentration:
    def test_within_limit(self):
        prm = _mock_portfolio_risk(100_000)
        ctrl = CorrelationRiskController(
            portfolio_risk=prm,
            max_symbol_concentration=0.4,
        )
        ctrl.update_positions({
            "eng_a": _make_positions([("BTC/USDT", "long", 20000)]),
            "eng_b": _make_positions([("BTC/USDT", "long", 10000)]),
        })
        allowed, reason = ctrl.check_symbol_concentration("BTC/USDT")
        # 30000/100000 = 0.30 < 0.40
        assert allowed is True
        assert reason == ""

    def test_exceeds_limit(self):
        prm = _mock_portfolio_risk(100_000)
        ctrl = CorrelationRiskController(
            portfolio_risk=prm,
            max_symbol_concentration=0.4,
        )
        ctrl.update_positions({
            "eng_a": _make_positions([("BTC/USDT", "long", 25000)]),
            "eng_b": _make_positions([("BTC/USDT", "long", 20000)]),
        })
        allowed, reason = ctrl.check_symbol_concentration("BTC/USDT")
        # 45000/100000 = 0.45 > 0.40
        assert allowed is False
        assert "BTC/USDT" in reason
        assert "exceeds" in reason

    def test_exact_limit(self):
        prm = _mock_portfolio_risk(100_000)
        ctrl = CorrelationRiskController(
            portfolio_risk=prm,
            max_symbol_concentration=0.4,
        )
        ctrl.update_positions({
            "eng_a": _make_positions([("BTC/USDT", "long", 20000)]),
            "eng_b": _make_positions([("BTC/USDT", "long", 20000)]),
        })
        allowed, reason = ctrl.check_symbol_concentration("BTC/USDT")
        # 40000/100000 = 0.40 → exactly at limit, not exceeding
        assert allowed is True

    def test_symbol_not_present(self):
        prm = _mock_portfolio_risk(100_000)
        ctrl = CorrelationRiskController(
            portfolio_risk=prm,
            max_symbol_concentration=0.4,
        )
        ctrl.update_positions({
            "eng_a": _make_positions([("BTC/USDT", "long", 50000)]),
        })
        allowed, reason = ctrl.check_symbol_concentration("ETH/USDT")
        # No exposure to ETH/USDT → 0%
        assert allowed is True

    def test_no_portfolio_risk(self):
        ctrl = CorrelationRiskController(
            portfolio_risk=None,
            max_symbol_concentration=0.4,
        )
        ctrl.update_positions({
            "eng_a": _make_positions([("BTC/USDT", "long", 999999)]),
        })
        allowed, reason = ctrl.check_symbol_concentration("BTC/USDT")
        # No portfolio value → can't calculate → allow
        assert allowed is True

    def test_zero_portfolio_value(self):
        prm = _mock_portfolio_risk(0.0)
        ctrl = CorrelationRiskController(
            portfolio_risk=prm,
            max_symbol_concentration=0.4,
        )
        allowed, reason = ctrl.check_symbol_concentration("BTC/USDT")
        assert allowed is True

    def test_engines_listed_in_reason(self):
        prm = _mock_portfolio_risk(100_000)
        ctrl = CorrelationRiskController(
            portfolio_risk=prm,
            max_symbol_concentration=0.1,
        )
        ctrl.update_positions({
            "eng_a": _make_positions([("BTC/USDT", "long", 6000)]),
            "eng_b": _make_positions([("BTC/USDT", "long", 6000)]),
        })
        allowed, reason = ctrl.check_symbol_concentration("BTC/USDT")
        assert allowed is False
        assert "eng_a" in reason
        assert "eng_b" in reason


# ──────────────────────────────────────────────────────────────
# get_concentration_report
# ──────────────────────────────────────────────────────────────


class TestGetConcentrationReport:
    def test_empty_report(self):
        ctrl = CorrelationRiskController()
        report = ctrl.get_concentration_report()
        assert report["per_symbol"] == {}
        assert report["cross_engine_correlations"] == {}
        assert report["alerts"] == []

    def test_report_structure(self):
        prm = _mock_portfolio_risk(100_000)
        ctrl = CorrelationRiskController(
            portfolio_risk=prm,
            max_symbol_concentration=0.4,
        )
        ctrl.update_positions({
            "eng_a": _make_positions([
                ("BTC/USDT", "long", 20000),
                ("ETH/USDT", "long", 10000),
            ]),
            "eng_b": _make_positions([
                ("BTC/USDT", "long", 15000),
            ]),
        })
        report = ctrl.get_concentration_report()

        # per_symbol
        assert "BTC/USDT" in report["per_symbol"]
        btc = report["per_symbol"]["BTC/USDT"]
        assert set(btc["engines"]) == {"eng_a", "eng_b"}
        assert btc["total_notional"] == 35000.0
        assert btc["pct_of_capital"] == pytest.approx(0.35, abs=0.01)

        assert "ETH/USDT" in report["per_symbol"]
        eth = report["per_symbol"]["ETH/USDT"]
        assert eth["engines"] == ["eng_a"]
        assert eth["total_notional"] == 10000.0

        # cross_engine_correlations
        assert "eng_a|eng_b" in report["cross_engine_correlations"]

        # alerts — no concentration exceeded with 0.4 limit
        assert report["alerts"] == []

    def test_report_with_concentration_alert(self):
        prm = _mock_portfolio_risk(100_000)
        ctrl = CorrelationRiskController(
            portfolio_risk=prm,
            max_symbol_concentration=0.3,
        )
        ctrl.update_positions({
            "eng_a": _make_positions([("BTC/USDT", "long", 20000)]),
            "eng_b": _make_positions([("BTC/USDT", "long", 20000)]),
        })
        report = ctrl.get_concentration_report()
        # 40000/100000 = 0.4 > 0.3 → alert
        assert len(report["alerts"]) >= 1
        assert any("BTC/USDT" in a for a in report["alerts"])

    def test_report_with_cross_engine_alert(self):
        prm = _mock_portfolio_risk(100_000)
        ctrl = CorrelationRiskController(
            portfolio_risk=prm,
            max_cross_engine_correlation=0.5,
        )
        ctrl.update_positions({
            "eng_a": _make_positions([("BTC/USDT", "long", 20000)]),
            "eng_b": _make_positions([("BTC/USDT", "long", 20000)]),
        })
        report = ctrl.get_concentration_report()
        # Full overlap → concentration_score = 1.0 > 0.5
        assert any("eng_a|eng_b" in a for a in report["alerts"])

    def test_report_no_portfolio_value(self):
        ctrl = CorrelationRiskController(portfolio_risk=None)
        ctrl.update_positions({
            "eng_a": _make_positions([("BTC/USDT", "long", 20000)]),
        })
        report = ctrl.get_concentration_report()
        assert report["per_symbol"]["BTC/USDT"]["pct_of_capital"] == 0.0
        assert report["alerts"] == []


# ──────────────────────────────────────────────────────────────
# BaseEngine._has_capacity integration
# ──────────────────────────────────────────────────────────────


class TestBaseEngineIntegration:
    def test_has_capacity_without_controller(self):
        """When no controller is set, _has_capacity works as before."""
        from bot.engines.base import BaseEngine

        class DummyEngine(BaseEngine):
            @property
            def name(self):
                return "dummy"

            @property
            def description(self):
                return "test"

            async def _run_cycle(self):
                pass

        pm = MagicMock()
        pm.request_capital.return_value = 10000
        pm.get_max_allocation.return_value = 10000
        engine = DummyEngine(portfolio_manager=pm, max_positions=2)
        assert engine._has_capacity() is True
        assert engine._has_capacity("BTC/USDT") is True

    def test_has_capacity_with_controller_allowed(self):
        from bot.engines.base import BaseEngine

        class DummyEngine(BaseEngine):
            @property
            def name(self):
                return "dummy"

            @property
            def description(self):
                return "test"

            async def _run_cycle(self):
                pass

        pm = MagicMock()
        pm.request_capital.return_value = 10000
        pm.get_max_allocation.return_value = 10000
        engine = DummyEngine(portfolio_manager=pm, max_positions=5)

        ctrl = MagicMock()
        ctrl.check_symbol_concentration.return_value = (True, "")
        engine.set_correlation_controller(ctrl)

        assert engine._has_capacity("BTC/USDT") is True
        ctrl.check_symbol_concentration.assert_called_once_with("BTC/USDT")

    def test_has_capacity_with_controller_blocked(self):
        from bot.engines.base import BaseEngine

        class DummyEngine(BaseEngine):
            @property
            def name(self):
                return "dummy"

            @property
            def description(self):
                return "test"

            async def _run_cycle(self):
                pass

        pm = MagicMock()
        pm.request_capital.return_value = 10000
        pm.get_max_allocation.return_value = 10000
        engine = DummyEngine(portfolio_manager=pm, max_positions=5)

        ctrl = MagicMock()
        ctrl.check_symbol_concentration.return_value = (
            False,
            "concentration too high",
        )
        engine.set_correlation_controller(ctrl)

        assert engine._has_capacity("BTC/USDT") is False

    def test_has_capacity_no_symbol_skips_controller(self):
        """When symbol=None, controller check is skipped."""
        from bot.engines.base import BaseEngine

        class DummyEngine(BaseEngine):
            @property
            def name(self):
                return "dummy"

            @property
            def description(self):
                return "test"

            async def _run_cycle(self):
                pass

        pm = MagicMock()
        pm.request_capital.return_value = 10000
        pm.get_max_allocation.return_value = 10000
        engine = DummyEngine(portfolio_manager=pm, max_positions=5)

        ctrl = MagicMock()
        ctrl.check_symbol_concentration.return_value = (
            False,
            "blocked",
        )
        engine.set_correlation_controller(ctrl)

        # No symbol → controller not checked
        assert engine._has_capacity() is True
        ctrl.check_symbol_concentration.assert_not_called()

    def test_has_capacity_max_positions_takes_precedence(self):
        """Max positions check fires before controller."""
        from bot.engines.base import BaseEngine

        class DummyEngine(BaseEngine):
            @property
            def name(self):
                return "dummy"

            @property
            def description(self):
                return "test"

            async def _run_cycle(self):
                pass

        pm = MagicMock()
        pm.request_capital.return_value = 10000
        pm.get_max_allocation.return_value = 10000
        engine = DummyEngine(portfolio_manager=pm, max_positions=1)
        engine._positions = {"BTC/USDT": {"symbol": "BTC/USDT"}}

        ctrl = MagicMock()
        engine.set_correlation_controller(ctrl)

        assert engine._has_capacity("ETH/USDT") is False
        # Controller not called since max_positions check failed first
        ctrl.check_symbol_concentration.assert_not_called()


# ──────────────────────────────────────────────────────────────
# EngineManager integration
# ──────────────────────────────────────────────────────────────


class TestEngineManagerIntegration:
    def test_set_correlation_controller(self):
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        mgr = EngineManager(pm)
        ctrl = MagicMock()
        mgr.set_correlation_controller(ctrl)
        assert mgr._correlation_controller is ctrl

    def test_sync_correlation_positions(self):
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        mgr = EngineManager(pm)
        ctrl = MagicMock()
        mgr.set_correlation_controller(ctrl)

        # Create a mock engine with positions
        engine = MagicMock()
        engine.name = "eng_a"
        engine.positions = {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "side": "long",
                "quantity": 0.5,
                "entry_price": 40000,
                "notional": 20000,
            }
        }
        mgr._engines = {"eng_a": engine}

        mgr._sync_correlation_positions()
        ctrl.update_positions.assert_called_once()
        call_args = ctrl.update_positions.call_args[0][0]
        assert "eng_a" in call_args
        assert len(call_args["eng_a"]) == 1
        assert call_args["eng_a"][0]["symbol"] == "BTC/USDT"
        assert call_args["eng_a"][0]["notional"] == 20000

    def test_sync_positions_estimates_notional(self):
        """When notional is 0, estimate from quantity * entry_price."""
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        mgr = EngineManager(pm)
        ctrl = MagicMock()
        mgr.set_correlation_controller(ctrl)

        engine = MagicMock()
        engine.name = "eng_a"
        engine.positions = {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "side": "long",
                "quantity": 0.5,
                "entry_price": 40000,
            }
        }
        mgr._engines = {"eng_a": engine}

        mgr._sync_correlation_positions()
        call_args = ctrl.update_positions.call_args[0][0]
        assert call_args["eng_a"][0]["notional"] == 20000

    def test_sync_positions_no_controller(self):
        """When no controller, _sync_correlation_positions is a no-op."""
        from bot.engines.manager import EngineManager

        pm = MagicMock()
        mgr = EngineManager(pm)
        mgr._sync_correlation_positions()  # Should not raise


# ──────────────────────────────────────────────────────────────
# Dashboard endpoint
# ──────────────────────────────────────────────────────────────


class TestDashboardEndpoint:
    @pytest.mark.asyncio
    async def test_correlation_endpoint_no_manager(self):
        from bot.dashboard.app import get_risk_correlation, set_engine_manager

        set_engine_manager(None)
        result = await get_risk_correlation()
        assert result == {"error": "not_available"}

    @pytest.mark.asyncio
    async def test_correlation_endpoint_no_controller(self):
        from bot.dashboard.app import get_risk_correlation, set_engine_manager

        mgr = MagicMock()
        mgr._correlation_controller = None
        # getattr fallback should return None
        del mgr._correlation_controller
        set_engine_manager(mgr)
        result = await get_risk_correlation()
        assert result == {"error": "not_available"}

    @pytest.mark.asyncio
    async def test_correlation_endpoint_with_controller(self):
        from bot.dashboard.app import get_risk_correlation, set_engine_manager

        mgr = MagicMock()
        ctrl = MagicMock()
        ctrl.get_concentration_report.return_value = {
            "per_symbol": {},
            "cross_engine_correlations": {},
            "alerts": [],
        }
        mgr._correlation_controller = ctrl
        set_engine_manager(mgr)
        result = await get_risk_correlation()
        assert result["per_symbol"] == {}
        assert result["alerts"] == []


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────


class TestConfig:
    def test_config_defaults(self):
        from bot.config import Settings

        s = Settings(
            binance_api_key="test",
            binance_secret_key="test",
        )
        assert s.cross_engine_correlation_enabled is True
        assert s.max_symbol_concentration_pct == 40.0

    def test_settings_metadata(self):
        from bot.config import SETTINGS_METADATA

        assert "cross_engine_correlation_enabled" in SETTINGS_METADATA
        meta = SETTINGS_METADATA["cross_engine_correlation_enabled"]
        assert meta["section"] == "Risk Management"
        assert meta["type"] == "bool"
        assert meta["requires_restart"] is False

        assert "max_symbol_concentration_pct" in SETTINGS_METADATA
        meta2 = SETTINGS_METADATA["max_symbol_concentration_pct"]
        assert meta2["section"] == "Risk Management"
        assert meta2["type"] == "float"


# ──────────────────────────────────────────────────────────────
# Import verification
# ──────────────────────────────────────────────────────────────


class TestImports:
    def test_import_from_risk_package(self):
        from bot.risk import CorrelationRiskController

        assert CorrelationRiskController is not None

    def test_import_direct(self):
        from bot.risk.correlation_controller import (
            CorrelationRiskController,
            EnginePosition,
        )

        assert CorrelationRiskController is not None
        assert EnginePosition is not None
