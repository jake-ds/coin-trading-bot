"""Tests for CostModel — trading cost calculator."""

import pytest

from bot.engines.cost_model import CostModel


class TestCostModelDefaults:
    """Verify default Binance VIP-0 fee values."""

    def test_default_maker_fee(self):
        cm = CostModel()
        assert cm.maker_fee_pct == 0.02

    def test_default_taker_fee(self):
        cm = CostModel()
        assert cm.taker_fee_pct == 0.04

    def test_default_slippage(self):
        cm = CostModel()
        assert cm.slippage_pct == 0.01


class TestRoundTripCost:
    """round_trip_cost(notional, legs, is_maker)."""

    def test_two_legs_taker(self):
        cm = CostModel()
        # cost = 10000 * (0.04 + 0.01) * 2 / 100 = 10.0
        assert cm.round_trip_cost(10000, legs=2) == pytest.approx(10.0)

    def test_four_legs_taker(self):
        cm = CostModel()
        # cost = 10000 * (0.04 + 0.01) * 4 / 100 = 20.0
        assert cm.round_trip_cost(10000, legs=4) == pytest.approx(20.0)

    def test_two_legs_maker(self):
        cm = CostModel()
        # cost = 10000 * (0.02 + 0.01) * 2 / 100 = 6.0
        assert cm.round_trip_cost(10000, legs=2, is_maker=True) == pytest.approx(6.0)

    def test_four_legs_maker(self):
        cm = CostModel()
        # cost = 10000 * (0.02 + 0.01) * 4 / 100 = 12.0
        assert cm.round_trip_cost(10000, legs=4, is_maker=True) == pytest.approx(12.0)

    def test_zero_notional(self):
        cm = CostModel()
        assert cm.round_trip_cost(0, legs=2) == pytest.approx(0.0)

    def test_custom_fees(self):
        cm = CostModel(maker_fee_pct=0.01, taker_fee_pct=0.03, slippage_pct=0.02)
        # cost = 5000 * (0.03 + 0.02) * 2 / 100 = 5.0
        assert cm.round_trip_cost(5000, legs=2) == pytest.approx(5.0)

    def test_large_notional(self):
        cm = CostModel()
        # cost = 1_000_000 * (0.04 + 0.01) * 2 / 100 = 1000.0
        assert cm.round_trip_cost(1_000_000, legs=2) == pytest.approx(1000.0)

    def test_single_leg(self):
        cm = CostModel()
        # cost = 10000 * (0.04 + 0.01) * 1 / 100 = 5.0
        assert cm.round_trip_cost(10000, legs=1) == pytest.approx(5.0)


class TestNetProfit:
    """net_profit(gross_pnl, notional, legs)."""

    def test_positive_net(self):
        cm = CostModel()
        # gross=50, cost=10 -> net=40
        assert cm.net_profit(50.0, 10000, legs=2) == pytest.approx(40.0)

    def test_negative_net(self):
        cm = CostModel()
        # gross=5, cost=10 -> net=-5
        assert cm.net_profit(5.0, 10000, legs=2) == pytest.approx(-5.0)

    def test_breakeven(self):
        cm = CostModel()
        # gross=10, cost=10 -> net=0
        assert cm.net_profit(10.0, 10000, legs=2) == pytest.approx(0.0)

    def test_negative_gross_pnl(self):
        cm = CostModel()
        # gross=-20, cost=10 -> net=-30
        assert cm.net_profit(-20.0, 10000, legs=2) == pytest.approx(-30.0)

    def test_four_legs(self):
        cm = CostModel()
        # gross=50, cost=20 -> net=30
        assert cm.net_profit(50.0, 10000, legs=4) == pytest.approx(30.0)

    def test_maker_fees(self):
        cm = CostModel()
        # gross=50, cost=6 -> net=44
        assert cm.net_profit(50.0, 10000, legs=2, is_maker=True) == pytest.approx(44.0)


class TestMinSpreadForProfit:
    """min_spread_for_profit(legs, is_maker)."""

    def test_two_legs_taker(self):
        cm = CostModel()
        # (0.04 + 0.01) * 2 = 0.10
        assert cm.min_spread_for_profit(legs=2) == pytest.approx(0.10)

    def test_four_legs_taker(self):
        cm = CostModel()
        # (0.04 + 0.01) * 4 = 0.20
        assert cm.min_spread_for_profit(legs=4) == pytest.approx(0.20)

    def test_two_legs_maker(self):
        cm = CostModel()
        # (0.02 + 0.01) * 2 = 0.06
        assert cm.min_spread_for_profit(legs=2, is_maker=True) == pytest.approx(0.06)

    def test_custom_fees(self):
        cm = CostModel(taker_fee_pct=0.10, slippage_pct=0.05)
        # (0.10 + 0.05) * 2 = 0.30
        assert cm.min_spread_for_profit(legs=2) == pytest.approx(0.30)


class TestIsProfitable:
    """is_profitable(gross_pnl, notional, legs)."""

    def test_profitable_trade(self):
        cm = CostModel()
        # gross=50, cost=10 -> net=40 > 0
        assert cm.is_profitable(50.0, 10000, legs=2) is True

    def test_unprofitable_trade(self):
        cm = CostModel()
        # gross=5, cost=10 -> net=-5 < 0
        assert cm.is_profitable(5.0, 10000, legs=2) is False

    def test_breakeven_is_not_profitable(self):
        cm = CostModel()
        # gross=10, cost=10 -> net=0, NOT > 0
        assert cm.is_profitable(10.0, 10000, legs=2) is False

    def test_profitable_with_maker(self):
        cm = CostModel()
        # gross=8, cost=6 (maker) -> net=2 > 0
        assert cm.is_profitable(8.0, 10000, legs=2, is_maker=True) is True

    def test_unprofitable_with_four_legs(self):
        cm = CostModel()
        # gross=15, cost=20 (4 legs) -> net=-5 < 0
        assert cm.is_profitable(15.0, 10000, legs=4) is False


class TestCustomCostModel:
    """Verify CostModel works with fully custom fee configuration."""

    def test_zero_fees(self):
        cm = CostModel(maker_fee_pct=0, taker_fee_pct=0, slippage_pct=0)
        assert cm.round_trip_cost(10000, legs=2) == pytest.approx(0.0)
        assert cm.net_profit(50.0, 10000) == pytest.approx(50.0)
        assert cm.min_spread_for_profit() == pytest.approx(0.0)
        assert cm.is_profitable(0.01, 10000) is True

    def test_high_fee_environment(self):
        cm = CostModel(maker_fee_pct=0.1, taker_fee_pct=0.2, slippage_pct=0.05)
        # cost = 10000 * (0.2 + 0.05) * 2 / 100 = 50.0
        assert cm.round_trip_cost(10000, legs=2) == pytest.approx(50.0)
        assert cm.is_profitable(49.0, 10000) is False
        assert cm.is_profitable(51.0, 10000) is True


class TestCostModelImport:
    """Verify CostModel is importable from the engines package."""

    def test_import_from_package(self):
        from bot.engines import CostModel as CostModelFromPackage

        cm = CostModelFromPackage()
        assert cm.taker_fee_pct == 0.04
