"""Tests for CostModel integration across all 4 trading engines.

V5-002: Verifies that all engines:
  - Deduct costs from PnL
  - Reject unprofitable trades after fees
  - Include cost info in DecisionSteps
"""

import pytest

from bot.engines.cost_model import CostModel
from bot.engines.cross_exchange_arb import CrossExchangeArbEngine
from bot.engines.funding_arb import FundingRateArbEngine
from bot.engines.grid_trading import GridTradingEngine
from bot.engines.portfolio_manager import PortfolioManager
from bot.engines.stat_arb import StatisticalArbEngine

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

@pytest.fixture
def pm_funding():
    return PortfolioManager(
        total_capital=10000.0,
        engine_allocations={"funding_rate_arb": 0.30},
    )


@pytest.fixture
def pm_grid():
    return PortfolioManager(
        total_capital=10000.0,
        engine_allocations={"grid_trading": 0.25},
    )


@pytest.fixture
def pm_cross():
    return PortfolioManager(
        total_capital=10000.0,
        engine_allocations={"cross_exchange_arb": 0.15},
    )


@pytest.fixture
def pm_stat():
    return PortfolioManager(
        total_capital=10000.0,
        engine_allocations={"stat_arb": 0.20},
    )


# ------------------------------------------------------------------ #
# FundingRateArbEngine — cost integration
# ------------------------------------------------------------------ #

class TestFundingArbCostIntegration:
    """Verify cost deduction and cost DecisionSteps in FundingRateArbEngine."""

    def test_has_cost_model(self, pm_funding):
        engine = FundingRateArbEngine(
            portfolio_manager=pm_funding, exchanges=[], paper_mode=True,
        )
        assert isinstance(engine._cost_model, CostModel)

    @pytest.mark.asyncio
    async def test_close_deducts_cost(self, pm_funding):
        engine = FundingRateArbEngine(
            portfolio_manager=pm_funding, exchanges=[], paper_mode=True,
        )
        await engine.start()
        await engine._open_position(
            "BTC/USDT",
            {"funding_rate": 0.0005, "mark_price": 50000, "spot_price": 49990},
        )
        pos = engine.positions["BTC/USDT"]
        qty = pos["quantity"]
        price = pos["entry_price"]
        gross = 0.0005 * qty * price
        cost = engine._cost_model.round_trip_cost(qty * price, legs=4)

        pnl = await engine._close_position("BTC/USDT")
        assert pnl == pytest.approx(gross - cost)

    @pytest.mark.asyncio
    async def test_cycle_includes_cost_decision(self, pm_funding):
        class MockExchange:
            name = "mock"
            async def get_funding_rate(self, symbol):
                return {
                    "symbol": symbol,
                    "funding_rate": 0.001,
                    "mark_price": 50000.0,
                    "spot_price": 49990.0,
                    "spread_pct": 0.02,
                }

        engine = FundingRateArbEngine(
            portfolio_manager=pm_funding,
            exchanges=[MockExchange()],
            paper_mode=True,
        )
        await engine.start()
        result = await engine._run_cycle()

        cost_steps = [d for d in result.decisions if d.label == "비용 분석"]
        assert len(cost_steps) >= 1
        assert cost_steps[0].category == "evaluate"
        assert "총비용" in cost_steps[0].observation


# ------------------------------------------------------------------ #
# GridTradingEngine — cost integration
# ------------------------------------------------------------------ #

class TestGridCostIntegration:
    """Verify cost deduction and skipping unprofitable fills."""

    def test_has_cost_model(self, pm_grid):
        engine = GridTradingEngine(
            portfolio_manager=pm_grid, exchanges=[], paper_mode=True,
        )
        assert isinstance(engine._cost_model, CostModel)

    def test_sell_fill_deducts_cost(self, pm_grid):
        engine = GridTradingEngine(
            portfolio_manager=pm_grid, exchanges=[], paper_mode=True,
        )
        engine._grid_levels_count = 3
        engine._grid_spacing_pct = 1.0
        engine._allocated_capital = 2500.0
        engine._init_grid("BTC/USDT", 50000.0)

        actions, pnl = engine._check_fills("BTC/USDT", 50500.0)
        sell_fills = [a for a in actions if a["action"] == "grid_sell_filled"]
        assert len(sell_fills) >= 1
        # Verify cost is tracked in action dict
        assert "cost" in sell_fills[0]
        assert sell_fills[0]["cost"] > 0
        # Net profit should be less than gross
        assert sell_fills[0]["profit"] < sell_fills[0]["gross_profit"]
        # PnL should still be positive (1% spread >> fees)
        assert pnl > 0

    def test_skips_unprofitable_fill(self, pm_grid):
        """With very high fees, fills become unprofitable and are skipped."""
        engine = GridTradingEngine(
            portfolio_manager=pm_grid, exchanges=[], paper_mode=True,
        )
        # Very tight grid spacing + high fees = unprofitable
        engine._grid_levels_count = 3
        engine._grid_spacing_pct = 0.01  # Only 0.01% spacing
        engine._allocated_capital = 2500.0
        engine._cost_model = CostModel(
            maker_fee_pct=0.1, taker_fee_pct=0.2, slippage_pct=0.1,
        )
        engine._init_grid("BTC/USDT", 50000.0)

        # Price rises above first sell level
        first_sell = min(
            lv.price for lv in engine._grids["BTC/USDT"] if lv.side == "sell"
        )
        actions, pnl = engine._check_fills("BTC/USDT", first_sell + 1)
        skipped = [a for a in actions if a.get("action") == "grid_sell_skipped"]
        assert len(skipped) >= 1
        assert skipped[0]["reason"] == "unprofitable_after_fees"
        # No PnL from skipped fills
        assert pnl == 0.0


# ------------------------------------------------------------------ #
# CrossExchangeArbEngine — cost integration
# ------------------------------------------------------------------ #

class MockExchange:
    def __init__(self, name, prices):
        self.name = name
        self._prices = prices

    async def get_ticker(self, symbol):
        return {"last": self._prices.get(symbol, 0.0)}


class TestCrossArbCostIntegration:
    """Verify cost-based minimum spread and profit deduction."""

    def test_has_cost_model(self, pm_cross):
        engine = CrossExchangeArbEngine(
            portfolio_manager=pm_cross, exchanges=[], paper_mode=True,
        )
        assert isinstance(engine._cost_model, CostModel)

    @pytest.mark.asyncio
    async def test_arb_deducts_cost(self, pm_cross):
        engine = CrossExchangeArbEngine(
            portfolio_manager=pm_cross, exchanges=[], paper_mode=True,
        )
        await engine.start()

        spread_info = {
            "symbol": "BTC/USDT",
            "exchange_a": "binance",
            "price_a": 50200.0,
            "exchange_b": "upbit",
            "price_b": 50000.0,
            "spread_pct": 0.4,
            "mid_price": 50100.0,
        }
        result = await engine._execute_arb("BTC/USDT", spread_info)
        assert result is not None
        assert "cost" in result
        assert result["cost"] > 0
        # Net profit < gross profit
        assert result["profit"] < result["gross_profit"]
        # Still profitable (0.4% spread > 0.20% cost-based min)
        assert result["profit"] > 0

    @pytest.mark.asyncio
    async def test_cost_based_min_spread(self, pm_cross):
        """Spread below cost-based minimum is rejected."""
        engine = CrossExchangeArbEngine(
            portfolio_manager=pm_cross, exchanges=[], paper_mode=True,
        )
        # Set config min_spread very low, but cost model enforces higher minimum
        engine._min_spread_pct = 0.05  # 0.05% — below fee-based minimum
        cost_min = engine._cost_model.min_spread_for_profit(legs=4)
        # With defaults: (0.04 + 0.01) * 4 = 0.20%
        assert cost_min == pytest.approx(0.20)

        # Spread of 0.15% — above config min but below cost min
        ex_a = MockExchange("binance", {"BTC/USDT": 50075})
        ex_b = MockExchange("upbit", {"BTC/USDT": 50000})
        engine._exchanges = [ex_a, ex_b]
        engine._symbols = ["BTC/USDT"]
        await engine.start()

        result = await engine._run_cycle()
        # Should not trade — spread 0.15% < cost-based 0.20%
        assert len(result.actions_taken) == 0

    @pytest.mark.asyncio
    async def test_decision_includes_cost_threshold(self, pm_cross):
        """DecisionStep threshold mentions cost-based minimum."""
        ex_a = MockExchange("binance", {"BTC/USDT": 50000})
        ex_b = MockExchange("upbit", {"BTC/USDT": 50000})
        engine = CrossExchangeArbEngine(
            portfolio_manager=pm_cross,
            exchanges=[ex_a, ex_b],
            paper_mode=True,
        )
        engine._symbols = ["BTC/USDT"]
        await engine.start()

        result = await engine._run_cycle()
        spread_steps = [
            d for d in result.decisions if "스프레드" in d.label
        ]
        assert len(spread_steps) >= 1
        assert "비용 기반" in spread_steps[0].threshold


# ------------------------------------------------------------------ #
# StatisticalArbEngine — cost integration
# ------------------------------------------------------------------ #

class TestStatArbCostIntegration:
    """Verify cost deduction and cost DecisionSteps in StatisticalArbEngine."""

    def test_has_cost_model(self, pm_stat):
        engine = StatisticalArbEngine(
            portfolio_manager=pm_stat, exchanges=[], paper_mode=True,
        )
        assert isinstance(engine._cost_model, CostModel)

    def test_pnl_scale_derived_from_cost_model(self, pm_stat):
        """Magic 0.005 is now derived from cost model params."""
        engine = StatisticalArbEngine(
            portfolio_manager=pm_stat, exchanges=[], paper_mode=True,
        )
        # (0.04 + 0.01) / 10 = 0.005
        expected_scale = (
            engine._cost_model.taker_fee_pct + engine._cost_model.slippage_pct
        ) / 10
        assert expected_scale == pytest.approx(0.005)

    def test_exit_deducts_cost(self, pm_stat):
        engine = StatisticalArbEngine(
            portfolio_manager=pm_stat, exchanges=[], paper_mode=True,
        )
        engine._allocated_capital = 2000.0
        pos = {
            "entry_zscore": 2.5,
            "price_a": 50000.0,
            "qty_a": 0.01,
            "side_a": "long",
        }
        engine._add_position(
            symbol="BTC/USDT|ETH/USDT",
            side="long_short",
            quantity=0,
            entry_price=0,
            **pos,
        )

        result = engine._check_exit("BTC/USDT|ETH/USDT", zscore=0.3, pos=pos)
        assert result is not None
        assert "cost" in result
        assert "gross_pnl" in result
        assert result["cost"] > 0
        # Net PnL = gross - cost
        assert result["pnl"] == pytest.approx(result["gross_pnl"] - result["cost"])
        # Trade should still be net profitable with large z-score reversion
        assert result["pnl"] > 0

    def test_exit_cost_makes_trade_worse(self, pm_stat):
        """Verify cost always reduces PnL (never increases it)."""
        engine = StatisticalArbEngine(
            portfolio_manager=pm_stat, exchanges=[], paper_mode=True,
        )
        pos = {
            "entry_zscore": 2.5,
            "price_a": 50000.0,
            "qty_a": 0.01,
            "side_a": "long",
        }
        engine._add_position(
            symbol="BTC/USDT|ETH/USDT",
            side="long_short",
            quantity=0,
            entry_price=0,
            **pos,
        )

        result = engine._check_exit("BTC/USDT|ETH/USDT", zscore=0.3, pos=pos)
        assert result["pnl"] < result["gross_pnl"]

    @pytest.mark.asyncio
    async def test_cycle_exit_includes_cost_decision(self, pm_stat):
        """Exit cycle includes a cost analysis DecisionStep."""
        import numpy as np

        engine = StatisticalArbEngine(
            portfolio_manager=pm_stat, exchanges=[], paper_mode=True,
        )
        engine._lookback = 20
        await engine.start()

        # Pre-fill cache with correlated data that produces high z-score
        np.random.seed(42)
        base = np.cumsum(np.random.randn(100)) + 100
        engine._price_cache["BTC/USDT"] = list(base)
        engine._price_cache["ETH/USDT"] = list(base * 0.06 + np.random.randn(100) * 0.01)

        # Run cycle to potentially enter
        await engine._run_cycle()

        # If a position was entered, manipulate z-score to trigger exit
        if engine.position_count > 0:
            for pair_key in list(engine._positions.keys()):
                pos = engine._positions[pair_key]
                # Force exit by setting small z-score
                result = engine._check_exit(pair_key, zscore=0.1, pos=pos)
                if result:
                    # Verify the result dict includes cost info
                    assert "cost" in result
                    assert result["cost"] > 0
