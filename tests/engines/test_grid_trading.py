"""Tests for GridTradingEngine."""

import pytest

from bot.engines.grid_trading import GridLevel, GridTradingEngine
from bot.engines.portfolio_manager import PortfolioManager


@pytest.fixture
def pm():
    return PortfolioManager(
        total_capital=10000.0,
        engine_allocations={"grid_trading": 0.25},
    )


def make_engine(pm, **overrides):
    engine = GridTradingEngine(
        portfolio_manager=pm,
        exchanges=[],
        paper_mode=True,
        settings=None,
    )
    for k, v in overrides.items():
        setattr(engine, k, v)
    return engine


class TestGridLevel:
    def test_initial_state(self):
        level = GridLevel(50000.0, "buy")
        assert level.price == 50000.0
        assert level.side == "buy"
        assert level.order_id is None
        assert level.filled is False


class TestGridTradingProperties:
    def test_name(self, pm):
        engine = make_engine(pm)
        assert engine.name == "grid_trading"

    def test_description(self, pm):
        engine = make_engine(pm)
        assert "grid" in engine.description.lower()


class TestGridInitialization:
    def test_init_grid(self, pm):
        engine = make_engine(pm, _grid_levels_count=5, _grid_spacing_pct=1.0)
        engine._init_grid("BTC/USDT", 50000.0)

        grid = engine._grids["BTC/USDT"]
        assert len(grid) == 10  # 5 buy + 5 sell

        buy_levels = [lv for lv in grid if lv.side == "buy"]
        sell_levels = [lv for lv in grid if lv.side == "sell"]
        assert len(buy_levels) == 5
        assert len(sell_levels) == 5

        # Buy levels should be below center
        for bl in buy_levels:
            assert bl.price < 50000.0
        # Sell levels should be above center
        for sl in sell_levels:
            assert sl.price > 50000.0

    def test_reset_grid(self, pm):
        engine = make_engine(pm, _grid_levels_count=3, _grid_spacing_pct=1.0)
        engine._init_grid("BTC/USDT", 50000.0)
        old_levels = len(engine._grids["BTC/USDT"])

        engine._reset_grid("BTC/USDT", 55000.0)
        new_grid = engine._grids["BTC/USDT"]
        assert len(new_grid) == old_levels
        # New grid should be centered around 55000
        sell_prices = [lv.price for lv in new_grid if lv.side == "sell"]
        assert all(p > 55000 for p in sell_prices)


class TestFillDetection:
    def test_buy_fill_when_price_drops(self, pm):
        engine = make_engine(pm, _grid_levels_count=3, _grid_spacing_pct=1.0)
        engine._allocated_capital = 2500.0
        engine._init_grid("BTC/USDT", 50000.0)

        # Price drops to first buy level (50000 * 0.99 = 49500)
        actions, pnl = engine._check_fills("BTC/USDT", 49500.0)
        buy_fills = [a for a in actions if a["action"] == "grid_buy_filled"]
        assert len(buy_fills) >= 1

    def test_sell_fill_when_price_rises(self, pm):
        engine = make_engine(pm, _grid_levels_count=3, _grid_spacing_pct=1.0)
        engine._allocated_capital = 2500.0
        engine._init_grid("BTC/USDT", 50000.0)

        # Price rises to first sell level (50000 * 1.01 = 50500)
        actions, pnl = engine._check_fills("BTC/USDT", 50500.0)
        sell_fills = [a for a in actions if a["action"] == "grid_sell_filled"]
        assert len(sell_fills) >= 1
        assert pnl > 0

    def test_fill_only_once(self, pm):
        engine = make_engine(pm, _grid_levels_count=3, _grid_spacing_pct=1.0)
        engine._allocated_capital = 2500.0
        engine._init_grid("BTC/USDT", 50000.0)

        engine._check_fills("BTC/USDT", 49000.0)
        # Second check at same price should not re-fill
        actions, _ = engine._check_fills("BTC/USDT", 49000.0)
        buy_fills = [a for a in actions if a["action"] == "grid_buy_filled"]
        assert len(buy_fills) == 0

    def test_no_fills_at_center(self, pm):
        engine = make_engine(pm, _grid_levels_count=3, _grid_spacing_pct=1.0)
        engine._allocated_capital = 2500.0
        engine._init_grid("BTC/USDT", 50000.0)

        actions, pnl = engine._check_fills("BTC/USDT", 50000.0)
        assert len(actions) == 0
        assert pnl == 0.0


class TestRangeDetection:
    def test_inside_range(self, pm):
        engine = make_engine(pm, _grid_levels_count=5, _grid_spacing_pct=1.0)
        engine._init_grid("BTC/USDT", 50000.0)
        assert not engine._is_outside_range("BTC/USDT", 50000.0)

    def test_above_range(self, pm):
        engine = make_engine(pm, _grid_levels_count=3, _grid_spacing_pct=1.0)
        engine._init_grid("BTC/USDT", 50000.0)
        # Grid max is 50000 * 1.03 = 51500
        assert engine._is_outside_range("BTC/USDT", 52000.0)

    def test_below_range(self, pm):
        engine = make_engine(pm, _grid_levels_count=3, _grid_spacing_pct=1.0)
        engine._init_grid("BTC/USDT", 50000.0)
        # Grid min is 50000 * 0.97 = 48500
        assert engine._is_outside_range("BTC/USDT", 48000.0)

    def test_no_grid(self, pm):
        engine = make_engine(pm)
        assert not engine._is_outside_range("BTC/USDT", 50000.0)


class TestRunCycle:
    @pytest.mark.asyncio
    async def test_cycle_no_exchanges(self, pm):
        engine = make_engine(pm)
        await engine.start()
        result = await engine._run_cycle()
        assert result.engine_name == "grid_trading"
        assert result.actions_taken == []

    @pytest.mark.asyncio
    async def test_cycle_creates_grid(self, pm):
        class MockExchange:
            name = "mock"

            async def get_ticker(self, symbol):
                return {"last": 50000.0}

        engine = make_engine(pm, _exchanges=[MockExchange()])
        await engine.start()

        result = await engine._run_cycle()
        grid_creates = [a for a in result.actions_taken if a["action"] == "grid_created"]
        assert len(grid_creates) == 2  # BTC + ETH

    @pytest.mark.asyncio
    async def test_cycle_detects_fills(self, pm):
        prices = {"BTC/USDT": [50000.0, 49000.0], "ETH/USDT": [3000.0, 3000.0]}
        call_index = {"BTC/USDT": 0, "ETH/USDT": 0}

        class MockExchange:
            name = "mock"

            async def get_ticker(self, symbol):
                idx = call_index[symbol]
                price = prices[symbol][min(idx, len(prices[symbol]) - 1)]
                call_index[symbol] = idx + 1
                return {"last": price}

        engine = make_engine(
            pm,
            _exchanges=[MockExchange()],
            _grid_levels_count=5,
            _grid_spacing_pct=1.0,
        )
        await engine.start()

        # First cycle: creates grids
        await engine._run_cycle()

        # Second cycle: BTC drops → buy fills
        r2 = await engine._run_cycle()
        buy_fills = [a for a in r2.actions_taken if a["action"] == "grid_buy_filled"]
        assert len(buy_fills) > 0

    @pytest.mark.asyncio
    async def test_cycle_resets_on_breakout(self, pm):
        prices = [50000.0, 60000.0]
        call_idx = [0]

        class MockExchange:
            name = "mock"

            async def get_ticker(self, symbol):
                p = prices[min(call_idx[0], len(prices) - 1)]
                call_idx[0] += 1
                return {"last": p}

        engine = make_engine(
            pm,
            _exchanges=[MockExchange()],
            _symbols=["BTC/USDT"],
            _grid_levels_count=3,
            _grid_spacing_pct=1.0,
        )
        await engine.start()

        # First cycle: creates grid
        await engine._run_cycle()

        # Second cycle: price way above range → grid reset
        r2 = await engine._run_cycle()
        resets = [a for a in r2.actions_taken if a["action"] == "grid_reset"]
        assert len(resets) == 1
