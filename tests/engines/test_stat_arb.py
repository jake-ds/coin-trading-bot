"""Tests for StatisticalArbEngine."""

import pytest

from bot.engines.portfolio_manager import PortfolioManager
from bot.engines.stat_arb import StatisticalArbEngine


class MockExchange:
    def __init__(self, prices):
        self.name = "mock"
        self._prices = prices
        self._call_count = 0

    async def get_ticker(self, symbol):
        price = self._prices.get(symbol, 0.0)
        if isinstance(price, list):
            idx = min(self._call_count, len(price) - 1)
            p = price[idx]
            self._call_count += 1
            return {"last": p}
        return {"last": price}


@pytest.fixture
def pm():
    return PortfolioManager(
        total_capital=10000.0,
        engine_allocations={"stat_arb": 0.20},
    )


def make_engine(pm, exchanges=None, **overrides):
    engine = StatisticalArbEngine(
        portfolio_manager=pm,
        exchanges=exchanges or [],
        paper_mode=True,
        settings=None,
    )
    for k, v in overrides.items():
        setattr(engine, k, v)
    return engine


class TestProperties:
    def test_name(self, pm):
        engine = make_engine(pm)
        assert engine.name == "stat_arb"

    def test_description(self, pm):
        engine = make_engine(pm)
        assert "pairs" in engine.description.lower() or "statistical" in engine.description.lower()


class TestEntryLogic:
    def test_entry_high_zscore(self, pm):
        engine = make_engine(pm)
        engine._allocated_capital = 2000.0

        result = engine._check_entry(
            "BTC/USDT|ETH/USDT", "BTC/USDT", "ETH/USDT",
            zscore=2.5, price_a=50000.0, price_b=3000.0,
        )
        assert result is not None
        assert result["action"] == "entry"
        assert result["side_a"] == "short"  # z > 0 → A overpriced
        assert result["side_b"] == "long"

    def test_entry_low_zscore(self, pm):
        engine = make_engine(pm)
        engine._allocated_capital = 2000.0

        result = engine._check_entry(
            "BTC/USDT|ETH/USDT", "BTC/USDT", "ETH/USDT",
            zscore=-2.5, price_a=50000.0, price_b=3000.0,
        )
        assert result is not None
        assert result["side_a"] == "long"  # z < 0 → A underpriced
        assert result["side_b"] == "short"

    def test_no_entry_below_threshold(self, pm):
        engine = make_engine(pm)
        result = engine._check_entry(
            "BTC/USDT|ETH/USDT", "BTC/USDT", "ETH/USDT",
            zscore=1.0, price_a=50000.0, price_b=3000.0,
        )
        assert result is None


class TestExitLogic:
    def test_exit_mean_reversion(self, pm):
        engine = make_engine(pm)
        pos = {
            "entry_zscore": 2.5,
            "price_a": 50000.0,
            "qty_a": 0.01,
            "side_a": "short",
        }

        result = engine._check_exit("BTC|ETH", zscore=0.3, pos=pos)
        assert result is not None
        assert result["reason"] == "mean_reversion"

    def test_exit_stop_loss(self, pm):
        engine = make_engine(pm)
        pos = {"entry_zscore": 2.5, "price_a": 50000.0, "qty_a": 0.01, "side_a": "short"}

        result = engine._check_exit("BTC|ETH", zscore=4.5, pos=pos)
        assert result is not None
        assert result["reason"] == "stop_loss"

    def test_exit_crossed_zero(self, pm):
        engine = make_engine(pm)
        pos = {"entry_zscore": 2.5, "price_a": 50000.0, "qty_a": 0.01, "side_a": "short"}

        result = engine._check_exit("BTC|ETH", zscore=-0.8, pos=pos)
        assert result is not None
        assert result["reason"] == "crossed_zero"

    def test_no_exit_still_extended(self, pm):
        engine = make_engine(pm)
        pos = {"entry_zscore": 2.5, "price_a": 50000.0, "qty_a": 0.01, "side_a": "short"}

        result = engine._check_exit("BTC|ETH", zscore=1.5, pos=pos)
        assert result is None


class TestPnLEstimation:
    def test_profit_on_reversion(self, pm):
        engine = make_engine(pm)
        pos = {
            "entry_zscore": 2.5,
            "price_a": 50000.0,
            "qty_a": 0.01,
            "side_a": "long",
        }
        pnl = engine._estimate_pairs_pnl(pos, exit_zscore=0.0)
        assert pnl > 0  # z went from 2.5 to 0 → profit

    def test_loss_on_extension(self, pm):
        engine = make_engine(pm)
        pos = {
            "entry_zscore": 2.0,
            "price_a": 50000.0,
            "qty_a": 0.01,
            "side_a": "long",
        }
        pnl = engine._estimate_pairs_pnl(pos, exit_zscore=4.0)
        assert pnl < 0  # z extended further → loss


class TestRunCycle:
    @pytest.mark.asyncio
    async def test_cycle_insufficient_data(self, pm):
        """With no exchanges, cache stays empty → insufficient data."""
        engine = make_engine(pm)
        await engine.start()

        result = await engine._run_cycle()
        assert result.engine_name == "stat_arb"
        insufficient = [s for s in result.signals if s.get("status") == "insufficient_data"]
        assert len(insufficient) >= 1

    @pytest.mark.asyncio
    async def test_cycle_with_enough_data(self, pm):
        """Pre-fill cache with correlated prices, check for signals."""
        import numpy as np

        engine = make_engine(pm, _lookback=20)
        await engine.start()

        # Pre-fill with correlated data
        np.random.seed(42)
        base = np.cumsum(np.random.randn(100)) + 100
        engine._price_cache["BTC/USDT"] = list(base)
        engine._price_cache["ETH/USDT"] = list(base * 0.06 + np.random.randn(100) * 0.01)

        result = await engine._run_cycle()
        active_signals = [s for s in result.signals if s.get("status") == "active"]
        assert len(active_signals) >= 1
        assert "zscore" in active_signals[0]
        assert "correlation" in active_signals[0]

    @pytest.mark.asyncio
    async def test_cycle_low_correlation(self, pm):
        """Uncorrelated pairs should be skipped."""
        import numpy as np

        engine = make_engine(pm, _lookback=20, _min_correlation=0.9)
        await engine.start()

        np.random.seed(99)
        engine._price_cache["BTC/USDT"] = list(np.random.randn(100) + 100)
        engine._price_cache["ETH/USDT"] = list(np.random.randn(100) + 50)

        result = await engine._run_cycle()
        low_corr = [s for s in result.signals if s.get("status") == "low_correlation"]
        assert len(low_corr) >= 1
