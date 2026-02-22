"""Tests for triangular arbitrage strategy."""

import pytest

from bot.models import SignalAction
from bot.strategies.base import strategy_registry
from bot.strategies.quant.triangular_arb import TriangularArbStrategy


@pytest.fixture(autouse=True)
def clean_registry():
    strategy_registry.clear()
    yield
    strategy_registry.clear()


def _make_ohlcv_list():
    from datetime import datetime

    from bot.models import OHLCV
    return [OHLCV(
        timestamp=datetime(2024, 1, 1),
        open=100, high=101, low=99, close=100,
        volume=1000, symbol="BTC/USDT",
    )]


def _make_arb_tickers():
    """Create tickers with a clear arbitrage opportunity.

    Cycle: USDT -> BTC -> ETH -> USDT
    Leg 1: Buy BTC with USDT at ask=50000 -> rate = 1/50000 = 0.00002 BTC per USDT
    Leg 2: Sell BTC for ETH: BTC->ETH via ETH/BTC buy at ask=0.05 -> rate = 1/0.05 = 20 ETH per BTC
    Leg 3: Sell ETH for USDT at bid=3000 -> rate = 3000 USDT per ETH

    Gross: 0.00002 * 20 * 3000 = 1.2 -> 20% gross profit
    """
    return {
        "BTC/USDT": {"bid": 50000, "ask": 50000},
        "ETH/BTC": {"bid": 0.05, "ask": 0.05},
        "ETH/USDT": {"bid": 3000, "ask": 2500},
    }


class TestTriangularArbStrategy:
    def test_name(self):
        s = TriangularArbStrategy()
        assert s.name == "triangular_arb"

    def test_required_history(self):
        s = TriangularArbStrategy()
        assert s.required_history_length == 1

    @pytest.mark.asyncio
    async def test_hold_no_tickers(self):
        """HOLD when no tickers provided."""
        s = TriangularArbStrategy()
        candles = _make_ohlcv_list()
        signal = await s.analyze(candles)
        assert signal.action == SignalAction.HOLD
        assert signal.metadata["reason"] == "insufficient_tickers"

    @pytest.mark.asyncio
    async def test_hold_insufficient_tickers(self):
        """HOLD when fewer than 3 tickers provided."""
        s = TriangularArbStrategy()
        candles = _make_ohlcv_list()
        tickers = {
            "BTC/USDT": {"bid": 50000, "ask": 50010},
            "ETH/USDT": {"bid": 3000, "ask": 3001},
        }
        signal = await s.analyze(candles, tickers=tickers)
        assert signal.action == SignalAction.HOLD
        assert signal.metadata["reason"] == "insufficient_tickers"

    @pytest.mark.asyncio
    async def test_hold_empty_tickers(self):
        """HOLD when tickers dict is empty."""
        s = TriangularArbStrategy()
        candles = _make_ohlcv_list()
        signal = await s.analyze(candles, tickers={})
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_detects_profitable_cycle(self):
        """BUY when cycle profit > threshold."""
        s = TriangularArbStrategy(min_profit_pct=0.1, default_fee_rate=0.001)
        tickers = _make_arb_tickers()
        candles = _make_ohlcv_list()
        signal = await s.analyze(candles, tickers=tickers)
        assert signal.action == SignalAction.BUY
        assert signal.metadata["profit_pct"] > 0
        assert "cycle" in signal.metadata
        assert "legs" in signal.metadata
        assert len(signal.metadata["legs"]) == 3

    @pytest.mark.asyncio
    async def test_hold_when_no_profitable_cycle(self):
        """HOLD when no cycle exceeds min_profit_pct."""
        s = TriangularArbStrategy(min_profit_pct=50.0)  # Absurdly high threshold
        tickers = _make_arb_tickers()
        candles = _make_ohlcv_list()
        signal = await s.analyze(candles, tickers=tickers)
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_hold_no_cycle_exists(self):
        """HOLD when tickers don't form a complete 3-leg cycle."""
        s = TriangularArbStrategy()
        # These pairs don't form a triangle
        tickers = {
            "BTC/USDT": {"bid": 50000, "ask": 50010},
            "ETH/USDT": {"bid": 3000, "ask": 3001},
            "SOL/USDT": {"bid": 100, "ask": 100.1},
        }
        candles = _make_ohlcv_list()
        signal = await s.analyze(candles, tickers=tickers)
        assert signal.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_confidence_bounded(self):
        """Confidence should be between 0 and 1."""
        s = TriangularArbStrategy(min_profit_pct=0.01, default_fee_rate=0.0001)
        tickers = _make_arb_tickers()
        candles = _make_ohlcv_list()
        signal = await s.analyze(candles, tickers=tickers)
        assert 0.0 <= signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_metadata_contains_fee_info(self):
        """Metadata should contain fee and profit details."""
        s = TriangularArbStrategy(min_profit_pct=0.01, default_fee_rate=0.001)
        tickers = _make_arb_tickers()
        candles = _make_ohlcv_list()
        signal = await s.analyze(candles, tickers=tickers)
        if signal.action == SignalAction.BUY:
            assert "total_fee_pct" in signal.metadata
            assert "gross_profit_pct" in signal.metadata
            assert signal.metadata["gross_profit_pct"] >= signal.metadata["profit_pct"]

    def test_graph_building(self):
        """Graph should contain all currencies from tickers."""
        s = TriangularArbStrategy()
        tickers = {
            "BTC/USDT": {"bid": 50000, "ask": 50010},
            "ETH/USDT": {"bid": 3000, "ask": 3001},
            "ETH/BTC": {"bid": 0.06, "ask": 0.06005},
        }
        graph = s._build_graph(tickers)
        assert "USDT" in graph
        assert "BTC" in graph
        assert "ETH" in graph

    def test_graph_skips_invalid_tickers(self):
        """Graph building should skip tickers with zero or negative prices."""
        s = TriangularArbStrategy()
        tickers = {
            "BTC/USDT": {"bid": 50000, "ask": 50010},
            "BAD/USDT": {"bid": 0, "ask": 0},
            "NEG/USDT": {"bid": -1, "ask": -1},
        }
        graph = s._build_graph(tickers)
        assert "BAD" not in graph.get("USDT", {})
        assert "NEG" not in graph.get("USDT", {})

    def test_graph_skips_malformed_symbols(self):
        """Graph building should skip symbols without / separator."""
        s = TriangularArbStrategy()
        tickers = {
            "BTC/USDT": {"bid": 50000, "ask": 50010},
            "BTCUSDT": {"bid": 50000, "ask": 50010},  # no slash
        }
        graph = s._build_graph(tickers)
        # BTCUSDT should not be in the graph
        assert "BTCUSDT" not in graph

    def test_registration(self):
        """Strategy should register with strategy_registry."""
        s = TriangularArbStrategy()
        strategy_registry.register(s)
        assert strategy_registry.get("triangular_arb") is not None

    @pytest.mark.asyncio
    async def test_custom_fee_rate(self):
        """Custom fee_rate kwarg should be used over default."""
        s = TriangularArbStrategy(min_profit_pct=0.01, default_fee_rate=0.5)  # 50% fee
        tickers = _make_arb_tickers()
        candles = _make_ohlcv_list()
        # With 50% fee, even 20% gross arb is not profitable
        signal = await s.analyze(candles, tickers=tickers)
        assert signal.action == SignalAction.HOLD
        # Override with low fee should find opportunity
        signal2 = await s.analyze(candles, tickers=tickers, fee_rate=0.0001)
        assert signal2.action == SignalAction.BUY

    @pytest.mark.asyncio
    async def test_strategy_name_in_signal(self):
        """Signal should carry the strategy name."""
        s = TriangularArbStrategy()
        candles = _make_ohlcv_list()
        signal = await s.analyze(candles)
        assert signal.strategy_name == "triangular_arb"

    @pytest.mark.asyncio
    async def test_cycle_path_starts_and_ends_same(self):
        """A profitable cycle should start and end at the same currency."""
        s = TriangularArbStrategy(min_profit_pct=0.01, default_fee_rate=0.0001)
        tickers = _make_arb_tickers()
        candles = _make_ohlcv_list()
        signal = await s.analyze(candles, tickers=tickers)
        if signal.action == SignalAction.BUY:
            cycle = signal.metadata["cycle"]
            assert cycle[0] == cycle[-1]
            assert len(cycle) == 4  # start, mid1, mid2, start
