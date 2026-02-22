"""Tests for CrossExchangeArbEngine."""

import pytest

from bot.engines.cross_exchange_arb import CrossExchangeArbEngine
from bot.engines.portfolio_manager import PortfolioManager


class MockExchange:
    """Mock exchange returning configurable prices."""

    def __init__(self, exchange_name, prices):
        self.name = exchange_name
        self._prices = prices

    async def get_ticker(self, symbol):
        return {"last": self._prices.get(symbol, 0.0)}


@pytest.fixture
def pm():
    return PortfolioManager(
        total_capital=10000.0,
        engine_allocations={"cross_exchange_arb": 0.15},
    )


def make_engine(pm, exchanges=None, **overrides):
    engine = CrossExchangeArbEngine(
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
        assert engine.name == "cross_exchange_arb"

    def test_description(self, pm):
        engine = make_engine(pm)
        assert "arbitrage" in engine.description.lower()


class TestSpreadDetection:
    @pytest.mark.asyncio
    async def test_positive_spread(self, pm):
        ex_a = MockExchange("binance", {"BTC/USDT": 50100})
        ex_b = MockExchange("upbit", {"BTC/USDT": 50000})
        engine = make_engine(pm, exchanges=[ex_a, ex_b])

        spread = await engine._check_spread("BTC/USDT")
        assert spread is not None
        assert spread["spread_pct"] > 0  # A more expensive

    @pytest.mark.asyncio
    async def test_negative_spread(self, pm):
        ex_a = MockExchange("binance", {"BTC/USDT": 50000})
        ex_b = MockExchange("upbit", {"BTC/USDT": 50200})
        engine = make_engine(pm, exchanges=[ex_a, ex_b])

        spread = await engine._check_spread("BTC/USDT")
        assert spread is not None
        assert spread["spread_pct"] < 0  # B more expensive

    @pytest.mark.asyncio
    async def test_no_spread_zero_prices(self, pm):
        ex_a = MockExchange("binance", {"BTC/USDT": 0})
        ex_b = MockExchange("upbit", {"BTC/USDT": 0})
        engine = make_engine(pm, exchanges=[ex_a, ex_b])

        spread = await engine._check_spread("BTC/USDT")
        assert spread is None

    @pytest.mark.asyncio
    async def test_single_exchange_returns_none(self, pm):
        ex_a = MockExchange("binance", {"BTC/USDT": 50000})
        engine = make_engine(pm, exchanges=[ex_a])

        result = await engine._run_cycle()
        assert result.metadata.get("error") == "need_at_least_2_exchanges"


class TestArbExecution:
    @pytest.mark.asyncio
    async def test_execute_arb_positive_spread(self, pm):
        engine = make_engine(pm)
        await engine.start()

        spread_info = {
            "symbol": "BTC/USDT",
            "exchange_a": "binance",
            "price_a": 50200.0,  # more expensive
            "exchange_b": "upbit",
            "price_b": 50000.0,  # cheaper
            "spread_pct": 0.4,
            "mid_price": 50100.0,
        }

        result = await engine._execute_arb("BTC/USDT", spread_info)
        assert result is not None
        assert result["action"] == "arb_trade"
        assert result["buy_exchange"] == "upbit"
        assert result["sell_exchange"] == "binance"
        assert result["profit"] > 0

    @pytest.mark.asyncio
    async def test_execute_arb_negative_spread(self, pm):
        engine = make_engine(pm)
        await engine.start()

        spread_info = {
            "symbol": "ETH/USDT",
            "exchange_a": "binance",
            "price_a": 3000.0,  # cheaper
            "exchange_b": "upbit",
            "price_b": 3020.0,  # more expensive
            "spread_pct": -0.67,
            "mid_price": 3010.0,
        }

        result = await engine._execute_arb("ETH/USDT", spread_info)
        assert result is not None
        assert result["buy_exchange"] == "binance"
        assert result["sell_exchange"] == "upbit"
        assert result["profit"] > 0


class TestRunCycle:
    @pytest.mark.asyncio
    async def test_cycle_with_profitable_spread(self, pm):
        ex_a = MockExchange("binance", {"BTC/USDT": 50300, "ETH/USDT": 3000})
        ex_b = MockExchange("upbit", {"BTC/USDT": 50000, "ETH/USDT": 3000})
        engine = make_engine(pm, exchanges=[ex_a, ex_b])
        await engine.start()

        result = await engine._run_cycle()
        assert len(result.signals) == 2
        # BTC spread ~0.6% > 0.3% min → should trade
        btc_trades = [a for a in result.actions_taken if a["symbol"] == "BTC/USDT"]
        assert len(btc_trades) == 1
        assert btc_trades[0]["profit"] > 0
        # ETH has no spread → no trade
        eth_trades = [a for a in result.actions_taken if a["symbol"] == "ETH/USDT"]
        assert len(eth_trades) == 0

    @pytest.mark.asyncio
    async def test_cycle_below_threshold(self, pm):
        # Spread < 0.3% → no trade
        ex_a = MockExchange("binance", {"BTC/USDT": 50050, "ETH/USDT": 3000})
        ex_b = MockExchange("upbit", {"BTC/USDT": 50000, "ETH/USDT": 3000})
        engine = make_engine(pm, exchanges=[ex_a, ex_b])
        await engine.start()

        result = await engine._run_cycle()
        assert len(result.actions_taken) == 0

    @pytest.mark.asyncio
    async def test_pnl_accumulates(self, pm):
        ex_a = MockExchange("binance", {"BTC/USDT": 50300, "ETH/USDT": 3020})
        ex_b = MockExchange("upbit", {"BTC/USDT": 50000, "ETH/USDT": 3000})
        engine = make_engine(pm, exchanges=[ex_a, ex_b])
        await engine.start()

        r1 = await engine._run_cycle()
        assert r1.pnl_update > 0

        r2 = await engine._run_cycle()
        assert r2.metadata["total_arb_pnl"] > r1.pnl_update
