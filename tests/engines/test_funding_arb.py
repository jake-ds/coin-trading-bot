"""Tests for FundingRateArbEngine."""

import pytest

from bot.engines.funding_arb import FundingRateArbEngine
from bot.engines.portfolio_manager import PortfolioManager


@pytest.fixture
def pm():
    return PortfolioManager(
        total_capital=10000.0,
        engine_allocations={"funding_rate_arb": 0.30},
    )


def make_engine(pm, **overrides):
    """Create a FundingRateArbEngine with test defaults."""
    engine = FundingRateArbEngine(
        portfolio_manager=pm,
        exchanges=[],
        paper_mode=True,
        settings=None,
    )
    for k, v in overrides.items():
        setattr(engine, k, v)
    return engine


class TestFundingArbEngineProperties:
    def test_name(self, pm):
        engine = make_engine(pm)
        assert engine.name == "funding_rate_arb"

    def test_description(self, pm):
        engine = make_engine(pm)
        assert "delta-neutral" in engine.description.lower()


class TestDecisionLogic:
    def test_should_open_high_rate(self, pm):
        engine = make_engine(pm)
        assert engine._should_open(funding_rate=0.001, spread_pct=0.1)

    def test_should_not_open_low_rate(self, pm):
        engine = make_engine(pm)
        assert not engine._should_open(funding_rate=0.0001, spread_pct=0.1)

    def test_should_not_open_wide_spread(self, pm):
        engine = make_engine(pm)
        assert not engine._should_open(funding_rate=0.001, spread_pct=0.6)

    def test_should_close_low_rate(self, pm):
        engine = make_engine(pm)
        assert engine._should_close(funding_rate=0.00005, spread_pct=0.1)

    def test_should_close_very_wide_spread(self, pm):
        engine = make_engine(pm)
        # max_spread_pct=0.5, close threshold is 2x = 1.0
        assert engine._should_close(funding_rate=0.001, spread_pct=1.1)

    def test_should_not_close_healthy(self, pm):
        engine = make_engine(pm)
        assert not engine._should_close(funding_rate=0.0005, spread_pct=0.2)


class TestPositionManagement:
    @pytest.mark.asyncio
    async def test_open_position_paper(self, pm):
        engine = make_engine(pm)
        await engine.start()

        opened = await engine._open_position(
            "BTC/USDT",
            {"funding_rate": 0.0005, "mark_price": 50000, "spot_price": 49990},
        )
        assert opened
        assert engine.position_count == 1
        pos = engine.positions["BTC/USDT"]
        assert pos["side"] == "delta_neutral"
        assert pos["funding_rate"] == 0.0005

    @pytest.mark.asyncio
    async def test_open_position_zero_price(self, pm):
        engine = make_engine(pm)
        await engine.start()

        opened = await engine._open_position(
            "BTC/USDT",
            {"funding_rate": 0.0005, "mark_price": 0, "spot_price": 0},
        )
        assert not opened
        assert engine.position_count == 0

    @pytest.mark.asyncio
    async def test_close_position_returns_pnl(self, pm):
        engine = make_engine(pm)
        await engine.start()

        await engine._open_position(
            "BTC/USDT",
            {"funding_rate": 0.0005, "mark_price": 50000, "spot_price": 49990},
        )
        pnl = await engine._close_position("BTC/USDT")
        assert pnl > 0  # funding_rate * quantity * price
        assert engine.position_count == 0

    @pytest.mark.asyncio
    async def test_close_nonexistent_position(self, pm):
        engine = make_engine(pm)
        pnl = await engine._close_position("NONEXISTENT/USDT")
        assert pnl == 0.0


class TestRunCycle:
    @pytest.mark.asyncio
    async def test_cycle_no_exchanges(self, pm):
        """Cycle with no exchanges returns empty signals."""
        engine = make_engine(pm)
        await engine.start()

        result = await engine._run_cycle()
        assert result.engine_name == "funding_rate_arb"
        assert result.signals == []
        assert result.actions_taken == []

    @pytest.mark.asyncio
    async def test_cycle_with_mock_exchange(self, pm):
        """Cycle with a mock exchange that returns high funding rate."""

        class MockFuturesExchange:
            name = "mock_futures"

            async def get_funding_rate(self, symbol):
                return {
                    "symbol": symbol,
                    "funding_rate": 0.001,
                    "mark_price": 50000.0,
                    "spot_price": 49990.0,
                    "spread_pct": 0.02,
                }

        engine = make_engine(pm, _exchanges=[MockFuturesExchange()])
        await engine.start()

        result = await engine._run_cycle()
        assert len(result.signals) == 2  # BTC + ETH
        assert len(result.actions_taken) == 2  # Both should be opened
        assert result.actions_taken[0]["action"] == "open"

    @pytest.mark.asyncio
    async def test_cycle_closes_low_rate(self, pm):
        """Cycle closes position when funding rate drops."""

        call_count = 0

        class MockExchange:
            name = "mock"

            async def get_funding_rate(self, symbol):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    # First calls: high rate → opens
                    return {
                        "symbol": symbol,
                        "funding_rate": 0.001,
                        "mark_price": 50000.0,
                        "spot_price": 49990.0,
                        "spread_pct": 0.02,
                    }
                else:
                    # Subsequent: low rate → closes
                    return {
                        "symbol": symbol,
                        "funding_rate": 0.00005,
                        "mark_price": 50000.0,
                        "spot_price": 49990.0,
                        "spread_pct": 0.02,
                    }

        engine = make_engine(pm, _exchanges=[MockExchange()])
        await engine.start()

        # First cycle: opens positions
        r1 = await engine._run_cycle()
        assert len([a for a in r1.actions_taken if a["action"] == "open"]) == 2

        # Second cycle: closes positions (low rate)
        r2 = await engine._run_cycle()
        closes = [a for a in r2.actions_taken if a["action"] == "close"]
        assert len(closes) == 2
        assert engine.position_count == 0

    @pytest.mark.asyncio
    async def test_respects_max_positions(self, pm):
        """Engine won't open more than max_positions."""

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

        engine = make_engine(pm, _exchanges=[MockExchange()], _max_positions=1)
        await engine.start()

        result = await engine._run_cycle()
        # Only 1 position should be opened despite 2 symbols available
        opens = [a for a in result.actions_taken if a["action"] == "open"]
        assert len(opens) == 1
        assert engine.position_count == 1
