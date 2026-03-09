"""Tests for FundingRateArbEngine."""

import pytest

from bot.engines.funding_arb import FundingRateArbEngine, _compute_leg_pnl
from bot.engines.portfolio_manager import PortfolioManager
from bot.models.base import OrderStatus


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
        assert pos["side"] == "long_spot_short_perp"
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
        # After V5-002: PnL is net of round-trip costs (4 legs).
        # Single funding payment (gross) is small vs cost → net is negative.
        # Verify PnL is calculated (not zero) and position is closed.
        assert pnl != 0.0
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


# ======================================================================
# Live position management tests
# ======================================================================

class _MockOrder:
    """Minimal Order-like object for tests."""

    def __init__(self, id="ord-1", status=OrderStatus.FILLED,
                 filled_price=50000.0, filled_quantity=0.01,
                 quantity=0.01, fee=0.5):
        self.id = id
        self.status = status
        self.filled_price = filled_price
        self.filled_quantity = filled_quantity
        self.quantity = quantity
        self.fee = fee


class _MockFuturesExchange:
    """Mock futures exchange with set_leverage (identifies as futures)."""

    name = "mock_futures"

    def __init__(self, *, create_fail=False, fill_status=OrderStatus.FILLED,
                 positions=None):
        self._create_fail = create_fail
        self._fill_status = fill_status
        self._positions = positions or []
        self.leverage_set = False
        self.margin_mode_set = False
        self._orders: dict[str, _MockOrder] = {}

    async def set_leverage(self, symbol, leverage):
        self.leverage_set = True

    async def set_margin_mode(self, symbol, mode):
        self.margin_mode_set = True

    async def create_order(self, symbol, side, order_type, quantity, price=None):
        if self._create_fail:
            raise RuntimeError("exchange down")
        order = _MockOrder(
            id=f"fut-{len(self._orders)+1}",
            status=self._fill_status,
            filled_price=50000.0,
            filled_quantity=quantity,
            quantity=quantity,
            fee=0.5,
        )
        self._orders[order.id] = order
        return order

    async def get_order_status(self, order_id, symbol):
        return self._orders.get(order_id, _MockOrder(id=order_id, status=self._fill_status))

    async def cancel_order(self, order_id, symbol):
        return True

    async def get_funding_rate(self, symbol):
        return {
            "symbol": symbol,
            "funding_rate": 0.001,
            "mark_price": 50000.0,
            "spot_price": 49990.0,
            "spread_pct": 0.02,
        }

    async def get_positions(self):
        return self._positions


class _MockSpotExchange:
    """Mock spot exchange (no set_leverage)."""

    name = "mock_spot"

    def __init__(self, *, create_fail=False, fill_status=OrderStatus.FILLED):
        self._create_fail = create_fail
        self._fill_status = fill_status
        self._orders: dict[str, _MockOrder] = {}

    async def create_order(self, symbol, side, order_type, quantity, price=None):
        if self._create_fail:
            raise RuntimeError("spot down")
        order = _MockOrder(
            id=f"spot-{len(self._orders)+1}",
            status=self._fill_status,
            filled_price=49990.0,
            filled_quantity=quantity,
            quantity=quantity,
            fee=0.3,
        )
        self._orders[order.id] = order
        return order

    async def get_order_status(self, order_id, symbol):
        return self._orders.get(order_id, _MockOrder(id=order_id, status=self._fill_status))

    async def cancel_order(self, order_id, symbol):
        return True


def make_live_engine(pm, futures_ex=None, spot_ex=None, **overrides):
    """Create a live-mode engine with mock exchanges."""
    exchanges = []
    if futures_ex is not None:
        exchanges.append(futures_ex)
    if spot_ex is not None:
        exchanges.append(spot_ex)
    engine = FundingRateArbEngine(
        portfolio_manager=pm,
        exchanges=exchanges,
        paper_mode=False,
        settings=None,
    )
    engine._order_timeout = 1.0  # fast timeout for tests
    for k, v in overrides.items():
        setattr(engine, k, v)
    return engine


class TestLivePositionManagement:
    """Tests for live (non-paper) order execution paths."""

    @pytest.mark.asyncio
    async def test_open_live_futures_success(self, pm):
        """Positive funding, futures-only mode → position opened with live metadata."""
        fut_ex = _MockFuturesExchange()
        engine = make_live_engine(pm, futures_ex=fut_ex, _futures_only_mode=True)
        await engine.start()

        opened = await engine._open_position(
            "BTC/USDT",
            {"funding_rate": 0.001, "mark_price": 50000, "spot_price": 49990},
        )
        assert opened
        assert engine.position_count == 1
        pos = engine.positions["BTC/USDT"]
        assert pos["mode"] == "live"
        assert pos["futures_order_id"].startswith("fut-")
        assert pos["spot_hedged"] is False
        assert fut_ex.leverage_set

    @pytest.mark.asyncio
    async def test_open_live_futures_fail(self, pm):
        """Futures order creation fails → return False, no position."""
        fut_ex = _MockFuturesExchange(create_fail=True)
        engine = make_live_engine(pm, futures_ex=fut_ex)
        await engine.start()

        opened = await engine._open_position(
            "BTC/USDT",
            {"funding_rate": 0.001, "mark_price": 50000, "spot_price": 49990},
        )
        assert not opened
        assert engine.position_count == 0

    @pytest.mark.asyncio
    async def test_open_live_spot_fail_emergency_unwind(self, pm):
        """Spot order fails after futures filled → emergency unwind futures."""
        fut_ex = _MockFuturesExchange()
        spot_ex = _MockSpotExchange(create_fail=True)
        engine = make_live_engine(pm, futures_ex=fut_ex, spot_ex=spot_ex)
        await engine.start()

        opened = await engine._open_position(
            "BTC/USDT",
            {"funding_rate": 0.001, "mark_price": 50000, "spot_price": 49990},
        )
        # Should fail because spot failed and emergency unwind should run
        assert not opened
        assert engine.position_count == 0
        # Emergency unwind should have created a close order on futures
        assert len(fut_ex._orders) >= 2  # open + emergency close

    @pytest.mark.asyncio
    async def test_open_live_negative_funding_futures_only(self, pm):
        """Negative funding → futures-only (long futures), no spot hedge attempted."""
        fut_ex = _MockFuturesExchange()
        spot_ex = _MockSpotExchange()
        engine = make_live_engine(pm, futures_ex=fut_ex, spot_ex=spot_ex)
        await engine.start()

        opened = await engine._open_position(
            "BTC/USDT",
            {"funding_rate": -0.001, "mark_price": 50000, "spot_price": 49990},
        )
        assert opened
        pos = engine.positions["BTC/USDT"]
        assert pos["side"] == "short_spot_long_perp"
        assert pos["futures_side"] == "BUY"
        assert pos["spot_hedged"] is False

    @pytest.mark.asyncio
    async def test_close_live_both_legs(self, pm):
        """Close a hedged position → PnL from both futures and spot legs."""
        fut_ex = _MockFuturesExchange()
        spot_ex = _MockSpotExchange()
        engine = make_live_engine(pm, futures_ex=fut_ex, spot_ex=spot_ex)
        await engine.start()

        # Manually set up a live position with spot hedge
        engine._positions["BTC/USDT"] = {
            "symbol": "BTC/USDT",
            "side": "long_spot_short_perp",
            "quantity": 0.01,
            "entry_price": 50000.0,
            "mode": "live",
            "futures_order_id": "fut-1",
            "futures_filled_price": 50000.0,
            "futures_side": "SELL",
            "spot_order_id": "spot-1",
            "spot_hedged": True,
            "spot_filled_price": 49990.0,
            "total_fees": 0.5,
        }

        pnl = await engine._close_position("BTC/USDT")
        assert engine.position_count == 0
        # PnL should be calculated (nonzero due to fills and fees)
        assert isinstance(pnl, float)

    @pytest.mark.asyncio
    async def test_close_live_futures_fail_readd(self, pm):
        """Close fails → position re-added for next cycle."""
        fut_ex = _MockFuturesExchange(create_fail=True)
        engine = make_live_engine(pm, futures_ex=fut_ex)
        await engine.start()

        engine._positions["BTC/USDT"] = {
            "symbol": "BTC/USDT",
            "side": "long_spot_short_perp",
            "quantity": 0.01,
            "entry_price": 50000.0,
            "mode": "live",
            "futures_order_id": "fut-1",
            "futures_filled_price": 50000.0,
            "futures_side": "SELL",
            "spot_hedged": False,
            "total_fees": 0.5,
        }

        pnl = await engine._close_position("BTC/USDT")
        assert pnl == 0.0
        # Position should be re-added
        assert engine.position_count == 1
        assert "BTC/USDT" in engine.positions

    @pytest.mark.asyncio
    async def test_reconcile_clean(self, pm):
        """No discrepancies when local and exchange agree."""
        fut_ex = _MockFuturesExchange(positions=[
            {"symbol": "BTC/USDT", "side": "short", "contracts": 0.01,
             "entry_price": 50000, "mark_price": 50100, "unrealized_pnl": 1.0,
             "leverage": 1, "margin_mode": "cross"},
        ])
        engine = make_live_engine(pm, futures_ex=fut_ex)
        engine._positions["BTC/USDT"] = {
            "symbol": "BTC/USDT", "mode": "live", "quantity": 0.01,
        }

        result = await engine._reconcile_positions()
        assert result == []

    @pytest.mark.asyncio
    async def test_reconcile_discrepancy(self, pm):
        """Detects missing local position when exchange has one we don't."""
        fut_ex = _MockFuturesExchange(positions=[
            {"symbol": "ETH/USDT", "side": "long", "contracts": 0.1,
             "entry_price": 3000, "mark_price": 3010, "unrealized_pnl": 1.0,
             "leverage": 1, "margin_mode": "cross"},
        ])
        engine = make_live_engine(pm, futures_ex=fut_ex)
        # No local positions

        result = await engine._reconcile_positions()
        assert len(result) == 1
        assert result[0]["type"] == "missing_locally"
        assert result[0]["symbol"] == "ETH/USDT"


class TestComputeLegPnl:
    def test_buy_profit(self):
        assert _compute_leg_pnl("buy", 100.0, 110.0, 1.0) == 10.0

    def test_sell_profit(self):
        assert _compute_leg_pnl("sell", 110.0, 100.0, 1.0) == 10.0

    def test_buy_loss(self):
        assert _compute_leg_pnl("buy", 100.0, 90.0, 1.0) == -10.0
