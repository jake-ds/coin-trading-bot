"""Tests for the FuturesShortEngine."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.engines.futures_short import FuturesShortEngine
from bot.engines.portfolio_manager import PortfolioManager
from bot.models import OrderSide
from bot.onchain.models import CompositeSignal, SignalAction, SignalScore


@pytest.fixture
def portfolio_manager():
    pm = PortfolioManager(
        total_capital=1000.0,
        engine_allocations={"futures_short": 0.30},
        max_drawdown_pct=20.0,
    )
    return pm


@pytest.fixture
def mock_futures_exchange():
    exchange = AsyncMock()
    exchange.name = "binance_futures"
    exchange.get_ticker = AsyncMock(return_value={
        "bid": 84900,
        "ask": 85100,
        "last": 85000,
        "volume": 50000,
    })
    exchange.create_order = AsyncMock(return_value=MagicMock(
        filled_price=85000,
        filled_quantity=0.002,
        id="test-order-456",
    ))
    exchange.ensure_leverage_and_margin = AsyncMock()
    exchange.close = AsyncMock()
    return exchange


@pytest.fixture
def engine(portfolio_manager, mock_futures_exchange):
    eng = FuturesShortEngine(
        portfolio_manager=portfolio_manager,
        futures_exchange=mock_futures_exchange,
        paper_mode=True,
        settings=None,
    )
    return eng


class TestFuturesShortProperties:
    def test_engine_name(self, engine):
        assert engine.name == "futures_short"

    def test_engine_description(self, engine):
        assert "short" in engine.description.lower()
        assert "2x" in engine.description

    def test_supports_live(self, engine):
        assert engine.supports_live is True

    def test_default_leverage(self, engine):
        assert engine._leverage == 2

    def test_default_margin_mode(self, engine):
        assert engine._margin_mode == "isolated"


class TestFuturesShortCycle:
    @pytest.mark.asyncio
    async def test_cycle_no_data(self, engine, portfolio_manager):
        """Engine should handle cycle when all fetchers return None."""
        engine._coingecko.fetch_market_data = AsyncMock(return_value=None)
        engine._fear_greed.fetch = AsyncMock(return_value=None)
        engine._defillama.fetch = AsyncMock(return_value=None)
        engine._coinglass.fetch_derivatives = AsyncMock(return_value=None)
        engine._coinglass.fetch_exchange_flow = AsyncMock(return_value=None)

        engine._allocated_capital = portfolio_manager.request_capital(
            "futures_short", 300.0
        )
        result = await engine._run_cycle()

        assert result.engine_name == "futures_short"
        assert result.metadata["leverage"] == 2

    @pytest.mark.asyncio
    async def test_cycle_sell_signal_opens_short(self, engine, portfolio_manager):
        """Strong SELL signal should trigger a short position."""
        engine._coingecko.fetch_market_data = AsyncMock(return_value=None)
        engine._fear_greed.fetch = AsyncMock(return_value=None)
        engine._defillama.fetch = AsyncMock(return_value=None)
        engine._coinglass.fetch_derivatives = AsyncMock(return_value=None)
        engine._coinglass.fetch_exchange_flow = AsyncMock(return_value=None)

        # Only watch BTC for simplicity
        engine._symbols = ["BTC/USDT"]

        engine._allocated_capital = portfolio_manager.request_capital(
            "futures_short", 300.0
        )

        # Inject a strong SELL signal
        # Note: sentiment score +60 (contrarian BUY for Fear) gets flipped
        # to -60 by the momentum adjustment, making overall score very negative
        sell_signal = CompositeSignal(
            symbol="BTC/USDT",
            action=SignalAction.SELL,
            score=-45.0,
            confidence=0.8,
            signals=[
                SignalScore(name="sentiment", score=60.0, confidence=0.7, reason="Fear contrarian"),
                SignalScore(name="market_trend", score=-80.0, confidence=0.8, reason="24h: -5%"),
                SignalScore(name="derivatives", score=-30.0, confidence=0.5, reason="Bearish OI"),
            ],
        )

        with patch(
            "bot.engines.futures_short.compute_composite_signal",
            return_value=sell_signal,
        ):
            result = await engine._run_cycle()

        assert len(engine._positions) == 1
        pos = engine._positions["BTC/USDT"]
        assert pos["side"] == "short"
        assert pos["leverage"] == 2
        assert pos["entry_price"] == 85000
        assert len(result.actions_taken) == 1
        assert result.actions_taken[0]["action"] == "open_short"

    @pytest.mark.asyncio
    async def test_cycle_buy_signal_no_short(self, engine, portfolio_manager):
        """BUY signal should NOT open a short position."""
        engine._coingecko.fetch_market_data = AsyncMock(return_value=None)
        engine._fear_greed.fetch = AsyncMock(return_value=None)
        engine._defillama.fetch = AsyncMock(return_value=None)
        engine._coinglass.fetch_derivatives = AsyncMock(return_value=None)
        engine._coinglass.fetch_exchange_flow = AsyncMock(return_value=None)

        engine._symbols = ["BTC/USDT"]
        engine._allocated_capital = 300.0

        buy_signal = CompositeSignal(
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            score=45.0,
            confidence=0.8,
        )

        with patch(
            "bot.engines.futures_short.compute_composite_signal",
            return_value=buy_signal,
        ):
            result = await engine._run_cycle()

        assert len(engine._positions) == 0
        assert len(result.actions_taken) == 0


class TestShortPositionPnL:
    """Test SHORT position PnL calculation (inverted from longs)."""

    def test_short_pnl_profit_when_price_drops(self, engine):
        """Short position profits when price drops."""
        engine._add_position(
            symbol="BTC/USDT",
            side="short",
            quantity=0.002,
            entry_price=85000.0,
            current_price=85000.0,
            low_price=85000.0,
        )
        pos = engine._positions["BTC/USDT"]

        # Price drops to 83000 — 2.35% profit for SHORT
        current_price = 83000.0
        pnl_pct = ((85000.0 - current_price) / 85000.0) * 100.0
        assert pnl_pct == pytest.approx(2.35, rel=0.01)
        pnl_usd = (85000.0 - current_price) * 0.002
        assert pnl_usd == pytest.approx(4.0)

    def test_short_pnl_loss_when_price_rises(self, engine):
        """Short position loses when price rises."""
        entry = 85000.0
        current = 87000.0
        pnl_pct = ((entry - current) / entry) * 100.0
        assert pnl_pct < 0  # Negative = loss for short


class TestShortExitConditions:
    def test_stop_loss_on_price_rise(self, engine):
        """SHORT stop-loss triggers when price rises (adverse)."""
        engine._add_position(
            symbol="BTC/USDT",
            side="short",
            quantity=0.002,
            entry_price=85000.0,
            current_price=85000.0,
            low_price=85000.0,
        )
        pos = engine._positions["BTC/USDT"]

        # Price rose 5% — should trigger stop loss
        current_price = 89250.0
        pnl_pct = ((85000.0 - current_price) / 85000.0) * 100.0
        assert pnl_pct < -4.0

        exit_reason = engine._check_exit("BTC/USDT", pos, current_price, pnl_pct)
        assert exit_reason is not None
        assert "stop_loss" in exit_reason

    def test_take_profit_on_price_drop(self, engine):
        """SHORT take-profit triggers when price drops (favorable)."""
        engine._add_position(
            symbol="BTC/USDT",
            side="short",
            quantity=0.002,
            entry_price=85000.0,
            current_price=85000.0,
            low_price=85000.0,
        )
        pos = engine._positions["BTC/USDT"]

        # Price dropped 6% — should trigger take profit
        current_price = 79900.0
        pnl_pct = ((85000.0 - current_price) / 85000.0) * 100.0
        assert pnl_pct > 5.0

        exit_reason = engine._check_exit("BTC/USDT", pos, current_price, pnl_pct)
        assert exit_reason is not None
        assert "take_profit" in exit_reason

    def test_hold_when_within_bounds(self, engine):
        """Position held when PnL is within stop/take bounds."""
        engine._add_position(
            symbol="BTC/USDT",
            side="short",
            quantity=0.002,
            entry_price=85000.0,
            current_price=84500.0,
            low_price=84500.0,
        )
        pos = engine._positions["BTC/USDT"]

        current_price = 84500.0
        pnl_pct = ((85000.0 - current_price) / 85000.0) * 100.0
        assert 0 < pnl_pct < 1  # small profit

        exit_reason = engine._check_exit("BTC/USDT", pos, current_price, pnl_pct)
        assert exit_reason is None

    def test_signal_reversal_exit(self, engine):
        """Short closes on signal reversal to BUY."""
        engine._add_position(
            symbol="BTC/USDT",
            side="short",
            quantity=0.002,
            entry_price=85000.0,
            current_price=84000.0,
            low_price=84000.0,
        )
        pos = engine._positions["BTC/USDT"]

        # Inject BUY signal
        engine._latest_signals["BTC/USDT"] = CompositeSignal(
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            score=40.0,
            confidence=0.7,
        )

        current_price = 84000.0
        pnl_pct = ((85000.0 - current_price) / 85000.0) * 100.0

        exit_reason = engine._check_exit("BTC/USDT", pos, current_price, pnl_pct)
        assert exit_reason is not None
        assert "signal_reversal" in exit_reason

    def test_trailing_stop_for_short(self, engine):
        """Trailing stop should close short when price bounces from low."""
        engine._add_position(
            symbol="BTC/USDT",
            side="short",
            quantity=0.002,
            entry_price=85000.0,
            current_price=82000.0,
            low_price=82000.0,  # tracked lowest point
        )
        pos = engine._positions["BTC/USDT"]

        # Price bounced from 82000 to 84100 — 2.56% bounce from low
        current_price = 84100.0
        pnl_pct = ((85000.0 - current_price) / 85000.0) * 100.0

        exit_reason = engine._check_exit("BTC/USDT", pos, current_price, pnl_pct)
        assert exit_reason is not None
        assert "trailing_stop" in exit_reason


class TestShortPositionSizing:
    @pytest.mark.asyncio
    async def test_position_with_leverage(self, engine, portfolio_manager):
        """Position notional should reflect leverage."""
        engine._symbols = ["BTC/USDT"]
        engine._allocated_capital = 300.0
        engine._max_position_pct = 10.0  # 10% of $300 = $30 margin
        engine._leverage = 2  # $60 notional

        signal = CompositeSignal(
            symbol="BTC/USDT",
            action=SignalAction.SELL,
            score=-45.0,
            confidence=0.8,
        )

        result, fail = await engine._open_short("BTC/USDT", signal, {"size_mult": 1.0})
        assert result is not None
        assert fail is None
        assert result["leverage"] == 2
        # Notional = margin * leverage
        assert result["notional"] > result["margin"]
        assert result["notional"] == pytest.approx(result["margin"] * 2, rel=0.01)

    @pytest.mark.asyncio
    async def test_position_minimum_order_check(self, engine):
        """Notional below $10 should be rejected."""
        engine._allocated_capital = 10.0
        engine._max_position_pct = 1.0  # $0.10 margin → $0.20 notional
        engine._leverage = 2

        signal = CompositeSignal(
            symbol="BTC/USDT",
            action=SignalAction.SELL,
            score=-45.0,
            confidence=0.5,
        )

        result, fail = await engine._open_short("BTC/USDT", signal, {"size_mult": 1.0})
        assert result is None
        assert fail is not None


class TestShortClosePosition:
    @pytest.mark.asyncio
    async def test_paper_close_short(self, engine):
        """Paper short close returns correct PnL."""
        engine._add_position(
            symbol="BTC/USDT",
            side="short",
            quantity=0.002,
            entry_price=85000.0,
        )

        # Price dropped to 83000 — profit for short
        pnl = await engine._close_position("BTC/USDT", 83000.0, "take_profit")
        expected_pnl = (85000.0 - 83000.0) * 0.002  # = 4.0
        assert pnl == pytest.approx(expected_pnl)
        assert "BTC/USDT" not in engine._positions

    @pytest.mark.asyncio
    async def test_paper_close_short_loss(self, engine):
        """Paper short close with loss (price went up)."""
        engine._add_position(
            symbol="BTC/USDT",
            side="short",
            quantity=0.002,
            entry_price=85000.0,
        )

        # Price rose to 87000 — loss for short
        pnl = await engine._close_position("BTC/USDT", 87000.0, "stop_loss")
        expected_pnl = (85000.0 - 87000.0) * 0.002  # = -4.0
        assert pnl == pytest.approx(expected_pnl)
        assert pnl < 0

    @pytest.mark.asyncio
    async def test_live_close_short(self, engine, mock_futures_exchange):
        """Live short close uses reduceOnly BUY order."""
        engine._paper_mode = False
        engine._add_position(
            symbol="BTC/USDT",
            side="short",
            quantity=0.002,
            entry_price=85000.0,
        )

        pnl = await engine._close_position("BTC/USDT", 83000.0, "take_profit")
        assert pnl == pytest.approx(4.0)
        mock_futures_exchange.create_order.assert_called_once()
        call_kwargs = mock_futures_exchange.create_order.call_args
        assert call_kwargs.kwargs.get("side") == OrderSide.BUY
        assert call_kwargs.kwargs.get("reduce_only") is True


class TestLiveShortOpening:
    @pytest.mark.asyncio
    async def test_live_open_short(self, engine, mock_futures_exchange):
        """Live short should call ensure_leverage_and_margin then create_order."""
        engine._paper_mode = False
        engine._allocated_capital = 300.0
        engine._symbols = ["BTC/USDT"]

        signal = CompositeSignal(
            symbol="BTC/USDT",
            action=SignalAction.SELL,
            score=-45.0,
            confidence=0.8,
        )

        result, fail = await engine._open_short("BTC/USDT", signal, {"size_mult": 1.0})
        assert result is not None
        assert fail is None
        assert result["action"] == "open_short"

        mock_futures_exchange.ensure_leverage_and_margin.assert_called_once_with(
            "BTC/USDT", 2, "isolated"
        )
        mock_futures_exchange.create_order.assert_called_once()
        call_kwargs = mock_futures_exchange.create_order.call_args
        assert call_kwargs.kwargs.get("side") == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_live_short_no_exchange_fails(self, portfolio_manager):
        """Live mode without futures exchange should not open positions."""
        engine = FuturesShortEngine(
            portfolio_manager=portfolio_manager,
            futures_exchange=None,
            paper_mode=False,
        )
        engine._allocated_capital = 300.0

        signal = CompositeSignal(
            symbol="BTC/USDT",
            action=SignalAction.SELL,
            score=-45.0,
            confidence=0.8,
        )

        result, fail = await engine._open_short("BTC/USDT", signal, {"size_mult": 1.0})
        assert result is None
        assert fail is not None


class TestCapacityCheck:
    @pytest.mark.asyncio
    async def test_max_positions_respected(self, engine, portfolio_manager):
        """Should not open more shorts than max_positions."""
        engine._max_positions = 2
        engine._allocated_capital = 300.0
        engine._symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

        # Fill 2 positions
        engine._add_position(
            symbol="BTC/USDT", side="short", quantity=0.001, entry_price=85000.0
        )
        engine._add_position(
            symbol="ETH/USDT", side="short", quantity=0.01, entry_price=3000.0
        )

        assert not engine._has_capacity()

    @pytest.mark.asyncio
    async def test_duplicate_symbol_prevented(self, engine, portfolio_manager):
        """Should not open a second short on the same symbol."""
        engine._allocated_capital = 300.0
        engine._add_position(
            symbol="BTC/USDT", side="short", quantity=0.001, entry_price=85000.0
        )

        # Try to open another BTC/USDT short — should be skipped in _run_cycle
        engine._coingecko.fetch_market_data = AsyncMock(return_value=None)
        engine._fear_greed.fetch = AsyncMock(return_value=None)
        engine._defillama.fetch = AsyncMock(return_value=None)
        engine._coinglass.fetch_derivatives = AsyncMock(return_value=None)
        engine._coinglass.fetch_exchange_flow = AsyncMock(return_value=None)
        engine._symbols = ["BTC/USDT"]

        sell_signal = CompositeSignal(
            symbol="BTC/USDT",
            action=SignalAction.SELL,
            score=-45.0,
            confidence=0.8,
        )

        with patch(
            "bot.engines.futures_short.compute_composite_signal",
            return_value=sell_signal,
        ):
            result = await engine._run_cycle()

        # Should still have only 1 position (no duplicate)
        assert len(engine._positions) == 1
        assert len(result.actions_taken) == 0
