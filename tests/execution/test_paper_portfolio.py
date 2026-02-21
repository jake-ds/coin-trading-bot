"""Tests for PaperPortfolio - paper trading balance tracking."""

from unittest.mock import AsyncMock, PropertyMock

import pytest

from bot.data.store import DataStore
from bot.execution.engine import ExecutionEngine
from bot.execution.paper_portfolio import PaperPortfolio
from bot.models import OrderStatus, SignalAction, TradingSignal


class TestPaperPortfolio:
    """Test PaperPortfolio balance tracking and position management."""

    def test_initial_state(self):
        portfolio = PaperPortfolio(initial_balance=10000.0)
        assert portfolio.cash == 10000.0
        assert portfolio.positions == {}
        assert portfolio.total_value == 10000.0
        assert portfolio.unrealized_pnl == 0.0
        assert portfolio.trade_history == []

    def test_custom_initial_balance(self):
        portfolio = PaperPortfolio(initial_balance=50000.0)
        assert portfolio.cash == 50000.0
        assert portfolio.total_value == 50000.0

    def test_custom_fee_pct(self):
        portfolio = PaperPortfolio(fee_pct=0.2)
        assert portfolio.fee_pct == 0.2

    def test_buy_reduces_balance(self):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)
        result = portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)

        assert result is True
        # cost = 0.1 * 50000 = 5000, fee = 5000 * 0.1% = 5.0
        expected_cash = 10000.0 - 5000.0 - 5.0
        assert portfolio.cash == pytest.approx(expected_cash)

    def test_buy_creates_position(self):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)
        portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)

        positions = portfolio.positions
        assert "BTC/USDT" in positions
        assert positions["BTC/USDT"]["qty"] == 0.1
        assert positions["BTC/USDT"]["entry_price"] == 50000.0

    def test_buy_insufficient_balance_rejected(self):
        portfolio = PaperPortfolio(initial_balance=1000.0, fee_pct=0.1)
        result = portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)

        assert result is False
        assert portfolio.cash == 1000.0
        assert portfolio.positions == {}

    def test_sell_increases_balance(self):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)
        portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)
        cash_after_buy = portfolio.cash

        result = portfolio.sell("BTC/USDT", qty=0.1, price=55000.0)

        assert result is True
        # proceeds = 0.1 * 55000 = 5500, fee = 5500 * 0.1% = 5.5
        expected_cash = cash_after_buy + 5500.0 - 5.5
        assert portfolio.cash == pytest.approx(expected_cash)

    def test_sell_removes_position(self):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)
        portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)
        portfolio.sell("BTC/USDT", qty=0.1, price=55000.0)

        assert "BTC/USDT" not in portfolio.positions

    def test_sell_no_position_rejected(self):
        portfolio = PaperPortfolio(initial_balance=10000.0)
        result = portfolio.sell("BTC/USDT", qty=0.1, price=50000.0)

        assert result is False
        assert portfolio.cash == 10000.0

    def test_sell_insufficient_qty_rejected(self):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)
        portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)

        result = portfolio.sell("BTC/USDT", qty=0.2, price=55000.0)

        assert result is False

    def test_partial_sell(self):
        portfolio = PaperPortfolio(initial_balance=20000.0, fee_pct=0.1)
        portfolio.buy("BTC/USDT", qty=0.2, price=50000.0)
        portfolio.sell("BTC/USDT", qty=0.1, price=55000.0)

        positions = portfolio.positions
        assert "BTC/USDT" in positions
        assert positions["BTC/USDT"]["qty"] == pytest.approx(0.1)
        assert positions["BTC/USDT"]["entry_price"] == 50000.0

    def test_multiple_buys_average_entry_price(self):
        portfolio = PaperPortfolio(initial_balance=20000.0, fee_pct=0.0)
        portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)
        portfolio.buy("BTC/USDT", qty=0.1, price=60000.0)

        positions = portfolio.positions
        assert positions["BTC/USDT"]["qty"] == pytest.approx(0.2)
        # avg = (0.1*50000 + 0.1*60000) / 0.2 = 55000
        assert positions["BTC/USDT"]["entry_price"] == pytest.approx(55000.0)

    def test_multiple_positions(self):
        portfolio = PaperPortfolio(initial_balance=20000.0, fee_pct=0.1)
        portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)
        portfolio.buy("ETH/USDT", qty=1.0, price=3000.0)

        assert "BTC/USDT" in portfolio.positions
        assert "ETH/USDT" in portfolio.positions
        assert len(portfolio.positions) == 2

    def test_total_value_with_positions(self):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.0)
        portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)

        # cash = 10000 - 5000 = 5000; position value = 0.1 * 50000 = 5000
        assert portfolio.total_value == pytest.approx(10000.0)

        # Update price
        portfolio.update_price("BTC/USDT", 60000.0)
        # cash = 5000; position value = 0.1 * 60000 = 6000
        assert portfolio.total_value == pytest.approx(11000.0)

    def test_unrealized_pnl(self):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.0)
        portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)

        # Same price - no PnL
        assert portfolio.unrealized_pnl == pytest.approx(0.0)

        # Price goes up
        portfolio.update_price("BTC/USDT", 55000.0)
        # pnl = (55000 - 50000) * 0.1 = 500
        assert portfolio.unrealized_pnl == pytest.approx(500.0)

        # Price goes down
        portfolio.update_price("BTC/USDT", 45000.0)
        # pnl = (45000 - 50000) * 0.1 = -500
        assert portfolio.unrealized_pnl == pytest.approx(-500.0)

    def test_trade_history_recorded(self):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)
        portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)
        portfolio.sell("BTC/USDT", qty=0.1, price=55000.0)

        history = portfolio.trade_history
        assert len(history) == 2
        assert history[0]["side"] == "BUY"
        assert history[0]["symbol"] == "BTC/USDT"
        assert history[0]["qty"] == 0.1
        assert history[0]["price"] == 50000.0
        assert history[0]["fee"] == pytest.approx(5.0)
        assert history[1]["side"] == "SELL"
        assert history[1]["fee"] == pytest.approx(5.5)

    def test_fee_deduction_on_buy(self):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.5)
        portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)

        # cost = 5000, fee = 5000 * 0.5% = 25
        assert portfolio.cash == pytest.approx(10000.0 - 5000.0 - 25.0)

    def test_fee_deduction_on_sell(self):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.5)
        portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)
        cash_after_buy = portfolio.cash

        portfolio.sell("BTC/USDT", qty=0.1, price=50000.0)

        # proceeds = 5000, fee = 5000 * 0.5% = 25, net = 4975
        assert portfolio.cash == pytest.approx(cash_after_buy + 5000.0 - 25.0)

    def test_zero_fee(self):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.0)
        portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)

        assert portfolio.cash == pytest.approx(5000.0)

        portfolio.sell("BTC/USDT", qty=0.1, price=50000.0)
        assert portfolio.cash == pytest.approx(10000.0)

    def test_exact_balance_buy_with_fee_rejected(self):
        """If you have exactly enough for cost but not fee, should reject."""
        portfolio = PaperPortfolio(initial_balance=5000.0, fee_pct=0.1)
        # cost = 5000, fee = 5, total = 5005 > 5000
        result = portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)
        assert result is False

    def test_exact_balance_buy_no_fee_succeeds(self):
        portfolio = PaperPortfolio(initial_balance=5000.0, fee_pct=0.0)
        result = portfolio.buy("BTC/USDT", qty=0.1, price=50000.0)
        assert result is True
        assert portfolio.cash == pytest.approx(0.0)


class TestExecutionEngineWithPaperPortfolio:
    """Test ExecutionEngine integration with PaperPortfolio."""

    @pytest.fixture
    def mock_exchange(self):
        exchange = AsyncMock()
        type(exchange).name = PropertyMock(return_value="mock")
        exchange.get_ticker = AsyncMock(
            return_value={"last": 50000.0, "bid": 49990, "ask": 50010}
        )
        return exchange

    @pytest.fixture
    async def store(self):
        ds = DataStore(database_url="sqlite+aiosqlite:///:memory:")
        await ds.initialize()
        yield ds
        await ds.close()

    def make_signal(self, action=SignalAction.BUY, symbol="BTC/USDT"):
        return TradingSignal(
            strategy_name="test", symbol=symbol, action=action, confidence=0.8,
        )

    @pytest.mark.asyncio
    async def test_paper_buy_with_portfolio_deducts_balance(
        self, mock_exchange, store
    ):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)
        engine = ExecutionEngine(
            mock_exchange, store, paper_trading=True, paper_portfolio=portfolio
        )

        signal = self.make_signal(SignalAction.BUY)
        order = await engine.execute_signal(signal, quantity=0.1, price=50000.0)

        assert order is not None
        assert order.status == OrderStatus.FILLED
        # cash = 10000 - 5000 - 5 = 4995
        assert portfolio.cash == pytest.approx(4995.0)
        assert "BTC/USDT" in portfolio.positions

    @pytest.mark.asyncio
    async def test_paper_sell_with_portfolio_adds_balance(
        self, mock_exchange, store
    ):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)
        engine = ExecutionEngine(
            mock_exchange, store, paper_trading=True, paper_portfolio=portfolio
        )

        # Buy first
        buy_signal = self.make_signal(SignalAction.BUY)
        await engine.execute_signal(buy_signal, quantity=0.1, price=50000.0)
        cash_after_buy = portfolio.cash

        # Sell
        sell_signal = self.make_signal(SignalAction.SELL)
        order = await engine.execute_signal(sell_signal, quantity=0.1, price=55000.0)

        assert order is not None
        # proceeds = 5500, fee = 5.5
        assert portfolio.cash == pytest.approx(cash_after_buy + 5500.0 - 5.5)
        assert "BTC/USDT" not in portfolio.positions

    @pytest.mark.asyncio
    async def test_paper_buy_insufficient_balance_rejected(
        self, mock_exchange, store
    ):
        portfolio = PaperPortfolio(initial_balance=1000.0, fee_pct=0.1)
        engine = ExecutionEngine(
            mock_exchange, store, paper_trading=True, paper_portfolio=portfolio
        )

        signal = self.make_signal(SignalAction.BUY)
        order = await engine.execute_signal(signal, quantity=0.1, price=50000.0)

        assert order is None
        assert portfolio.cash == 1000.0

    @pytest.mark.asyncio
    async def test_paper_sell_no_position_rejected(self, mock_exchange, store):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)
        engine = ExecutionEngine(
            mock_exchange, store, paper_trading=True, paper_portfolio=portfolio
        )

        signal = self.make_signal(SignalAction.SELL)
        order = await engine.execute_signal(signal, quantity=0.1, price=50000.0)

        assert order is None

    @pytest.mark.asyncio
    async def test_engine_without_portfolio_works_as_before(
        self, mock_exchange, store
    ):
        """ExecutionEngine without paper_portfolio should use old unlimited behavior."""
        engine = ExecutionEngine(
            mock_exchange, store, paper_trading=True
        )

        signal = self.make_signal(SignalAction.BUY)
        order = await engine.execute_signal(signal, quantity=100.0, price=50000.0)

        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.fee == 0.0

    @pytest.mark.asyncio
    async def test_paper_order_includes_fee(self, mock_exchange, store):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)
        engine = ExecutionEngine(
            mock_exchange, store, paper_trading=True, paper_portfolio=portfolio
        )

        signal = self.make_signal(SignalAction.BUY)
        order = await engine.execute_signal(signal, quantity=0.1, price=50000.0)

        assert order is not None
        # fee = 0.1 * 50000 * 0.1% = 5.0
        assert order.fee == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_hold_signal_not_affected_by_portfolio(
        self, mock_exchange, store
    ):
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)
        engine = ExecutionEngine(
            mock_exchange, store, paper_trading=True, paper_portfolio=portfolio
        )

        signal = self.make_signal(SignalAction.HOLD)
        order = await engine.execute_signal(signal, quantity=0.1)

        assert order is None
        assert portfolio.cash == 10000.0
