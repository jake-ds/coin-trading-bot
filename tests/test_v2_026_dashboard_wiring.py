"""Integration tests for V2-026: Dashboard wiring in main.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.config import Settings, TradingMode
from bot.dashboard.app import get_state, set_strategy_registry, update_state
from bot.main import TradingBot


def make_settings(**overrides):
    defaults = {
        "trading_mode": TradingMode.PAPER,
        "binance_api_key": "",
        "upbit_api_key": "",
        "database_url": "sqlite+aiosqlite:///:memory:",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "symbols": ["BTC/USDT"],
        "loop_interval_seconds": 1,
        "signal_min_agreement": 1,
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture(autouse=True)
def reset_dashboard():
    """Reset dashboard state before each test."""
    update_state(
        status="stopped",
        started_at=None,
        trades=[],
        metrics={},
        portfolio={"balances": {}, "positions": [], "total_value": 0.0},
        cycle_metrics={
            "cycle_count": 0,
            "average_cycle_duration": 0.0,
            "last_cycle_time": None,
        },
        strategy_stats={},
        equity_curve=[],
        open_positions=[],
        regime=None,
    )
    set_strategy_registry(None)


class TestBuildOpenPositions:
    def test_no_position_manager(self):
        """Returns empty list when no position manager."""
        bot = TradingBot(settings=make_settings())
        result = bot._build_open_positions()
        assert result == []

    def test_with_positions(self):
        """Builds position data from PositionManager."""
        from bot.execution.position_manager import PositionManager

        bot = TradingBot(settings=make_settings())
        bot._position_manager = PositionManager(
            stop_loss_pct=3.0, take_profit_pct=5.0
        )
        bot._position_manager.add_position("BTC/USDT", 50000.0, 0.1)

        result = bot._build_open_positions()
        assert len(result) == 1
        pos = result[0]
        assert pos["symbol"] == "BTC/USDT"
        assert pos["quantity"] == 0.1
        assert pos["entry_price"] == 50000.0
        assert pos["stop_loss"] == pytest.approx(50000 * 0.97)
        assert pos["take_profit"] == pytest.approx(50000 * 1.05)

    def test_with_ws_price(self):
        """Uses WebSocket price when available."""
        from bot.execution.position_manager import PositionManager

        bot = TradingBot(settings=make_settings())
        bot._position_manager = PositionManager()
        bot._position_manager.add_position("ETH/USDT", 3000.0, 1.0)

        mock_ws = MagicMock()
        mock_ws.get_latest_price.return_value = 3200.0
        bot._ws_feed = mock_ws

        result = bot._build_open_positions()
        assert result[0]["current_price"] == 3200.0
        assert result[0]["unrealized_pnl"] == pytest.approx(200.0)


class TestMaybeSavePortfolioSnapshot:
    @pytest.mark.asyncio
    async def test_skip_non_10th_cycle(self):
        """Skips snapshot when cycle count is not multiple of 10."""
        bot = TradingBot(settings=make_settings())
        bot._cycle_count = 5
        bot._store = MagicMock()
        bot._store.save_portfolio_snapshot = AsyncMock()

        await bot._maybe_save_portfolio_snapshot()
        bot._store.save_portfolio_snapshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_cycle_zero(self):
        """Skips snapshot at cycle 0."""
        bot = TradingBot(settings=make_settings())
        bot._cycle_count = 0
        bot._store = MagicMock()
        bot._store.save_portfolio_snapshot = AsyncMock()

        await bot._maybe_save_portfolio_snapshot()
        bot._store.save_portfolio_snapshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_saves_at_10th_cycle(self):
        """Saves snapshot at cycle 10."""
        from bot.execution.paper_portfolio import PaperPortfolio

        bot = TradingBot(settings=make_settings())
        bot._cycle_count = 10
        bot._store = MagicMock()
        bot._store.save_portfolio_snapshot = AsyncMock()
        bot._paper_portfolio = PaperPortfolio(initial_balance=10500.0)

        await bot._maybe_save_portfolio_snapshot()
        bot._store.save_portfolio_snapshot.assert_called_once()
        call_kwargs = bot._store.save_portfolio_snapshot.call_args
        assert call_kwargs.kwargs["total_value"] == 10500.0

    @pytest.mark.asyncio
    async def test_saves_at_20th_cycle(self):
        """Saves snapshot at cycle 20."""
        bot = TradingBot(settings=make_settings())
        bot._cycle_count = 20
        bot._store = MagicMock()
        bot._store.save_portfolio_snapshot = AsyncMock()
        bot._paper_portfolio = MagicMock()
        bot._paper_portfolio.total_value = 11000.0
        bot._paper_portfolio.unrealized_pnl = 500.0

        await bot._maybe_save_portfolio_snapshot()
        bot._store.save_portfolio_snapshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_updates_equity_curve_in_dashboard(self):
        """Snapshot saving appends to dashboard equity curve."""
        bot = TradingBot(settings=make_settings())
        bot._cycle_count = 10
        bot._store = MagicMock()
        bot._store.save_portfolio_snapshot = AsyncMock()
        bot._paper_portfolio = MagicMock()
        bot._paper_portfolio.total_value = 10200.0
        bot._paper_portfolio.unrealized_pnl = 200.0

        await bot._maybe_save_portfolio_snapshot()

        state = get_state()
        assert len(state["equity_curve"]) == 1
        assert state["equity_curve"][0]["total_value"] == 10200.0

    @pytest.mark.asyncio
    async def test_no_store_skips(self):
        """Skips when no store available."""
        bot = TradingBot(settings=make_settings())
        bot._cycle_count = 10
        bot._store = None
        # Should not raise
        await bot._maybe_save_portfolio_snapshot()

    @pytest.mark.asyncio
    async def test_error_handled_gracefully(self):
        """Errors in snapshot saving are caught."""
        bot = TradingBot(settings=make_settings())
        bot._cycle_count = 10
        bot._store = MagicMock()
        bot._store.save_portfolio_snapshot = AsyncMock(
            side_effect=RuntimeError("DB error")
        )
        bot._paper_portfolio = MagicMock()
        bot._paper_portfolio.total_value = 10000.0
        bot._paper_portfolio.unrealized_pnl = 0.0

        # Should not raise
        await bot._maybe_save_portfolio_snapshot()


class TestCurrentRegimeTracking:
    def test_initial_regime_is_none(self):
        """Bot starts with no regime."""
        bot = TradingBot(settings=make_settings())
        assert bot._current_regime is None


class TestDashboardRegistryWiring:
    @pytest.mark.asyncio
    async def test_strategy_registry_set_on_initialize(self):
        """Strategy registry is passed to dashboard during initialize."""
        bot = TradingBot(settings=make_settings())
        bot._store = MagicMock()
        bot._store.initialize = AsyncMock()
        bot._store.close = AsyncMock()

        with patch("bot.main.dashboard_module") as mock_dash:
            mock_dash.update_state = MagicMock()
            mock_dash.get_state = MagicMock(return_value=get_state())
            mock_dash.set_strategy_registry = MagicMock()
            mock_dash.app = MagicMock()

            # Partially initialize — skip uvicorn
            with patch("bot.main.TradingBot._start_dashboard"):
                with patch("bot.main.TradingBot._init_ws_feed"):
                    with patch("bot.main.TradingBot._load_strategies"):
                        await bot.initialize()

            mock_dash.set_strategy_registry.assert_called_once()
