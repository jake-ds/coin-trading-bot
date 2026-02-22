"""Tests for V2-026: Enhanced dashboard with real-time charts and equity curve."""

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from bot.dashboard.app import (
    _build_equity_labels_js,
    _build_equity_values_js,
    _build_strategies_list_html,
    _build_strategy_colors_js,
    _build_strategy_names_js,
    _build_strategy_pnl_js,
    _build_trade_markers_js,
    _find_closest_index,
    _regime_color,
    app,
    get_state,
    set_strategy_registry,
    update_state,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset bot state before each test."""
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
    # Reset strategy registry
    set_strategy_registry(None)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --- New API Endpoint Tests ---


class TestEquityCurveEndpoint:
    @pytest.mark.asyncio
    async def test_equity_curve_empty(self, client):
        resp = await client.get("/equity-curve")
        assert resp.status_code == 200
        data = resp.json()
        assert data["equity_curve"] == []

    @pytest.mark.asyncio
    async def test_equity_curve_with_data(self, client):
        update_state(equity_curve=[
            {"timestamp": "2026-01-01T00:00:00", "total_value": 10000.0},
            {"timestamp": "2026-01-01T01:00:00", "total_value": 10050.0},
            {"timestamp": "2026-01-01T02:00:00", "total_value": 10100.0},
        ])
        resp = await client.get("/equity-curve")
        data = resp.json()
        assert len(data["equity_curve"]) == 3
        assert data["equity_curve"][0]["total_value"] == 10000.0
        assert data["equity_curve"][2]["total_value"] == 10100.0


class TestOpenPositionsEndpoint:
    @pytest.mark.asyncio
    async def test_open_positions_empty(self, client):
        resp = await client.get("/open-positions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["positions"] == []

    @pytest.mark.asyncio
    async def test_open_positions_with_data(self, client):
        update_state(open_positions=[
            {
                "symbol": "BTC/USDT",
                "quantity": 0.1,
                "entry_price": 50000,
                "current_price": 51000,
                "unrealized_pnl": 100.0,
                "stop_loss": 48500,
                "take_profit": 52500,
            }
        ])
        resp = await client.get("/open-positions")
        data = resp.json()
        assert len(data["positions"]) == 1
        assert data["positions"][0]["symbol"] == "BTC/USDT"
        assert data["positions"][0]["unrealized_pnl"] == 100.0


class TestRegimeEndpoint:
    @pytest.mark.asyncio
    async def test_regime_default(self, client):
        resp = await client.get("/regime")
        assert resp.status_code == 200
        data = resp.json()
        assert data["regime"] is None

    @pytest.mark.asyncio
    async def test_regime_with_data(self, client):
        update_state(regime="TRENDING_UP")
        resp = await client.get("/regime")
        data = resp.json()
        assert data["regime"] == "TRENDING_UP"


class TestStrategyToggle:
    @pytest.mark.asyncio
    async def test_toggle_no_registry(self, client):
        """Toggle without registry returns error."""
        resp = await client.post("/strategies/test_strategy/toggle")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "not available" in data["error"]

    @pytest.mark.asyncio
    async def test_toggle_strategy_not_found(self, client):
        """Toggle non-existent strategy returns error."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        set_strategy_registry(mock_registry)

        resp = await client.post("/strategies/nonexistent/toggle")
        data = resp.json()
        assert data["success"] is False
        assert "not found" in data["error"]

    @pytest.mark.asyncio
    async def test_toggle_disable_active_strategy(self, client):
        """Toggle an active strategy disables it."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock(name="ma_crossover")
        mock_registry.is_active.return_value = True
        mock_registry.disable.return_value = True
        set_strategy_registry(mock_registry)

        resp = await client.post("/strategies/ma_crossover/toggle")
        data = resp.json()
        assert data["success"] is True
        assert data["active"] is False
        assert data["name"] == "ma_crossover"
        mock_registry.disable.assert_called_once_with("ma_crossover")

    @pytest.mark.asyncio
    async def test_toggle_enable_inactive_strategy(self, client):
        """Toggle an inactive strategy enables it."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock(name="rsi")
        mock_registry.is_active.return_value = False
        mock_registry.enable.return_value = True
        set_strategy_registry(mock_registry)

        resp = await client.post("/strategies/rsi/toggle")
        data = resp.json()
        assert data["success"] is True
        assert data["active"] is True
        assert data["name"] == "rsi"
        mock_registry.enable.assert_called_once_with("rsi")


# --- Dashboard HTML Tests ---


class TestEnhancedDashboardHTML:
    @pytest.mark.asyncio
    async def test_dashboard_includes_chartjs(self, client):
        """Dashboard HTML includes Chart.js CDN."""
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "chart.js" in resp.text.lower() or "Chart" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_includes_equity_chart(self, client):
        """Dashboard HTML includes equity curve chart canvas."""
        resp = await client.get("/")
        assert "equityChart" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_includes_strategy_chart(self, client):
        """Dashboard HTML includes strategy performance chart canvas."""
        resp = await client.get("/")
        assert "strategyChart" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_includes_positions_table(self, client):
        """Dashboard HTML includes open positions table."""
        resp = await client.get("/")
        assert "Open Positions" in resp.text
        assert "Stop Loss" in resp.text
        assert "Take Profit" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_shows_regime(self, client):
        """Dashboard HTML shows market regime badge."""
        update_state(regime="TRENDING_UP")
        resp = await client.get("/")
        assert "TRENDING_UP" in resp.text
        assert "regime-badge" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_shows_open_positions(self, client):
        """Dashboard HTML renders open positions data."""
        update_state(open_positions=[
            {
                "symbol": "ETH/USDT",
                "quantity": 1.0,
                "entry_price": 3000.0,
                "current_price": 3100.0,
                "unrealized_pnl": 100.0,
                "stop_loss": 2910.0,
                "take_profit": 3150.0,
            }
        ])
        resp = await client.get("/")
        assert "ETH/USDT" in resp.text
        assert "3,000.00" in resp.text
        assert "3,100.00" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_strategy_list(self, client):
        """Dashboard HTML shows strategy list with toggle buttons."""
        update_state(strategy_stats={
            "ma_crossover": {"total_pnl": 150.0, "win_rate": 55.0, "active": True},
            "rsi": {"total_pnl": -50.0, "win_rate": 40.0, "active": False},
        })
        resp = await client.get("/")
        assert "ma_crossover" in resp.text
        assert "rsi" in resp.text
        assert "toggleStrategy" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_trade_markers_buy_sell(self, client):
        """Dashboard shows BUY/SELL styling on trades."""
        update_state(trades=[
            {"timestamp": "2026-01-01T00:00:00", "symbol": "BTC/USDT",
             "side": "BUY", "quantity": 0.1, "price": 50000},
            {"timestamp": "2026-01-01T01:00:00", "symbol": "BTC/USDT",
             "side": "SELL", "quantity": 0.1, "price": 51000},
        ])
        resp = await client.get("/")
        text = resp.text
        assert 'class="buy"' in text
        assert 'class="sell"' in text


# --- Helper Function Unit Tests ---


class TestRegimeColor:
    def test_trending_up(self):
        assert _regime_color("TRENDING_UP") == "#22c55e"

    def test_trending_down(self):
        assert _regime_color("TRENDING_DOWN") == "#ef4444"

    def test_ranging(self):
        assert _regime_color("RANGING") == "#f59e0b"

    def test_high_volatility(self):
        assert _regime_color("HIGH_VOLATILITY") == "#8b5cf6"

    def test_none(self):
        assert _regime_color(None) == "#6b7280"

    def test_unknown(self):
        assert _regime_color("OTHER") == "#6b7280"


class TestStrategyJSBuilders:
    def test_empty_strategy_stats(self):
        assert _build_strategy_names_js({}) == "[]"
        assert _build_strategy_pnl_js({}) == "[]"
        assert _build_strategy_colors_js({}) == "[]"

    def test_strategy_names(self):
        stats = {"rsi": {"total_pnl": 100}, "macd": {"total_pnl": -50}}
        result = _build_strategy_names_js(stats)
        assert '"rsi"' in result
        assert '"macd"' in result

    def test_strategy_pnl(self):
        stats = {"rsi": {"total_pnl": 100.5}, "macd": {"total_pnl": -50.0}}
        result = _build_strategy_pnl_js(stats)
        assert "100.5" in result
        assert "-50.0" in result

    def test_strategy_colors_positive_negative(self):
        stats = {"rsi": {"total_pnl": 100}, "macd": {"total_pnl": -50}}
        result = _build_strategy_colors_js(stats)
        assert '"#22c55e"' in result  # green for positive
        assert '"#ef4444"' in result  # red for negative


class TestEquityCurveJSBuilders:
    def test_empty_equity_curve(self):
        assert _build_equity_labels_js([]) == "[]"
        assert _build_equity_values_js([]) == "[]"

    def test_equity_labels(self):
        curve = [
            {"timestamp": "2026-01-01", "total_value": 10000},
            {"timestamp": "2026-01-02", "total_value": 10100},
        ]
        result = _build_equity_labels_js(curve)
        assert '"2026-01-01"' in result
        assert '"2026-01-02"' in result

    def test_equity_values(self):
        curve = [
            {"timestamp": "2026-01-01", "total_value": 10000},
            {"timestamp": "2026-01-02", "total_value": 10100},
        ]
        result = _build_equity_values_js(curve)
        assert "10000" in result
        assert "10100" in result


class TestTradeMarkers:
    def test_no_trades_or_no_curve(self):
        assert _build_trade_markers_js([], []) == ""
        assert _build_trade_markers_js([{"side": "BUY"}], []) == ""
        assert _build_trade_markers_js([], [{"timestamp": "t1"}]) == ""

    def test_buy_marker(self):
        trades = [{"timestamp": "2026-01-01", "side": "BUY", "symbol": "BTC"}]
        curve = [{"timestamp": "2026-01-01", "total_value": 10000}]
        result = _build_trade_markers_js(trades, curve)
        assert "BUY" in result
        assert "#22c55e" in result
        assert "10000" in result

    def test_sell_marker(self):
        trades = [{"timestamp": "2026-01-01", "side": "SELL", "symbol": "BTC"}]
        curve = [{"timestamp": "2026-01-01", "total_value": 10500}]
        result = _build_trade_markers_js(trades, curve)
        assert "SELL" in result
        assert "#ef4444" in result

    def test_multiple_markers(self):
        trades = [
            {"timestamp": "2026-01-01", "side": "BUY", "symbol": "BTC"},
            {"timestamp": "2026-01-02", "side": "SELL", "symbol": "BTC"},
        ]
        curve = [
            {"timestamp": "2026-01-01", "total_value": 10000},
            {"timestamp": "2026-01-02", "total_value": 10500},
        ]
        result = _build_trade_markers_js(trades, curve)
        assert "BUY" in result
        assert "SELL" in result


class TestFindClosestIndex:
    def test_exact_match(self):
        assert _find_closest_index("b", ["a", "b", "c"]) == 1

    def test_closest_before(self):
        assert _find_closest_index("b5", ["a", "b", "c"]) == 1

    def test_empty_list(self):
        assert _find_closest_index("a", []) is None

    def test_empty_target(self):
        assert _find_closest_index("", ["a"]) is None

    def test_before_all(self):
        # Target is before all timestamps — returns 0
        assert _find_closest_index("0", ["a", "b", "c"]) == 0


class TestStrategiesListHTML:
    def test_empty_stats(self):
        result = _build_strategies_list_html({})
        assert "No strategy data" in result

    def test_active_strategy(self):
        stats = {"rsi": {"total_pnl": 100, "win_rate": 60, "active": True}}
        result = _build_strategies_list_html(stats)
        assert "rsi" in result
        assert "Enabled" in result
        assert "60" in result

    def test_inactive_strategy(self):
        stats = {"macd": {"total_pnl": -50, "win_rate": 35, "active": False}}
        result = _build_strategies_list_html(stats)
        assert "macd" in result
        assert "Disabled" in result

    def test_html_escaping(self):
        """Strategy names with special characters are escaped."""
        stats = {"<script>": {"total_pnl": 0, "win_rate": 0, "active": True}}
        result = _build_strategies_list_html(stats)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


# --- State Management Tests ---


class TestNewStateFields:
    def test_equity_curve_in_state(self):
        state = get_state()
        assert "equity_curve" in state
        assert state["equity_curve"] == []

    def test_open_positions_in_state(self):
        state = get_state()
        assert "open_positions" in state
        assert state["open_positions"] == []

    def test_regime_in_state(self):
        state = get_state()
        assert "regime" in state
        assert state["regime"] is None

    def test_update_equity_curve(self):
        update_state(equity_curve=[
            {"timestamp": "t1", "total_value": 10000}
        ])
        state = get_state()
        assert len(state["equity_curve"]) == 1
        assert state["equity_curve"][0]["total_value"] == 10000

    def test_update_open_positions(self):
        update_state(open_positions=[
            {"symbol": "BTC/USDT", "entry_price": 50000}
        ])
        state = get_state()
        assert len(state["open_positions"]) == 1

    def test_update_regime(self):
        update_state(regime="RANGING")
        state = get_state()
        assert state["regime"] == "RANGING"


# --- DataStore Portfolio Snapshots Tests ---


class TestDataStorePortfolioSnapshots:
    @pytest.mark.asyncio
    async def test_get_portfolio_snapshots_empty(self):
        """Get snapshots from empty DB returns empty list."""
        from bot.data.store import DataStore

        store = DataStore(database_url="sqlite+aiosqlite:///:memory:")
        await store.initialize()
        try:
            snapshots = await store.get_portfolio_snapshots()
            assert snapshots == []
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_get_portfolio_snapshots_with_data(self):
        """Get snapshots returns time-series data."""
        from bot.data.store import DataStore

        store = DataStore(database_url="sqlite+aiosqlite:///:memory:")
        await store.initialize()
        try:
            await store.save_portfolio_snapshot(total_value=10000.0)
            await store.save_portfolio_snapshot(total_value=10100.0)
            await store.save_portfolio_snapshot(total_value=10200.0)

            snapshots = await store.get_portfolio_snapshots()
            assert len(snapshots) == 3
            assert snapshots[0]["total_value"] == 10000.0
            assert snapshots[2]["total_value"] == 10200.0
            assert "timestamp" in snapshots[0]
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_get_portfolio_snapshots_with_limit(self):
        """Snapshots respect limit parameter."""
        from bot.data.store import DataStore

        store = DataStore(database_url="sqlite+aiosqlite:///:memory:")
        await store.initialize()
        try:
            for i in range(5):
                await store.save_portfolio_snapshot(total_value=10000.0 + i * 100)

            snapshots = await store.get_portfolio_snapshots(limit=3)
            # Returns last 3 due to DESC order + reversed
            assert len(snapshots) == 3
        finally:
            await store.close()


# --- Integration: set_strategy_registry ---


class TestSetStrategyRegistry:
    def test_set_and_use_registry(self):
        mock_reg = MagicMock()
        set_strategy_registry(mock_reg)
        # The global is set — toggle endpoint will use it
        from bot.dashboard.app import _strategy_registry
        assert _strategy_registry is mock_reg

    def test_set_none(self):
        set_strategy_registry(None)
        from bot.dashboard.app import _strategy_registry
        assert _strategy_registry is None
