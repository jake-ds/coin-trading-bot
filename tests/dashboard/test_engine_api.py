"""Tests for engine API endpoints — V5-003 descriptions and params."""

import pytest
from httpx import ASGITransport, AsyncClient

from bot.config import ENGINE_DESCRIPTIONS, load_settings
from bot.dashboard.app import app, set_engine_manager, set_settings


class MockEngineManager:
    """Minimal EngineManager mock for API tests."""

    def __init__(self):
        self._status = {
            "funding_rate_arb": {
                "name": "funding_rate_arb",
                "description": "Delta-neutral funding rate arbitrage",
                "status": "stopped",
                "cycle_count": 0,
                "total_pnl": 0.0,
                "allocated_capital": 3000.0,
                "position_count": 0,
                "max_positions": 3,
                "loop_interval": 300.0,
                "error": None,
            },
            "grid_trading": {
                "name": "grid_trading",
                "description": "Automated grid trading",
                "status": "stopped",
                "cycle_count": 0,
                "total_pnl": 0.0,
                "allocated_capital": 2500.0,
                "position_count": 0,
                "max_positions": 20,
                "loop_interval": 30.0,
                "error": None,
            },
        }

    def get_status(self):
        # Return copies to avoid mutation issues
        return {k: dict(v) for k, v in self._status.items()}


@pytest.fixture
def mock_manager():
    return MockEngineManager()


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture(autouse=True)
def setup_engine_manager(mock_manager, settings):
    set_engine_manager(mock_manager)
    set_settings(settings)
    yield
    set_engine_manager(None)
    set_settings(None)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestListEnginesWithDescriptions:
    @pytest.mark.asyncio
    async def test_engines_include_role_ko(self, client):
        resp = await client.get("/api/engines")
        assert resp.status_code == 200
        data = resp.json()
        assert "funding_rate_arb" in data
        assert data["funding_rate_arb"]["role_ko"] == "델타중립 펀딩비 차익거래"

    @pytest.mark.asyncio
    async def test_engines_include_symbols(self, client):
        resp = await client.get("/api/engines")
        data = resp.json()
        symbols = data["funding_rate_arb"]["symbols"]
        assert "BTC/USDT" in symbols
        assert "SOL/USDT" in symbols

    @pytest.mark.asyncio
    async def test_engines_include_description_ko(self, client):
        resp = await client.get("/api/engines")
        data = resp.json()
        assert data["grid_trading"]["description_ko"] != ""
        assert data["grid_trading"]["role_en"] == "Automated grid trading"


class TestEngineParamsEndpoint:
    @pytest.mark.asyncio
    async def test_params_returns_config(self, client):
        resp = await client.get("/api/engines/funding_rate_arb/params")
        assert resp.status_code == 200
        data = resp.json()
        assert data["engine"] == "funding_rate_arb"
        assert "params" in data
        assert "funding_arb_min_rate" in data["params"]
        assert "funding_arb_symbols" in data["params"]

    @pytest.mark.asyncio
    async def test_params_returns_description(self, client):
        resp = await client.get("/api/engines/grid_trading/params")
        data = resp.json()
        assert data["description"]["role_ko"] == "그리드 자동매매"

    @pytest.mark.asyncio
    async def test_params_unknown_engine(self, client):
        resp = await client.get("/api/engines/nonexistent/params")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_params_stat_arb(self, client):
        resp = await client.get("/api/engines/stat_arb/params")
        assert resp.status_code == 200
        data = resp.json()
        assert "stat_arb_entry_zscore" in data["params"]
        assert "stat_arb_pairs" in data["params"]
        # Verify expanded pairs
        pairs = data["params"]["stat_arb_pairs"]
        assert len(pairs) == 2  # BTC/ETH and SOL/ETH

    @pytest.mark.asyncio
    async def test_params_cross_arb(self, client):
        resp = await client.get("/api/engines/cross_exchange_arb/params")
        assert resp.status_code == 200
        data = resp.json()
        assert "cross_arb_min_spread_pct" in data["params"]
        symbols = data["params"]["cross_arb_symbols"]
        assert "SOL/USDT" in symbols


class TestEngineDescriptionsMetadata:
    def test_all_engines_have_descriptions(self):
        expected = ["funding_rate_arb", "grid_trading", "cross_exchange_arb", "stat_arb"]
        for name in expected:
            assert name in ENGINE_DESCRIPTIONS
            desc = ENGINE_DESCRIPTIONS[name]
            assert "role_ko" in desc
            assert "role_en" in desc
            assert "description_ko" in desc
            assert "key_params" in desc

    def test_default_symbol_expansion(self):
        s = load_settings()
        assert len(s.funding_arb_symbols) == 5
        assert "DOGE/USDT" in s.funding_arb_symbols
        assert len(s.grid_symbols) == 3
        assert "SOL/USDT" in s.grid_symbols
        assert len(s.cross_arb_symbols) == 3
        assert len(s.stat_arb_pairs) == 2
