"""Tests for engine API endpoints — onchain_trader descriptions and params."""

import pytest
from httpx import ASGITransport, AsyncClient

from bot.config import ENGINE_DESCRIPTIONS, load_settings
from bot.dashboard.app import app, set_engine_manager, set_settings


class MockEngineManager:
    """Minimal EngineManager mock for API tests."""

    def __init__(self):
        self._status = {
            "onchain_trader": {
                "name": "onchain_trader",
                "description": "On-chain data autonomous trader",
                "status": "stopped",
                "cycle_count": 0,
                "total_pnl": 0.0,
                "allocated_capital": 1000.0,
                "position_count": 0,
                "max_positions": 5,
                "loop_interval": 300.0,
                "error": None,
            },
        }

    def get_status(self):
        return {k: dict(v) for k, v in self._status.items()}

    def get_engine(self, name):
        return None


@pytest.fixture
def mock_manager():
    return MockEngineManager()


@pytest.fixture
def settings():
    """Load settings with auth disabled (changeme password)."""
    s = load_settings()
    s.dashboard_password = "changeme"
    return s


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
        assert "onchain_trader" in data
        desc = ENGINE_DESCRIPTIONS.get("onchain_trader", {})
        assert data["onchain_trader"]["role_ko"] == desc["role_ko"]

    @pytest.mark.asyncio
    async def test_engines_include_symbols(self, client):
        resp = await client.get("/api/engines")
        data = resp.json()
        symbols = data["onchain_trader"]["symbols"]
        assert "BTC/USDT" in symbols


class TestEngineParamsEndpoint:
    @pytest.mark.asyncio
    async def test_params_returns_config(self, client):
        resp = await client.get("/api/engines/onchain_trader/params")
        assert resp.status_code == 200
        data = resp.json()
        assert data["engine"] == "onchain_trader"
        assert "params" in data

    @pytest.mark.asyncio
    async def test_params_unknown_engine(self, client):
        resp = await client.get("/api/engines/nonexistent/params")
        assert resp.status_code == 404


class TestEngineDescriptionsMetadata:
    def test_onchain_trader_has_description(self):
        assert "onchain_trader" in ENGINE_DESCRIPTIONS
        desc = ENGINE_DESCRIPTIONS["onchain_trader"]
        assert "role_ko" in desc
        assert "role_en" in desc
        assert "description_ko" in desc
        assert "key_params" in desc

    def test_default_onchain_symbols(self):
        s = load_settings()
        assert len(s.onchain_symbols) >= 3
        assert "BTC/USDT" in s.onchain_symbols
