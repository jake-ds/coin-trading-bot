"""Tests for V4-007: JWT authentication with login page."""

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from bot.dashboard import auth as auth_module
from bot.dashboard.app import app, set_settings, update_state


def _make_settings(password="changeme", username="admin", jwt_secret="test-secret"):
    """Create a mock settings object."""
    return SimpleNamespace(
        dashboard_username=username,
        dashboard_password=password,
        jwt_secret=jwt_secret,
    )


@pytest.fixture(autouse=True)
def reset_state():
    """Reset bot state and auth state before each test."""
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
    auth_module.clear_blacklist()
    # Reset settings to None (auth disabled by default)
    set_settings(None)
    yield
    set_settings(None)
    auth_module.clear_blacklist()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def auth_settings():
    """Settings with auth enabled (non-default password)."""
    return _make_settings(password="secure-password-123")


@pytest.fixture
def dev_settings():
    """Settings with auth disabled (default password)."""
    return _make_settings(password="changeme")


# ---------------------------------------------------------------------------
# Auth module unit tests
# ---------------------------------------------------------------------------


class TestAuthModule:
    def test_is_auth_enabled_default_password(self, dev_settings):
        assert auth_module.is_auth_enabled(dev_settings) is False

    def test_is_auth_enabled_custom_password(self, auth_settings):
        assert auth_module.is_auth_enabled(auth_settings) is True

    def test_verify_credentials_correct(self, auth_settings):
        assert auth_module.verify_credentials(auth_settings, "admin", "secure-password-123") is True

    def test_verify_credentials_wrong_password(self, auth_settings):
        assert auth_module.verify_credentials(auth_settings, "admin", "wrong") is False

    def test_verify_credentials_wrong_username(self, auth_settings):
        result = auth_module.verify_credentials(
            auth_settings, "hacker", "secure-password-123"
        )
        assert result is False

    def test_create_access_token(self, auth_settings):
        token = auth_module.create_access_token(auth_settings, "admin")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_validate_access_token(self, auth_settings):
        token = auth_module.create_access_token(auth_settings, "admin")
        username = auth_module.validate_access_token(auth_settings, token)
        assert username == "admin"

    def test_validate_access_token_invalid(self, auth_settings):
        result = auth_module.validate_access_token(auth_settings, "invalid-token")
        assert result is None

    def test_validate_access_token_rejects_refresh(self, auth_settings):
        """Access token validator should reject refresh tokens."""
        refresh = auth_module.create_refresh_token(auth_settings, "admin")
        result = auth_module.validate_access_token(auth_settings, refresh)
        assert result is None

    def test_create_refresh_token(self, auth_settings):
        token = auth_module.create_refresh_token(auth_settings, "admin")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_validate_refresh_token(self, auth_settings):
        token = auth_module.create_refresh_token(auth_settings, "admin")
        username = auth_module.validate_refresh_token(auth_settings, token)
        assert username == "admin"

    def test_validate_refresh_token_rejects_access(self, auth_settings):
        """Refresh token validator should reject access tokens."""
        access = auth_module.create_access_token(auth_settings, "admin")
        result = auth_module.validate_refresh_token(auth_settings, access)
        assert result is None

    def test_blacklist_refresh_token(self, auth_settings):
        token = auth_module.create_refresh_token(auth_settings, "admin")
        # Initially valid
        assert auth_module.validate_refresh_token(auth_settings, token) == "admin"
        # Blacklist it
        assert auth_module.blacklist_refresh_token(auth_settings, token) is True
        # Now invalid
        assert auth_module.validate_refresh_token(auth_settings, token) is None

    def test_blacklist_invalid_token(self, auth_settings):
        result = auth_module.blacklist_refresh_token(auth_settings, "invalid")
        assert result is False

    def test_clear_blacklist(self, auth_settings):
        token = auth_module.create_refresh_token(auth_settings, "admin")
        auth_module.blacklist_refresh_token(auth_settings, token)
        assert auth_module.validate_refresh_token(auth_settings, token) is None
        auth_module.clear_blacklist()
        assert auth_module.validate_refresh_token(auth_settings, token) == "admin"

    def test_wrong_secret_rejects_token(self):
        settings_a = _make_settings(password="pass", jwt_secret="secret-a")
        settings_b = _make_settings(password="pass", jwt_secret="secret-b")
        token = auth_module.create_access_token(settings_a, "admin")
        # Validate with different secret should fail
        assert auth_module.validate_access_token(settings_b, token) is None


# ---------------------------------------------------------------------------
# Auth API endpoint tests
# ---------------------------------------------------------------------------


class TestAuthEndpoints:
    @pytest.mark.asyncio
    async def test_auth_status_disabled(self, client):
        """Auth status returns disabled when no settings or default password."""
        resp = await client.get("/api/auth/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["auth_enabled"] is False
        assert data["dev_mode"] is True

    @pytest.mark.asyncio
    async def test_auth_status_enabled(self, client, auth_settings):
        set_settings(auth_settings)
        resp = await client.get("/api/auth/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["auth_enabled"] is True
        assert data["dev_mode"] is False

    @pytest.mark.asyncio
    async def test_login_success(self, client, auth_settings):
        set_settings(auth_settings)
        resp = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "secure-password-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client, auth_settings):
        set_settings(auth_settings)
        resp = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "wrong"},
        )
        assert resp.status_code == 401
        assert "Invalid credentials" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_login_wrong_username(self, client, auth_settings):
        set_settings(auth_settings)
        resp = await client.post(
            "/api/auth/login",
            json={"username": "hacker", "password": "secure-password-123"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_login_disabled_auth(self, client, dev_settings):
        """Login returns 400 when auth is disabled."""
        set_settings(dev_settings)
        resp = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "changeme"},
        )
        assert resp.status_code == 400
        assert "disabled" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_refresh_success(self, client, auth_settings):
        set_settings(auth_settings)
        # Login first
        login_resp = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "secure-password-123"},
        )
        refresh_token = login_resp.json()["refresh_token"]

        # Refresh
        resp = await client.post(
            "/api/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data

    @pytest.mark.asyncio
    async def test_refresh_invalid_token(self, client, auth_settings):
        set_settings(auth_settings)
        resp = await client.post(
            "/api/auth/refresh",
            json={"refresh_token": "invalid-token"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_logout_invalidates_refresh(self, client, auth_settings):
        set_settings(auth_settings)
        # Login
        login_resp = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "secure-password-123"},
        )
        refresh_token = login_resp.json()["refresh_token"]

        # Logout
        resp = await client.post(
            "/api/auth/logout",
            json={"refresh_token": refresh_token},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        # Refresh should now fail
        resp = await client.post(
            "/api/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Protected route tests
# ---------------------------------------------------------------------------


class TestProtectedRoutes:
    @pytest.mark.asyncio
    async def test_api_accessible_when_auth_disabled(self, client):
        """All API routes accessible when auth is disabled (default)."""
        resp = await client.get("/api/status")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_api_accessible_with_dev_settings(self, client, dev_settings):
        """API routes accessible with default password (auth disabled)."""
        set_settings(dev_settings)
        resp = await client.get("/api/status")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_api_rejected_without_token(self, client, auth_settings):
        """API routes reject requests without token when auth is enabled."""
        set_settings(auth_settings)
        resp = await client.get("/api/status")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_api_rejected_with_invalid_token(self, client, auth_settings):
        set_settings(auth_settings)
        resp = await client.get(
            "/api/status",
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_api_accessible_with_valid_token(self, client, auth_settings):
        set_settings(auth_settings)
        # Login
        login_resp = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "secure-password-123"},
        )
        token = login_resp.json()["access_token"]

        # Access protected route
        resp = await client.get(
            "/api/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_health_accessible_without_auth(self, client, auth_settings):
        """Health endpoint at root should always be accessible."""
        set_settings(auth_settings)
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_auth_endpoints_accessible_without_token(self, client, auth_settings):
        """Auth endpoints should not require authentication."""
        set_settings(auth_settings)
        # auth/status should be accessible
        resp = await client.get("/api/auth/status")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_trades_protected(self, client, auth_settings):
        set_settings(auth_settings)
        resp = await client.get("/api/trades")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_portfolio_protected(self, client, auth_settings):
        set_settings(auth_settings)
        resp = await client.get("/api/portfolio")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_strategies_protected(self, client, auth_settings):
        set_settings(auth_settings)
        resp = await client.get("/api/strategies")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_analytics_protected(self, client, auth_settings):
        set_settings(auth_settings)
        resp = await client.get("/api/analytics")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_multiple_routes_with_token(self, client, auth_settings):
        """Verify multiple API routes work with valid token."""
        set_settings(auth_settings)
        login_resp = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "secure-password-123"},
        )
        token = login_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # All should return 200
        for path in ["/api/status", "/api/trades", "/api/portfolio", "/api/metrics",
                     "/api/strategies", "/api/analytics", "/api/regime"]:
            resp = await client.get(path, headers=headers)
            assert resp.status_code == 200, f"Failed for {path}: {resp.status_code}"

    @pytest.mark.asyncio
    async def test_toggle_protected(self, client, auth_settings):
        """POST endpoints also require auth."""
        set_settings(auth_settings)
        resp = await client.post("/api/strategies/test/toggle")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_custom_username(self, client):
        """Auth works with a custom username."""
        settings = _make_settings(
            password="mypass", username="trader", jwt_secret="test-secret"
        )
        set_settings(settings)
        # Login with custom username
        resp = await client.post(
            "/api/auth/login",
            json={"username": "trader", "password": "mypass"},
        )
        assert resp.status_code == 200
        token = resp.json()["access_token"]

        # Use token
        resp = await client.get(
            "/api/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
