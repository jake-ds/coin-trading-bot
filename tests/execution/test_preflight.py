"""Tests for V4-011: Live trading pre-flight checks and safety gates."""

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from bot.dashboard import auth as auth_module
from bot.dashboard.app import app, set_settings, update_state
from bot.execution.preflight import (
    CheckResult,
    CheckStatus,
    PreFlightChecker,
    PreFlightResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_settings(
    stop_loss_pct=3.0,
    daily_loss_limit_pct=5.0,
    dashboard_password="changeme",
    **kwargs,
):
    """Create a mock settings object for pre-flight tests."""
    return SimpleNamespace(
        stop_loss_pct=stop_loss_pct,
        daily_loss_limit_pct=daily_loss_limit_pct,
        dashboard_password=dashboard_password,
        dashboard_username="admin",
        jwt_secret="",
        **kwargs,
    )


@pytest.fixture
def checker():
    """Create a default PreFlightChecker."""
    return PreFlightChecker(min_balance_usd=100.0)


@pytest.fixture
def mock_exchange():
    """Create a mock exchange adapter with valid responses."""
    exchange = AsyncMock()
    exchange.name = "binance"
    exchange.get_balance = AsyncMock(return_value={"USDT": 500.0, "BTC": 0.01})
    exchange.get_ticker = AsyncMock(return_value={"last": 50000.0, "bid": 49999.0, "ask": 50001.0})
    return exchange


@pytest.fixture
def settings():
    """Create default settings for pre-flight tests."""
    return _make_settings()


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
        reconciliation={},
        preflight={},
    )
    auth_module.clear_blacklist()
    set_settings(None)
    yield
    set_settings(None)
    auth_module.clear_blacklist()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# CheckResult / PreFlightResult model tests
# ---------------------------------------------------------------------------


class TestCheckResult:
    def test_to_dict(self):
        result = CheckResult(
            name="test_check",
            status=CheckStatus.PASS,
            message="All good",
            details={"key": "value"},
        )
        d = result.to_dict()
        assert d["name"] == "test_check"
        assert d["status"] == "PASS"
        assert d["message"] == "All good"
        assert d["details"] == {"key": "value"}

    def test_default_details(self):
        result = CheckResult(name="x", status=CheckStatus.FAIL, message="bad")
        assert result.details == {}


class TestPreFlightResult:
    def test_empty_result(self):
        result = PreFlightResult()
        assert result.overall == CheckStatus.PASS
        assert not result.has_failures
        assert not result.has_warnings
        assert result.checks == []

    def test_has_failures(self):
        result = PreFlightResult(checks=[
            CheckResult(name="a", status=CheckStatus.PASS, message="ok"),
            CheckResult(name="b", status=CheckStatus.FAIL, message="bad"),
        ])
        assert result.has_failures
        assert not result.has_warnings

    def test_has_warnings(self):
        result = PreFlightResult(checks=[
            CheckResult(name="a", status=CheckStatus.PASS, message="ok"),
            CheckResult(name="b", status=CheckStatus.WARN, message="caution"),
        ])
        assert not result.has_failures
        assert result.has_warnings

    def test_to_dict(self):
        result = PreFlightResult(
            overall=CheckStatus.FAIL,
            checks=[
                CheckResult(name="a", status=CheckStatus.FAIL, message="bad"),
            ],
        )
        d = result.to_dict()
        assert d["overall"] == "FAIL"
        assert d["has_failures"] is True
        assert len(d["checks"]) == 1

    def test_format_summary(self):
        result = PreFlightResult(
            overall=CheckStatus.PASS,
            checks=[
                CheckResult(name="check1", status=CheckStatus.PASS, message="ok"),
                CheckResult(name="check2", status=CheckStatus.WARN, message="caution"),
            ],
        )
        summary = result.format_summary()
        assert "PASS" in summary
        assert "check1" in summary
        assert "check2" in summary
        assert "WARN" in summary


# ---------------------------------------------------------------------------
# Individual check tests
# ---------------------------------------------------------------------------


class TestApiKeyCheck:
    @pytest.mark.asyncio
    async def test_pass_with_valid_exchange(self, checker, mock_exchange):
        result = await checker._check_api_keys([mock_exchange])
        assert result.status == CheckStatus.PASS
        assert "binance" in result.message

    @pytest.mark.asyncio
    async def test_fail_no_exchanges(self, checker):
        result = await checker._check_api_keys(None)
        assert result.status == CheckStatus.FAIL
        assert "No exchange" in result.message

    @pytest.mark.asyncio
    async def test_fail_empty_exchanges(self, checker):
        result = await checker._check_api_keys([])
        assert result.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_fail_connection_error(self, checker):
        exchange = AsyncMock()
        exchange.name = "binance"
        exchange.get_balance = AsyncMock(side_effect=Exception("Connection refused"))
        result = await checker._check_api_keys([exchange])
        assert result.status == CheckStatus.FAIL
        assert "Connection refused" in result.message


class TestBalanceCheck:
    @pytest.mark.asyncio
    async def test_pass_sufficient_balance(self, checker, mock_exchange):
        result = await checker._check_balance([mock_exchange])
        assert result.status == CheckStatus.PASS
        assert "500.00" in result.message

    @pytest.mark.asyncio
    async def test_fail_insufficient_balance(self, checker):
        exchange = AsyncMock()
        exchange.name = "binance"
        exchange.get_balance = AsyncMock(return_value={"USDT": 50.0})
        result = await checker._check_balance([exchange])
        assert result.status == CheckStatus.FAIL
        assert "Insufficient" in result.message

    @pytest.mark.asyncio
    async def test_fail_no_exchanges(self, checker):
        result = await checker._check_balance(None)
        assert result.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_multiple_usd_currencies(self, checker):
        exchange = AsyncMock()
        exchange.name = "binance"
        exchange.get_balance = AsyncMock(return_value={"USDT": 60.0, "USDC": 50.0})
        result = await checker._check_balance([exchange])
        assert result.status == CheckStatus.PASS
        assert result.details["balance_usd"] == 110.0

    @pytest.mark.asyncio
    async def test_non_usd_currency_ignored(self, checker):
        exchange = AsyncMock()
        exchange.name = "binance"
        exchange.get_balance = AsyncMock(return_value={"BTC": 10.0, "ETH": 50.0})
        result = await checker._check_balance([exchange])
        assert result.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_balance_error_skips_exchange(self, checker, mock_exchange):
        bad_exchange = AsyncMock()
        bad_exchange.name = "bad"
        bad_exchange.get_balance = AsyncMock(side_effect=Exception("API error"))
        # First exchange fails, second passes
        result = await checker._check_balance([bad_exchange, mock_exchange])
        assert result.status == CheckStatus.PASS


class TestSymbolCheck:
    @pytest.mark.asyncio
    async def test_pass_all_symbols_available(self, checker, mock_exchange):
        result = await checker._check_symbols(
            [mock_exchange], ["BTC/USDT", "ETH/USDT"]
        )
        assert result.status == CheckStatus.PASS
        assert "2 symbols" in result.message

    @pytest.mark.asyncio
    async def test_fail_missing_symbol(self, checker):
        exchange = AsyncMock()
        exchange.name = "binance"
        exchange.get_ticker = AsyncMock(side_effect=Exception("Symbol not found"))
        result = await checker._check_symbols(
            [exchange], ["INVALID/USDT"]
        )
        assert result.status == CheckStatus.FAIL
        assert "INVALID/USDT" in result.message

    @pytest.mark.asyncio
    async def test_fail_no_exchanges(self, checker):
        result = await checker._check_symbols(None, ["BTC/USDT"])
        assert result.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_warn_no_symbols(self, checker, mock_exchange):
        result = await checker._check_symbols([mock_exchange], [])
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_partial_availability(self, checker):
        exchange = AsyncMock()
        exchange.name = "binance"

        async def mock_ticker(symbol):
            if symbol == "BTC/USDT":
                return {"last": 50000}
            raise Exception("Not found")

        exchange.get_ticker = mock_ticker
        result = await checker._check_symbols(
            [exchange], ["BTC/USDT", "INVALID/USDT"]
        )
        assert result.status == CheckStatus.FAIL
        assert "INVALID/USDT" in result.details["missing"]
        assert "BTC/USDT" in result.details["verified"]


class TestRateLimitCheck:
    def test_pass_enabled(self, checker):
        result = checker._check_rate_limit(True)
        assert result.status == CheckStatus.PASS

    def test_warn_disabled(self, checker):
        result = checker._check_rate_limit(False)
        assert result.status == CheckStatus.WARN
        assert "disabled" in result.message


class TestStopLossCheck:
    def test_pass_configured(self, checker):
        settings = _make_settings(stop_loss_pct=3.0)
        result = checker._check_stop_loss(settings)
        assert result.status == CheckStatus.PASS
        assert "3.0%" in result.message

    def test_fail_zero(self, checker):
        settings = _make_settings(stop_loss_pct=0.0)
        result = checker._check_stop_loss(settings)
        assert result.status == CheckStatus.FAIL
        assert "not configured" in result.message


class TestDailyLossLimitCheck:
    def test_pass_configured(self, checker):
        settings = _make_settings(daily_loss_limit_pct=5.0)
        result = checker._check_daily_loss_limit(settings)
        assert result.status == CheckStatus.PASS
        assert "5.0%" in result.message

    def test_fail_zero(self, checker):
        settings = _make_settings(daily_loss_limit_pct=0.0)
        result = checker._check_daily_loss_limit(settings)
        assert result.status == CheckStatus.FAIL
        assert "not configured" in result.message


class TestPasswordCheck:
    def test_pass_changed(self, checker):
        settings = _make_settings(dashboard_password="securepassword123")
        result = checker._check_password_changed(settings)
        assert result.status == CheckStatus.PASS

    def test_warn_default(self, checker):
        settings = _make_settings(dashboard_password="changeme")
        result = checker._check_password_changed(settings)
        assert result.status == CheckStatus.WARN
        assert "changeme" in result.message


class TestValidationReportCheck:
    def test_warn_no_directory(self):
        checker = PreFlightChecker(validation_report_dir="/nonexistent/path")
        result = checker._check_validation_report()
        assert result.status == CheckStatus.WARN
        assert "No validation report directory" in result.message

    def test_warn_no_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = PreFlightChecker(validation_report_dir=tmpdir)
            result = checker._check_validation_report()
            assert result.status == CheckStatus.WARN
            assert "No validation reports found" in result.message

    def test_pass_go_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = {
                "recommendation": "GO",
                "total_return_pct": 5.0,
                "win_rate_pct": 60.0,
                "sharpe_ratio": 1.5,
            }
            report_path = os.path.join(
                tmpdir, "validation_report_20260222_120000.json"
            )
            with open(report_path, "w") as f:
                json.dump(report, f)

            checker = PreFlightChecker(validation_report_dir=tmpdir)
            result = checker._check_validation_report()
            assert result.status == CheckStatus.PASS
            assert "GO" in result.message
            assert result.details["recommendation"] == "GO"

    def test_warn_nogo_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = {"recommendation": "NO-GO"}
            report_path = os.path.join(
                tmpdir, "validation_report_20260222_120000.json"
            )
            with open(report_path, "w") as f:
                json.dump(report, f)

            checker = PreFlightChecker(validation_report_dir=tmpdir)
            result = checker._check_validation_report()
            assert result.status == CheckStatus.WARN
            assert "NO-GO" in result.message

    def test_warn_corrupt_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(
                tmpdir, "validation_report_20260222_120000.json"
            )
            with open(report_path, "w") as f:
                f.write("not valid json{{{")

            checker = PreFlightChecker(validation_report_dir=tmpdir)
            result = checker._check_validation_report()
            assert result.status == CheckStatus.WARN
            assert "Could not read" in result.message

    def test_picks_latest_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Old report: NO-GO
            old_path = os.path.join(
                tmpdir, "validation_report_20260220_120000.json"
            )
            with open(old_path, "w") as f:
                json.dump({"recommendation": "NO-GO"}, f)

            # New report: GO
            new_path = os.path.join(
                tmpdir, "validation_report_20260222_120000.json"
            )
            with open(new_path, "w") as f:
                json.dump({"recommendation": "GO"}, f)

            checker = PreFlightChecker(validation_report_dir=tmpdir)
            result = checker._check_validation_report()
            assert result.status == CheckStatus.PASS
            assert "GO" in result.message


# ---------------------------------------------------------------------------
# Full run_all_checks tests
# ---------------------------------------------------------------------------


class TestRunAllChecks:
    @pytest.mark.asyncio
    async def test_all_pass(self, mock_exchange):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(
                tmpdir, "validation_report_20260222_120000.json"
            )
            with open(report_path, "w") as f:
                json.dump({"recommendation": "GO"}, f)

            checker = PreFlightChecker(
                min_balance_usd=100.0,
                validation_report_dir=tmpdir,
            )
            settings = _make_settings(
                stop_loss_pct=3.0,
                daily_loss_limit_pct=5.0,
                dashboard_password="securepassword",
            )
            result = await checker.run_all_checks(
                settings=settings,
                exchanges=[mock_exchange],
                symbols=["BTC/USDT"],
                rate_limit_enabled=True,
            )
            assert result.overall == CheckStatus.PASS
            assert not result.has_failures
            assert not result.has_warnings
            assert len(result.checks) == 8

    @pytest.mark.asyncio
    async def test_fail_prevents_overall_pass(self, mock_exchange):
        checker = PreFlightChecker()
        settings = _make_settings(stop_loss_pct=0.0)  # Will fail
        result = await checker.run_all_checks(
            settings=settings,
            exchanges=[mock_exchange],
            symbols=["BTC/USDT"],
        )
        assert result.overall == CheckStatus.FAIL
        assert result.has_failures

    @pytest.mark.asyncio
    async def test_warn_allows_startup(self, mock_exchange):
        checker = PreFlightChecker()
        settings = _make_settings(
            stop_loss_pct=3.0,
            daily_loss_limit_pct=5.0,
            dashboard_password="changeme",  # Will warn
        )
        result = await checker.run_all_checks(
            settings=settings,
            exchanges=[mock_exchange],
            symbols=["BTC/USDT"],
            rate_limit_enabled=True,
        )
        # Password and validation report checks will be WARN
        assert result.overall == CheckStatus.WARN
        assert result.has_warnings
        assert not result.has_failures

    @pytest.mark.asyncio
    async def test_last_result_stored(self, mock_exchange):
        checker = PreFlightChecker()
        settings = _make_settings()
        assert checker.last_result is None
        await checker.run_all_checks(
            settings=settings,
            exchanges=[mock_exchange],
            symbols=["BTC/USDT"],
        )
        assert checker.last_result is not None
        assert isinstance(checker.last_result, PreFlightResult)

    @pytest.mark.asyncio
    async def test_no_exchanges(self):
        checker = PreFlightChecker()
        settings = _make_settings()
        result = await checker.run_all_checks(
            settings=settings,
            exchanges=None,
            symbols=["BTC/USDT"],
        )
        assert result.overall == CheckStatus.FAIL
        # api_key, balance, symbol checks should all fail
        fail_names = [c.name for c in result.checks if c.status == CheckStatus.FAIL]
        assert "api_key_validity" in fail_names
        assert "sufficient_balance" in fail_names
        assert "symbol_availability" in fail_names

    @pytest.mark.asyncio
    async def test_default_symbols_empty(self, mock_exchange):
        checker = PreFlightChecker()
        settings = _make_settings()
        result = await checker.run_all_checks(
            settings=settings,
            exchanges=[mock_exchange],
            symbols=None,  # Will default to []
        )
        # Should have a WARN for no symbols
        symbol_check = next(
            c for c in result.checks if c.name == "symbol_availability"
        )
        assert symbol_check.status == CheckStatus.WARN


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestPreflightEndpoint:
    @pytest.mark.asyncio
    async def test_empty_preflight(self, client):
        resp = await client.get("/api/preflight")
        assert resp.status_code == 200
        data = resp.json()
        assert "preflight" in data
        assert data["preflight"] == {}

    @pytest.mark.asyncio
    async def test_preflight_with_data(self, client):
        update_state(preflight={
            "overall": "PASS",
            "checks": [
                {"name": "api_key_validity", "status": "PASS", "message": "ok", "details": {}},
            ],
            "has_failures": False,
            "has_warnings": False,
        })
        resp = await client.get("/api/preflight")
        assert resp.status_code == 200
        data = resp.json()
        assert data["preflight"]["overall"] == "PASS"
        assert len(data["preflight"]["checks"]) == 1

    @pytest.mark.asyncio
    async def test_preflight_with_failures(self, client):
        update_state(preflight={
            "overall": "FAIL",
            "checks": [
                {"name": "stop_loss", "status": "FAIL", "message": "not configured", "details": {}},
            ],
            "has_failures": True,
            "has_warnings": False,
        })
        resp = await client.get("/api/preflight")
        assert resp.status_code == 200
        data = resp.json()
        assert data["preflight"]["overall"] == "FAIL"
        assert data["preflight"]["has_failures"] is True


# ---------------------------------------------------------------------------
# CheckStatus enum tests
# ---------------------------------------------------------------------------


class TestCheckStatus:
    def test_values(self):
        assert CheckStatus.PASS.value == "PASS"
        assert CheckStatus.WARN.value == "WARN"
        assert CheckStatus.FAIL.value == "FAIL"

    def test_string_enum(self):
        assert str(CheckStatus.PASS) == "CheckStatus.PASS"
        assert CheckStatus.PASS == "PASS"
