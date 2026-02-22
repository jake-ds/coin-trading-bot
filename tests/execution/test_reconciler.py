"""Tests for V4-010: Position reconciliation."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from bot.dashboard.app import app, update_state
from bot.execution.reconciler import (
    DiscrepancyType,
    PositionDiscrepancy,
    PositionReconciler,
    ReconciliationResult,
    _base_currency,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reconciler():
    """Create a default PositionReconciler."""
    return PositionReconciler(tolerance_pct=1.0, auto_fix=False)


@pytest.fixture
def mock_exchange():
    """Create a mock exchange adapter."""
    exchange = AsyncMock()
    exchange.name = "binance"
    exchange.get_balance = AsyncMock(return_value={})
    return exchange


@pytest.fixture(autouse=True)
def reset_dashboard_state():
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
    )


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Unit tests: _base_currency helper
# ---------------------------------------------------------------------------


class TestBaseCurrency:
    def test_btc_usdt(self):
        assert _base_currency("BTC/USDT") == "BTC"

    def test_eth_usdt(self):
        assert _base_currency("ETH/USDT") == "ETH"

    def test_no_slash(self):
        assert _base_currency("BTCUSDT") == "BTCUSDT"

    def test_triple_slash(self):
        assert _base_currency("A/B/C") == "A"


# ---------------------------------------------------------------------------
# Unit tests: ReconciliationResult
# ---------------------------------------------------------------------------


class TestReconciliationResult:
    def test_empty_result_no_discrepancies(self):
        result = ReconciliationResult()
        assert not result.has_discrepancies
        assert result.total_discrepancies == 0

    def test_has_discrepancies_local_only(self):
        result = ReconciliationResult(
            local_only=[
                PositionDiscrepancy(
                    symbol="BTC/USDT",
                    discrepancy_type=DiscrepancyType.LOCAL_ONLY,
                    local_qty=1.0,
                    exchange_qty=0.0,
                )
            ]
        )
        assert result.has_discrepancies
        assert result.total_discrepancies == 1

    def test_has_discrepancies_exchange_only(self):
        result = ReconciliationResult(
            exchange_only=[
                PositionDiscrepancy(
                    symbol="ETH/USDT",
                    discrepancy_type=DiscrepancyType.EXCHANGE_ONLY,
                    local_qty=0.0,
                    exchange_qty=2.5,
                )
            ]
        )
        assert result.has_discrepancies
        assert result.total_discrepancies == 1

    def test_has_discrepancies_qty_mismatch(self):
        result = ReconciliationResult(
            qty_mismatch=[
                PositionDiscrepancy(
                    symbol="BTC/USDT",
                    discrepancy_type=DiscrepancyType.QTY_MISMATCH,
                    local_qty=1.0,
                    exchange_qty=0.8,
                )
            ]
        )
        assert result.has_discrepancies
        assert result.total_discrepancies == 1

    def test_to_dict_structure(self):
        result = ReconciliationResult(
            timestamp="2026-02-22T10:00:00Z",
            exchange_name="binance",
            matched=["BTC/USDT"],
            local_only=[
                PositionDiscrepancy(
                    symbol="ETH/USDT",
                    discrepancy_type=DiscrepancyType.LOCAL_ONLY,
                    local_qty=1.5,
                    exchange_qty=0.0,
                )
            ],
        )
        d = result.to_dict()
        assert d["timestamp"] == "2026-02-22T10:00:00Z"
        assert d["exchange_name"] == "binance"
        assert d["matched"] == ["BTC/USDT"]
        assert len(d["local_only"]) == 1
        assert d["local_only"][0]["symbol"] == "ETH/USDT"
        assert d["local_only"][0]["type"] == "local_only"
        assert d["has_discrepancies"] is True
        assert d["total_discrepancies"] == 1
        assert d["error"] is None

    def test_total_discrepancies_counts_all(self):
        result = ReconciliationResult(
            local_only=[
                PositionDiscrepancy("A/USDT", DiscrepancyType.LOCAL_ONLY, 1.0, 0.0),
            ],
            exchange_only=[
                PositionDiscrepancy("B/USDT", DiscrepancyType.EXCHANGE_ONLY, 0.0, 2.0),
            ],
            qty_mismatch=[
                PositionDiscrepancy("C/USDT", DiscrepancyType.QTY_MISMATCH, 3.0, 2.5),
            ],
        )
        assert result.total_discrepancies == 3


# ---------------------------------------------------------------------------
# Unit tests: PositionDiscrepancy
# ---------------------------------------------------------------------------


class TestPositionDiscrepancy:
    def test_to_dict(self):
        d = PositionDiscrepancy(
            symbol="BTC/USDT",
            discrepancy_type=DiscrepancyType.QTY_MISMATCH,
            local_qty=1.0,
            exchange_qty=0.95,
            details="mismatch",
        )
        result = d.to_dict()
        assert result["symbol"] == "BTC/USDT"
        assert result["type"] == "qty_mismatch"
        assert result["local_qty"] == 1.0
        assert result["exchange_qty"] == 0.95
        assert result["details"] == "mismatch"


# ---------------------------------------------------------------------------
# Unit tests: PositionReconciler.reconcile()
# ---------------------------------------------------------------------------


class TestReconcileMatching:
    """Test reconciliation when positions match."""

    @pytest.mark.asyncio
    async def test_no_positions_clean(self, reconciler, mock_exchange):
        """No local positions and no exchange positions = clean."""
        mock_exchange.get_balance.return_value = {"USDT": 10000.0}
        result = await reconciler.reconcile(mock_exchange, {})
        assert not result.has_discrepancies
        assert result.matched == []
        assert result.error is None

    @pytest.mark.asyncio
    async def test_matching_positions(self, reconciler, mock_exchange):
        """Matching local and exchange positions."""
        mock_exchange.get_balance.return_value = {
            "BTC": 0.5,
            "ETH": 2.0,
            "USDT": 5000.0,
        }
        local = {
            "BTC/USDT": {"quantity": 0.5, "entry_price": 50000},
            "ETH/USDT": {"quantity": 2.0, "entry_price": 3000},
        }
        result = await reconciler.reconcile(mock_exchange, local)
        assert not result.has_discrepancies
        assert "BTC/USDT" in result.matched
        assert "ETH/USDT" in result.matched

    @pytest.mark.asyncio
    async def test_within_tolerance(self, reconciler, mock_exchange):
        """Qty within tolerance is considered matching."""
        mock_exchange.get_balance.return_value = {
            "BTC": 0.504,  # 0.8% more than local 0.5 — within 1% tolerance
            "USDT": 5000.0,
        }
        local = {"BTC/USDT": {"quantity": 0.5, "entry_price": 50000}}
        result = await reconciler.reconcile(mock_exchange, local)
        assert "BTC/USDT" in result.matched
        assert not result.has_discrepancies


class TestReconcileDiscrepancies:
    """Test reconciliation detects discrepancies."""

    @pytest.mark.asyncio
    async def test_local_only_detected(self, reconciler, mock_exchange):
        """Local position not found on exchange."""
        mock_exchange.get_balance.return_value = {"USDT": 10000.0}
        local = {"BTC/USDT": {"quantity": 0.5, "entry_price": 50000}}
        result = await reconciler.reconcile(mock_exchange, local)
        assert result.has_discrepancies
        assert len(result.local_only) == 1
        assert result.local_only[0].symbol == "BTC/USDT"
        assert result.local_only[0].local_qty == 0.5
        assert result.local_only[0].exchange_qty == 0.0

    @pytest.mark.asyncio
    async def test_exchange_only_detected(self, reconciler, mock_exchange):
        """Exchange has position not tracked locally."""
        mock_exchange.get_balance.return_value = {
            "SOL": 10.0,
            "USDT": 5000.0,
        }
        local = {}
        result = await reconciler.reconcile(mock_exchange, local)
        assert result.has_discrepancies
        assert len(result.exchange_only) == 1
        assert result.exchange_only[0].exchange_qty == 10.0

    @pytest.mark.asyncio
    async def test_qty_mismatch_detected(self, reconciler, mock_exchange):
        """Quantity mismatch beyond tolerance."""
        mock_exchange.get_balance.return_value = {
            "BTC": 0.3,  # 40% less than local 0.5 — well beyond 1% tolerance
            "USDT": 5000.0,
        }
        local = {"BTC/USDT": {"quantity": 0.5, "entry_price": 50000}}
        result = await reconciler.reconcile(mock_exchange, local)
        assert result.has_discrepancies
        assert len(result.qty_mismatch) == 1
        assert result.qty_mismatch[0].local_qty == 0.5
        assert result.qty_mismatch[0].exchange_qty == 0.3

    @pytest.mark.asyncio
    async def test_multiple_discrepancies(self, reconciler, mock_exchange):
        """Multiple types of discrepancies in one check."""
        mock_exchange.get_balance.return_value = {
            "ETH": 5.0,  # local_only for BTC, qty_mismatch for ETH
            "SOL": 10.0,  # exchange_only
            "USDT": 5000.0,
        }
        local = {
            "BTC/USDT": {"quantity": 0.5, "entry_price": 50000},
            "ETH/USDT": {"quantity": 2.0, "entry_price": 3000},
        }
        result = await reconciler.reconcile(mock_exchange, local)
        assert result.has_discrepancies
        assert len(result.local_only) == 1  # BTC
        assert len(result.qty_mismatch) == 1  # ETH
        assert len(result.exchange_only) == 1  # SOL
        assert result.total_discrepancies == 3

    @pytest.mark.asyncio
    async def test_ignores_quote_currencies(self, reconciler, mock_exchange):
        """Quote currencies like USDT, USD, BUSD are not flagged as exchange_only."""
        mock_exchange.get_balance.return_value = {
            "USDT": 10000.0,
            "BUSD": 5000.0,
            "USDC": 3000.0,
        }
        result = await reconciler.reconcile(mock_exchange, {})
        assert not result.has_discrepancies
        assert len(result.exchange_only) == 0

    @pytest.mark.asyncio
    async def test_ignores_dust_balance(self, reconciler, mock_exchange):
        """Dust balances (< 1e-8) are ignored."""
        mock_exchange.get_balance.return_value = {
            "BTC": 0.000000001,  # dust
            "USDT": 10000.0,
        }
        result = await reconciler.reconcile(mock_exchange, {})
        assert not result.has_discrepancies

    @pytest.mark.asyncio
    async def test_ignores_zero_local_qty(self, reconciler, mock_exchange):
        """Local positions with qty <= 0 are ignored."""
        mock_exchange.get_balance.return_value = {"USDT": 10000.0}
        local = {"BTC/USDT": {"quantity": 0.0, "entry_price": 50000}}
        result = await reconciler.reconcile(mock_exchange, local)
        assert not result.has_discrepancies


class TestReconcileErrorHandling:
    """Test graceful error handling during reconciliation."""

    @pytest.mark.asyncio
    async def test_exchange_error_sets_error_field(self, reconciler, mock_exchange):
        """Exchange errors are caught and reported in result."""
        mock_exchange.get_balance.side_effect = ConnectionError("Network error")
        result = await reconciler.reconcile(
            mock_exchange,
            {"BTC/USDT": {"quantity": 0.5, "entry_price": 50000}},
        )
        assert result.error is not None
        assert "Network error" in result.error
        assert not result.has_discrepancies  # No discrepancies because we couldn't check

    @pytest.mark.asyncio
    async def test_exchange_error_stores_last_result(self, reconciler, mock_exchange):
        """Even on error, last_result is updated."""
        mock_exchange.get_balance.side_effect = Exception("timeout")
        await reconciler.reconcile(mock_exchange, {})
        assert reconciler.last_result is not None
        assert reconciler.last_result.error is not None


class TestReconcilerState:
    """Test reconciler state management."""

    @pytest.mark.asyncio
    async def test_last_result_initially_none(self):
        r = PositionReconciler()
        assert r.last_result is None

    @pytest.mark.asyncio
    async def test_last_result_updated_after_reconcile(self, reconciler, mock_exchange):
        mock_exchange.get_balance.return_value = {"USDT": 10000.0}
        result = await reconciler.reconcile(mock_exchange, {})
        assert reconciler.last_result is result

    @pytest.mark.asyncio
    async def test_exchange_name_captured(self, reconciler, mock_exchange):
        mock_exchange.name = "upbit"
        mock_exchange.get_balance.return_value = {"USDT": 1000.0}
        result = await reconciler.reconcile(mock_exchange, {})
        assert result.exchange_name == "upbit"

    @pytest.mark.asyncio
    async def test_timestamp_set(self, reconciler, mock_exchange):
        mock_exchange.get_balance.return_value = {"USDT": 1000.0}
        result = await reconciler.reconcile(mock_exchange, {})
        assert result.timestamp  # Not empty


class TestAlertFormatting:
    """Test alert message formatting."""

    def test_clean_result_message(self, reconciler):
        result = ReconciliationResult(
            exchange_name="binance",
            matched=["BTC/USDT", "ETH/USDT"],
        )
        msg = reconciler.format_alert_message(result)
        assert "OK" in msg
        assert "2 positions matched" in msg

    def test_discrepancy_message_local_only(self, reconciler):
        result = ReconciliationResult(
            exchange_name="binance",
            local_only=[
                PositionDiscrepancy(
                    "BTC/USDT", DiscrepancyType.LOCAL_ONLY, 0.5, 0.0
                ),
            ],
        )
        msg = reconciler.format_alert_message(result)
        assert "ALERT" in msg
        assert "LOCAL ONLY" in msg
        assert "BTC/USDT" in msg

    def test_discrepancy_message_qty_mismatch(self, reconciler):
        result = ReconciliationResult(
            exchange_name="binance",
            qty_mismatch=[
                PositionDiscrepancy(
                    "ETH/USDT", DiscrepancyType.QTY_MISMATCH, 2.0, 1.5
                ),
            ],
        )
        msg = reconciler.format_alert_message(result)
        assert "QTY MISMATCH" in msg
        assert "local=2.0" in msg
        assert "exchange=1.5" in msg


class TestCustomTolerance:
    """Test reconciliation with custom tolerance."""

    @pytest.mark.asyncio
    async def test_strict_tolerance(self, mock_exchange):
        """Zero tolerance flags even tiny differences."""
        reconciler = PositionReconciler(tolerance_pct=0.0)
        mock_exchange.get_balance.return_value = {
            "BTC": 0.500001,
            "USDT": 5000.0,
        }
        local = {"BTC/USDT": {"quantity": 0.5, "entry_price": 50000}}
        result = await reconciler.reconcile(mock_exchange, local)
        assert len(result.qty_mismatch) == 1

    @pytest.mark.asyncio
    async def test_loose_tolerance(self, mock_exchange):
        """High tolerance allows bigger differences."""
        reconciler = PositionReconciler(tolerance_pct=50.0)
        mock_exchange.get_balance.return_value = {
            "BTC": 0.3,  # 40% less — within 50% tolerance
            "USDT": 5000.0,
        }
        local = {"BTC/USDT": {"quantity": 0.5, "entry_price": 50000}}
        result = await reconciler.reconcile(mock_exchange, local)
        assert "BTC/USDT" in result.matched
        assert not result.has_discrepancies


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestReconciliationEndpoint:
    """Test GET /api/reconciliation endpoint."""

    @pytest.mark.asyncio
    async def test_empty_reconciliation(self, client):
        resp = await client.get("/api/reconciliation")
        assert resp.status_code == 200
        data = resp.json()
        assert "reconciliation" in data
        assert data["reconciliation"] == {}

    @pytest.mark.asyncio
    async def test_reconciliation_with_data(self, client):
        """Endpoint returns stored reconciliation result."""
        update_state(reconciliation={
            "timestamp": "2026-02-22T10:00:00Z",
            "exchange_name": "binance",
            "matched": ["BTC/USDT"],
            "local_only": [],
            "exchange_only": [],
            "qty_mismatch": [],
            "has_discrepancies": False,
            "total_discrepancies": 0,
            "error": None,
        })
        resp = await client.get("/api/reconciliation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["reconciliation"]["exchange_name"] == "binance"
        assert data["reconciliation"]["matched"] == ["BTC/USDT"]
        assert data["reconciliation"]["has_discrepancies"] is False

    @pytest.mark.asyncio
    async def test_reconciliation_with_discrepancies(self, client):
        """Endpoint returns discrepancy details."""
        update_state(reconciliation={
            "timestamp": "2026-02-22T10:00:00Z",
            "exchange_name": "binance",
            "matched": [],
            "local_only": [{
                "symbol": "BTC/USDT", "type": "local_only",
                "local_qty": 0.5, "exchange_qty": 0.0, "details": "",
            }],
            "exchange_only": [],
            "qty_mismatch": [{
                "symbol": "ETH/USDT", "type": "qty_mismatch",
                "local_qty": 2.0, "exchange_qty": 1.5, "details": "",
            }],
            "has_discrepancies": True,
            "total_discrepancies": 2,
            "error": None,
        })
        resp = await client.get("/api/reconciliation")
        data = resp.json()
        assert data["reconciliation"]["has_discrepancies"] is True
        assert data["reconciliation"]["total_discrepancies"] == 2
        assert len(data["reconciliation"]["local_only"]) == 1
        assert len(data["reconciliation"]["qty_mismatch"]) == 1


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------


class TestReconciliationConfig:
    """Test reconciliation config fields exist and work."""

    def test_config_defaults(self):
        """Reconciliation config fields have correct defaults."""
        from bot.config import Settings

        # Use nonexistent config file to avoid YAML overrides
        s = Settings(config_file="__nonexistent__")
        assert s.reconciliation_enabled is True
        assert s.reconciliation_interval_cycles == 10
        assert s.reconciliation_auto_fix is False

    def test_config_in_metadata(self):
        """Reconciliation settings are in SETTINGS_METADATA."""
        from bot.config import SETTINGS_METADATA

        assert "reconciliation_enabled" in SETTINGS_METADATA
        assert "reconciliation_interval_cycles" in SETTINGS_METADATA
        assert "reconciliation_auto_fix" in SETTINGS_METADATA

        # All should be hot-reloadable (requires_restart=False)
        assert SETTINGS_METADATA["reconciliation_enabled"]["requires_restart"] is False
        assert SETTINGS_METADATA["reconciliation_interval_cycles"]["requires_restart"] is False
        assert SETTINGS_METADATA["reconciliation_auto_fix"]["requires_restart"] is False

    def test_config_hot_reload(self):
        """Reconciliation settings can be hot-reloaded."""
        from bot.config import Settings

        s = Settings(config_file="__nonexistent__")
        changed = s.reload({"reconciliation_enabled": False})
        assert "reconciliation_enabled" in changed
        assert s.reconciliation_enabled is False

        changed = s.reload({"reconciliation_interval_cycles": 5})
        assert "reconciliation_interval_cycles" in changed
        assert s.reconciliation_interval_cycles == 5
