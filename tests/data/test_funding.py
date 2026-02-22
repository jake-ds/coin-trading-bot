"""Tests for FundingRateMonitor."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from bot.data.funding import FundingRateMonitor


@pytest.fixture
def mock_exchange():
    """Create a mock exchange adapter with no _exchange attribute."""
    exchange = MagicMock(spec=["name"])
    exchange.name = "binance"
    return exchange


@pytest.fixture
def mock_store():
    """Create a mock DataStore."""
    store = AsyncMock()
    store.save_funding_rate = AsyncMock()
    store.get_funding_rates = AsyncMock(return_value=[])
    return store


@pytest.fixture
def monitor(mock_exchange, mock_store):
    """Create a FundingRateMonitor with mocked dependencies."""
    return FundingRateMonitor(
        exchange=mock_exchange,
        store=mock_store,
        funding_rate_history=10,
    )


@pytest.fixture
def monitor_no_store(mock_exchange):
    """Create a FundingRateMonitor without a DataStore."""
    return FundingRateMonitor(
        exchange=mock_exchange,
        store=None,
    )


def _make_ccxt_mock_with_funding():
    """Create a ccxt mock that has fetch_funding_rate."""
    mock = MagicMock(spec=["fetch_funding_rate", "fetch_funding_rate_history"])
    return mock


class TestFundingRateMonitorInit:
    """Test FundingRateMonitor initialization."""

    def test_initial_state(self, monitor):
        assert monitor.latest_rates == {}
        assert monitor.get_latest_rate("BTC/USDT:USDT") is None
        assert monitor.get_rate_history("BTC/USDT:USDT") == []

    def test_get_average_rate_no_history(self, monitor):
        assert monitor.get_average_rate("BTC/USDT:USDT") is None


class TestUpdateRate:
    """Test manual rate updates."""

    def test_update_rate_basic(self, monitor):
        ts = datetime(2026, 2, 22, 12, 0, 0)
        monitor.update_rate("BTC/USDT:USDT", 0.0005, ts, 50000.0, 49950.0)

        latest = monitor.get_latest_rate("BTC/USDT:USDT")
        assert latest is not None
        assert latest["funding_rate"] == 0.0005
        assert latest["mark_price"] == 50000.0
        assert latest["spot_price"] == 49950.0
        assert latest["spread_pct"] == pytest.approx(
            (50000.0 - 49950.0) / 49950.0 * 100, rel=1e-4
        )

    def test_update_rate_negative(self, monitor):
        monitor.update_rate("ETH/USDT:USDT", -0.001)
        latest = monitor.get_latest_rate("ETH/USDT:USDT")
        assert latest is not None
        assert latest["funding_rate"] == -0.001

    def test_update_rate_builds_history(self, monitor):
        for i in range(5):
            ts = datetime(2026, 2, 22) + timedelta(hours=8 * i)
            monitor.update_rate("BTC/USDT:USDT", 0.0001 * (i + 1), ts)

        history = monitor.get_rate_history("BTC/USDT:USDT")
        assert len(history) == 5
        assert history[0]["funding_rate"] == 0.0001
        assert history[4]["funding_rate"] == 0.0005

    def test_update_rate_trims_history(self, monitor):
        # Monitor configured with funding_rate_history=10
        for i in range(15):
            ts = datetime(2026, 2, 22) + timedelta(hours=8 * i)
            monitor.update_rate("BTC/USDT:USDT", 0.0001 * i, ts)

        history = monitor.get_rate_history("BTC/USDT:USDT")
        assert len(history) == 10

    def test_update_rate_default_timestamp(self, monitor):
        monitor.update_rate("BTC/USDT:USDT", 0.0005)
        latest = monitor.get_latest_rate("BTC/USDT:USDT")
        assert latest is not None
        assert latest["funding_timestamp"] is not None

    def test_update_rate_zero_spread_when_no_prices(self, monitor):
        monitor.update_rate("BTC/USDT:USDT", 0.0005)
        latest = monitor.get_latest_rate("BTC/USDT:USDT")
        assert latest["spread_pct"] == 0.0

    def test_latest_rates_returns_copy(self, monitor):
        monitor.update_rate("BTC/USDT:USDT", 0.0005)
        rates = monitor.latest_rates
        rates["FAKE"] = {"funding_rate": 99}
        assert "FAKE" not in monitor.latest_rates


class TestGetAverageRate:
    """Test average rate calculation."""

    def test_average_rate(self, monitor):
        rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
        for i, r in enumerate(rates):
            ts = datetime(2026, 2, 22) + timedelta(hours=8 * i)
            monitor.update_rate("BTC/USDT:USDT", r, ts)

        avg = monitor.get_average_rate("BTC/USDT:USDT", periods=5)
        assert avg == pytest.approx(0.0003)

    def test_average_rate_limited_periods(self, monitor):
        rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0010]
        for i, r in enumerate(rates):
            ts = datetime(2026, 2, 22) + timedelta(hours=8 * i)
            monitor.update_rate("BTC/USDT:USDT", r, ts)

        avg = monitor.get_average_rate("BTC/USDT:USDT", periods=2)
        assert avg == pytest.approx((0.0004 + 0.0010) / 2)

    def test_average_rate_unknown_symbol(self, monitor):
        assert monitor.get_average_rate("UNKNOWN/USDT:USDT") is None


class TestFetchFundingRate:
    """Test fetching funding rates from exchange."""

    @pytest.mark.asyncio
    async def test_fetch_funding_rate_success(self, mock_store):
        # Create exchange with ccxt mock that has fetch_funding_rate
        ccxt_mock = _make_ccxt_mock_with_funding()
        ccxt_mock.fetch_funding_rate = AsyncMock(return_value={
            "fundingRate": 0.0005,
            "datetime": "2026-02-22T08:00:00Z",
            "markPrice": 50000.0,
            "indexPrice": 49950.0,
            "fundingTimestamp": 1740211200000,
        })

        # Exchange adapter wrapping ccxt
        exchange = MagicMock(spec=["name", "_exchange"])
        exchange.name = "binance"
        exchange._exchange = ccxt_mock

        monitor = FundingRateMonitor(exchange=exchange, store=mock_store,
                                     funding_rate_history=10)
        result = await monitor.fetch_funding_rate("BTC/USDT:USDT")

        assert result is not None
        assert result["funding_rate"] == 0.0005
        assert result["mark_price"] == 50000.0
        assert result["spot_price"] == 49950.0
        assert result["spread_pct"] != 0.0
        assert result["symbol"] == "BTC/USDT:USDT"
        # Verify stored in memory
        assert monitor.get_latest_rate("BTC/USDT:USDT") is not None
        # Verify persisted to store
        mock_store.save_funding_rate.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_funding_rate_no_ccxt(self, monitor_no_store):
        # exchange has no _exchange — can't traverse to ccxt
        result = await monitor_no_store.fetch_funding_rate("BTC/USDT:USDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_funding_rate_no_support(self):
        # ccxt exchange without fetch_funding_rate
        ccxt_mock = MagicMock(spec=["has"])
        exchange = MagicMock(spec=["name", "_exchange"])
        exchange.name = "upbit"
        exchange._exchange = ccxt_mock

        monitor = FundingRateMonitor(exchange=exchange, store=None)
        result = await monitor.fetch_funding_rate("BTC/USDT:USDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_funding_rate_none_rate(self):
        ccxt_mock = _make_ccxt_mock_with_funding()
        ccxt_mock.fetch_funding_rate = AsyncMock(return_value={
            "fundingRate": None,
            "datetime": "2026-02-22T08:00:00Z",
        })
        exchange = MagicMock(spec=["name", "_exchange"])
        exchange.name = "binance"
        exchange._exchange = ccxt_mock

        monitor = FundingRateMonitor(exchange=exchange, store=None)
        result = await monitor.fetch_funding_rate("BTC/USDT:USDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_funding_rate_exception(self):
        ccxt_mock = _make_ccxt_mock_with_funding()
        ccxt_mock.fetch_funding_rate = AsyncMock(side_effect=Exception("API error"))
        exchange = MagicMock(spec=["name", "_exchange"])
        exchange.name = "binance"
        exchange._exchange = ccxt_mock

        monitor = FundingRateMonitor(exchange=exchange, store=None)
        result = await monitor.fetch_funding_rate("BTC/USDT:USDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_funding_rate_no_store(self):
        ccxt_mock = _make_ccxt_mock_with_funding()
        ccxt_mock.fetch_funding_rate = AsyncMock(return_value={
            "fundingRate": 0.0003,
            "datetime": "2026-02-22T08:00:00Z",
            "markPrice": 50000.0,
            "indexPrice": 49950.0,
        })
        exchange = MagicMock(spec=["name", "_exchange"])
        exchange.name = "binance"
        exchange._exchange = ccxt_mock

        monitor = FundingRateMonitor(exchange=exchange, store=None)
        result = await monitor.fetch_funding_rate("BTC/USDT:USDT")
        assert result is not None
        assert result["funding_rate"] == 0.0003
        assert monitor.get_latest_rate("BTC/USDT:USDT") is not None

    @pytest.mark.asyncio
    async def test_fetch_funding_rate_timestamp_int(self, mock_store):
        ccxt_mock = _make_ccxt_mock_with_funding()
        ccxt_mock.fetch_funding_rate = AsyncMock(return_value={
            "fundingRate": 0.0002,
            "timestamp": 1740211200000,
            "markPrice": 50000.0,
            "indexPrice": 49950.0,
        })
        exchange = MagicMock(spec=["name", "_exchange"])
        exchange.name = "binance"
        exchange._exchange = ccxt_mock

        monitor = FundingRateMonitor(exchange=exchange, store=mock_store,
                                     funding_rate_history=10)
        result = await monitor.fetch_funding_rate("BTC/USDT:USDT")
        assert result is not None
        assert result["funding_rate"] == 0.0002

    @pytest.mark.asyncio
    async def test_fetch_trims_history(self, mock_store):
        ccxt_mock = _make_ccxt_mock_with_funding()
        exchange = MagicMock(spec=["name", "_exchange"])
        exchange.name = "binance"
        exchange._exchange = ccxt_mock

        monitor = FundingRateMonitor(exchange=exchange, store=mock_store,
                                     funding_rate_history=10)

        # Fetch 15 times (history limit is 10)
        for i in range(15):
            ts = datetime(2026, 2, 22) + timedelta(hours=8 * i)
            ccxt_mock.fetch_funding_rate = AsyncMock(return_value={
                "fundingRate": 0.0001 * (i + 1),
                "datetime": ts.isoformat() + "Z",
                "markPrice": 50000.0,
                "indexPrice": 49950.0,
            })
            await monitor.fetch_funding_rate("BTC/USDT:USDT")

        history = monitor.get_rate_history("BTC/USDT:USDT")
        assert len(history) == 10


class TestFetchFundingRateHistory:
    """Test fetching historical funding rates."""

    @pytest.mark.asyncio
    async def test_fetch_history_success(self):
        ccxt_mock = _make_ccxt_mock_with_funding()
        ccxt_mock.fetch_funding_rate_history = AsyncMock(return_value=[
            {
                "fundingRate": 0.0001,
                "datetime": "2026-02-22T00:00:00Z",
                "markPrice": 50000.0,
                "indexPrice": 49950.0,
            },
            {
                "fundingRate": 0.0003,
                "datetime": "2026-02-22T08:00:00Z",
                "markPrice": 50100.0,
                "indexPrice": 50050.0,
            },
        ])
        exchange = MagicMock(spec=["name", "_exchange"])
        exchange.name = "binance"
        exchange._exchange = ccxt_mock

        monitor = FundingRateMonitor(exchange=exchange, store=None)
        results = await monitor.fetch_funding_rate_history("BTC/USDT:USDT", limit=10)
        assert len(results) == 2
        assert results[0]["funding_rate"] == 0.0001
        assert results[1]["funding_rate"] == 0.0003

    @pytest.mark.asyncio
    async def test_fetch_history_no_support(self):
        ccxt_mock = MagicMock(spec=["has"])  # No fetch_funding_rate_history
        exchange = MagicMock(spec=["name", "_exchange"])
        exchange.name = "upbit"
        exchange._exchange = ccxt_mock

        monitor = FundingRateMonitor(exchange=exchange, store=None)
        results = await monitor.fetch_funding_rate_history("BTC/USDT:USDT")
        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_history_exception(self):
        ccxt_mock = _make_ccxt_mock_with_funding()
        ccxt_mock.fetch_funding_rate_history = AsyncMock(
            side_effect=Exception("API error")
        )
        exchange = MagicMock(spec=["name", "_exchange"])
        exchange.name = "binance"
        exchange._exchange = ccxt_mock

        monitor = FundingRateMonitor(exchange=exchange, store=None)
        results = await monitor.fetch_funding_rate_history("BTC/USDT:USDT")
        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_history_updates_memory(self):
        ccxt_mock = _make_ccxt_mock_with_funding()
        ccxt_mock.fetch_funding_rate_history = AsyncMock(return_value=[
            {
                "fundingRate": 0.0002,
                "datetime": "2026-02-22T00:00:00Z",
                "markPrice": 50000.0,
                "indexPrice": 49950.0,
            },
        ])
        exchange = MagicMock(spec=["name", "_exchange"])
        exchange.name = "binance"
        exchange._exchange = ccxt_mock

        monitor = FundingRateMonitor(exchange=exchange, store=None)
        await monitor.fetch_funding_rate_history("BTC/USDT:USDT")
        history = monitor.get_rate_history("BTC/USDT:USDT")
        assert len(history) == 1
        assert history[0]["funding_rate"] == 0.0002


class TestDataStoreFundingRate:
    """Test DataStore funding rate persistence."""

    @pytest_asyncio.fixture
    async def store(self, tmp_path):
        from bot.data.store import DataStore
        db_path = tmp_path / "test.db"
        store = DataStore(database_url=f"sqlite+aiosqlite:///{db_path}")
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_save_and_get_funding_rate(self, store):
        ts = datetime(2026, 2, 22, 8, 0, 0)
        await store.save_funding_rate(
            symbol="BTC/USDT:USDT",
            funding_rate=0.0005,
            funding_timestamp=ts,
            mark_price=50000.0,
            spot_price=49950.0,
            spread_pct=0.1,
        )

        rates = await store.get_funding_rates("BTC/USDT:USDT")
        assert len(rates) == 1
        assert rates[0]["funding_rate"] == 0.0005
        assert rates[0]["mark_price"] == 50000.0
        assert rates[0]["spot_price"] == 49950.0
        assert rates[0]["spread_pct"] == 0.1

    @pytest.mark.asyncio
    async def test_save_duplicate_funding_rate_ignored(self, store):
        ts = datetime(2026, 2, 22, 8, 0, 0)
        await store.save_funding_rate(
            symbol="BTC/USDT:USDT",
            funding_rate=0.0005,
            funding_timestamp=ts,
        )
        # Save duplicate — should be ignored
        await store.save_funding_rate(
            symbol="BTC/USDT:USDT",
            funding_rate=0.0010,
            funding_timestamp=ts,
        )

        rates = await store.get_funding_rates("BTC/USDT:USDT")
        assert len(rates) == 1
        assert rates[0]["funding_rate"] == 0.0005

    @pytest.mark.asyncio
    async def test_get_funding_rates_with_date_range(self, store):
        for i in range(5):
            ts = datetime(2026, 2, 22) + timedelta(hours=8 * i)
            await store.save_funding_rate(
                symbol="BTC/USDT:USDT",
                funding_rate=0.0001 * (i + 1),
                funding_timestamp=ts,
            )

        start = datetime(2026, 2, 22, 8, 0, 0)
        end = datetime(2026, 2, 23, 8, 0, 0)
        rates = await store.get_funding_rates(
            "BTC/USDT:USDT", start=start, end=end
        )
        assert len(rates) == 4  # hours 8, 16, 24, 32

    @pytest.mark.asyncio
    async def test_get_funding_rates_limit(self, store):
        for i in range(10):
            ts = datetime(2026, 2, 22) + timedelta(hours=8 * i)
            await store.save_funding_rate(
                symbol="BTC/USDT:USDT",
                funding_rate=0.0001 * (i + 1),
                funding_timestamp=ts,
            )

        rates = await store.get_funding_rates("BTC/USDT:USDT", limit=3)
        assert len(rates) == 3

    @pytest.mark.asyncio
    async def test_get_funding_rates_empty(self, store):
        rates = await store.get_funding_rates("UNKNOWN/USDT:USDT")
        assert rates == []

    @pytest.mark.asyncio
    async def test_save_funding_rate_defaults(self, store):
        ts = datetime(2026, 2, 22, 8, 0, 0)
        await store.save_funding_rate(
            symbol="ETH/USDT:USDT",
            funding_rate=-0.0003,
            funding_timestamp=ts,
        )

        rates = await store.get_funding_rates("ETH/USDT:USDT")
        assert len(rates) == 1
        assert rates[0]["mark_price"] == 0.0
        assert rates[0]["spot_price"] == 0.0
        assert rates[0]["spread_pct"] == 0.0
