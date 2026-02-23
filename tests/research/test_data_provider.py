"""Tests for HistoricalDataProvider."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.models import OHLCV
from bot.research.data_provider import HistoricalDataProvider


def _make_candles(n: int, base_close: float = 100.0) -> list[OHLCV]:
    """Create n OHLCV candles with incrementing prices."""
    candles = []
    for i in range(n):
        price = base_close + i
        candles.append(
            OHLCV(
                timestamp=datetime(2025, 1, 1, i, 0, tzinfo=timezone.utc),
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000.0 + i,
                symbol="BTC/USDT",
                timeframe="1h",
            )
        )
    return candles


def _make_funding_rates(n: int) -> list[dict]:
    """Create n funding rate dicts."""
    return [
        {
            "symbol": "BTC/USDT",
            "timestamp": datetime(2025, 1, 1, i * 8, 0, tzinfo=timezone.utc),
            "funding_rate": 0.0001 * (i + 1),
            "funding_timestamp": datetime(2025, 1, 1, i * 8, 0, tzinfo=timezone.utc),
            "mark_price": 50000.0 + i * 10,
            "spot_price": 49990.0 + i * 10,
            "spread_pct": 0.02,
        }
        for i in range(n)
    ]


def _make_mock_store(
    candles: list[OHLCV] | None = None,
    funding_rates: list[dict] | None = None,
    available_symbols: list[str] | None = None,
) -> MagicMock:
    """Create a mock DataStore with configurable returns."""
    store = MagicMock()
    store.get_candles = AsyncMock(return_value=candles or [])
    store.get_funding_rates = AsyncMock(return_value=funding_rates or [])
    store.get_available_symbols = AsyncMock(return_value=available_symbols or [])
    return store


class TestGetPrices:
    @pytest.mark.asyncio
    async def test_returns_close_prices(self):
        candles = _make_candles(5)
        store = _make_mock_store(candles=candles)
        provider = HistoricalDataProvider(data_store=store)

        prices = await provider.get_prices("BTC/USDT", "1h", lookback_days=30)

        assert prices == [100.0, 101.0, 102.0, 103.0, 104.0]
        store.get_candles.assert_called_once()
        call_kwargs = store.get_candles.call_args[1]
        assert call_kwargs["symbol"] == "BTC/USDT"
        assert call_kwargs["timeframe"] == "1h"
        assert call_kwargs["limit"] == 24 * 30  # 1h × 30 days

    @pytest.mark.asyncio
    async def test_empty_data_returns_empty_list(self):
        store = _make_mock_store(candles=[])
        provider = HistoricalDataProvider(data_store=store)

        prices = await provider.get_prices("UNKNOWN/USDT")

        assert prices == []

    @pytest.mark.asyncio
    async def test_4h_timeframe_limit(self):
        store = _make_mock_store(candles=[])
        provider = HistoricalDataProvider(data_store=store)

        await provider.get_prices("BTC/USDT", timeframe="4h", lookback_days=10)

        call_kwargs = store.get_candles.call_args[1]
        assert call_kwargs["limit"] == 6 * 10  # 4h → 6 candles/day

    @pytest.mark.asyncio
    async def test_1d_timeframe_limit(self):
        store = _make_mock_store(candles=[])
        provider = HistoricalDataProvider(data_store=store)

        await provider.get_prices("BTC/USDT", timeframe="1d", lookback_days=60)

        call_kwargs = store.get_candles.call_args[1]
        assert call_kwargs["limit"] == 1 * 60  # 1d → 1 candle/day

    @pytest.mark.asyncio
    async def test_unknown_timeframe_defaults_to_24(self):
        store = _make_mock_store(candles=[])
        provider = HistoricalDataProvider(data_store=store)

        await provider.get_prices("BTC/USDT", timeframe="3h", lookback_days=5)

        call_kwargs = store.get_candles.call_args[1]
        assert call_kwargs["limit"] == 24 * 5  # fallback 24/day


class TestGetOhlcv:
    @pytest.mark.asyncio
    async def test_returns_ohlcv_models(self):
        candles = _make_candles(3)
        store = _make_mock_store(candles=candles)
        provider = HistoricalDataProvider(data_store=store)

        result = await provider.get_ohlcv("BTC/USDT")

        assert len(result) == 3
        assert all(isinstance(c, OHLCV) for c in result)
        assert result[0].close == 100.0

    @pytest.mark.asyncio
    async def test_empty_returns_empty(self):
        store = _make_mock_store(candles=[])
        provider = HistoricalDataProvider(data_store=store)

        result = await provider.get_ohlcv("BTC/USDT")

        assert result == []


class TestGetReturns:
    @pytest.mark.asyncio
    async def test_calculates_returns_correctly(self):
        candles = _make_candles(4, base_close=100.0)
        store = _make_mock_store(candles=candles)
        provider = HistoricalDataProvider(data_store=store)

        returns = await provider.get_returns("BTC/USDT")

        assert len(returns) == 3  # len(prices) - 1
        # 100→101: 1/100 = 0.01
        assert abs(returns[0] - 0.01) < 1e-9
        # 101→102: 1/101 ≈ 0.00990099
        assert abs(returns[1] - 1.0 / 101.0) < 1e-9

    @pytest.mark.asyncio
    async def test_single_price_returns_empty(self):
        candles = _make_candles(1)
        store = _make_mock_store(candles=candles)
        provider = HistoricalDataProvider(data_store=store)

        returns = await provider.get_returns("BTC/USDT")

        assert returns == []

    @pytest.mark.asyncio
    async def test_no_data_returns_empty(self):
        store = _make_mock_store(candles=[])
        provider = HistoricalDataProvider(data_store=store)

        returns = await provider.get_returns("BTC/USDT")

        assert returns == []


class TestGetFundingRates:
    @pytest.mark.asyncio
    async def test_returns_formatted_dicts(self):
        raw = _make_funding_rates(3)
        store = _make_mock_store(funding_rates=raw)
        provider = HistoricalDataProvider(data_store=store)

        result = await provider.get_funding_rates("BTC/USDT", lookback_days=10)

        assert len(result) == 3
        for r in result:
            assert "timestamp" in r
            assert "funding_rate" in r
            assert "mark_price" in r
            assert "spot_price" in r
            # Should NOT include extra fields like spread_pct or funding_timestamp
            assert "spread_pct" not in r
            assert "funding_timestamp" not in r

    @pytest.mark.asyncio
    async def test_empty_returns_empty(self):
        store = _make_mock_store(funding_rates=[])
        provider = HistoricalDataProvider(data_store=store)

        result = await provider.get_funding_rates("BTC/USDT")

        assert result == []

    @pytest.mark.asyncio
    async def test_lookback_limit_calculation(self):
        store = _make_mock_store(funding_rates=[])
        provider = HistoricalDataProvider(data_store=store)

        await provider.get_funding_rates("BTC/USDT", lookback_days=20)

        call_kwargs = store.get_funding_rates.call_args[1]
        assert call_kwargs["limit"] == 20 * 3  # 3 per day (8h interval)


class TestGetMultiPrices:
    @pytest.mark.asyncio
    async def test_fetches_multiple_symbols(self):
        async def mock_get_candles(symbol, timeframe="1h", start=None, limit=500):
            if symbol == "BTC/USDT":
                return _make_candles(3, base_close=50000.0)
            elif symbol == "ETH/USDT":
                return _make_candles(3, base_close=3000.0)
            return []

        store = MagicMock()
        store.get_candles = AsyncMock(side_effect=mock_get_candles)
        provider = HistoricalDataProvider(data_store=store)

        result = await provider.get_multi_prices(
            ["BTC/USDT", "ETH/USDT"], lookback_days=5
        )

        assert "BTC/USDT" in result
        assert "ETH/USDT" in result
        assert result["BTC/USDT"] == [50000.0, 50001.0, 50002.0]
        assert result["ETH/USDT"] == [3000.0, 3001.0, 3002.0]

    @pytest.mark.asyncio
    async def test_empty_symbols_list(self):
        store = _make_mock_store()
        provider = HistoricalDataProvider(data_store=store)

        result = await provider.get_multi_prices([])

        assert result == {}

    @pytest.mark.asyncio
    async def test_some_symbols_missing_data(self):
        async def mock_get_candles(symbol, timeframe="1h", start=None, limit=500):
            if symbol == "BTC/USDT":
                return _make_candles(3)
            return []

        store = MagicMock()
        store.get_candles = AsyncMock(side_effect=mock_get_candles)
        provider = HistoricalDataProvider(data_store=store)

        result = await provider.get_multi_prices(["BTC/USDT", "UNKNOWN/USDT"])

        assert len(result["BTC/USDT"]) == 3
        assert result["UNKNOWN/USDT"] == []


class TestGetAvailableSymbols:
    @pytest.mark.asyncio
    async def test_delegates_to_store(self):
        store = _make_mock_store(available_symbols=["BTC/USDT", "ETH/USDT"])
        provider = HistoricalDataProvider(data_store=store)

        result = await provider.get_available_symbols(timeframe="1h", min_candles=100)

        assert result == ["BTC/USDT", "ETH/USDT"]
        store.get_available_symbols.assert_called_once_with(
            timeframe="1h", min_count=100
        )

    @pytest.mark.asyncio
    async def test_default_params(self):
        store = _make_mock_store(available_symbols=[])
        provider = HistoricalDataProvider(data_store=store)

        result = await provider.get_available_symbols()

        assert result == []
        store.get_available_symbols.assert_called_once_with(
            timeframe="1h", min_count=100
        )

    @pytest.mark.asyncio
    async def test_custom_min_candles(self):
        store = _make_mock_store(available_symbols=["BTC/USDT"])
        provider = HistoricalDataProvider(data_store=store)

        await provider.get_available_symbols(timeframe="4h", min_candles=50)

        store.get_available_symbols.assert_called_once_with(
            timeframe="4h", min_count=50
        )


class TestLookbackFiltering:
    @pytest.mark.asyncio
    async def test_start_time_is_set(self):
        store = _make_mock_store(candles=[])
        provider = HistoricalDataProvider(data_store=store)

        await provider.get_prices("BTC/USDT", lookback_days=7)

        call_kwargs = store.get_candles.call_args[1]
        assert call_kwargs["start"] is not None
        # Should be roughly 7 days ago
        delta = datetime.now(timezone.utc) - call_kwargs["start"]
        assert 6.99 < delta.total_seconds() / 86400 < 7.01
