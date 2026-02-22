"""Tests for multi-symbol data alignment."""

import numpy as np
import pytest

from bot.data.pair_data import (
    PairDataProvider,
    compute_correlation_matrix,
    compute_log_returns,
    compute_spread,
)


def _make_candles(closes, symbol="BTC/USDT"):
    from datetime import datetime, timedelta

    from bot.models import OHLCV

    base = datetime(2024, 1, 1)
    return [
        OHLCV(
            timestamp=base + timedelta(hours=i),
            open=c,
            high=c * 1.01,
            low=c * 0.99,
            close=c,
            volume=1000.0,
            symbol=symbol,
        )
        for i, c in enumerate(closes)
    ]


class TestPairDataProvider:
    def test_aligned_closes_from_candles(self):
        candles_a = _make_candles([100, 101, 102, 103], symbol="A")
        candles_b = _make_candles([200, 201, 202, 203], symbol="B")
        result = PairDataProvider.get_aligned_closes_from_candles(
            {
                "A": candles_a,
                "B": candles_b,
            }
        )
        assert "A" in result
        assert "B" in result
        assert len(result["A"]) == 4
        assert len(result["B"]) == 4

    def test_aligned_closes_partial_overlap(self):
        from datetime import datetime, timedelta

        from bot.models import OHLCV

        base = datetime(2024, 1, 1)
        candles_a = [
            OHLCV(
                timestamp=base + timedelta(hours=i),
                open=100 + i,
                high=101 + i,
                low=99 + i,
                close=100 + i,
                volume=1000,
                symbol="A",
            )
            for i in range(5)
        ]
        candles_b = [
            OHLCV(
                timestamp=base + timedelta(hours=i + 2),
                open=200 + i,
                high=201 + i,
                low=199 + i,
                close=200 + i,
                volume=1000,
                symbol="B",
            )
            for i in range(5)
        ]
        result = PairDataProvider.get_aligned_closes_from_candles(
            {
                "A": candles_a,
                "B": candles_b,
            }
        )
        # Only overlapping timestamps
        assert len(result["A"]) == 3
        assert len(result["B"]) == 3

    def test_empty_input(self):
        result = PairDataProvider.get_aligned_closes_from_candles({})
        assert result == {}

    def test_no_overlap(self):
        """Test that non-overlapping timestamps return empty result."""
        from datetime import datetime, timedelta

        from bot.models import OHLCV

        candles_a = [
            OHLCV(
                timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
                open=100,
                high=101,
                low=99,
                close=100,
                volume=1000,
                symbol="A",
            )
            for i in range(3)
        ]
        candles_b = [
            OHLCV(
                timestamp=datetime(2024, 2, 1) + timedelta(hours=i),
                open=200,
                high=201,
                low=199,
                close=200,
                volume=1000,
                symbol="B",
            )
            for i in range(3)
        ]
        result = PairDataProvider.get_aligned_closes_from_candles(
            {"A": candles_a, "B": candles_b}
        )
        assert result == {}

    def test_returns_dict_of_symbol_to_prices(self):
        """Verify the return type is dict of symbol -> numpy arrays."""
        candles_a = _make_candles([100, 101, 102], symbol="BTC/USDT")
        candles_b = _make_candles([50, 51, 52], symbol="ETH/USDT")
        result = PairDataProvider.get_aligned_closes_from_candles(
            {"BTC/USDT": candles_a, "ETH/USDT": candles_b}
        )
        assert isinstance(result, dict)
        assert isinstance(result["BTC/USDT"], np.ndarray)
        assert isinstance(result["ETH/USDT"], np.ndarray)
        np.testing.assert_array_almost_equal(result["BTC/USDT"], [100, 101, 102])
        np.testing.assert_array_almost_equal(result["ETH/USDT"], [50, 51, 52])

    @pytest.mark.asyncio
    async def test_get_aligned_closes_from_store(self):
        """Test async get_aligned_closes fetches from store and aligns."""
        from datetime import datetime, timedelta
        from unittest.mock import AsyncMock

        from bot.models import OHLCV

        base = datetime(2024, 1, 1)

        def make_store_candles(symbol, closes):
            return [
                OHLCV(
                    timestamp=base + timedelta(hours=i),
                    open=c,
                    high=c * 1.01,
                    low=c * 0.99,
                    close=c,
                    volume=1000,
                    symbol=symbol,
                )
                for i, c in enumerate(closes)
            ]

        mock_store = AsyncMock()
        mock_store.get_candles = AsyncMock(
            side_effect=lambda symbol, **kwargs: make_store_candles(
                symbol, [100, 101, 102] if symbol == "A" else [200, 201, 202]
            )
        )

        result = await PairDataProvider.get_aligned_closes(
            symbols=["A", "B"], store=mock_store
        )
        assert "A" in result
        assert "B" in result
        assert len(result["A"]) == 3

    @pytest.mark.asyncio
    async def test_get_aligned_closes_missing_data(self):
        """Test that symbols with no data are excluded."""
        from unittest.mock import AsyncMock

        mock_store = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=[])

        result = await PairDataProvider.get_aligned_closes(
            symbols=["A", "B"], store=mock_store
        )
        assert result == {}


class TestComputeSpread:
    def test_basic_spread(self):
        a = np.array([100, 101, 102])
        b = np.array([50, 50.5, 51])
        spread = compute_spread(a, b, hedge_ratio=2.0)
        np.testing.assert_array_almost_equal(spread, [0, 0, 0])

    def test_array_hedge_ratio(self):
        a = np.array([100, 101, 102])
        b = np.array([50, 50.5, 51])
        hedge = np.array([2.0, 2.0, 2.0])
        spread = compute_spread(a, b, hedge)
        np.testing.assert_array_almost_equal(spread, [0, 0, 0])

    def test_nonzero_spread(self):
        a = np.array([100, 110, 120])
        b = np.array([50, 50, 50])
        spread = compute_spread(a, b, hedge_ratio=1.0)
        np.testing.assert_array_almost_equal(spread, [50, 60, 70])


class TestComputeLogReturns:
    def test_basic(self):
        prices = np.array([100, 105, 110])
        returns = compute_log_returns(prices)
        assert len(returns) == 2
        assert returns[0] == pytest.approx(np.log(105 / 100))

    def test_single_price(self):
        returns = compute_log_returns(np.array([100]))
        assert len(returns) == 0

    def test_handles_zeros(self):
        """Verify log(0) is handled gracefully."""
        prices = np.array([0, 100, 200])
        returns = compute_log_returns(prices)
        assert len(returns) == 2
        assert np.isfinite(returns).all()


class TestCorrelationMatrix:
    def test_perfect_correlation(self):
        returns = {
            "A": np.array(
                [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01]
            ),
            "B": np.array(
                [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01]
            ),
        }
        matrix = compute_correlation_matrix(returns)
        assert matrix["A"]["B"] == pytest.approx(1.0)

    def test_single_symbol(self):
        returns = {"A": np.random.randn(20)}
        matrix = compute_correlation_matrix(returns)
        assert matrix["A"]["A"] == 1.0

    def test_with_window(self):
        np.random.seed(42)
        returns = {
            "A": np.random.randn(100),
            "B": np.random.randn(100),
        }
        matrix = compute_correlation_matrix(returns, window=20)
        assert "A" in matrix
        assert "B" in matrix["A"]

    def test_insufficient_data(self):
        """Short series returns zero correlations."""
        returns = {
            "A": np.array([0.01, -0.01]),
            "B": np.array([0.02, -0.02]),
        }
        matrix = compute_correlation_matrix(returns)
        assert matrix["A"]["B"] == 0.0
