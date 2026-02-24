"""Tests for V6-003: Research experiments with real data via HistoricalDataProvider.

Tests cover: real data mode, fallback to synthetic, data_source field,
time pattern analysis (funding), and backward compatibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from bot.research.experiments.cointegration import CointegrationExperiment
from bot.research.experiments.funding_prediction import FundingPredictionExperiment
from bot.research.experiments.optimal_grid import OptimalGridExperiment
from bot.research.experiments.volatility_regime import VolatilityRegimeExperiment


def _make_data_provider(
    prices: dict[str, list[float]] | None = None,
    funding_rates: list[dict] | None = None,
):
    """Create a mock HistoricalDataProvider."""
    provider = MagicMock()

    default_prices = list(np.cumsum(np.random.default_rng(42).normal(0, 0.01, 200)) + 100)

    async def _get_prices(symbol, timeframe="1h", lookback_days=30):
        if prices and symbol in prices:
            return prices[symbol]
        return default_prices

    async def _get_multi_prices(symbols, timeframe="1h", lookback_days=30):
        result = {}
        for sym in symbols:
            if prices and sym in prices:
                result[sym] = prices[sym]
            else:
                result[sym] = default_prices
        return result

    async def _get_funding_rates(symbol, lookback_days=30):
        if funding_rates is not None:
            return funding_rates
        return []

    provider.get_prices = AsyncMock(side_effect=_get_prices)
    provider.get_multi_prices = AsyncMock(side_effect=_get_multi_prices)
    provider.get_funding_rates = AsyncMock(side_effect=_get_funding_rates)

    return provider


# ------------------------------------------------------------------ #
# VolatilityRegimeExperiment — real data
# ------------------------------------------------------------------ #


class TestVolatilityRegimeRealData:
    def test_uses_real_data_when_provider_available(self):
        real_prices = list(
            np.cumsum(np.random.default_rng(10).normal(0, 0.01, 300)) + 50000
        )
        provider = _make_data_provider(prices={"BTC/USDT": real_prices})
        exp = VolatilityRegimeExperiment(data_provider=provider)
        report = exp.run_experiment()
        assert report.results["data_source"] == "real"
        assert "fixed_sharpe" in report.results

    def test_fallback_to_synthetic_when_provider_returns_empty(self):
        provider = _make_data_provider(prices={"BTC/USDT": []})
        exp = VolatilityRegimeExperiment(data_provider=provider)
        report = exp.run_experiment()
        assert report.results["data_source"] == "synthetic"

    def test_fallback_to_synthetic_when_no_provider(self):
        exp = VolatilityRegimeExperiment()
        report = exp.run_experiment()
        assert report.results["data_source"] == "synthetic"

    def test_kwargs_override_provider(self):
        provider = _make_data_provider()
        exp = VolatilityRegimeExperiment(data_provider=provider)
        custom_prices = [100.0 + i * 0.5 for i in range(200)]
        report = exp.run_experiment(prices=custom_prices)
        assert report.results["data_source"] == "kwargs"

    def test_provider_exception_falls_back(self):
        provider = MagicMock()
        provider.get_prices = AsyncMock(side_effect=RuntimeError("DB error"))
        exp = VolatilityRegimeExperiment(data_provider=provider)
        report = exp.run_experiment()
        assert report.results["data_source"] == "synthetic"

    def test_insufficient_real_data_falls_back(self):
        # Only 10 prices — below the 30 threshold
        provider = _make_data_provider(prices={"BTC/USDT": [100.0] * 10})
        exp = VolatilityRegimeExperiment(data_provider=provider)
        report = exp.run_experiment()
        assert report.results["data_source"] == "synthetic"


# ------------------------------------------------------------------ #
# CointegrationExperiment — real data
# ------------------------------------------------------------------ #


class TestCointegrationRealData:
    def test_uses_real_data_when_provider_available(self):
        rng = np.random.default_rng(42)
        n = 200
        common = np.cumsum(rng.normal(0, 1, n))
        prices = {
            "BTC/USDT": (common + rng.normal(0, 0.5, n) + 50000).tolist(),
            "ETH/USDT": (0.8 * common + rng.normal(0, 0.5, n) + 3000).tolist(),
        }
        provider = _make_data_provider(prices=prices)
        exp = CointegrationExperiment(
            data_provider=provider,
            stat_arb_pairs=[["BTC/USDT", "ETH/USDT"]],
        )
        report = exp.run_experiment()
        assert report.results["data_source"] == "real"
        assert "BTC/USDT-ETH/USDT" in report.results

    def test_fallback_to_synthetic_when_no_provider(self):
        exp = CointegrationExperiment()
        report = exp.run_experiment()
        assert report.results["data_source"] == "synthetic"

    def test_fallback_when_insufficient_data(self):
        provider = _make_data_provider(
            prices={"BTC/USDT": [100.0] * 5, "ETH/USDT": [50.0] * 5}
        )
        exp = CointegrationExperiment(
            data_provider=provider,
            stat_arb_pairs=[["BTC/USDT", "ETH/USDT"]],
        )
        report = exp.run_experiment()
        assert report.results["data_source"] == "synthetic"

    def test_kwargs_override_provider(self):
        provider = _make_data_provider()
        exp = CointegrationExperiment(data_provider=provider)
        rng = np.random.default_rng(42)
        n = 200
        common = np.cumsum(rng.normal(0, 1, n))
        pairs = {
            "TEST-PAIR": (
                (common + rng.normal(0, 0.1, n)).tolist(),
                (0.9 * common + rng.normal(0, 0.1, n) + 5).tolist(),
            )
        }
        report = exp.run_experiment(pairs=pairs)
        assert report.results["data_source"] == "kwargs"

    def test_default_stat_arb_pairs_used(self):
        exp = CointegrationExperiment()
        assert exp._stat_arb_pairs == [
            ["BTC/USDT", "ETH/USDT"],
            ["SOL/USDT", "ETH/USDT"],
        ]

    def test_custom_stat_arb_pairs(self):
        exp = CointegrationExperiment(
            stat_arb_pairs=[["DOGE/USDT", "SHIB/USDT"]]
        )
        assert exp._stat_arb_pairs == [["DOGE/USDT", "SHIB/USDT"]]

    def test_provider_exception_falls_back(self):
        provider = MagicMock()
        provider.get_multi_prices = AsyncMock(side_effect=RuntimeError("DB error"))
        exp = CointegrationExperiment(data_provider=provider)
        report = exp.run_experiment()
        assert report.results["data_source"] == "synthetic"


# ------------------------------------------------------------------ #
# OptimalGridExperiment — real data
# ------------------------------------------------------------------ #


class TestOptimalGridRealData:
    def test_uses_real_data_when_provider_available(self):
        real_prices = list(
            np.cumsum(np.random.default_rng(10).normal(0, 0.01, 300)) + 50000
        )
        provider = _make_data_provider(prices={"BTC/USDT": real_prices})
        exp = OptimalGridExperiment(data_provider=provider, grid_symbols=["BTC/USDT"])
        report = exp.run_experiment()
        assert report.results["data_source"] == "real"

    def test_fallback_to_synthetic_when_no_provider(self):
        exp = OptimalGridExperiment()
        report = exp.run_experiment()
        assert report.results["data_source"] == "synthetic"

    def test_kwargs_override_provider(self):
        provider = _make_data_provider()
        exp = OptimalGridExperiment(data_provider=provider)
        custom_prices = [100.0 + i * 0.2 for i in range(200)]
        report = exp.run_experiment(prices=custom_prices)
        assert report.results["data_source"] == "kwargs"

    def test_uses_first_grid_symbol(self):
        provider = _make_data_provider(
            prices={"ETH/USDT": [100.0 + i * 0.1 for i in range(200)]}
        )
        exp = OptimalGridExperiment(
            data_provider=provider,
            grid_symbols=["ETH/USDT", "BTC/USDT"],
        )
        report = exp.run_experiment()
        assert report.results["data_source"] == "real"
        provider.get_prices.assert_called_once()
        call_args = provider.get_prices.call_args
        assert call_args[0][0] == "ETH/USDT"

    def test_provider_exception_falls_back(self):
        provider = MagicMock()
        provider.get_prices = AsyncMock(side_effect=RuntimeError("DB error"))
        exp = OptimalGridExperiment(data_provider=provider)
        report = exp.run_experiment()
        assert report.results["data_source"] == "synthetic"

    def test_default_grid_symbols(self):
        exp = OptimalGridExperiment()
        assert exp._grid_symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


# ------------------------------------------------------------------ #
# FundingPredictionExperiment — real data
# ------------------------------------------------------------------ #


class TestFundingPredictionRealData:
    def _make_funding_records(self, n: int = 90) -> list[dict]:
        """Generate mock funding rate records with timestamps."""
        rng = np.random.default_rng(42)
        records = []
        base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(n):
            hour = (i * 8) % 24  # 0, 8, 16 UTC cycle
            day = i // 3
            ts = base_ts.replace(day=1 + (day % 28), hour=hour)
            records.append({
                "timestamp": ts,
                "funding_rate": float(rng.normal(0.0003, 0.0001)),
                "mark_price": 50000.0 + rng.normal(0, 100),
                "spot_price": 49990.0 + rng.normal(0, 100),
            })
        return records

    def test_uses_real_data_when_provider_available(self):
        records = self._make_funding_records()
        provider = _make_data_provider(funding_rates=records)
        exp = FundingPredictionExperiment(data_provider=provider)
        report = exp.run_experiment()
        assert report.results["data_source"] == "real"
        assert "mean_rate" in report.results

    def test_real_data_includes_time_patterns(self):
        records = self._make_funding_records()
        provider = _make_data_provider(funding_rates=records)
        exp = FundingPredictionExperiment(data_provider=provider)
        report = exp.run_experiment()
        assert report.results["data_source"] == "real"
        assert "best_entry_hour" in report.results
        assert "avg_positive_rate" in report.results

    def test_real_data_includes_daily_pattern(self):
        records = self._make_funding_records()
        provider = _make_data_provider(funding_rates=records)
        exp = FundingPredictionExperiment(data_provider=provider)
        report = exp.run_experiment()
        assert "daily_pattern" in report.results

    def test_real_data_includes_hourly_pattern(self):
        records = self._make_funding_records()
        provider = _make_data_provider(funding_rates=records)
        exp = FundingPredictionExperiment(data_provider=provider)
        report = exp.run_experiment()
        assert "hourly_pattern" in report.results

    def test_fallback_to_synthetic_when_no_provider(self):
        exp = FundingPredictionExperiment()
        report = exp.run_experiment()
        assert report.results["data_source"] == "synthetic"

    def test_fallback_when_insufficient_data(self):
        # Only 5 records — below 24 threshold
        records = self._make_funding_records(5)
        provider = _make_data_provider(funding_rates=records)
        exp = FundingPredictionExperiment(data_provider=provider)
        report = exp.run_experiment()
        assert report.results["data_source"] == "synthetic"

    def test_kwargs_override_provider(self):
        records = self._make_funding_records()
        provider = _make_data_provider(funding_rates=records)
        exp = FundingPredictionExperiment(data_provider=provider)
        custom_rates = [0.0005] * 100
        report = exp.run_experiment(funding_rates=custom_rates)
        assert report.results["data_source"] == "kwargs"

    def test_provider_exception_falls_back(self):
        provider = MagicMock()
        provider.get_funding_rates = AsyncMock(side_effect=RuntimeError("DB error"))
        exp = FundingPredictionExperiment(data_provider=provider)
        report = exp.run_experiment()
        assert report.results["data_source"] == "synthetic"

    def test_timestamp_as_epoch_ms(self):
        """Test that integer timestamps (epoch ms) are handled."""
        rng = np.random.default_rng(42)
        records = []
        for i in range(30):
            records.append({
                "timestamp": 1704067200000 + i * 28800000,  # 8h intervals in ms
                "funding_rate": float(rng.normal(0.0003, 0.0001)),
                "mark_price": 50000.0,
                "spot_price": 49990.0,
            })
        provider = _make_data_provider(funding_rates=records)
        exp = FundingPredictionExperiment(data_provider=provider)
        report = exp.run_experiment()
        assert report.results["data_source"] == "real"
        assert "best_entry_hour" in report.results


# ------------------------------------------------------------------ #
# Backward compatibility
# ------------------------------------------------------------------ #


class TestBackwardCompatibility:
    """Ensure experiments still work exactly as before when no data_provider."""

    def test_volatility_regime_no_provider(self):
        exp = VolatilityRegimeExperiment()
        assert exp.data_provider is None
        report = exp.run_experiment()
        assert report.experiment_name == "volatility_regime"
        assert report.results["data_source"] == "synthetic"
        assert "fixed_sharpe" in report.results

    def test_cointegration_no_provider(self):
        exp = CointegrationExperiment()
        assert exp.data_provider is None
        report = exp.run_experiment()
        assert report.experiment_name == "cointegration"
        assert report.results["data_source"] == "synthetic"

    def test_optimal_grid_no_provider(self):
        exp = OptimalGridExperiment()
        assert exp.data_provider is None
        report = exp.run_experiment()
        assert report.experiment_name == "optimal_grid"
        assert report.results["data_source"] == "synthetic"

    def test_funding_prediction_no_provider(self):
        exp = FundingPredictionExperiment()
        assert exp.data_provider is None
        report = exp.run_experiment()
        assert report.experiment_name == "funding_prediction"
        assert report.results["data_source"] == "synthetic"

    def test_research_task_base_init_default(self):
        """ResearchTask stores data_provider from constructor."""

        class DummyTask(VolatilityRegimeExperiment):
            pass

        task = DummyTask()
        assert task.data_provider is None

        mock_provider = MagicMock()
        task2 = DummyTask(data_provider=mock_provider)
        assert task2.data_provider is mock_provider
