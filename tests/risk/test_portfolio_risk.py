"""Tests for PortfolioRiskManager: correlation, exposure limits, heat map."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from bot.models import OHLCV
from bot.risk.portfolio_risk import PortfolioRiskManager


def make_candles(
    closes: list[float],
    base_time: datetime | None = None,
    symbol: str = "BTC/USDT",
) -> list[OHLCV]:
    """Create OHLCV candles from a list of close prices."""
    if base_time is None:
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i, close in enumerate(closes):
        high = close * 1.01
        low = close * 0.99
        candles.append(
            OHLCV(
                symbol=symbol,
                timeframe="1h",
                timestamp=base_time + timedelta(hours=i),
                open=close,
                high=high,
                low=low,
                close=close,
                volume=1000.0,
            )
        )
    return candles


class TestExposureLimit:
    def test_no_positions_allows_new(self):
        mgr = PortfolioRiskManager(max_total_exposure_pct=60.0)
        mgr.update_portfolio_value(10000.0)
        allowed, reason = mgr.check_exposure_limit(3000.0)
        assert allowed
        assert reason == ""

    def test_within_limit_allows(self):
        mgr = PortfolioRiskManager(max_total_exposure_pct=60.0)
        mgr.update_portfolio_value(10000.0)
        mgr.add_position("BTC/USDT", 2000.0)
        # 2000 + 3000 = 5000 = 50% < 60%
        allowed, reason = mgr.check_exposure_limit(3000.0)
        assert allowed

    def test_exceeds_limit_blocks(self):
        mgr = PortfolioRiskManager(max_total_exposure_pct=60.0)
        mgr.update_portfolio_value(10000.0)
        mgr.add_position("BTC/USDT", 4000.0)
        # 4000 + 3000 = 7000 = 70% > 60%
        allowed, reason = mgr.check_exposure_limit(3000.0)
        assert not allowed
        assert "total_exposure" in reason
        assert "70.0%" in reason

    def test_exactly_at_limit_blocks(self):
        mgr = PortfolioRiskManager(max_total_exposure_pct=60.0)
        mgr.update_portfolio_value(10000.0)
        mgr.add_position("BTC/USDT", 3000.0)
        # 3000 + 3001 = 6001 > 60%
        allowed, _ = mgr.check_exposure_limit(3001.0)
        assert not allowed

    def test_zero_portfolio_allows(self):
        mgr = PortfolioRiskManager(max_total_exposure_pct=60.0)
        mgr.update_portfolio_value(0)
        allowed, _ = mgr.check_exposure_limit(1000.0)
        assert allowed

    def test_get_total_exposure(self):
        mgr = PortfolioRiskManager()
        mgr.update_portfolio_value(10000.0)
        mgr.add_position("BTC/USDT", 2000.0)
        mgr.add_position("ETH/USDT", 1500.0)
        assert mgr.get_total_exposure() == pytest.approx(35.0)


class TestCorrelation:
    def test_perfect_positive_correlation(self):
        mgr = PortfolioRiskManager(correlation_window=30)
        # Both move up in lockstep
        prices_a = [100 + i for i in range(35)]
        prices_b = [50 + i * 0.5 for i in range(35)]
        candles_a = make_candles(prices_a, symbol="A/USDT")
        candles_b = make_candles(prices_b, symbol="B/USDT")
        mgr.update_price_history("A/USDT", candles_a)
        mgr.update_price_history("B/USDT", candles_b)

        corr = mgr.calculate_correlation("A/USDT", "B/USDT")
        assert corr is not None
        assert corr > 0.95

    def test_negative_correlation(self):
        mgr = PortfolioRiskManager(correlation_window=30)
        # A alternates up/down, B inverts A's pattern
        np.random.seed(123)
        prices_a = []
        prices_b = []
        base_a = 100.0
        base_b = 100.0
        for _ in range(35):
            change = np.random.normal(0, 3)
            base_a += change
            base_b -= change  # opposite direction
            prices_a.append(max(base_a, 10.0))
            prices_b.append(max(base_b, 10.0))
        candles_a = make_candles(prices_a, symbol="A/USDT")
        candles_b = make_candles(prices_b, symbol="B/USDT")
        mgr.update_price_history("A/USDT", candles_a)
        mgr.update_price_history("B/USDT", candles_b)

        corr = mgr.calculate_correlation("A/USDT", "B/USDT")
        assert corr is not None
        assert corr < -0.9

    def test_no_history_returns_none(self):
        mgr = PortfolioRiskManager()
        corr = mgr.calculate_correlation("A/USDT", "B/USDT")
        assert corr is None

    def test_insufficient_data_returns_none(self):
        mgr = PortfolioRiskManager()
        # Only 3 candles -> 2 returns, need at least 5
        candles_a = make_candles([100, 101, 102], symbol="A/USDT")
        candles_b = make_candles([50, 51, 52], symbol="B/USDT")
        mgr.update_price_history("A/USDT", candles_a)
        mgr.update_price_history("B/USDT", candles_b)

        corr = mgr.calculate_correlation("A/USDT", "B/USDT")
        assert corr is None

    def test_high_correlation_blocks_new_position(self):
        mgr = PortfolioRiskManager(max_correlation=0.8)
        # Create highly correlated price histories
        prices_a = [100 + i * 2 for i in range(35)]
        prices_b = [50 + i for i in range(35)]
        candles_a = make_candles(prices_a, symbol="BTC/USDT")
        candles_b = make_candles(prices_b, symbol="ETH/USDT")
        mgr.update_price_history("BTC/USDT", candles_a)
        mgr.update_price_history("ETH/USDT", candles_b)

        mgr.add_position("BTC/USDT", 5000.0)

        allowed, reason = mgr.check_correlation("ETH/USDT")
        assert not allowed
        assert "high_correlation" in reason
        assert "BTC/USDT" in reason

    def test_low_correlation_allows(self):
        mgr = PortfolioRiskManager(max_correlation=0.8)
        # Create uncorrelated data with random-like patterns
        np.random.seed(42)
        base = 100.0
        prices_a = []
        prices_b = []
        for _ in range(35):
            base_a = base + np.random.normal(0, 5)
            base_b = base + np.random.normal(0, 5)
            prices_a.append(max(base_a, 10.0))
            prices_b.append(max(base_b, 10.0))

        candles_a = make_candles(prices_a, symbol="BTC/USDT")
        candles_b = make_candles(prices_b, symbol="DOGE/USDT")
        mgr.update_price_history("BTC/USDT", candles_a)
        mgr.update_price_history("DOGE/USDT", candles_b)

        mgr.add_position("BTC/USDT", 5000.0)

        allowed, reason = mgr.check_correlation("DOGE/USDT")
        assert allowed

    def test_no_existing_positions_allows(self):
        mgr = PortfolioRiskManager(max_correlation=0.8)
        allowed, reason = mgr.check_correlation("ETH/USDT")
        assert allowed

    def test_same_symbol_not_checked(self):
        mgr = PortfolioRiskManager(max_correlation=0.8)
        # Even with perfect self-correlation, same-symbol check skips
        mgr.add_position("BTC/USDT", 5000.0)
        prices = [100 + i for i in range(35)]
        candles = make_candles(prices, symbol="BTC/USDT")
        mgr.update_price_history("BTC/USDT", candles)

        allowed, _ = mgr.check_correlation("BTC/USDT")
        assert allowed

    def test_constant_returns_returns_none(self):
        mgr = PortfolioRiskManager()
        # All same price -> zero std -> no valid correlation
        candles_a = make_candles([100.0] * 35, symbol="A/USDT")
        candles_b = make_candles([50 + i for i in range(35)], symbol="B/USDT")
        mgr.update_price_history("A/USDT", candles_a)
        mgr.update_price_history("B/USDT", candles_b)

        corr = mgr.calculate_correlation("A/USDT", "B/USDT")
        assert corr is None


class TestSectorLimit:
    def test_no_sector_map_allows(self):
        mgr = PortfolioRiskManager(max_positions_per_sector=2)
        mgr.add_position("BTC/USDT", 5000.0)
        allowed, _ = mgr.check_sector_limit("ETH/USDT")
        assert allowed

    def test_within_sector_limit_allows(self):
        mgr = PortfolioRiskManager(
            max_positions_per_sector=3,
            sector_map={
                "BTC/USDT": "large_cap",
                "ETH/USDT": "large_cap",
                "SOL/USDT": "large_cap",
            },
        )
        mgr.add_position("BTC/USDT", 3000.0)
        mgr.add_position("ETH/USDT", 2000.0)
        # 2 positions in large_cap, limit is 3
        allowed, _ = mgr.check_sector_limit("SOL/USDT")
        assert allowed

    def test_at_sector_limit_blocks(self):
        mgr = PortfolioRiskManager(
            max_positions_per_sector=2,
            sector_map={
                "BTC/USDT": "large_cap",
                "ETH/USDT": "large_cap",
                "SOL/USDT": "large_cap",
            },
        )
        mgr.add_position("BTC/USDT", 3000.0)
        mgr.add_position("ETH/USDT", 2000.0)
        # 2 positions in large_cap, limit is 2
        allowed, reason = mgr.check_sector_limit("SOL/USDT")
        assert not allowed
        assert "large_cap" in reason

    def test_different_sector_allows(self):
        mgr = PortfolioRiskManager(
            max_positions_per_sector=2,
            sector_map={
                "BTC/USDT": "large_cap",
                "ETH/USDT": "large_cap",
                "DOGE/USDT": "meme",
            },
        )
        mgr.add_position("BTC/USDT", 3000.0)
        mgr.add_position("ETH/USDT", 2000.0)
        # DOGE is in "meme" sector, 0 positions
        allowed, _ = mgr.check_sector_limit("DOGE/USDT")
        assert allowed

    def test_unmapped_symbol_allowed(self):
        mgr = PortfolioRiskManager(
            max_positions_per_sector=1,
            sector_map={"BTC/USDT": "large_cap"},
        )
        mgr.add_position("BTC/USDT", 5000.0)
        # XRP not in sector_map -> allowed
        allowed, _ = mgr.check_sector_limit("XRP/USDT")
        assert allowed


class TestPortfolioHeat:
    def test_no_positions_zero_heat(self):
        mgr = PortfolioRiskManager()
        mgr.update_portfolio_value(10000.0)
        assert mgr.calculate_portfolio_heat() == 0.0

    def test_heat_calculation(self):
        mgr = PortfolioRiskManager()
        mgr.update_portfolio_value(10000.0)
        # position value=5000, atr_ratio=0.02
        # heat = 5000 * 0.02 / 10000 = 0.01
        mgr.add_position("BTC/USDT", 5000.0, atr=0.02)
        heat = mgr.calculate_portfolio_heat()
        assert heat == pytest.approx(0.01)

    def test_heat_multiple_positions(self):
        mgr = PortfolioRiskManager()
        mgr.update_portfolio_value(10000.0)
        mgr.add_position("BTC/USDT", 3000.0, atr=0.02)
        mgr.add_position("ETH/USDT", 2000.0, atr=0.03)
        # heat = (3000*0.02 + 2000*0.03) / 10000 = (60 + 60) / 10000 = 0.012
        heat = mgr.calculate_portfolio_heat()
        assert heat == pytest.approx(0.012)

    def test_position_without_atr_excluded(self):
        mgr = PortfolioRiskManager()
        mgr.update_portfolio_value(10000.0)
        mgr.add_position("BTC/USDT", 5000.0, atr=0.02)
        mgr.add_position("ETH/USDT", 3000.0)  # no ATR
        # Only BTC contributes: 5000 * 0.02 / 10000 = 0.01
        heat = mgr.calculate_portfolio_heat()
        assert heat == pytest.approx(0.01)

    def test_heat_exceeds_limit_blocks(self):
        mgr = PortfolioRiskManager(max_portfolio_heat=0.05)
        mgr.update_portfolio_value(10000.0)
        mgr.add_position("BTC/USDT", 5000.0, atr=0.03)
        # Current heat: 5000 * 0.03 / 10000 = 0.015
        # New: 4000 * 0.04 / 10000 = 0.016
        # Total: 0.031 → still under 0.05
        allowed, _ = mgr.check_portfolio_heat(4000.0, 0.04)
        assert allowed

        # Add more heat to exceed
        mgr.add_position("ETH/USDT", 5000.0, atr=0.05)
        # Current: 0.015 + 5000*0.05/10000 = 0.015 + 0.025 = 0.04
        # New: 3000 * 0.04 / 10000 = 0.012
        # Total: 0.04 + 0.012 = 0.052 > 0.05
        allowed, reason = mgr.check_portfolio_heat(3000.0, 0.04)
        assert not allowed
        assert "portfolio_heat" in reason

    def test_heat_without_atr_allows(self):
        mgr = PortfolioRiskManager(max_portfolio_heat=0.01)
        mgr.update_portfolio_value(10000.0)
        allowed, _ = mgr.check_portfolio_heat(5000.0, None)
        assert allowed


class TestValidateNewPosition:
    def test_all_checks_pass(self):
        mgr = PortfolioRiskManager(
            max_total_exposure_pct=80.0,
            max_correlation=0.8,
            max_positions_per_sector=5,
            max_portfolio_heat=1.0,
        )
        mgr.update_portfolio_value(10000.0)
        allowed, reason = mgr.validate_new_position("BTC/USDT", 3000.0)
        assert allowed
        assert reason == ""

    def test_exposure_blocks_first(self):
        mgr = PortfolioRiskManager(max_total_exposure_pct=30.0)
        mgr.update_portfolio_value(10000.0)
        mgr.add_position("ETH/USDT", 2500.0)
        # 2500 + 1000 = 3500 = 35% > 30%
        allowed, reason = mgr.validate_new_position("BTC/USDT", 1000.0)
        assert not allowed
        assert "total_exposure" in reason

    def test_correlation_blocks(self):
        mgr = PortfolioRiskManager(
            max_total_exposure_pct=100.0,
            max_correlation=0.8,
        )
        mgr.update_portfolio_value(10000.0)
        # Create correlated histories
        prices_a = [100 + i * 2 for i in range(35)]
        prices_b = [50 + i for i in range(35)]
        candles_a = make_candles(prices_a, symbol="BTC/USDT")
        candles_b = make_candles(prices_b, symbol="ETH/USDT")
        mgr.update_price_history("BTC/USDT", candles_a)
        mgr.update_price_history("ETH/USDT", candles_b)

        mgr.add_position("BTC/USDT", 3000.0)
        allowed, reason = mgr.validate_new_position("ETH/USDT", 2000.0)
        assert not allowed
        assert "high_correlation" in reason

    def test_sector_blocks(self):
        mgr = PortfolioRiskManager(
            max_total_exposure_pct=100.0,
            max_positions_per_sector=1,
            sector_map={
                "BTC/USDT": "large_cap",
                "ETH/USDT": "large_cap",
            },
        )
        mgr.update_portfolio_value(10000.0)
        mgr.add_position("BTC/USDT", 3000.0)
        allowed, reason = mgr.validate_new_position("ETH/USDT", 2000.0)
        assert not allowed
        assert "large_cap" in reason

    def test_heat_blocks(self):
        mgr = PortfolioRiskManager(
            max_total_exposure_pct=100.0,
            max_portfolio_heat=0.01,
        )
        mgr.update_portfolio_value(10000.0)
        mgr.add_position("BTC/USDT", 5000.0, atr=0.02)
        # Current heat: 0.01
        # New: 5000 * 0.02 / 10000 = 0.01
        # Total: 0.02 > 0.01
        allowed, reason = mgr.validate_new_position(
            "ETH/USDT", 5000.0, atr=0.02
        )
        assert not allowed
        assert "portfolio_heat" in reason


class TestPositionManagement:
    def test_add_and_remove_position(self):
        mgr = PortfolioRiskManager()
        mgr.add_position("BTC/USDT", 5000.0, atr=0.02)
        assert "BTC/USDT" in mgr.positions
        assert mgr.positions["BTC/USDT"]["value"] == 5000.0
        assert mgr.positions["BTC/USDT"]["atr"] == 0.02

        mgr.remove_position("BTC/USDT")
        assert "BTC/USDT" not in mgr.positions

    def test_remove_nonexistent_position_no_error(self):
        mgr = PortfolioRiskManager()
        mgr.remove_position("NONEXIST/USDT")  # Should not raise

    def test_update_position_value(self):
        mgr = PortfolioRiskManager()
        mgr.add_position("BTC/USDT", 5000.0)
        mgr.update_position_value("BTC/USDT", 6000.0)
        assert mgr.positions["BTC/USDT"]["value"] == 6000.0

    def test_update_nonexistent_position_no_error(self):
        mgr = PortfolioRiskManager()
        mgr.update_position_value("NONEXIST/USDT", 1000.0)  # no-op

    def test_portfolio_value_property(self):
        mgr = PortfolioRiskManager()
        assert mgr.portfolio_value == 0.0
        mgr.update_portfolio_value(25000.0)
        assert mgr.portfolio_value == 25000.0


class TestPriceHistory:
    def test_update_price_history_stores_returns(self):
        mgr = PortfolioRiskManager(correlation_window=10)
        prices = [100.0, 102.0, 101.0, 103.0, 105.0]
        candles = make_candles(prices, symbol="BTC/USDT")
        mgr.update_price_history("BTC/USDT", candles)
        # 4 returns from 5 prices
        assert len(mgr._price_history["BTC/USDT"]) == 4

    def test_window_limit(self):
        mgr = PortfolioRiskManager(correlation_window=5)
        prices = [100 + i for i in range(20)]
        candles = make_candles(prices, symbol="BTC/USDT")
        mgr.update_price_history("BTC/USDT", candles)
        # Should be capped at 5
        assert len(mgr._price_history["BTC/USDT"]) == 5

    def test_single_candle_no_history(self):
        mgr = PortfolioRiskManager()
        candles = make_candles([100.0], symbol="BTC/USDT")
        mgr.update_price_history("BTC/USDT", candles)
        assert "BTC/USDT" not in mgr._price_history


class TestEdgeCases:
    def test_config_defaults(self):
        mgr = PortfolioRiskManager()
        assert mgr._max_total_exposure_pct == 60.0
        assert mgr._max_correlation == 0.8
        assert mgr._correlation_window == 30
        assert mgr._max_positions_per_sector == 3
        assert mgr._max_portfolio_heat == 0.15

    def test_custom_config(self):
        mgr = PortfolioRiskManager(
            max_total_exposure_pct=50.0,
            max_correlation=0.7,
            correlation_window=20,
            max_positions_per_sector=2,
            max_portfolio_heat=0.10,
            sector_map={"BTC/USDT": "crypto"},
        )
        assert mgr._max_total_exposure_pct == 50.0
        assert mgr._max_correlation == 0.7
        assert mgr._correlation_window == 20
        assert mgr._max_positions_per_sector == 2
        assert mgr._max_portfolio_heat == 0.10
        assert mgr._sector_map == {"BTC/USDT": "crypto"}
