"""Tests for per-strategy performance tracking and auto-disable."""

import time
from unittest.mock import MagicMock

import pytest

from bot.monitoring.strategy_tracker import StrategyStats, StrategyTracker
from bot.strategies.base import StrategyRegistry
from bot.strategies.regime import MarketRegime


class TestStrategyStats:
    def test_initial_state(self):
        stats = StrategyStats()
        assert stats.total_trades == 0
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.total_pnl == 0.0
        assert stats.consecutive_losses == 0
        assert stats.win_rate == 0.0
        assert stats.avg_pnl == 0.0
        assert stats.sharpe_ratio == 0.0
        assert stats.profit_factor == 0.0
        assert stats.disabled is False
        assert stats.disabled_at is None
        assert stats.disabled_regime is None

    def test_win_rate_calculation(self):
        stats = StrategyStats()
        stats.total_trades = 10
        stats.wins = 7
        stats.losses = 3
        assert stats.win_rate == 70.0

    def test_avg_pnl(self):
        stats = StrategyStats()
        stats.total_trades = 4
        stats.total_pnl = 200.0
        assert stats.avg_pnl == 50.0

    def test_sharpe_ratio(self):
        stats = StrategyStats()
        stats.trade_pnls = [10.0, 20.0, 15.0, 5.0, 25.0]
        sharpe = stats.sharpe_ratio
        assert sharpe > 0  # All positive PnLs

    def test_sharpe_ratio_insufficient_data(self):
        stats = StrategyStats()
        stats.trade_pnls = [10.0]
        assert stats.sharpe_ratio == 0.0

    def test_profit_factor(self):
        stats = StrategyStats()
        stats.trade_pnls = [100.0, -50.0, 75.0, -25.0]
        # gross_profit = 175, gross_loss = 75
        assert stats.profit_factor == pytest.approx(175.0 / 75.0)

    def test_profit_factor_no_losses(self):
        stats = StrategyStats()
        stats.trade_pnls = [10.0, 20.0]
        assert stats.profit_factor == 0.0

    def test_to_dict(self):
        stats = StrategyStats()
        stats.total_trades = 5
        stats.wins = 3
        stats.losses = 2
        stats.total_pnl = 150.0
        stats.consecutive_losses = 1
        stats.trade_pnls = [50.0, -20.0, 60.0, -10.0, 70.0]
        d = stats.to_dict()
        assert d["total_trades"] == 5
        assert d["wins"] == 3
        assert d["losses"] == 2
        assert d["total_pnl"] == 150.0
        assert d["win_rate"] == 60.0
        assert d["avg_pnl"] == 30.0
        assert d["consecutive_losses"] == 1
        assert d["disabled"] is False
        assert "sharpe_ratio" in d
        assert "profit_factor" in d


class TestStrategyTracker:
    def test_init_defaults(self):
        tracker = StrategyTracker()
        assert tracker.max_consecutive_losses == 5
        assert tracker.min_win_rate_pct == 40.0
        assert tracker.min_trades_for_evaluation == 20
        assert tracker.re_enable_check_hours == 24.0

    def test_init_custom(self):
        tracker = StrategyTracker(
            max_consecutive_losses=3,
            min_win_rate_pct=50.0,
            min_trades_for_evaluation=10,
            re_enable_check_hours=12.0,
        )
        assert tracker.max_consecutive_losses == 3
        assert tracker.min_win_rate_pct == 50.0
        assert tracker.min_trades_for_evaluation == 10
        assert tracker.re_enable_check_hours == 12.0

    def test_record_winning_trade(self):
        tracker = StrategyTracker()
        tracker.record_trade("rsi", 100.0)
        stats = tracker.get_stats("rsi")
        assert stats.total_trades == 1
        assert stats.wins == 1
        assert stats.losses == 0
        assert stats.total_pnl == 100.0
        assert stats.consecutive_losses == 0

    def test_record_losing_trade(self):
        tracker = StrategyTracker()
        tracker.record_trade("rsi", -50.0)
        stats = tracker.get_stats("rsi")
        assert stats.total_trades == 1
        assert stats.wins == 0
        assert stats.losses == 1
        assert stats.total_pnl == -50.0
        assert stats.consecutive_losses == 1

    def test_consecutive_losses_reset_on_win(self):
        tracker = StrategyTracker()
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -20.0)
        tracker.record_trade("rsi", -15.0)
        assert tracker.get_stats("rsi").consecutive_losses == 3
        tracker.record_trade("rsi", 50.0)
        assert tracker.get_stats("rsi").consecutive_losses == 0

    def test_multiple_strategies_tracked_separately(self):
        tracker = StrategyTracker()
        tracker.record_trade("rsi", 100.0)
        tracker.record_trade("macd", -50.0)
        tracker.record_trade("rsi", -20.0)

        rsi_stats = tracker.get_stats("rsi")
        macd_stats = tracker.get_stats("macd")
        assert rsi_stats.total_trades == 2
        assert macd_stats.total_trades == 1
        assert rsi_stats.total_pnl == 80.0
        assert macd_stats.total_pnl == -50.0

    def test_get_all_stats(self):
        tracker = StrategyTracker()
        tracker.record_trade("rsi", 100.0)
        tracker.record_trade("macd", -50.0)
        all_stats = tracker.get_all_stats()
        assert "rsi" in all_stats
        assert "macd" in all_stats
        assert all_stats["rsi"]["total_trades"] == 1
        assert all_stats["macd"]["total_trades"] == 1

    def test_get_all_stats_empty(self):
        tracker = StrategyTracker()
        assert tracker.get_all_stats() == {}


class TestAutoDisable:
    def test_disable_on_consecutive_losses(self):
        registry = StrategyRegistry.__new__(StrategyRegistry)
        registry._strategies = {"rsi": MagicMock()}
        registry._active = {"rsi"}

        tracker = StrategyTracker(
            max_consecutive_losses=3,
            registry=registry,
        )

        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -10.0)
        assert tracker.get_stats("rsi").disabled is False

        tracker.record_trade("rsi", -10.0)  # 3rd consecutive loss
        stats = tracker.get_stats("rsi")
        assert stats.disabled is True
        assert stats.disabled_at is not None
        assert "rsi" not in registry._active

    def test_disable_on_low_win_rate(self):
        registry = StrategyRegistry.__new__(StrategyRegistry)
        registry._strategies = {"macd": MagicMock()}
        registry._active = {"macd"}

        tracker = StrategyTracker(
            min_win_rate_pct=40.0,
            min_trades_for_evaluation=5,
            max_consecutive_losses=100,  # High so it doesn't trigger
            registry=registry,
        )

        # 1 win, 4 losses = 20% win rate after 5 trades
        tracker.record_trade("macd", 50.0)  # win
        tracker.record_trade("macd", -10.0)
        tracker.record_trade("macd", -10.0)
        tracker.record_trade("macd", -10.0)
        assert tracker.get_stats("macd").disabled is False

        tracker.record_trade("macd", -10.0)  # 5th trade, win_rate = 20%
        stats = tracker.get_stats("macd")
        assert stats.disabled is True
        assert "macd" not in registry._active

    def test_no_disable_with_good_win_rate(self):
        registry = StrategyRegistry.__new__(StrategyRegistry)
        registry._strategies = {"rsi": MagicMock()}
        registry._active = {"rsi"}

        tracker = StrategyTracker(
            min_win_rate_pct=40.0,
            min_trades_for_evaluation=5,
            max_consecutive_losses=100,
            registry=registry,
        )

        # 3 wins, 2 losses = 60% win rate
        tracker.record_trade("rsi", 50.0)
        tracker.record_trade("rsi", 50.0)
        tracker.record_trade("rsi", 50.0)
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -10.0)
        assert tracker.get_stats("rsi").disabled is False

    def test_no_disable_before_min_trades(self):
        tracker = StrategyTracker(
            min_win_rate_pct=40.0,
            min_trades_for_evaluation=10,
            max_consecutive_losses=100,
        )

        # 0 wins, 5 losses = 0% win rate, but < 10 trades
        for _ in range(5):
            tracker.record_trade("rsi", -10.0)
        assert tracker.get_stats("rsi").disabled is False

    def test_already_disabled_not_re_checked(self):
        tracker = StrategyTracker(max_consecutive_losses=2)
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -10.0)  # disabled at 2
        stats = tracker.get_stats("rsi")
        assert stats.disabled is True

        # More losses shouldn't change disabled_at
        first_disabled_at = stats.disabled_at
        tracker.record_trade("rsi", -10.0)
        assert stats.disabled_at == first_disabled_at

    def test_disable_without_registry(self):
        tracker = StrategyTracker(max_consecutive_losses=2)
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -10.0)
        stats = tracker.get_stats("rsi")
        assert stats.disabled is True
        # No crash even without registry


class TestReEnable:
    def test_re_enable_on_regime_change(self):
        registry = StrategyRegistry.__new__(StrategyRegistry)
        registry._strategies = {"rsi": MagicMock()}
        registry._active = {"rsi"}

        tracker = StrategyTracker(
            max_consecutive_losses=2,
            registry=registry,
        )

        # Disable via consecutive losses in RANGING regime
        tracker.update_regime(MarketRegime.RANGING)
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -10.0)
        assert tracker.get_stats("rsi").disabled is True
        assert "rsi" not in registry._active

        # Regime changes to TRENDING_UP → re-enable
        tracker.update_regime(MarketRegime.TRENDING_UP)
        stats = tracker.get_stats("rsi")
        assert stats.disabled is False
        assert stats.consecutive_losses == 0
        assert "rsi" in registry._active

    def test_no_re_enable_same_regime(self):
        registry = StrategyRegistry.__new__(StrategyRegistry)
        registry._strategies = {"rsi": MagicMock()}
        registry._active = {"rsi"}

        tracker = StrategyTracker(
            max_consecutive_losses=2,
            registry=registry,
        )

        tracker.update_regime(MarketRegime.RANGING)
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -10.0)
        assert tracker.get_stats("rsi").disabled is True

        # Same regime update → no re-enable
        tracker.update_regime(MarketRegime.RANGING)
        assert tracker.get_stats("rsi").disabled is True

    def test_time_based_re_enable_with_regime_change(self):
        registry = StrategyRegistry.__new__(StrategyRegistry)
        registry._strategies = {"rsi": MagicMock()}
        registry._active = {"rsi"}

        tracker = StrategyTracker(
            max_consecutive_losses=2,
            re_enable_check_hours=1.0,
            registry=registry,
        )

        tracker.update_regime(MarketRegime.RANGING)
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -10.0)
        assert tracker.get_stats("rsi").disabled is True

        # Simulate time passing and regime change
        stats = tracker.get_stats("rsi")
        stats.disabled_at = time.time() - 3700  # 1+ hour ago
        tracker.update_regime(MarketRegime.TRENDING_UP)  # New regime
        # Should have been re-enabled by regime change already
        assert stats.disabled is False

    def test_time_based_check_deferred_same_regime(self):
        registry = StrategyRegistry.__new__(StrategyRegistry)
        registry._strategies = {"rsi": MagicMock()}
        registry._active = {"rsi"}

        tracker = StrategyTracker(
            max_consecutive_losses=2,
            re_enable_check_hours=1.0,
            registry=registry,
        )

        tracker.update_regime(MarketRegime.RANGING)
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -10.0)
        assert tracker.get_stats("rsi").disabled is True

        # Simulate time passing but same regime
        stats = tracker.get_stats("rsi")
        stats.disabled_at = time.time() - 3700  # 1+ hour ago

        tracker.check_re_enable()
        # Same regime → deferred, not re-enabled
        assert stats.disabled is True
        # disabled_at should be reset (timer deferred)
        assert time.time() - stats.disabled_at < 5

    def test_check_re_enable_not_enough_time(self):
        tracker = StrategyTracker(
            max_consecutive_losses=2,
            re_enable_check_hours=24.0,
        )

        tracker.update_regime(MarketRegime.RANGING)
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -10.0)
        stats = tracker.get_stats("rsi")
        assert stats.disabled is True

        # Not enough time has passed
        tracker.check_re_enable()
        assert stats.disabled is True

    def test_re_enable_resets_consecutive_losses(self):
        registry = StrategyRegistry.__new__(StrategyRegistry)
        registry._strategies = {"rsi": MagicMock()}
        registry._active = {"rsi"}

        tracker = StrategyTracker(
            max_consecutive_losses=2,
            registry=registry,
        )

        tracker.update_regime(MarketRegime.RANGING)
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -10.0)
        assert tracker.get_stats("rsi").consecutive_losses == 2

        tracker.update_regime(MarketRegime.TRENDING_UP)
        stats = tracker.get_stats("rsi")
        assert stats.consecutive_losses == 0
        assert stats.disabled_at is None
        assert stats.disabled_regime is None

    def test_re_enable_without_registry(self):
        tracker = StrategyTracker(max_consecutive_losses=2)
        tracker.update_regime(MarketRegime.RANGING)
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -10.0)

        tracker.update_regime(MarketRegime.TRENDING_UP)
        assert tracker.get_stats("rsi").disabled is False
        # No crash without registry

    def test_set_registry(self):
        tracker = StrategyTracker()
        registry = MagicMock()
        tracker.set_registry(registry)
        assert tracker._registry is registry


class TestUpdateRegime:
    def test_update_regime(self):
        tracker = StrategyTracker()
        tracker.update_regime(MarketRegime.TRENDING_UP)
        assert tracker._current_regime == MarketRegime.TRENDING_UP

    def test_regime_change_triggers_re_enable_check(self):
        registry = StrategyRegistry.__new__(StrategyRegistry)
        registry._strategies = {"rsi": MagicMock()}
        registry._active = {"rsi"}

        tracker = StrategyTracker(
            max_consecutive_losses=2,
            registry=registry,
        )

        tracker.update_regime(MarketRegime.HIGH_VOLATILITY)
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -10.0)
        assert tracker.get_stats("rsi").disabled is True

        # Different regime → re-enable triggered
        tracker.update_regime(MarketRegime.TRENDING_DOWN)
        assert tracker.get_stats("rsi").disabled is False

    def test_multiple_strategies_re_enabled(self):
        registry = StrategyRegistry.__new__(StrategyRegistry)
        registry._strategies = {"rsi": MagicMock(), "macd": MagicMock()}
        registry._active = {"rsi", "macd"}

        tracker = StrategyTracker(
            max_consecutive_losses=2,
            registry=registry,
        )

        tracker.update_regime(MarketRegime.RANGING)
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("rsi", -10.0)
        tracker.record_trade("macd", -10.0)
        tracker.record_trade("macd", -10.0)
        assert tracker.get_stats("rsi").disabled is True
        assert tracker.get_stats("macd").disabled is True

        tracker.update_regime(MarketRegime.TRENDING_UP)
        assert tracker.get_stats("rsi").disabled is False
        assert tracker.get_stats("macd").disabled is False


class TestStrategyStatsEdgeCases:
    def test_zero_pnl_is_loss(self):
        tracker = StrategyTracker()
        tracker.record_trade("rsi", 0.0)
        stats = tracker.get_stats("rsi")
        assert stats.losses == 1
        assert stats.wins == 0
        assert stats.consecutive_losses == 1

    def test_sharpe_ratio_all_same_pnl(self):
        stats = StrategyStats()
        stats.trade_pnls = [10.0, 10.0, 10.0]
        # std_dev = 0, so sharpe = 0
        assert stats.sharpe_ratio == 0.0

    def test_profit_factor_all_losses(self):
        stats = StrategyStats()
        stats.trade_pnls = [-10.0, -20.0, -5.0]
        # gross_profit = 0, gross_loss = 35
        assert stats.profit_factor == 0.0

    def test_disabled_regime_stored(self):
        tracker = StrategyTracker(max_consecutive_losses=1)
        tracker.update_regime(MarketRegime.HIGH_VOLATILITY)
        tracker.record_trade("rsi", -10.0)
        stats = tracker.get_stats("rsi")
        assert stats.disabled is True
        assert stats.disabled_regime == MarketRegime.HIGH_VOLATILITY
