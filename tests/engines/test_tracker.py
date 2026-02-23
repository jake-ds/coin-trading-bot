"""Tests for EngineTracker — V5-005 performance tracking and metrics."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from bot.engines.tracker import EngineMetrics, EngineTracker, TradeRecord

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _past_iso(hours: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.isoformat()


def _make_trade(
    engine: str = "test_engine",
    symbol: str = "BTC/USDT",
    net_pnl: float = 10.0,
    cost: float = 1.0,
    hold_seconds: float = 3600.0,
    exit_time: str | None = None,
) -> TradeRecord:
    gross = net_pnl + cost
    return TradeRecord(
        engine_name=engine,
        symbol=symbol,
        side="buy",
        entry_price=50000.0,
        exit_price=50010.0,
        quantity=0.01,
        pnl=gross,
        cost=cost,
        net_pnl=net_pnl,
        entry_time=_past_iso(1),
        exit_time=exit_time or _now_iso(),
        hold_time_seconds=hold_seconds,
    )


# ------------------------------------------------------------------ #
# TradeRecord dataclass
# ------------------------------------------------------------------ #

class TestTradeRecord:
    def test_fields(self):
        t = _make_trade()
        assert t.engine_name == "test_engine"
        assert t.symbol == "BTC/USDT"
        assert t.cost == 1.0
        assert t.net_pnl == 10.0
        assert t.pnl == 11.0  # gross

    def test_hold_time_default(self):
        t = TradeRecord(
            engine_name="e", symbol="X", side="buy",
            entry_price=1, exit_price=2, quantity=1,
            pnl=1, cost=0, net_pnl=1,
            entry_time="", exit_time="",
        )
        assert t.hold_time_seconds == 0.0


# ------------------------------------------------------------------ #
# EngineMetrics dataclass
# ------------------------------------------------------------------ #

class TestEngineMetrics:
    def test_defaults_are_zero(self):
        m = EngineMetrics()
        assert m.total_trades == 0
        assert m.sharpe_ratio == 0.0
        assert m.max_drawdown == 0.0

    def test_to_dict(self):
        m = EngineMetrics(total_trades=5, win_rate=0.6)
        d = m.to_dict()
        assert d["total_trades"] == 5
        assert d["win_rate"] == 0.6


# ------------------------------------------------------------------ #
# EngineTracker — recording
# ------------------------------------------------------------------ #

class TestRecording:
    def test_record_trade(self):
        tracker = EngineTracker()
        trade = _make_trade()
        tracker.record_trade("test_engine", trade)
        assert len(tracker._trades["test_engine"]) == 1

    def test_record_multiple_engines(self):
        tracker = EngineTracker()
        tracker.record_trade("engine_a", _make_trade(engine="engine_a"))
        tracker.record_trade("engine_b", _make_trade(engine="engine_b"))
        assert "engine_a" in tracker._trades
        assert "engine_b" in tracker._trades

    def test_record_cycle(self):
        tracker = EngineTracker()

        class FakeResult:
            cycle_num = 1
            timestamp = _now_iso()
            pnl_update = 5.0
            actions_taken = [{"action": "buy"}]
            duration_ms = 150.0

        tracker.record_cycle("test_engine", FakeResult())
        assert len(tracker._cycles["test_engine"]) == 1
        assert tracker._cycles["test_engine"][0]["pnl_update"] == 5.0


# ------------------------------------------------------------------ #
# EngineTracker — metrics with known trades
# ------------------------------------------------------------------ #

class TestMetricsCalculation:
    def test_empty_returns_zero_metrics(self):
        tracker = EngineTracker()
        m = tracker.get_metrics("nonexistent")
        assert m.total_trades == 0
        assert m.sharpe_ratio == 0.0

    def test_single_winning_trade(self):
        tracker = EngineTracker()
        tracker.record_trade("e", _make_trade(net_pnl=10.0, cost=1.0))
        m = tracker.get_metrics("e")
        assert m.total_trades == 1
        assert m.winning_trades == 1
        assert m.losing_trades == 0
        assert m.win_rate == 1.0
        assert m.total_pnl == 10.0
        assert m.total_cost == 1.0
        # Single trade: sharpe = 0 (need >= 2)
        assert m.sharpe_ratio == 0.0

    def test_single_losing_trade(self):
        tracker = EngineTracker()
        tracker.record_trade("e", _make_trade(net_pnl=-5.0, cost=1.0))
        m = tracker.get_metrics("e")
        assert m.total_trades == 1
        assert m.winning_trades == 0
        assert m.losing_trades == 1
        assert m.win_rate == 0.0

    def test_win_rate_calculation(self):
        tracker = EngineTracker()
        # 3 wins, 2 losses
        for pnl in [10, 5, 15, -3, -7]:
            tracker.record_trade("e", _make_trade(net_pnl=pnl))
        m = tracker.get_metrics("e")
        assert m.total_trades == 5
        assert m.winning_trades == 3
        assert m.losing_trades == 2
        assert m.win_rate == pytest.approx(0.6)

    def test_profit_factor(self):
        tracker = EngineTracker()
        # Wins: 10 + 20 = 30, Losses: |-5| + |-10| = 15
        for pnl in [10, -5, 20, -10]:
            tracker.record_trade("e", _make_trade(net_pnl=pnl))
        m = tracker.get_metrics("e")
        assert m.profit_factor == pytest.approx(30.0 / 15.0)

    def test_profit_factor_no_losses(self):
        tracker = EngineTracker()
        tracker.record_trade("e", _make_trade(net_pnl=10.0))
        tracker.record_trade("e", _make_trade(net_pnl=5.0))
        m = tracker.get_metrics("e")
        assert m.profit_factor == 0.0  # 0 when no losses (div by zero guard)

    def test_best_worst_trade(self):
        tracker = EngineTracker()
        for pnl in [10, -5, 20, -3]:
            tracker.record_trade("e", _make_trade(net_pnl=pnl))
        m = tracker.get_metrics("e")
        assert m.best_trade == 20.0
        assert m.worst_trade == -5.0

    def test_avg_profit_per_trade(self):
        tracker = EngineTracker()
        for pnl in [10, -2, 8]:
            tracker.record_trade("e", _make_trade(net_pnl=pnl))
        m = tracker.get_metrics("e")
        assert m.avg_profit_per_trade == pytest.approx(16.0 / 3)

    def test_avg_hold_time(self):
        tracker = EngineTracker()
        tracker.record_trade("e", _make_trade(hold_seconds=600))
        tracker.record_trade("e", _make_trade(hold_seconds=1200))
        m = tracker.get_metrics("e")
        assert m.avg_hold_time_min == pytest.approx(15.0)  # (600+1200)/2/60

    def test_cost_ratio(self):
        tracker = EngineTracker()
        # gross = 11, cost = 1 -> ratio = 1/11
        tracker.record_trade("e", _make_trade(net_pnl=10, cost=1))
        m = tracker.get_metrics("e")
        assert m.cost_ratio == pytest.approx(1.0 / 11.0)

    def test_total_cost(self):
        tracker = EngineTracker()
        tracker.record_trade("e", _make_trade(cost=2.5))
        tracker.record_trade("e", _make_trade(cost=1.5))
        m = tracker.get_metrics("e")
        assert m.total_cost == pytest.approx(4.0)


# ------------------------------------------------------------------ #
# Sharpe ratio
# ------------------------------------------------------------------ #

class TestSharpeRatio:
    def test_sharpe_with_two_trades(self):
        tracker = EngineTracker()
        tracker.record_trade("e", _make_trade(net_pnl=10))
        tracker.record_trade("e", _make_trade(net_pnl=10))
        m = tracker.get_metrics("e")
        # Both same return → std=0 → sharpe=0
        assert m.sharpe_ratio == 0.0

    def test_sharpe_positive(self):
        tracker = EngineTracker()
        # Positive mean, some variance
        for pnl in [10, 12, 8, 11, 9]:
            tracker.record_trade("e", _make_trade(net_pnl=pnl))
        m = tracker.get_metrics("e")
        assert m.sharpe_ratio > 0

    def test_sharpe_negative(self):
        tracker = EngineTracker()
        # Negative mean
        for pnl in [-10, -12, -8, -11, -9]:
            tracker.record_trade("e", _make_trade(net_pnl=pnl))
        m = tracker.get_metrics("e")
        assert m.sharpe_ratio < 0

    def test_sharpe_annualization(self):
        tracker = EngineTracker()
        pnls = [10, 12, 8, 11, 9]
        for pnl in pnls:
            tracker.record_trade("e", _make_trade(net_pnl=pnl))
        m = tracker.get_metrics("e")

        # Manual calculation
        mean_r = sum(pnls) / len(pnls)
        var = sum((p - mean_r) ** 2 for p in pnls) / (len(pnls) - 1)
        std_r = math.sqrt(var)
        expected = (mean_r / std_r) * math.sqrt(365 * 24)
        assert m.sharpe_ratio == pytest.approx(expected)

    def test_sharpe_insufficient_data(self):
        tracker = EngineTracker()
        tracker.record_trade("e", _make_trade(net_pnl=10))
        m = tracker.get_metrics("e")
        assert m.sharpe_ratio == 0.0  # < 2 trades


# ------------------------------------------------------------------ #
# Max drawdown
# ------------------------------------------------------------------ #

class TestMaxDrawdown:
    def test_no_drawdown(self):
        tracker = EngineTracker()
        for pnl in [10, 10, 10]:
            tracker.record_trade("e", _make_trade(net_pnl=pnl))
        m = tracker.get_metrics("e")
        assert m.max_drawdown == 0.0

    def test_known_drawdown(self):
        tracker = EngineTracker()
        # cumulative: 10, 20, 10, 5, 15
        # peak:       10, 20, 20, 20, 20
        # drawdown:   0,  0, 50%, 75%, 25%
        for pnl in [10, 10, -10, -5, 10]:
            tracker.record_trade("e", _make_trade(net_pnl=pnl))
        m = tracker.get_metrics("e")
        assert m.max_drawdown == pytest.approx(0.75)

    def test_drawdown_all_losses(self):
        tracker = EngineTracker()
        # All losses → no peak above 0 → drawdown stays 0 (no peak to draw down from)
        for pnl in [-5, -5, -5]:
            tracker.record_trade("e", _make_trade(net_pnl=pnl))
        m = tracker.get_metrics("e")
        assert m.max_drawdown == 0.0

    def test_drawdown_recovery(self):
        tracker = EngineTracker()
        # cumulative: 20, 10, 30
        # peak:       20, 20, 30
        # drawdown:   0,  50%, 0
        for pnl in [20, -10, 20]:
            tracker.record_trade("e", _make_trade(net_pnl=pnl))
        m = tracker.get_metrics("e")
        assert m.max_drawdown == pytest.approx(0.5)


# ------------------------------------------------------------------ #
# PnL history
# ------------------------------------------------------------------ #

class TestPnlHistory:
    def test_empty_history(self):
        tracker = EngineTracker()
        assert tracker.get_pnl_history("nonexistent") == []

    def test_cumulative_pnl(self):
        tracker = EngineTracker()
        tracker.record_trade("e", _make_trade(net_pnl=10))
        tracker.record_trade("e", _make_trade(net_pnl=-3))
        tracker.record_trade("e", _make_trade(net_pnl=5))

        history = tracker.get_pnl_history("e")
        assert len(history) == 3
        assert history[0]["cumulative_pnl"] == pytest.approx(10.0)
        assert history[1]["cumulative_pnl"] == pytest.approx(7.0)
        assert history[2]["cumulative_pnl"] == pytest.approx(12.0)

    def test_history_includes_symbol(self):
        tracker = EngineTracker()
        tracker.record_trade("e", _make_trade(symbol="ETH/USDT"))
        history = tracker.get_pnl_history("e")
        assert history[0]["symbol"] == "ETH/USDT"


# ------------------------------------------------------------------ #
# Window filtering
# ------------------------------------------------------------------ #

class TestWindowFiltering:
    def test_recent_trades_included(self):
        tracker = EngineTracker()
        tracker.record_trade("e", _make_trade(exit_time=_now_iso()))
        m = tracker.get_metrics("e", window_hours=1)
        assert m.total_trades == 1

    def test_old_trades_excluded(self):
        tracker = EngineTracker()
        tracker.record_trade("e", _make_trade(exit_time=_past_iso(48)))
        m = tracker.get_metrics("e", window_hours=24)
        assert m.total_trades == 0

    def test_mixed_window(self):
        tracker = EngineTracker()
        tracker.record_trade("e", _make_trade(net_pnl=10, exit_time=_now_iso()))
        tracker.record_trade("e", _make_trade(net_pnl=20, exit_time=_past_iso(48)))
        m = tracker.get_metrics("e", window_hours=24)
        assert m.total_trades == 1
        assert m.total_pnl == pytest.approx(10.0)


# ------------------------------------------------------------------ #
# get_all_metrics
# ------------------------------------------------------------------ #

class TestGetAllMetrics:
    def test_all_metrics_returns_all_engines(self):
        tracker = EngineTracker()
        tracker.record_trade("a", _make_trade(engine="a"))
        tracker.record_trade("b", _make_trade(engine="b"))
        all_m = tracker.get_all_metrics()
        assert "a" in all_m
        assert "b" in all_m
        assert all_m["a"].total_trades == 1
        assert all_m["b"].total_trades == 1

    def test_all_metrics_empty(self):
        tracker = EngineTracker()
        assert tracker.get_all_metrics() == {}

    def test_all_metrics_includes_cycle_only_engines(self):
        tracker = EngineTracker()

        class FakeResult:
            cycle_num = 1
            timestamp = _now_iso()
            pnl_update = 0.0
            actions_taken = []
            duration_ms = 10.0

        tracker.record_cycle("engine_c", FakeResult())
        all_m = tracker.get_all_metrics()
        assert "engine_c" in all_m
        assert all_m["engine_c"].total_trades == 0
