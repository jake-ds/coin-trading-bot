"""Tests for metrics collector."""

from bot.monitoring.metrics import MetricsCollector


class TestMetricsCollector:
    def test_initial_state(self):
        mc = MetricsCollector(initial_capital=10000.0)
        result = mc.calculate()
        assert result.total_return_pct == 0.0
        assert result.total_trades == 0
        assert result.win_rate == 0.0

    def test_positive_return(self):
        mc = MetricsCollector(initial_capital=10000.0)
        mc.record_portfolio_value(12000.0)
        result = mc.calculate()
        assert result.total_return_pct == 20.0

    def test_negative_return(self):
        mc = MetricsCollector(initial_capital=10000.0)
        mc.record_portfolio_value(8000.0)
        result = mc.calculate()
        assert result.total_return_pct == -20.0

    def test_win_rate(self):
        mc = MetricsCollector()
        mc.record_trade(100)
        mc.record_trade(200)
        mc.record_trade(-50)
        result = mc.calculate()
        assert result.total_trades == 3
        assert result.winning_trades == 2
        assert result.losing_trades == 1
        assert abs(result.win_rate - 66.67) < 0.1

    def test_max_drawdown(self):
        mc = MetricsCollector(initial_capital=10000.0)
        mc.record_portfolio_value(12000.0)  # peak
        mc.record_portfolio_value(9000.0)  # 25% drawdown from peak
        mc.record_portfolio_value(11000.0)
        result = mc.calculate()
        assert result.max_drawdown_pct == 25.0

    def test_sharpe_ratio_with_trades(self):
        mc = MetricsCollector()
        for pnl in [100, 200, 50, -30, 150, 80]:
            mc.record_trade(pnl)
        result = mc.calculate()
        assert result.sharpe_ratio > 0  # Positive since net positive returns

    def test_no_trades_zero_sharpe(self):
        mc = MetricsCollector()
        result = mc.calculate()
        assert result.sharpe_ratio == 0.0
