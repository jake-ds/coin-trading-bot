"""Tests for risk management engine."""

import pytest

from bot.models import SignalAction, TradingSignal
from bot.risk.manager import RiskManager


def make_signal(
    action: SignalAction = SignalAction.BUY,
    symbol: str = "BTC/USDT",
    confidence: float = 0.8,
) -> TradingSignal:
    return TradingSignal(
        strategy_name="test",
        symbol=symbol,
        action=action,
        confidence=confidence,
    )


class TestRiskManager:
    @pytest.fixture
    def rm(self):
        return RiskManager(
            max_position_size_pct=10.0,
            stop_loss_pct=3.0,
            daily_loss_limit_pct=5.0,
            max_drawdown_pct=15.0,
            max_concurrent_positions=3,
        )

    def test_hold_passes_through(self, rm):
        signal = make_signal(SignalAction.HOLD)
        result = rm.validate_signal(signal)
        assert result.action == SignalAction.HOLD

    def test_buy_passes_when_no_limits(self, rm):
        rm.update_portfolio_value(10000.0)
        signal = make_signal(SignalAction.BUY)
        result = rm.validate_signal(signal)
        assert result.action == SignalAction.BUY

    def test_sell_passes_through(self, rm):
        rm.update_portfolio_value(10000.0)
        signal = make_signal(SignalAction.SELL)
        result = rm.validate_signal(signal)
        assert result.action == SignalAction.SELL

    def test_max_concurrent_positions(self, rm):
        rm.update_portfolio_value(10000.0)
        rm.add_position("BTC/USDT", 1.0, 50000.0)
        rm.add_position("ETH/USDT", 10.0, 3000.0)
        rm.add_position("SOL/USDT", 100.0, 100.0)

        signal = make_signal(SignalAction.BUY, symbol="ADA/USDT")
        result = rm.validate_signal(signal)
        assert result.action == SignalAction.HOLD
        assert result.metadata["reject_reason"] == "max_concurrent_positions"

    def test_position_already_exists(self, rm):
        rm.update_portfolio_value(10000.0)
        rm.add_position("BTC/USDT", 1.0, 50000.0)

        signal = make_signal(SignalAction.BUY, symbol="BTC/USDT")
        result = rm.validate_signal(signal)
        assert result.action == SignalAction.HOLD
        assert result.metadata["reject_reason"] == "position_already_exists"

    def test_daily_loss_limit(self, rm):
        rm.update_portfolio_value(10000.0)
        rm.record_trade_pnl(-500.0)  # 5% loss

        signal = make_signal(SignalAction.BUY)
        result = rm.validate_signal(signal)
        assert result.action == SignalAction.HOLD
        assert rm.is_halted

    def test_max_drawdown(self, rm):
        rm.update_portfolio_value(10000.0)  # peak
        rm.update_portfolio_value(8400.0)  # 16% drawdown (> 15% limit)

        signal = make_signal(SignalAction.BUY)
        result = rm.validate_signal(signal)
        assert result.action == SignalAction.HOLD
        assert rm.is_halted
        assert rm.halt_reason == "max_drawdown"

    def test_position_sizing(self, rm):
        size = rm.calculate_position_size(10000.0, 50000.0)
        # 10% of 10000 = 1000, at price 50000 = 0.02
        assert abs(size - 0.02) < 0.001

    def test_position_sizing_zero_price(self, rm):
        size = rm.calculate_position_size(10000.0, 0)
        assert size == 0.0

    def test_stop_loss_buy(self, rm):
        sl = rm.calculate_stop_loss(50000.0, "BUY")
        assert sl == 48500.0  # 3% below

    def test_stop_loss_sell(self, rm):
        sl = rm.calculate_stop_loss(50000.0, "SELL")
        assert sl == 51500.0  # 3% above

    def test_reset_daily(self, rm):
        rm.update_portfolio_value(10000.0)
        rm.record_trade_pnl(-500.0)  # triggers halt
        rm.validate_signal(make_signal())
        assert rm.is_halted

        rm.reset_daily()
        assert not rm.is_halted

    def test_resume_trading(self, rm):
        rm.update_portfolio_value(10000.0)
        rm.update_portfolio_value(8400.0)  # triggers drawdown halt
        rm.validate_signal(make_signal())
        assert rm.is_halted

        rm.resume_trading()
        assert not rm.is_halted

    def test_remove_position(self, rm):
        rm.update_portfolio_value(10000.0)
        rm.add_position("BTC/USDT", 1.0, 50000.0)
        rm.remove_position("BTC/USDT")
        # Should be able to open position again
        signal = make_signal(SignalAction.BUY, symbol="BTC/USDT")
        result = rm.validate_signal(signal)
        assert result.action == SignalAction.BUY

    def test_portfolio_peak_tracking(self, rm):
        rm.update_portfolio_value(10000.0)
        rm.update_portfolio_value(12000.0)
        rm.update_portfolio_value(11000.0)
        assert rm._portfolio_peak == 12000.0
