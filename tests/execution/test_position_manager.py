"""Tests for PositionManager: stop-loss, take-profit, and trailing stop."""

import pytest

from bot.execution.position_manager import (
    ExitType,
    ManagedPosition,
    PositionManager,
)


class TestManagedPosition:
    def test_creates_correct_price_levels(self):
        pos = ManagedPosition(
            symbol="BTC/USDT",
            entry_price=50000.0,
            quantity=1.0,
            stop_loss_pct=3.0,
            take_profit_pct=5.0,
            tp1_pct=3.0,
        )
        assert pos.stop_loss_price == 48500.0  # 50000 * 0.97
        assert pos.tp1_price == 51500.0  # 50000 * 1.03
        assert pos.tp2_price == 52500.0  # 50000 * 1.05
        assert pos.highest_price_since_entry == 50000.0
        assert pos.tp1_hit is False
        assert pos.original_quantity == 1.0

    def test_custom_percentages(self):
        pos = ManagedPosition(
            symbol="ETH/USDT",
            entry_price=3000.0,
            quantity=10.0,
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
            tp1_pct=4.0,
        )
        assert pos.stop_loss_price == 2850.0  # 3000 * 0.95
        assert pos.tp1_price == 3120.0  # 3000 * 1.04
        assert pos.tp2_price == pytest.approx(3300.0)  # 3000 * 1.10


class TestPositionManagerBasic:
    def test_add_position(self):
        pm = PositionManager(stop_loss_pct=3.0, take_profit_pct=5.0)
        pos = pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        assert pos.symbol == "BTC/USDT"
        assert pos.entry_price == 50000.0
        assert pos.quantity == 1.0
        assert "BTC/USDT" in pm.managed_symbols

    def test_remove_position(self):
        pm = PositionManager()
        pm.add_position("BTC/USDT", 50000.0, 1.0)
        pm.remove_position("BTC/USDT")
        assert "BTC/USDT" not in pm.managed_symbols
        assert pm.get_position("BTC/USDT") is None

    def test_remove_nonexistent_position(self):
        pm = PositionManager()
        pm.remove_position("BTC/USDT")  # Should not raise

    def test_get_position(self):
        pm = PositionManager()
        pm.add_position("BTC/USDT", 50000.0, 1.0)
        pos = pm.get_position("BTC/USDT")
        assert pos is not None
        assert pos.entry_price == 50000.0

    def test_get_nonexistent_position(self):
        pm = PositionManager()
        assert pm.get_position("BTC/USDT") is None

    def test_multiple_positions(self):
        pm = PositionManager()
        pm.add_position("BTC/USDT", 50000.0, 1.0)
        pm.add_position("ETH/USDT", 3000.0, 10.0)
        assert len(pm.positions) == 2
        assert set(pm.managed_symbols) == {"BTC/USDT", "ETH/USDT"}

    def test_no_exit_in_range(self):
        pm = PositionManager(stop_loss_pct=3.0, take_profit_pct=5.0)
        pm.add_position("BTC/USDT", 50000.0, 1.0)
        result = pm.check_exits("BTC/USDT", 50000.0)
        assert result is None

    def test_no_exit_nonexistent_symbol(self):
        pm = PositionManager()
        result = pm.check_exits("BTC/USDT", 50000.0)
        assert result is None


class TestStopLoss:
    def test_stop_loss_triggers_at_correct_price(self):
        pm = PositionManager(stop_loss_pct=3.0, take_profit_pct=5.0)
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        # Stop-loss at 48500 (50000 * 0.97)
        result = pm.check_exits("BTC/USDT", 48500.0)
        assert result is not None
        assert result.exit_type == ExitType.STOP_LOSS
        assert result.quantity == 1.0
        assert result.exit_price == 48500.0
        assert result.symbol == "BTC/USDT"

    def test_stop_loss_triggers_below_threshold(self):
        pm = PositionManager(stop_loss_pct=3.0, take_profit_pct=5.0)
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        result = pm.check_exits("BTC/USDT", 47000.0)
        assert result is not None
        assert result.exit_type == ExitType.STOP_LOSS
        assert result.quantity == 1.0

    def test_stop_loss_does_not_trigger_above_threshold(self):
        pm = PositionManager(stop_loss_pct=3.0, take_profit_pct=5.0)
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        # Just above stop-loss
        result = pm.check_exits("BTC/USDT", 48501.0)
        assert result is None

    def test_stop_loss_custom_percentage(self):
        pm = PositionManager(stop_loss_pct=5.0, take_profit_pct=10.0)
        pm.add_position("ETH/USDT", entry_price=3000.0, quantity=10.0)
        # Stop-loss at 2850 (3000 * 0.95)
        result = pm.check_exits("ETH/USDT", 2850.0)
        assert result is not None
        assert result.exit_type == ExitType.STOP_LOSS
        assert result.quantity == 10.0

    def test_stop_loss_exits_full_quantity(self):
        pm = PositionManager(stop_loss_pct=3.0, take_profit_pct=5.0)
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=2.5)
        result = pm.check_exits("BTC/USDT", 48000.0)
        assert result is not None
        assert result.quantity == 2.5


class TestTakeProfit:
    def test_tp1_triggers_partial_exit(self):
        pm = PositionManager(
            stop_loss_pct=3.0, take_profit_pct=5.0, tp1_pct=3.0
        )
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        # TP1 at 51500 (50000 * 1.03)
        result = pm.check_exits("BTC/USDT", 51500.0)
        assert result is not None
        assert result.exit_type == ExitType.TAKE_PROFIT_1
        assert result.quantity == 0.5  # 50% of 1.0
        assert result.exit_price == 51500.0

    def test_tp1_reduces_managed_quantity(self):
        pm = PositionManager(
            stop_loss_pct=3.0, take_profit_pct=5.0, tp1_pct=3.0
        )
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        pm.check_exits("BTC/USDT", 51500.0)
        pos = pm.get_position("BTC/USDT")
        assert pos is not None
        assert pos.quantity == 0.5
        assert pos.tp1_hit is True

    def test_tp2_triggers_after_tp1(self):
        pm = PositionManager(
            stop_loss_pct=3.0, take_profit_pct=5.0, tp1_pct=3.0
        )
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        # First trigger TP1
        pm.check_exits("BTC/USDT", 51500.0)
        # Now trigger TP2 at 52500 (50000 * 1.05)
        result = pm.check_exits("BTC/USDT", 52500.0)
        assert result is not None
        assert result.exit_type == ExitType.TAKE_PROFIT_2
        assert result.quantity == 0.5  # remaining 50%
        assert result.exit_price == 52500.0

    def test_tp1_does_not_trigger_below_threshold(self):
        pm = PositionManager(
            stop_loss_pct=3.0, take_profit_pct=5.0, tp1_pct=3.0
        )
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        result = pm.check_exits("BTC/USDT", 51499.0)
        assert result is None

    def test_tp2_does_not_trigger_without_tp1(self):
        pm = PositionManager(
            stop_loss_pct=3.0, take_profit_pct=5.0, tp1_pct=3.0
        )
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        # Price above TP2 but TP1 not hit yet — should trigger TP1 first
        result = pm.check_exits("BTC/USDT", 53000.0)
        assert result is not None
        assert result.exit_type == ExitType.TAKE_PROFIT_1
        assert result.quantity == 0.5

    def test_tp1_only_triggers_once(self):
        pm = PositionManager(
            stop_loss_pct=3.0, take_profit_pct=5.0, tp1_pct=3.0
        )
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        # Trigger TP1
        pm.check_exits("BTC/USDT", 51500.0)
        # Price dips back and comes back above TP1 — should not trigger again
        result = pm.check_exits("BTC/USDT", 51600.0)
        assert result is None  # TP1 already hit, price below TP2


class TestTrailingStop:
    def test_trailing_stop_moves_up(self):
        pm = PositionManager(
            stop_loss_pct=3.0,
            take_profit_pct=5.0,
            trailing_stop_enabled=True,
            trailing_stop_pct=2.0,
        )
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        pos = pm.get_position("BTC/USDT")

        # Price moves up to 52000
        pm.check_exits("BTC/USDT", 52000.0)
        assert pos.highest_price_since_entry == 52000.0
        # Trailing stop at 52000 * 0.98 = 50960
        assert pos.stop_loss_price == pytest.approx(50960.0)

    def test_trailing_stop_does_not_move_down(self):
        pm = PositionManager(
            stop_loss_pct=3.0,
            take_profit_pct=5.0,
            trailing_stop_enabled=True,
            trailing_stop_pct=2.0,
        )
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        pos = pm.get_position("BTC/USDT")

        # Price goes up then down
        pm.check_exits("BTC/USDT", 52000.0)
        stop_after_up = pos.stop_loss_price
        pm.check_exits("BTC/USDT", 51000.0)
        assert pos.stop_loss_price == stop_after_up  # Should not decrease

    def test_trailing_stop_triggers_exit(self):
        pm = PositionManager(
            stop_loss_pct=3.0,
            take_profit_pct=20.0,
            tp1_pct=15.0,
            trailing_stop_enabled=True,
            trailing_stop_pct=2.0,
        )
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)

        # Price moves up to 55000 (below TP1 at 57500)
        pm.check_exits("BTC/USDT", 55000.0)
        # Trailing stop at 55000 * 0.98 = 53900
        # Price drops to 53900
        result = pm.check_exits("BTC/USDT", 53900.0)
        assert result is not None
        assert result.exit_type == ExitType.TRAILING_STOP
        assert result.quantity == 1.0
        assert result.exit_price == pytest.approx(53900.0)

    def test_trailing_stop_not_used_when_disabled(self):
        pm = PositionManager(
            stop_loss_pct=3.0,
            take_profit_pct=5.0,
            trailing_stop_enabled=False,
            trailing_stop_pct=2.0,
        )
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        pos = pm.get_position("BTC/USDT")

        # Price goes up — stop-loss should NOT move
        pm.check_exits("BTC/USDT", 55000.0)
        # Original stop-loss at 48500
        assert pos.stop_loss_price == 48500.0

    def test_trailing_stop_vs_original_stop_loss(self):
        """When trailing stop is enabled but hasn't moved above original SL,
        exit type should be STOP_LOSS not TRAILING_STOP."""
        pm = PositionManager(
            stop_loss_pct=3.0,
            take_profit_pct=5.0,
            trailing_stop_enabled=True,
            trailing_stop_pct=2.0,
        )
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)

        # Price drops immediately without going up
        result = pm.check_exits("BTC/USDT", 48500.0)
        assert result is not None
        assert result.exit_type == ExitType.STOP_LOSS

    def test_trailing_stop_per_position_override(self):
        pm = PositionManager(
            stop_loss_pct=3.0,
            take_profit_pct=5.0,
            trailing_stop_enabled=False,
        )
        # Override trailing stop for this specific position
        pos = pm.add_position(
            "BTC/USDT",
            entry_price=50000.0,
            quantity=1.0,
            trailing_stop_enabled=True,
        )
        assert pos.trailing_stop_enabled is True

    def test_trailing_stop_preserves_after_tp1(self):
        """Trailing stop should continue working after TP1 is hit."""
        pm = PositionManager(
            stop_loss_pct=3.0,
            take_profit_pct=5.0,
            tp1_pct=3.0,
            trailing_stop_enabled=True,
            trailing_stop_pct=2.0,
        )
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)

        # Trigger TP1 at 51500
        result = pm.check_exits("BTC/USDT", 51500.0)
        assert result.exit_type == ExitType.TAKE_PROFIT_1
        assert result.quantity == 0.5

        # Price continues up to 52000
        pm.check_exits("BTC/USDT", 52000.0)
        pos = pm.get_position("BTC/USDT")
        # Trailing stop should be at 52000 * 0.98 = 50960
        assert pos.stop_loss_price == pytest.approx(50960.0)

        # Price drops to trailing stop
        result = pm.check_exits("BTC/USDT", 50960.0)
        assert result is not None
        assert result.exit_type == ExitType.TRAILING_STOP
        assert result.quantity == 0.5  # remaining after TP1


class TestEdgeCases:
    def test_stop_loss_after_tp1(self):
        """After TP1 partial sell, stop-loss should sell remaining."""
        pm = PositionManager(
            stop_loss_pct=3.0, take_profit_pct=5.0, tp1_pct=3.0
        )
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)

        # Trigger TP1
        pm.check_exits("BTC/USDT", 51500.0)
        # Price crashes to stop-loss
        result = pm.check_exits("BTC/USDT", 48500.0)
        assert result is not None
        assert result.exit_type == ExitType.STOP_LOSS
        assert result.quantity == 0.5  # remaining after TP1

    def test_replace_existing_position(self):
        """Adding a position for same symbol replaces the old one."""
        pm = PositionManager()
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        pm.add_position("BTC/USDT", entry_price=55000.0, quantity=2.0)
        pos = pm.get_position("BTC/USDT")
        assert pos.entry_price == 55000.0
        assert pos.quantity == 2.0

    def test_price_exactly_at_entry(self):
        pm = PositionManager()
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        result = pm.check_exits("BTC/USDT", 50000.0)
        assert result is None

    def test_highest_price_tracking(self):
        pm = PositionManager()
        pm.add_position("BTC/USDT", entry_price=50000.0, quantity=1.0)
        pos = pm.get_position("BTC/USDT")

        pm.check_exits("BTC/USDT", 51000.0)
        assert pos.highest_price_since_entry == 51000.0

        pm.check_exits("BTC/USDT", 50500.0)
        assert pos.highest_price_since_entry == 51000.0  # doesn't decrease

        pm.check_exits("BTC/USDT", 52000.0)
        assert pos.highest_price_since_entry == 52000.0
