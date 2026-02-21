"""Position exit management: stop-loss, take-profit, and trailing stop."""

from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger()


class ExitType(str, Enum):
    """Type of exit signal."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT_1 = "take_profit_1"
    TAKE_PROFIT_2 = "take_profit_2"
    TRAILING_STOP = "trailing_stop"


@dataclass
class ExitSignal:
    """Signal to exit a position (fully or partially)."""

    symbol: str
    exit_type: ExitType
    quantity: float
    exit_price: float


class ManagedPosition:
    """A position managed with stop-loss and take-profit levels."""

    def __init__(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        stop_loss_pct: float = 3.0,
        take_profit_pct: float = 5.0,
        tp1_pct: float = 3.0,
        trailing_stop_enabled: bool = False,
        trailing_stop_pct: float = 2.0,
    ):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.original_quantity = quantity

        # Calculated price levels
        self.stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
        self.tp1_price = entry_price * (1 + tp1_pct / 100)
        self.tp2_price = entry_price * (1 + take_profit_pct / 100)

        # Trailing stop config
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_pct = trailing_stop_pct

        # Track highest price for trailing stop
        self.highest_price_since_entry = entry_price

        # Track which TP levels have been hit
        self.tp1_hit = False

        # Store original stop-loss for exit type detection
        self._original_stop_loss_price = self.stop_loss_price


class PositionManager:
    """Manages stop-loss, take-profit, and trailing stop for open positions.

    When a position is opened, stores exit levels and monitors price against them.
    Supports:
    - Fixed stop-loss (configurable % below entry)
    - Two take-profit levels: TP1 (partial exit, 50%) and TP2 (full exit)
    - Trailing stop: moves stop-loss up as price rises
    """

    def __init__(
        self,
        stop_loss_pct: float = 3.0,
        take_profit_pct: float = 5.0,
        tp1_pct: float = 3.0,
        trailing_stop_enabled: bool = False,
        trailing_stop_pct: float = 2.0,
    ):
        self._stop_loss_pct = stop_loss_pct
        self._take_profit_pct = take_profit_pct
        self._tp1_pct = tp1_pct
        self._trailing_stop_enabled = trailing_stop_enabled
        self._trailing_stop_pct = trailing_stop_pct
        self._positions: dict[str, ManagedPosition] = {}

    def add_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        trailing_stop_enabled: bool | None = None,
    ) -> ManagedPosition:
        """Register a new position for exit monitoring."""
        use_trailing = (
            trailing_stop_enabled
            if trailing_stop_enabled is not None
            else self._trailing_stop_enabled
        )
        position = ManagedPosition(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss_pct=self._stop_loss_pct,
            take_profit_pct=self._take_profit_pct,
            tp1_pct=self._tp1_pct,
            trailing_stop_enabled=use_trailing,
            trailing_stop_pct=self._trailing_stop_pct,
        )
        self._positions[symbol] = position
        logger.info(
            "position_managed",
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=position.stop_loss_price,
            tp1=position.tp1_price,
            tp2=position.tp2_price,
            trailing_stop=use_trailing,
        )
        return position

    def remove_position(self, symbol: str) -> None:
        """Remove a position from monitoring."""
        self._positions.pop(symbol, None)

    def get_position(self, symbol: str) -> ManagedPosition | None:
        """Get a managed position by symbol."""
        return self._positions.get(symbol)

    def check_exits(self, symbol: str, current_price: float) -> ExitSignal | None:
        """Check if a position should be exited based on current price.

        Returns an ExitSignal if an exit condition is met, None otherwise.
        Checks in order: stop-loss → TP2 (if TP1 hit) → TP1.
        """
        position = self._positions.get(symbol)
        if position is None:
            return None

        # Update highest price seen (for trailing stop)
        if current_price > position.highest_price_since_entry:
            position.highest_price_since_entry = current_price

        # Update trailing stop if enabled and price has moved above entry
        if (
            position.trailing_stop_enabled
            and position.highest_price_since_entry > position.entry_price
        ):
            trailing_stop_price = position.highest_price_since_entry * (
                1 - position.trailing_stop_pct / 100
            )
            if trailing_stop_price > position.stop_loss_price:
                position.stop_loss_price = trailing_stop_price

        # Check stop-loss (including trailing stop)
        if current_price <= position.stop_loss_price:
            # Determine if exit was due to trailing stop or fixed stop-loss
            exit_type = (
                ExitType.TRAILING_STOP
                if position.trailing_stop_enabled
                and position.stop_loss_price > position._original_stop_loss_price
                else ExitType.STOP_LOSS
            )
            logger.info(
                "exit_triggered",
                symbol=symbol,
                exit_type=exit_type.value,
                current_price=current_price,
                stop_loss_price=position.stop_loss_price,
            )
            return ExitSignal(
                symbol=symbol,
                exit_type=exit_type,
                quantity=position.quantity,
                exit_price=position.stop_loss_price,
            )

        # Check TP2 (full exit) if TP1 already hit
        if position.tp1_hit and current_price >= position.tp2_price:
            logger.info(
                "exit_triggered",
                symbol=symbol,
                exit_type=ExitType.TAKE_PROFIT_2.value,
                current_price=current_price,
                tp2_price=position.tp2_price,
            )
            return ExitSignal(
                symbol=symbol,
                exit_type=ExitType.TAKE_PROFIT_2,
                quantity=position.quantity,
                exit_price=position.tp2_price,
            )

        # Check TP1 (partial exit — sell 50%)
        if not position.tp1_hit and current_price >= position.tp1_price:
            sell_qty = position.quantity * 0.5
            position.tp1_hit = True
            position.quantity -= sell_qty
            logger.info(
                "exit_triggered",
                symbol=symbol,
                exit_type=ExitType.TAKE_PROFIT_1.value,
                current_price=current_price,
                tp1_price=position.tp1_price,
                sell_qty=sell_qty,
                remaining_qty=position.quantity,
            )
            return ExitSignal(
                symbol=symbol,
                exit_type=ExitType.TAKE_PROFIT_1,
                quantity=sell_qty,
                exit_price=position.tp1_price,
            )

        return None

    @property
    def positions(self) -> dict[str, ManagedPosition]:
        """All managed positions."""
        return dict(self._positions)

    @property
    def managed_symbols(self) -> list[str]:
        """All symbols with managed positions."""
        return list(self._positions.keys())
