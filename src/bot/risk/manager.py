"""Risk management engine."""

from datetime import datetime

import structlog

from bot.models import SignalAction, TradingSignal

logger = structlog.get_logger()


class RiskManager:
    """Validates all signals before execution with risk controls."""

    def __init__(
        self,
        max_position_size_pct: float = 10.0,
        stop_loss_pct: float = 3.0,
        daily_loss_limit_pct: float = 5.0,
        max_drawdown_pct: float = 15.0,
        max_concurrent_positions: int = 5,
    ):
        self._max_position_size_pct = max_position_size_pct
        self._stop_loss_pct = stop_loss_pct
        self._daily_loss_limit_pct = daily_loss_limit_pct
        self._max_drawdown_pct = max_drawdown_pct
        self._max_concurrent_positions = max_concurrent_positions

        # Tracking state
        self._portfolio_peak: float = 0.0
        self._current_portfolio_value: float = 0.0
        self._daily_pnl: float = 0.0
        self._daily_pnl_reset_date: datetime | None = None
        self._open_positions: dict[str, dict] = {}
        self._trading_halted: bool = False
        self._halt_reason: str = ""

    def update_portfolio_value(self, value: float) -> None:
        """Update current portfolio value and track peak."""
        self._current_portfolio_value = value
        if value > self._portfolio_peak:
            self._portfolio_peak = value

    def record_trade_pnl(self, pnl: float) -> None:
        """Record a realized P&L from a trade."""
        today = datetime.utcnow().date()
        if self._daily_pnl_reset_date != today:
            self._daily_pnl = 0.0
            self._daily_pnl_reset_date = today
        self._daily_pnl += pnl

    def add_position(self, symbol: str, quantity: float, entry_price: float) -> None:
        """Track an open position."""
        self._open_positions[symbol] = {
            "quantity": quantity,
            "entry_price": entry_price,
        }

    def get_position(self, symbol: str) -> dict | None:
        """Get position data for a symbol, or None if no position."""
        return self._open_positions.get(symbol)

    def remove_position(self, symbol: str) -> None:
        """Remove a closed position."""
        self._open_positions.pop(symbol, None)

    def validate_signal(self, signal: TradingSignal) -> TradingSignal:
        """Validate a trading signal against risk rules.

        Returns HOLD if the signal is rejected, otherwise returns the original signal.
        """
        if signal.action == SignalAction.HOLD:
            return signal

        # Check if trading is halted
        if self._trading_halted:
            logger.warning(
                "signal_rejected_halted",
                reason=self._halt_reason,
                symbol=signal.symbol,
            )
            return self._reject(signal, f"trading_halted: {self._halt_reason}")

        # Check daily loss limit
        if self._current_portfolio_value > 0:
            daily_loss_pct = abs(self._daily_pnl) / self._current_portfolio_value * 100
            if self._daily_pnl < 0 and daily_loss_pct >= self._daily_loss_limit_pct:
                self._trading_halted = True
                self._halt_reason = "daily_loss_limit"
                logger.warning(
                    "trading_halted",
                    reason="daily_loss_limit",
                    daily_loss_pct=daily_loss_pct,
                )
                return self._reject(signal, "daily_loss_limit_exceeded")

        # Check max drawdown
        if self._portfolio_peak > 0:
            drawdown_pct = (
                (self._portfolio_peak - self._current_portfolio_value) / self._portfolio_peak * 100
            )
            if drawdown_pct >= self._max_drawdown_pct:
                self._trading_halted = True
                self._halt_reason = "max_drawdown"
                logger.warning(
                    "trading_halted",
                    reason="max_drawdown",
                    drawdown_pct=drawdown_pct,
                )
                return self._reject(signal, "max_drawdown_exceeded")

        # For BUY signals, apply position-related checks
        if signal.action == SignalAction.BUY:
            # Max concurrent positions
            if len(self._open_positions) >= self._max_concurrent_positions:
                return self._reject(signal, "max_concurrent_positions")

            # Already have position in this symbol
            if signal.symbol in self._open_positions:
                return self._reject(signal, "position_already_exists")

        return signal

    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
    ) -> float:
        """Calculate position size based on risk parameters.

        Uses fixed percentage of portfolio.
        """
        if price <= 0 or portfolio_value <= 0:
            return 0.0

        max_value = portfolio_value * (self._max_position_size_pct / 100)
        return max_value / price

    def calculate_dynamic_position_size(
        self,
        portfolio_value: float,
        price: float,
        atr: float,
        risk_per_trade_pct: float = 1.0,
        atr_multiplier: float = 2.0,
    ) -> float:
        """Calculate volatility-adjusted position size using ATR.

        Risks the same dollar amount per trade regardless of volatility:
        - In volatile markets (high ATR), take smaller positions.
        - In quiet markets (low ATR), take larger positions.

        Formula:
            risk_amount = portfolio_value * risk_per_trade_pct / 100
            position_size = risk_amount / (atr * atr_multiplier)

        The result is capped at max_position_size_pct of portfolio.

        Args:
            portfolio_value: Total portfolio value in quote currency.
            price: Current asset price.
            atr: Average True Range value.
            risk_per_trade_pct: Percentage of portfolio to risk per trade (default 1.0%).
            atr_multiplier: Multiplier for ATR to set risk distance (default 2.0).

        Returns:
            Position size in base currency (quantity), or 0.0 if inputs are invalid.
        """
        if price <= 0 or portfolio_value <= 0 or atr <= 0 or atr_multiplier <= 0:
            return 0.0

        risk_amount = portfolio_value * risk_per_trade_pct / 100
        position_size = risk_amount / (atr * atr_multiplier)

        # Cap at max_position_size_pct
        max_value = portfolio_value * (self._max_position_size_pct / 100)
        max_qty = max_value / price
        return min(position_size, max_qty)

    def calculate_stop_loss(self, entry_price: float, side: str = "BUY") -> float:
        """Calculate stop-loss price."""
        if side == "BUY":
            return entry_price * (1 - self._stop_loss_pct / 100)
        return entry_price * (1 + self._stop_loss_pct / 100)

    def reset_daily(self) -> None:
        """Reset daily tracking (call at start of new trading day)."""
        self._daily_pnl = 0.0
        self._daily_pnl_reset_date = datetime.utcnow().date()
        if self._halt_reason == "daily_loss_limit":
            self._trading_halted = False
            self._halt_reason = ""

    def check_and_reset_daily(self) -> bool:
        """Check if date has changed and auto-reset daily PnL if so.

        Returns True if a reset was performed.
        """
        today = datetime.utcnow().date()
        if self._daily_pnl_reset_date is not None and self._daily_pnl_reset_date == today:
            return False
        logger.info("daily_pnl_reset", new_date=str(today))
        self.reset_daily()
        return True

    def resume_trading(self) -> None:
        """Resume trading after a halt."""
        self._trading_halted = False
        self._halt_reason = ""

    @property
    def is_halted(self) -> bool:
        return self._trading_halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @staticmethod
    def _reject(signal: TradingSignal, reason: str) -> TradingSignal:
        """Create a HOLD signal from a rejected signal."""
        return TradingSignal(
            strategy_name=signal.strategy_name,
            symbol=signal.symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            metadata={**signal.metadata, "rejected": True, "reject_reason": reason},
        )
