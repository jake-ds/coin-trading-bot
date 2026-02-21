"""Paper trading portfolio simulator with finite capital and fee tracking."""

from datetime import datetime, timezone

import structlog

logger = structlog.get_logger()


class PaperPortfolio:
    """Simulates a real portfolio with finite capital, fee deduction, and position tracking.

    Tracks cash balance, open positions, and trade history for paper trading mode.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        fee_pct: float = 0.1,
    ):
        self._initial_balance = initial_balance
        self._cash = initial_balance
        self._fee_pct = fee_pct
        self._positions: dict[str, dict] = {}
        self._trade_history: list[dict] = []
        self._current_prices: dict[str, float] = {}

    def buy(self, symbol: str, qty: float, price: float) -> bool:
        """Buy a quantity of a symbol at a given price.

        Deducts cost + fee from cash and adds/updates the position.
        Returns False if insufficient balance.
        """
        cost = qty * price
        fee = cost * self._fee_pct / 100
        total_cost = cost + fee

        if total_cost > self._cash:
            logger.warning(
                "paper_buy_insufficient_balance",
                symbol=symbol,
                required=total_cost,
                available=self._cash,
            )
            return False

        self._cash -= total_cost
        now = datetime.now(timezone.utc)

        if symbol in self._positions:
            existing = self._positions[symbol]
            total_qty = existing["qty"] + qty
            avg_price = (
                (existing["qty"] * existing["entry_price"] + qty * price) / total_qty
            )
            self._positions[symbol] = {
                "qty": total_qty,
                "entry_price": avg_price,
                "entry_time": existing["entry_time"],
            }
        else:
            self._positions[symbol] = {
                "qty": qty,
                "entry_price": price,
                "entry_time": now,
            }

        self._current_prices[symbol] = price
        self._trade_history.append(
            {
                "symbol": symbol,
                "side": "BUY",
                "qty": qty,
                "price": price,
                "fee": fee,
                "timestamp": now,
            }
        )

        logger.info(
            "paper_buy_executed",
            symbol=symbol,
            qty=qty,
            price=price,
            fee=fee,
            cash_remaining=self._cash,
        )
        return True

    def sell(self, symbol: str, qty: float, price: float) -> bool:
        """Sell a quantity of a symbol at a given price.

        Adds proceeds minus fee to cash and reduces/removes the position.
        Returns False if no position or insufficient quantity.
        """
        if symbol not in self._positions:
            logger.warning("paper_sell_no_position", symbol=symbol)
            return False

        position = self._positions[symbol]
        if position["qty"] < qty - 1e-10:
            logger.warning(
                "paper_sell_insufficient_qty",
                symbol=symbol,
                requested=qty,
                available=position["qty"],
            )
            return False

        proceeds = qty * price
        fee = proceeds * self._fee_pct / 100
        net_proceeds = proceeds - fee

        self._cash += net_proceeds
        now = datetime.now(timezone.utc)

        remaining = position["qty"] - qty
        if remaining < 1e-10:
            del self._positions[symbol]
        else:
            self._positions[symbol] = {
                "qty": remaining,
                "entry_price": position["entry_price"],
                "entry_time": position["entry_time"],
            }

        self._current_prices[symbol] = price
        self._trade_history.append(
            {
                "symbol": symbol,
                "side": "SELL",
                "qty": qty,
                "price": price,
                "fee": fee,
                "timestamp": now,
            }
        )

        logger.info(
            "paper_sell_executed",
            symbol=symbol,
            qty=qty,
            price=price,
            fee=fee,
            cash_remaining=self._cash,
        )
        return True

    def update_price(self, symbol: str, price: float) -> None:
        """Update the current price for a symbol (used for total_value calculations)."""
        self._current_prices[symbol] = price

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    @property
    def positions(self) -> dict[str, dict]:
        """Current open positions."""
        return dict(self._positions)

    @property
    def total_value(self) -> float:
        """Total portfolio value: cash + sum of position values at current prices."""
        position_value = sum(
            pos["qty"] * self._current_prices.get(symbol, pos["entry_price"])
            for symbol, pos in self._positions.items()
        )
        return self._cash + position_value

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss across all open positions."""
        pnl = 0.0
        for symbol, pos in self._positions.items():
            current_price = self._current_prices.get(symbol, pos["entry_price"])
            pnl += (current_price - pos["entry_price"]) * pos["qty"]
        return pnl

    @property
    def trade_history(self) -> list[dict]:
        """List of all executed trades."""
        return list(self._trade_history)

    @property
    def fee_pct(self) -> float:
        """Fee percentage."""
        return self._fee_pct
