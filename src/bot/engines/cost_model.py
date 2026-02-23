"""Trading cost calculator for fee and slippage modeling.

Provides a reusable CostModel that accurately calculates trading fees,
slippage, and net profitability for any trade.  This is the foundation
for making all engines cost-aware.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    """Models trading costs (fees + slippage) for profitability analysis.

    All fee/slippage values are stored as *percentages* —
    ``0.02`` means **0.02 %**, NOT 0.0002.

    Defaults match Binance VIP-0 spot fees.
    """

    maker_fee_pct: float = 0.02
    taker_fee_pct: float = 0.04
    slippage_pct: float = 0.01

    # ------------------------------------------------------------------ #
    #  Core calculations                                                  #
    # ------------------------------------------------------------------ #

    def round_trip_cost(
        self,
        notional: float,
        legs: int = 2,
        *,
        is_maker: bool = False,
    ) -> float:
        """Total cost in USD for a round-trip trade.

        ``cost = notional * (fee_pct + slippage_pct) * legs / 100``

        Parameters
        ----------
        notional:
            Trade notional value in USD.
        legs:
            Number of order legs (2 for simple buy+sell, 4 for
            arb with entry+exit on two sides).
        is_maker:
            If ``True``, use the maker fee; otherwise use taker fee.
        """
        fee_pct = self.maker_fee_pct if is_maker else self.taker_fee_pct
        return notional * (fee_pct + self.slippage_pct) * legs / 100

    def net_profit(
        self,
        gross_pnl: float,
        notional: float,
        legs: int = 2,
        *,
        is_maker: bool = False,
    ) -> float:
        """Gross PnL minus round-trip trading cost."""
        return gross_pnl - self.round_trip_cost(notional, legs, is_maker=is_maker)

    def min_spread_for_profit(
        self,
        legs: int = 2,
        *,
        is_maker: bool = False,
    ) -> float:
        """Minimum spread percentage needed to break even.

        ``min_spread = (fee_pct + slippage_pct) * legs``

        The result is in the same percentage unit as the fee fields
        (i.e. ``0.10`` means 0.10 %).
        """
        fee_pct = self.maker_fee_pct if is_maker else self.taker_fee_pct
        return (fee_pct + self.slippage_pct) * legs

    def is_profitable(
        self,
        gross_pnl: float,
        notional: float,
        legs: int = 2,
        *,
        is_maker: bool = False,
    ) -> bool:
        """Return ``True`` when net profit (after costs) is positive."""
        return self.net_profit(gross_pnl, notional, legs, is_maker=is_maker) > 0
