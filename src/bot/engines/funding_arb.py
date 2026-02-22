"""Funding rate arbitrage engine — delta-neutral strategy.

When perpetual funding rate is positive (longs pay shorts):
  - Buy spot (long) + Sell perpetual (short) → collect funding every 8h.

When funding drops below exit threshold or basis spread widens:
  - Close both legs.

This is a market-neutral strategy: PnL comes from funding payments,
not directional price movement.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from bot.engines.base import BaseEngine, EngineCycleResult

if TYPE_CHECKING:
    from bot.config import Settings
    from bot.engines.portfolio_manager import PortfolioManager
    from bot.exchanges.base import ExchangeAdapter

logger = structlog.get_logger(__name__)


class FundingRateArbEngine(BaseEngine):
    """Delta-neutral funding rate arbitrage engine.

    Monitors funding rates on configured symbols and opens hedged positions
    when the annualised rate exceeds a threshold.
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        exchanges: list[ExchangeAdapter] | None = None,
        paper_mode: bool = True,
        settings: Settings | None = None,
    ):
        s = settings
        super().__init__(
            portfolio_manager=portfolio_manager,
            exchanges=exchanges,
            loop_interval=300.0,  # 5 min
            max_positions=s.funding_arb_max_positions if s else 3,
            paper_mode=paper_mode,
        )
        self._min_funding_rate = s.funding_arb_min_rate if s else 0.0003
        self._exit_funding_rate = s.funding_arb_exit_rate if s else 0.0001
        self._max_spread_pct = s.funding_arb_max_spread_pct if s else 0.5
        self._leverage = s.funding_arb_leverage if s else 1
        self._symbols = list(s.funding_arb_symbols) if s else ["BTC/USDT", "ETH/USDT"]
        self._funding_monitor = None

    # ------------------------------------------------------------------
    # ABC implementation
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "funding_rate_arb"

    @property
    def description(self) -> str:
        return "Delta-neutral funding rate arbitrage (spot long + perp short)"

    async def _run_cycle(self) -> EngineCycleResult:
        """Check funding rates, open/close positions accordingly."""
        cycle_start = datetime.now(timezone.utc)
        actions: list[dict] = []
        signals: list[dict] = []
        pnl_update = 0.0

        for symbol in self._symbols:
            rate_info = await self._fetch_funding_rate(symbol)
            if rate_info is None:
                continue

            funding_rate = rate_info.get("funding_rate", 0)
            spread_pct = rate_info.get("spread_pct", 0)

            signals.append({
                "symbol": symbol,
                "funding_rate": funding_rate,
                "spread_pct": spread_pct,
                "annualised_pct": funding_rate * 3 * 365 * 100,
            })

            # Check if we should close existing position
            if symbol in self._positions:
                if self._should_close(funding_rate, spread_pct):
                    pnl = await self._close_position(symbol)
                    pnl_update += pnl
                    actions.append({
                        "action": "close",
                        "symbol": symbol,
                        "reason": (
                            "rate_below_exit"
                            if funding_rate < self._exit_funding_rate
                            else "spread_too_wide"
                        ),
                        "pnl": pnl,
                    })
                continue

            # Check if we should open a new position
            if self._should_open(funding_rate, spread_pct):
                if not self._has_capacity():
                    continue
                opened = await self._open_position(symbol, rate_info)
                if opened:
                    actions.append({
                        "action": "open",
                        "symbol": symbol,
                        "funding_rate": funding_rate,
                        "annualised_pct": funding_rate * 3 * 365 * 100,
                    })

        return EngineCycleResult(
            engine_name=self.name,
            cycle_num=self._cycle_count + 1,
            timestamp=cycle_start.isoformat(),
            duration_ms=0.0,
            actions_taken=actions,
            positions=list(self._positions.values()),
            signals=signals,
            pnl_update=pnl_update,
            metadata={
                "symbols_monitored": len(self._symbols),
                "open_positions": len(self._positions),
            },
        )

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _should_open(self, funding_rate: float, spread_pct: float) -> bool:
        """Decide whether to open a new delta-neutral position."""
        if funding_rate < self._min_funding_rate:
            return False
        if abs(spread_pct) > self._max_spread_pct:
            return False
        return True

    def _should_close(self, funding_rate: float, spread_pct: float) -> bool:
        """Decide whether to close an existing position."""
        if funding_rate < self._exit_funding_rate:
            return True
        if abs(spread_pct) > self._max_spread_pct * 2:
            return True
        return False

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    async def _open_position(
        self, symbol: str, rate_info: dict[str, Any]
    ) -> bool:
        """Open a delta-neutral position: spot long + perp short."""
        if self._paper_mode:
            # Simulate position
            price = rate_info.get("mark_price", 0) or rate_info.get("spot_price", 0)
            if price <= 0:
                return False
            # Size: allocate equal share of capital across max_positions
            position_capital = self._allocated_capital / max(self._max_positions, 1)
            quantity = position_capital / price if price > 0 else 0
            if quantity <= 0:
                return False

            self._add_position(
                symbol=symbol,
                side="delta_neutral",
                quantity=quantity,
                entry_price=price,
                funding_rate=rate_info.get("funding_rate", 0),
                spot_price=rate_info.get("spot_price", 0),
                mark_price=rate_info.get("mark_price", 0),
            )
            logger.info(
                "funding_arb_position_opened",
                symbol=symbol,
                quantity=quantity,
                price=price,
                funding_rate=rate_info.get("funding_rate"),
            )
            return True

        # Live mode: would place actual orders on spot + futures exchanges
        logger.warning("funding_arb_live_not_implemented", symbol=symbol)
        return False

    async def _close_position(self, symbol: str) -> float:
        """Close a delta-neutral position. Returns realised PnL."""
        pos = self._remove_position(symbol)
        if pos is None:
            return 0.0

        if self._paper_mode:
            # Simulate: PnL from funding payments accumulated
            # (Simplified: estimate based on time held and initial rate)
            funding_rate = pos.get("funding_rate", 0)
            quantity = pos.get("quantity", 0)
            entry_price = pos.get("entry_price", 0)
            # Approximate one funding payment
            pnl = funding_rate * quantity * entry_price
            logger.info(
                "funding_arb_position_closed",
                symbol=symbol,
                estimated_pnl=round(pnl, 4),
            )
            return pnl

        return 0.0

    async def _fetch_funding_rate(self, symbol: str) -> dict[str, Any] | None:
        """Fetch funding rate — uses exchange if available, or monitor."""
        # Try the first exchange that has fetch_funding_rate
        for exchange in self._exchanges:
            # Check for BinanceFuturesAdapter with get_funding_rate
            if hasattr(exchange, "get_funding_rate"):
                try:
                    return await exchange.get_funding_rate(symbol)
                except Exception as e:
                    logger.debug(
                        "funding_rate_fetch_failed",
                        exchange=getattr(exchange, "name", "unknown"),
                        symbol=symbol,
                        error=str(e),
                    )

            # Try via ccxt underlying
            inner = getattr(exchange, "_exchange", None)
            if inner and hasattr(inner, "fetch_funding_rate"):
                try:
                    data = await inner.fetch_funding_rate(symbol)
                    if isinstance(data, dict):
                        mark = float(data.get("markPrice") or 0)
                        index = float(data.get("indexPrice") or 0)
                        spread = (mark - index) / index * 100 if index > 0 else 0
                        return {
                            "symbol": symbol,
                            "funding_rate": float(data.get("fundingRate", 0)),
                            "mark_price": mark,
                            "spot_price": index,
                            "spread_pct": spread,
                        }
                except Exception:
                    pass

        return None
