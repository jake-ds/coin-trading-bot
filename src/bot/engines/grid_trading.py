"""Grid trading engine — places buy/sell limit orders at fixed intervals.

Places a grid of buy orders below current price and sell orders above.
When a buy fills, a corresponding sell is placed above; when a sell fills,
a buy is placed below.  Profits come from capturing the spread between
grid levels during sideways (ranging) markets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from bot.engines.base import BaseEngine, DecisionStep, EngineCycleResult
from bot.engines.cost_model import CostModel

if TYPE_CHECKING:
    from bot.config import Settings
    from bot.engines.portfolio_manager import PortfolioManager
    from bot.exchanges.base import ExchangeAdapter

logger = structlog.get_logger(__name__)


class GridLevel:
    """Represents a single grid level with its price and order state."""

    __slots__ = ("price", "side", "order_id", "filled")

    def __init__(self, price: float, side: str, order_id: str | None = None):
        self.price = price
        self.side = side  # "buy" or "sell"
        self.order_id = order_id
        self.filled = False


class GridTradingEngine(BaseEngine):
    """Automated grid trading engine.

    Places limit orders at evenly-spaced price levels around the current
    market price.  Captures profit from price oscillation within a range.
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
            loop_interval=30.0,
            max_positions=s.grid_max_open_orders if s else 20,
            paper_mode=paper_mode,
        )
        self._grid_levels_count = s.grid_levels if s else 10
        self._grid_spacing_pct = s.grid_spacing_pct if s else 0.5
        self._auto_range = s.grid_auto_range if s else True
        self._range_atr_multiplier = s.grid_range_atr_multiplier if s else 3.0
        self._max_open_orders = s.grid_max_open_orders if s else 20
        self._symbols = list(s.grid_symbols) if s else ["BTC/USDT", "ETH/USDT"]

        self._cost_model = CostModel()

        # Grid state per symbol: list of GridLevel
        self._grids: dict[str, list[GridLevel]] = {}
        self._grid_pnl: dict[str, float] = {}
        self._last_prices: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "grid_trading"

    @property
    def description(self) -> str:
        return "Automated grid trading with limit order grid"

    @property
    def grids(self) -> dict[str, list[GridLevel]]:
        return dict(self._grids)

    async def _run_cycle(self) -> EngineCycleResult:
        """Check grid fills, place new orders, rebalance grids."""
        cycle_start = datetime.now(timezone.utc)
        actions: list[dict] = []
        signals: list[dict] = []
        decisions: list[DecisionStep] = []
        pnl_update = 0.0

        for symbol in self._symbols:
            price = await self._get_current_price(symbol)
            if price is None or price <= 0:
                decisions.append(DecisionStep(
                    label=f"{symbol} 가격 조회",
                    observation="가격 데이터 없음",
                    threshold="N/A",
                    result="SKIP - 가격 조회 실패",
                    category="skip",
                ))
                continue

            self._last_prices[symbol] = price
            signals.append({"symbol": symbol, "price": price})

            # Initialize grid if not yet created
            if symbol not in self._grids:
                self._init_grid(symbol, price)
                actions.append({
                    "action": "grid_created",
                    "symbol": symbol,
                    "center_price": price,
                    "levels": len(self._grids[symbol]),
                })
                decisions.append(DecisionStep(
                    label=f"{symbol} 그리드 초기화",
                    observation=f"현재가 ${price:,.2f}, 그리드 없음",
                    threshold=f"간격 {self._grid_spacing_pct}%, 레벨 {self._grid_levels_count}개",
                    result=f"INIT - {len(self._grids[symbol])}개 레벨 생성",
                    category="execute",
                ))

            # Check for fills and react
            grid_actions, grid_pnl = self._check_fills(symbol, price)
            actions.extend(grid_actions)
            pnl_update += grid_pnl

            grid = self._grids.get(symbol, [])
            filled_count = sum(1 for lvl in grid if lvl.filled)
            buy_fills = sum(1 for a in grid_actions if a.get("action") == "grid_buy_filled")
            sell_fills = sum(1 for a in grid_actions if a.get("action") == "grid_sell_filled")
            skipped_fills = sum(1 for a in grid_actions if a.get("action") == "grid_sell_skipped")
            total_cost = sum(a.get("cost", 0) for a in grid_actions if a.get("cost"))

            decisions.append(DecisionStep(
                label=f"{symbol} 그리드 체크",
                observation=(
                    f"현재가 ${price:,.2f}, "
                    f"그리드 {len(grid)}개 레벨 ({filled_count}개 체결됨)"
                ),
                threshold=f"간격 {self._grid_spacing_pct}%",
                result=(
                    f"FILL - 매수 {buy_fills}건, 매도 {sell_fills}건, PnL: {grid_pnl:.4f}"
                    if buy_fills or sell_fills
                    else "HOLD - 체결 없음"
                ),
                category="execute" if buy_fills or sell_fills else "evaluate",
            ))

            # Cost analysis for fills
            if sell_fills or skipped_fills:
                decisions.append(DecisionStep(
                    label=f"{symbol} 비용 분석",
                    observation=(
                        f"매도 체결 {sell_fills}건, "
                        f"비용초과 스킵 {skipped_fills}건, "
                        f"총비용=${total_cost:.2f}"
                    ),
                    threshold="순수익 > 0",
                    result="수익" if grid_pnl > 0 else "손실",
                    category="evaluate",
                ))

            # Check if price is outside the grid range — reset
            if self._is_outside_range(symbol, price):
                self._reset_grid(symbol, price)
                actions.append({
                    "action": "grid_reset",
                    "symbol": symbol,
                    "new_center": price,
                    "reason": "price_outside_range",
                })
                decisions.append(DecisionStep(
                    label=f"{symbol} 범위 이탈 체크",
                    observation=f"현재가 ${price:,.2f}가 그리드 범위 밖",
                    threshold="가격이 그리드 최소/최대 범위 이내",
                    result=f"RESET - 새 중심가 ${price:,.2f}로 그리드 재설정",
                    category="execute",
                ))

        return EngineCycleResult(
            engine_name=self.name,
            cycle_num=self._cycle_count + 1,
            timestamp=cycle_start.isoformat(),
            duration_ms=0.0,
            actions_taken=actions,
            positions=[
                {"symbol": s, "levels": len(g), "pnl": self._grid_pnl.get(s, 0)}
                for s, g in self._grids.items()
            ],
            signals=signals,
            pnl_update=pnl_update,
            metadata={
                "total_grid_levels": sum(len(g) for g in self._grids.values()),
            },
            decisions=decisions,
        )

    # ------------------------------------------------------------------
    # Grid management
    # ------------------------------------------------------------------

    def _init_grid(self, symbol: str, center_price: float) -> None:
        """Create a new grid around center_price."""
        levels: list[GridLevel] = []
        spacing = self._grid_spacing_pct / 100.0

        for i in range(1, self._grid_levels_count + 1):
            buy_price = center_price * (1 - spacing * i)
            sell_price = center_price * (1 + spacing * i)
            levels.append(GridLevel(round(buy_price, 8), "buy"))
            levels.append(GridLevel(round(sell_price, 8), "sell"))

        self._grids[symbol] = levels
        self._grid_pnl.setdefault(symbol, 0.0)
        logger.info(
            "grid_initialized",
            symbol=symbol,
            center=center_price,
            levels=len(levels),
            spacing_pct=self._grid_spacing_pct,
        )

    def _reset_grid(self, symbol: str, new_center: float) -> None:
        """Reset the grid around a new center price."""
        self._grids.pop(symbol, None)
        self._init_grid(symbol, new_center)

    def _check_fills(
        self, symbol: str, current_price: float
    ) -> tuple[list[dict], float]:
        """Check if any grid levels have been 'filled' by the current price.

        In paper mode, a buy level is filled when price <= level price,
        and a sell level is filled when price >= level price.

        Returns (actions, pnl_delta).
        """
        grid = self._grids.get(symbol, [])
        actions: list[dict] = []
        pnl = 0.0

        for level in grid:
            if level.filled:
                continue

            if level.side == "buy" and current_price <= level.price:
                level.filled = True
                # Profit: we'll sell at the corresponding sell level later
                actions.append({
                    "action": "grid_buy_filled",
                    "symbol": symbol,
                    "price": level.price,
                })
                logger.debug(
                    "grid_buy_filled",
                    symbol=symbol,
                    level_price=level.price,
                    current_price=current_price,
                )

            elif level.side == "sell" and current_price >= level.price:
                level.filled = True
                # Calculate profit: difference between this sell and
                # the spacing below (approximation for paper mode)
                spacing = self._grid_spacing_pct / 100.0
                buy_price = level.price / (1 + spacing)
                gross_profit = level.price - buy_price
                # Scale by a notional quantity (capital / max_orders / price)
                if self._allocated_capital > 0 and level.price > 0:
                    notional_qty = (
                        self._allocated_capital / self._max_open_orders / level.price
                    )
                    gross_profit *= notional_qty
                    # Deduct maker fees (grid uses limit orders)
                    fill_notional = notional_qty * level.price
                    cost = self._cost_model.round_trip_cost(
                        fill_notional, legs=2, is_maker=True,
                    )
                    net_profit = gross_profit - cost
                    # Skip unprofitable fills
                    if net_profit <= 0:
                        actions.append({
                            "action": "grid_sell_skipped",
                            "symbol": symbol,
                            "price": level.price,
                            "gross_profit": round(gross_profit, 4),
                            "cost": round(cost, 4),
                            "reason": "unprofitable_after_fees",
                        })
                        continue
                    pnl += net_profit
                else:
                    net_profit = gross_profit
                    cost = 0.0
                    pnl += gross_profit
                actions.append({
                    "action": "grid_sell_filled",
                    "symbol": symbol,
                    "price": level.price,
                    "gross_profit": round(gross_profit, 4),
                    "cost": round(cost, 4),
                    "profit": round(net_profit, 4),
                })
                logger.debug(
                    "grid_sell_filled",
                    symbol=symbol,
                    level_price=level.price,
                    gross_profit=round(gross_profit, 4),
                    cost=round(cost, 4),
                    net_profit=round(net_profit, 4),
                )

        self._grid_pnl[symbol] = self._grid_pnl.get(symbol, 0) + pnl
        return actions, pnl

    def _is_outside_range(self, symbol: str, current_price: float) -> bool:
        """Check if price has moved beyond the grid's outer levels."""
        grid = self._grids.get(symbol, [])
        if not grid:
            return False

        prices = [lvl.price for lvl in grid]
        grid_low = min(prices)
        grid_high = max(prices)
        return current_price < grid_low or current_price > grid_high

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    async def _get_current_price(self, symbol: str) -> float | None:
        """Get the current price from the first available exchange."""
        for exchange in self._exchanges:
            try:
                ticker = await exchange.get_ticker(symbol)
                return ticker.get("last", 0.0)
            except Exception as e:
                logger.debug(
                    "grid_price_fetch_failed",
                    symbol=symbol,
                    error=str(e),
                )
        return None
