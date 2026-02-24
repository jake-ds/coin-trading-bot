"""Cross-exchange arbitrage engine.

Monitors the same symbol across two exchanges.  When the price spread
exceeds fees + minimum profit, simultaneously buys on the cheaper
exchange and sells on the more expensive one.

Requires pre-funded balances on both exchanges (no actual asset
transfer during the trade — inventory is rebalanced periodically).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from bot.engines.base import BaseEngine, DecisionStep, EngineCycleResult
from bot.engines.cost_model import CostModel
from bot.engines.opportunity_registry import OpportunityRegistry, OpportunityType

if TYPE_CHECKING:
    from bot.config import Settings
    from bot.engines.portfolio_manager import PortfolioManager
    from bot.exchanges.base import ExchangeAdapter

logger = structlog.get_logger(__name__)


class CrossExchangeArbEngine(BaseEngine):
    """Cross-exchange spot arbitrage engine."""

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
            loop_interval=5.0,
            max_positions=10,
            paper_mode=paper_mode,
        )
        self._min_spread_pct = s.cross_arb_min_spread_pct if s else 0.3
        self._symbols = list(s.cross_arb_symbols) if s else ["BTC/USDT", "ETH/USDT"]
        self._max_position_per_symbol = (
            s.cross_arb_max_position_per_symbol if s else 1000.0
        )
        self._rebalance_threshold_pct = (
            s.cross_arb_rebalance_threshold_pct if s else 20.0
        )

        self._cost_model = CostModel()
        self._registry: OpportunityRegistry | None = None

        # Inventory tracking per exchange per symbol
        self._inventory: dict[str, dict[str, float]] = {}
        self._arb_pnl = 0.0

    @property
    def name(self) -> str:
        return "cross_exchange_arb"

    @property
    def description(self) -> str:
        return "Cross-exchange spot arbitrage (buy cheap, sell expensive)"

    def set_registry(self, registry: OpportunityRegistry) -> None:
        """Attach a shared OpportunityRegistry for dynamic symbol discovery."""
        self._registry = registry

    async def _run_cycle(self) -> EngineCycleResult:
        cycle_start = datetime.now(timezone.utc)
        actions: list[dict] = []
        signals: list[dict] = []
        decisions: list[DecisionStep] = []
        pnl_update = 0.0

        # Regime adaptation
        regime_adj = self._get_regime_adjustments()
        regime_label = "NORMAL"
        if self._regime_detector:
            regime_label = self._regime_detector.get_current_regime().value
        t_mult = regime_adj["threshold_mult"]
        s_mult = regime_adj["size_mult"]
        regime_result = "정상 운영"
        if regime_label == "HIGH":
            regime_result = "보수적 모드 적용"
        elif regime_label == "CRISIS":
            regime_result = "위기 모드 — 신규 진입 중단"
        decisions.append(DecisionStep(
            label="시장 레짐",
            observation=f"현재: {regime_label}, "
                        f"threshold×{t_mult:.1f}, size×{s_mult:.1f}",
            threshold="LOW: t×0.8/s×1.2, NORMAL: t×1.0/s×1.0, "
                      "HIGH: t×1.3/s×0.7, CRISIS: 진입 중단",
            result=regime_result,
            category="evaluate",
        ))

        effective_min_spread = self._min_spread_pct * regime_adj["threshold_mult"]
        is_crisis = regime_label == "CRISIS"

        if len(self._exchanges) < 2:
            decisions.append(DecisionStep(
                label="거래소 확인",
                observation=f"연결된 거래소 {len(self._exchanges)}개",
                threshold="최소 2개 거래소 필요",
                result="SKIP - 거래소 부족",
                category="skip",
            ))
            return EngineCycleResult(
                engine_name=self.name,
                cycle_num=self._cycle_count + 1,
                timestamp=cycle_start.isoformat(),
                duration_ms=0.0,
                metadata={"error": "need_at_least_2_exchanges"},
                decisions=decisions,
            )

        # Build symbol list: static config + dynamic from registry
        symbols = list(self._symbols)
        if self._registry:
            discovered = self._registry.get_symbols(
                OpportunityType.CROSS_EXCHANGE_SPREAD, n=10, min_score=20.0,
            )
            for sym in discovered:
                if sym not in symbols:
                    symbols.append(sym)

        for symbol in symbols:
            spread_info = await self._check_spread(symbol)
            if spread_info is None:
                decisions.append(DecisionStep(
                    label=f"{symbol} 가격 비교",
                    observation="가격 조회 실패",
                    threshold=f"최소 스프레드 {self._min_spread_pct}%",
                    result="SKIP - 가격 조회 실패",
                    category="skip",
                ))
                continue

            signals.append(spread_info)
            spread_abs = abs(spread_info["spread_pct"])

            # Use cost-based minimum spread (at least break-even after fees)
            cost_min_spread = self._cost_model.min_spread_for_profit(legs=4)
            effective_min = max(effective_min_spread, cost_min_spread)

            decisions.append(DecisionStep(
                label=f"{symbol} 거래소간 스프레드",
                observation=(
                    f"{spread_info['exchange_a']}: ${spread_info['price_a']:,.2f} | "
                    f"{spread_info['exchange_b']}: ${spread_info['price_b']:,.2f} | "
                    f"스프레드: {spread_info['spread_pct']:+.4f}%"
                ),
                threshold=(
                    f"최소 스프레드 >= {effective_min}% "
                    f"(비용 기반: {cost_min_spread:.2f}%)"
                ),
                result="",  # filled below
                category="evaluate",
            ))

            if is_crisis:
                decisions[-1].result = "SKIP - CRISIS 레짐, 신규 진입 중단"
                decisions[-1].category = "skip"
                continue

            if spread_abs >= effective_min:
                # Log dynamic sizing if available
                if self._dynamic_sizer and self._allocated_capital > 0:
                    mid = spread_info.get("mid_price", 0)
                    if mid > 0:
                        ps = self._dynamic_sizer.calculate_size(
                            symbol=symbol,
                            price=mid,
                            portfolio_value=self._allocated_capital,
                        )
                        decisions.append(DecisionStep(
                            label="포지션 사이징",
                            observation=(
                                f"방법: {ps.method}, 변동성 배수: {ps.vol_multiplier:.2f}, "
                                f"수량: {ps.quantity:.6f}"
                            ),
                            threshold="변동성 배수 범위: [0.25, 2.0]",
                            result=f"사이즈 결정: ${ps.notional_value:.2f}",
                            category="evaluate",
                        ))
                arb_result = await self._execute_arb(symbol, spread_info)
                if arb_result:
                    actions.append(arb_result)
                    pnl_update += arb_result.get("profit", 0)
                    decisions[-1].result = (
                        f"ARB - 매수@{arb_result['buy_exchange']} "
                        f"매도@{arb_result['sell_exchange']}, "
                        f"수익: ${arb_result['profit']:.4f}"
                    )
                    decisions[-1].category = "execute"
                else:
                    decisions[-1].result = "SKIP - 차익거래 실행 실패"
                    decisions[-1].category = "skip"
            else:
                decisions[-1].result = (
                    f"NO ACTION - 스프레드 {spread_abs:.4f}% < "
                    f"최소 {effective_min:.2f}%"
                )
                decisions[-1].category = "skip"

        return EngineCycleResult(
            engine_name=self.name,
            cycle_num=self._cycle_count + 1,
            timestamp=cycle_start.isoformat(),
            duration_ms=0.0,
            actions_taken=actions,
            positions=[],
            signals=signals,
            pnl_update=pnl_update,
            metadata={
                "total_arb_pnl": round(self._arb_pnl + pnl_update, 2),
            },
            decisions=decisions,
        )

    async def _check_spread(self, symbol: str) -> dict[str, Any] | None:
        """Fetch prices from both exchanges and compute spread."""
        prices: dict[str, float] = {}

        for exchange in self._exchanges[:2]:
            try:
                ticker = await exchange.get_ticker(symbol)
                prices[exchange.name] = ticker.get("last", 0.0)
            except Exception as e:
                logger.debug(
                    "cross_arb_price_error",
                    exchange=exchange.name,
                    symbol=symbol,
                    error=str(e),
                )
                return None

        names = list(prices.keys())
        if len(names) < 2:
            return None

        price_a = prices[names[0]]
        price_b = prices[names[1]]
        if price_a <= 0 or price_b <= 0:
            return None

        mid = (price_a + price_b) / 2
        spread_pct = (price_a - price_b) / mid * 100

        return {
            "symbol": symbol,
            "exchange_a": names[0],
            "price_a": price_a,
            "exchange_b": names[1],
            "price_b": price_b,
            "spread_pct": round(spread_pct, 4),
            "mid_price": mid,
        }

    async def _execute_arb(
        self, symbol: str, spread_info: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Execute an arbitrage trade: buy cheap, sell expensive."""
        spread = spread_info["spread_pct"]

        if spread > 0:
            # exchange_a is more expensive → sell on A, buy on B
            buy_exchange = spread_info["exchange_b"]
            sell_exchange = spread_info["exchange_a"]
            buy_price = spread_info["price_b"]
            sell_price = spread_info["price_a"]
        else:
            # exchange_b is more expensive → sell on B, buy on A
            buy_exchange = spread_info["exchange_a"]
            sell_exchange = spread_info["exchange_b"]
            buy_price = spread_info["price_a"]
            sell_price = spread_info["price_b"]

        # Calculate quantity based on max position (dynamic sizer if available)
        if self._dynamic_sizer and self._allocated_capital > 0 and buy_price > 0:
            ps = self._dynamic_sizer.calculate_size(
                symbol=symbol,
                price=buy_price,
                portfolio_value=self._allocated_capital,
            )
            quantity = ps.quantity
        else:
            quantity = self._max_position_per_symbol / buy_price if buy_price > 0 else 0
        if quantity <= 0:
            return None

        gross_profit = (sell_price - buy_price) * quantity
        # Deduct round-trip costs (4 legs: buy+sell on each exchange)
        notional = quantity * buy_price
        cost = self._cost_model.round_trip_cost(notional, legs=4)
        net_profit = gross_profit - cost

        if self._paper_mode:
            self._arb_pnl += net_profit
            logger.info(
                "cross_arb_executed",
                symbol=symbol,
                buy_exchange=buy_exchange,
                sell_exchange=sell_exchange,
                buy_price=buy_price,
                sell_price=sell_price,
                quantity=round(quantity, 6),
                gross_profit=round(gross_profit, 4),
                cost=round(cost, 4),
                net_profit=round(net_profit, 4),
            )
            return {
                "action": "arb_trade",
                "symbol": symbol,
                "buy_exchange": buy_exchange,
                "sell_exchange": sell_exchange,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "quantity": round(quantity, 6),
                "gross_profit": round(gross_profit, 4),
                "cost": round(cost, 4),
                "profit": round(net_profit, 4),
                "spread_pct": abs(spread),
            }

        logger.warning("cross_arb_live_not_implemented", symbol=symbol)
        return None
