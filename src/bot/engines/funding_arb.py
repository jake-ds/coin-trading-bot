"""Funding rate arbitrage engine — bidirectional delta-neutral strategy.

Positive funding (longs pay shorts):
  - Buy spot (long) + Sell perpetual (short) → collect funding every 8h.

Negative funding (shorts pay longs):
  - Sell/short spot + Buy perpetual (long) → collect funding every 8h.

When abs(funding) drops below exit threshold or basis spread widens:
  - Close both legs.

This is a market-neutral strategy: PnL comes from funding payments,
not directional price movement.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from bot.engines.base import BaseEngine, DecisionStep, EngineCycleResult
from bot.engines.cost_model import CostModel
from bot.engines.opportunity_registry import OpportunityRegistry, OpportunityType
from bot.models.base import OrderSide, OrderStatus, OrderType

if TYPE_CHECKING:
    from bot.config import Settings
    from bot.engines.portfolio_manager import PortfolioManager
    from bot.exchanges.base import ExchangeAdapter

logger = structlog.get_logger(__name__)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _compute_leg_pnl(
    side: str, open_price: float, close_price: float, qty: float
) -> float:
    """Compute PnL for a single leg (futures or spot).

    Args:
        side: 'buy' or 'sell' — the side of the opening trade.
        open_price: Fill price when the leg was opened.
        close_price: Fill price when the leg was closed.
        qty: Position quantity.

    Returns:
        Realised PnL in quote currency.
    """
    if side == "buy":
        return (close_price - open_price) * qty
    else:  # sell / short
        return (open_price - close_price) * qty


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
        self._order_timeout = s.funding_arb_order_timeout_seconds if s else 10.0
        self._futures_only_mode = s.funding_arb_futures_only_mode if s else False
        self._reconcile_interval = (
            s.funding_arb_reconcile_interval_cycles if s else 12
        )
        self._funding_monitor = None
        self._cost_model = CostModel()
        self._registry: OpportunityRegistry | None = None
        self._settings = s

    def set_registry(self, registry: OpportunityRegistry) -> None:
        """Attach a shared OpportunityRegistry for dynamic symbol discovery."""
        self._registry = registry

    # ------------------------------------------------------------------
    # Exchange identification
    # ------------------------------------------------------------------

    @property
    def _futures_exchange(self) -> Any | None:
        """Return the first exchange that supports set_leverage (futures)."""
        for ex in self._exchanges:
            if hasattr(ex, "set_leverage"):
                return ex
        return None

    @property
    def _spot_exchange(self) -> Any | None:
        """Return the first exchange that does NOT have set_leverage (spot)."""
        for ex in self._exchanges:
            if not hasattr(ex, "set_leverage"):
                return ex
        return None

    # ------------------------------------------------------------------
    # ABC implementation
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "funding_rate_arb"

    @property
    def description(self) -> str:
        return "Bidirectional delta-neutral funding rate arbitrage"

    async def _run_cycle(self) -> EngineCycleResult:
        """Check funding rates, open/close positions accordingly."""
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

        effective_min_rate = self._min_funding_rate * regime_adj["threshold_mult"]
        is_crisis = regime_label == "CRISIS"

        # Build symbol list: static config + dynamic from registry
        symbols = list(self._symbols)
        if self._registry:
            discovered = self._registry.get_symbols(
                OpportunityType.FUNDING_RATE, n=10, min_score=20.0,
            )
            for sym in discovered:
                # Normalize futures symbol format (e.g. "AGLD/USDT:USDT" → "AGLD/USDT")
                normalized = sym.split(":")[0] if ":" in sym else sym
                if normalized not in symbols:
                    symbols.append(normalized)

        for symbol in symbols:
            rate_info = await self._fetch_funding_rate(symbol)
            if rate_info is None:
                decisions.append(DecisionStep(
                    label=f"{symbol} 펀딩비 조회",
                    observation="데이터 없음 (거래소 응답 실패)",
                    threshold="N/A",
                    result="SKIP - 데이터 없음",
                    category="skip",
                ))
                continue

            funding_rate = rate_info.get("funding_rate", 0)
            spread_pct = rate_info.get("spread_pct", 0)
            abs_rate = abs(funding_rate)
            ann_pct = abs_rate * 3 * 365 * 100
            direction = "숏선물+롱현물" if funding_rate > 0 else "롱선물+숏현물"

            signals.append({
                "symbol": symbol,
                "funding_rate": funding_rate,
                "spread_pct": spread_pct,
                "annualised_pct": ann_pct,
                "direction": direction,
            })

            rate_step = DecisionStep(
                label=f"{symbol} 펀딩비 체크",
                observation=(
                    f"펀딩비 {funding_rate:.6f} ({ann_pct:.1f}% 연환산), "
                    f"방향: {direction}, "
                    f"스프레드 {spread_pct:.4f}%"
                ),
                threshold=(
                    f"진입: |rate| >= {effective_min_rate:.6f}, "
                    f"spread <= {self._max_spread_pct}% | "
                    f"청산: |rate| < {self._exit_funding_rate} "
                    f"or spread > {self._max_spread_pct * 2}%"
                ),
                result="",  # filled below
                category="evaluate",
            )
            decisions.append(rate_step)

            # Cost analysis
            position_capital = self._allocated_capital / max(self._max_positions, 1)
            cost = self._cost_model.round_trip_cost(position_capital, legs=4)
            expected_daily = abs_rate * position_capital * 3  # 3 periods/day
            decisions.append(DecisionStep(
                label="비용 분석",
                observation=f"총비용=${cost:.2f}, 일일 예상수익=${expected_daily:.2f}",
                threshold="일일 예상수익 > 총비용",
                result="수익" if expected_daily > cost else "손실",
                category="evaluate",
            ))

            # Check if we should close existing position
            if symbol in self._positions:
                if self._should_close(funding_rate, spread_pct):
                    pnl = await self._close_position(symbol)
                    pnl_update += pnl
                    reason = (
                        "rate_below_exit"
                        if abs_rate < self._exit_funding_rate
                        else "spread_too_wide"
                    )
                    actions.append({
                        "action": "close",
                        "symbol": symbol,
                        "reason": reason,
                        "pnl": pnl,
                    })
                    rate_step.result = f"CLOSE - {reason}, PnL: {pnl:.4f}"
                    rate_step.category = "execute"
                else:
                    rate_step.result = "HOLD - 기존 포지션 유지"
                    rate_step.category = "decide"
                continue

            # Check if we should open a new position
            if is_crisis:
                rate_step.result = "SKIP - CRISIS 레짐, 신규 진입 중단"
                rate_step.category = "skip"
                continue

            if self._should_open_with_threshold(funding_rate, spread_pct, effective_min_rate):
                if not self._has_capacity(symbol):
                    rate_step.result = "SKIP - 기준 충족이나 포지션 한도 초과"
                    rate_step.category = "skip"
                    continue
                # Log dynamic sizing decision
                if self._dynamic_sizer:
                    price = rate_info.get("mark_price", 0) or rate_info.get("spot_price", 0)
                    if price > 0 and self._allocated_capital > 0:
                        ps = self._dynamic_sizer.calculate_size(
                            symbol=symbol,
                            price=price,
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
                opened = await self._open_position(symbol, rate_info)
                if opened:
                    actions.append({
                        "action": "open",
                        "symbol": symbol,
                        "funding_rate": funding_rate,
                        "annualised_pct": ann_pct,
                        "direction": direction,
                    })
                    rate_step.result = f"OPEN - {direction}, 델타중립 포지션 오픈"
                    rate_step.category = "execute"
                else:
                    rate_step.result = "SKIP - 포지션 오픈 실패"
                    rate_step.category = "skip"
            else:
                rate_step.result = "SKIP - 진입 기준 미달"
                rate_step.category = "skip"

        # Live mode reconciliation
        if not self._paper_mode and self._cycle_count > 0:
            if self._cycle_count % self._reconcile_interval == 0:
                discrepancies = await self._reconcile_positions()
                if discrepancies:
                    decisions.append(DecisionStep(
                        label="포지션 정합성 검증",
                        observation=f"불일치 {len(discrepancies)}건 감지",
                        threshold="exchange vs local 포지션 일치",
                        result=f"WARNING: {discrepancies}",
                        category="execute",
                    ))
                    logger.warning(
                        "funding_arb_reconcile_discrepancy",
                        discrepancies=discrepancies,
                    )

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
                "symbols_monitored": len(symbols),
                "symbols_from_scanner": len(symbols) - len(self._symbols),
                "open_positions": len(self._positions),
            },
            decisions=decisions,
        )

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _should_open(self, funding_rate: float, spread_pct: float) -> bool:
        """Decide whether to open a new delta-neutral position."""
        if abs(funding_rate) < self._min_funding_rate:
            return False
        if abs(spread_pct) > self._max_spread_pct:
            return False
        return True

    def _should_open_with_threshold(
        self, funding_rate: float, spread_pct: float, min_rate: float
    ) -> bool:
        """Decide whether to open, using a regime-adjusted min rate."""
        if abs(funding_rate) < min_rate:
            return False
        if abs(spread_pct) > self._max_spread_pct:
            return False
        return True

    def _should_close(self, funding_rate: float, spread_pct: float) -> bool:
        """Decide whether to close an existing position."""
        if abs(funding_rate) < self._exit_funding_rate:
            return True
        if abs(spread_pct) > self._max_spread_pct * 2:
            return True
        return False

    # ------------------------------------------------------------------
    # Sizing helper (shared by paper & live)
    # ------------------------------------------------------------------

    def _compute_quantity(self, price: float) -> float:
        """Compute position quantity from allocated capital and price."""
        if price <= 0 or self._allocated_capital <= 0:
            return 0.0
        if self._dynamic_sizer:
            ps = self._dynamic_sizer.calculate_size(
                symbol="",  # symbol not needed for quantity
                price=price,
                portfolio_value=self._allocated_capital,
            )
            return ps.quantity
        position_capital = self._allocated_capital / max(self._max_positions, 1)
        return position_capital / price

    # ------------------------------------------------------------------
    # Live order helpers
    # ------------------------------------------------------------------

    async def _place_order_with_retry(
        self,
        exchange: Any,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: float,
        max_retries: int = 3,
    ) -> Any | None:
        """Place an order with exponential backoff on ConnectionError.

        Returns the Order on success, None on failure.
        """
        for attempt in range(max_retries):
            try:
                order = await exchange.create_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=qty,
                )
                logger.info(
                    "funding_arb_order_placed",
                    symbol=symbol,
                    side=side.value,
                    qty=qty,
                    order_id=order.id,
                    attempt=attempt + 1,
                )
                return order
            except ConnectionError as e:
                wait = 2 ** attempt
                logger.warning(
                    "funding_arb_order_retry",
                    symbol=symbol,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait_seconds=wait,
                    error=str(e),
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait)
            except Exception as e:
                logger.error(
                    "funding_arb_order_failed",
                    symbol=symbol,
                    side=side.value,
                    error=str(e),
                )
                return None
        return None

    async def _wait_for_fill(
        self,
        exchange: Any,
        order_id: str,
        symbol: str,
        timeout: float | None = None,
    ) -> Any | None:
        """Poll order status until FILLED, CANCELLED, or FAILED.

        Returns the final Order on fill, None on timeout/cancel/fail.
        """
        if timeout is None:
            timeout = self._order_timeout
        elapsed = 0.0
        poll_interval = 0.5
        while elapsed < timeout:
            try:
                order = await exchange.get_order_status(order_id, symbol)
                if order.status == OrderStatus.FILLED:
                    return order
                if order.status in (OrderStatus.CANCELLED, OrderStatus.FAILED):
                    logger.warning(
                        "funding_arb_order_terminal",
                        order_id=order_id,
                        status=order.status.value,
                    )
                    return None
            except Exception as e:
                logger.debug(
                    "funding_arb_poll_error",
                    order_id=order_id,
                    error=str(e),
                )
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        # Timeout — try to cancel
        logger.warning("funding_arb_order_timeout", order_id=order_id, symbol=symbol)
        await self._safe_cancel(exchange, order_id, symbol)
        return None

    async def _safe_cancel(
        self, exchange: Any, order_id: str, symbol: str
    ) -> bool:
        """Cancel an order, suppressing errors."""
        try:
            return await exchange.cancel_order(order_id, symbol)
        except Exception as e:
            logger.debug(
                "funding_arb_cancel_failed",
                order_id=order_id,
                error=str(e),
            )
            return False

    async def _emergency_unwind_futures(
        self,
        futures_ex: Any,
        symbol: str,
        futures_order: Any,
        futures_side: OrderSide,
    ) -> bool:
        """Emergency close of a futures position when spot hedge fails.

        Retries up to 5 times. Logs CRITICAL on total failure.
        """
        close_side = OrderSide.BUY if futures_side == OrderSide.SELL else OrderSide.SELL
        qty = futures_order.filled_quantity or futures_order.quantity

        for attempt in range(5):
            try:
                order = await futures_ex.create_order(
                    symbol=symbol,
                    side=close_side,
                    order_type=OrderType.MARKET,
                    quantity=qty,
                )
                if order and order.status in (OrderStatus.FILLED, OrderStatus.SUBMITTED):
                    logger.warning(
                        "funding_arb_emergency_unwind_ok",
                        symbol=symbol,
                        order_id=order.id,
                        attempt=attempt + 1,
                    )
                    return True
            except Exception as e:
                logger.error(
                    "funding_arb_emergency_unwind_retry",
                    symbol=symbol,
                    attempt=attempt + 1,
                    error=str(e),
                )
            await asyncio.sleep(2 ** attempt)

        logger.critical(
            "funding_arb_emergency_unwind_failed",
            symbol=symbol,
            futures_order_id=futures_order.id,
            message="MANUAL INTERVENTION REQUIRED — open futures position not closed",
        )
        return False

    async def _reconcile_positions(self) -> list[dict]:
        """Compare exchange positions with internal state.

        Returns a list of discrepancy dicts (empty if clean).
        """
        futures_ex = self._futures_exchange
        if futures_ex is None:
            return []

        discrepancies: list[dict] = []
        try:
            exchange_positions = await futures_ex.get_positions()
        except Exception as e:
            logger.error("funding_arb_reconcile_fetch_failed", error=str(e))
            return [{"type": "fetch_failed", "error": str(e)}]

        # Build lookup: symbol → exchange position
        ex_by_symbol: dict[str, dict] = {}
        for ep in exchange_positions:
            sym = ep.get("symbol", "")
            if sym:
                ex_by_symbol[sym] = ep

        # Check local positions exist on exchange
        for symbol, local_pos in self._positions.items():
            if local_pos.get("mode") != "live":
                continue
            if symbol not in ex_by_symbol:
                discrepancies.append({
                    "type": "missing_on_exchange",
                    "symbol": symbol,
                    "local": local_pos,
                })

        # Check exchange positions exist locally
        for sym, ex_pos in ex_by_symbol.items():
            if sym not in self._positions:
                discrepancies.append({
                    "type": "missing_locally",
                    "symbol": sym,
                    "exchange": ex_pos,
                })

        return discrepancies

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    async def _open_position(
        self, symbol: str, rate_info: dict[str, Any]
    ) -> bool:
        """Open a delta-neutral position (bidirectional).

        Positive rate → long spot + short perp (collect from longs)
        Negative rate → short spot + long perp (collect from shorts)
        """
        if self._paper_mode:
            # Simulate position
            price = rate_info.get("mark_price", 0) or rate_info.get("spot_price", 0)
            if price <= 0:
                return False

            funding_rate = rate_info.get("funding_rate", 0)
            side = "long_spot_short_perp" if funding_rate > 0 else "short_spot_long_perp"

            # Dynamic sizing if available, else fixed
            if self._dynamic_sizer and self._allocated_capital > 0:
                from bot.risk.dynamic_sizer import PositionSize

                ps: PositionSize = self._dynamic_sizer.calculate_size(
                    symbol=symbol,
                    price=price,
                    portfolio_value=self._allocated_capital,
                )
                quantity = ps.quantity
            else:
                # Size: allocate equal share of capital across max_positions
                position_capital = self._allocated_capital / max(self._max_positions, 1)
                quantity = position_capital / price if price > 0 else 0
            if quantity <= 0:
                return False

            self._add_position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                funding_rate=funding_rate,
                spot_price=rate_info.get("spot_price", 0),
                mark_price=rate_info.get("mark_price", 0),
            )
            logger.info(
                "funding_arb_position_opened",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                funding_rate=funding_rate,
            )
            return True

        # ------------------------------------------------------------------
        # Live mode: place actual orders on futures (+ optional spot hedge)
        # ------------------------------------------------------------------
        futures_ex = self._futures_exchange
        if futures_ex is None:
            logger.error("funding_arb_no_futures_exchange")
            return False

        price = rate_info.get("mark_price", 0) or rate_info.get("spot_price", 0)
        if price <= 0:
            return False

        funding_rate = rate_info.get("funding_rate", 0)
        quantity = self._compute_quantity(price)
        if quantity <= 0:
            return False

        # Determine futures side based on funding direction
        # Positive funding → short futures (collect from longs), buy spot
        # Negative funding → long futures (collect from shorts)
        if funding_rate > 0:
            futures_side = OrderSide.SELL
            spot_side = OrderSide.BUY
            side_label = "long_spot_short_perp"
        else:
            futures_side = OrderSide.BUY
            spot_side = OrderSide.SELL
            side_label = "short_spot_long_perp"

        # 1. Set leverage & margin mode (best effort — don't fail on error)
        try:
            await futures_ex.set_leverage(symbol, self._leverage)
        except Exception as e:
            logger.debug("funding_arb_set_leverage_failed", error=str(e))
        try:
            await futures_ex.set_margin_mode(symbol, "cross")
        except Exception as e:
            logger.debug("funding_arb_set_margin_mode_failed", error=str(e))

        # 2. Place futures MARKET order
        futures_order = await self._place_order_with_retry(
            futures_ex, symbol, futures_side, OrderType.MARKET, quantity
        )
        if futures_order is None:
            logger.error("funding_arb_futures_order_failed", symbol=symbol)
            return False

        # 3. Wait for futures fill
        filled_futures = await self._wait_for_fill(
            futures_ex, futures_order.id, symbol
        )
        if filled_futures is None:
            logger.error(
                "funding_arb_futures_fill_timeout",
                symbol=symbol,
                order_id=futures_order.id,
            )
            return False

        # 4. Attempt spot hedge (only for positive funding, spot available, not futures-only)
        spot_hedged = False
        spot_order_id = None
        spot_filled_price = None
        spot_ex = self._spot_exchange

        want_spot_hedge = (
            funding_rate > 0
            and spot_ex is not None
            and not self._futures_only_mode
        )

        if want_spot_hedge:
            spot_order = await self._place_order_with_retry(
                spot_ex, symbol, spot_side, OrderType.MARKET, quantity
            )
            if spot_order is not None:
                filled_spot = await self._wait_for_fill(
                    spot_ex, spot_order.id, symbol
                )
                if filled_spot is not None:
                    spot_hedged = True
                    spot_order_id = filled_spot.id
                    spot_filled_price = filled_spot.filled_price or price
                else:
                    # Spot failed after futures filled — emergency unwind
                    logger.error(
                        "funding_arb_spot_fill_failed_emergency_unwind",
                        symbol=symbol,
                    )
                    await self._emergency_unwind_futures(
                        futures_ex, symbol, filled_futures, futures_side
                    )
                    return False
            else:
                # Spot order placement failed — emergency unwind
                logger.error(
                    "funding_arb_spot_order_failed_emergency_unwind",
                    symbol=symbol,
                )
                await self._emergency_unwind_futures(
                    futures_ex, symbol, filled_futures, futures_side
                )
                return False

        # 5. Record position
        futures_filled_price = filled_futures.filled_price or price
        total_fees = filled_futures.fee
        if spot_hedged and filled_futures.fee:
            # We'll approximate spot fee from futures fee
            total_fees = filled_futures.fee * 2

        self._add_position(
            symbol=symbol,
            side=side_label,
            quantity=quantity,
            entry_price=futures_filled_price,
            funding_rate=funding_rate,
            spot_price=rate_info.get("spot_price", 0),
            mark_price=rate_info.get("mark_price", 0),
            # Live-specific fields
            mode="live",
            futures_order_id=filled_futures.id,
            futures_filled_price=futures_filled_price,
            futures_side=futures_side.value,
            spot_order_id=spot_order_id,
            spot_hedged=spot_hedged,
            spot_filled_price=spot_filled_price,
            funding_payments_collected=0.0,
            total_fees=total_fees,
        )

        logger.info(
            "funding_arb_live_position_opened",
            symbol=symbol,
            side=side_label,
            quantity=quantity,
            futures_price=futures_filled_price,
            spot_hedged=spot_hedged,
            futures_order_id=filled_futures.id,
        )
        return True

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
            # Approximate one funding payment — always positive (we collect)
            gross_pnl = abs(funding_rate) * quantity * entry_price
            # Deduct round-trip trading costs (4 legs: open+close on spot+perp)
            notional = quantity * entry_price
            net_pnl = self._cost_model.net_profit(gross_pnl, notional, legs=4)
            logger.info(
                "funding_arb_position_closed",
                symbol=symbol,
                side=pos.get("side", "unknown"),
                gross_pnl=round(gross_pnl, 4),
                cost=round(gross_pnl - net_pnl, 4),
                net_pnl=round(net_pnl, 4),
            )
            return net_pnl

        # ------------------------------------------------------------------
        # Live mode: close actual exchange positions
        # ------------------------------------------------------------------
        futures_ex = self._futures_exchange
        if futures_ex is None:
            # Can't close — re-add position for next cycle
            self._positions[symbol] = pos
            logger.error("funding_arb_close_no_futures_exchange", symbol=symbol)
            return 0.0

        quantity = pos.get("quantity", 0)
        futures_side_str = pos.get("futures_side", "")
        futures_open_price = pos.get("futures_filled_price", pos.get("entry_price", 0))

        # Close futures: opposite side
        close_side = (
            OrderSide.BUY if futures_side_str == "SELL" else OrderSide.SELL
        )

        close_order = await self._place_order_with_retry(
            futures_ex, symbol, close_side, OrderType.MARKET, quantity
        )
        if close_order is None:
            # Re-add position for retry next cycle
            self._positions[symbol] = pos
            logger.error(
                "funding_arb_close_futures_failed",
                symbol=symbol,
                message="position re-added for next cycle",
            )
            return 0.0

        filled_close = await self._wait_for_fill(
            futures_ex, close_order.id, symbol
        )
        if filled_close is None:
            self._positions[symbol] = pos
            logger.error(
                "funding_arb_close_futures_timeout",
                symbol=symbol,
                message="position re-added for next cycle",
            )
            return 0.0

        futures_close_price = filled_close.filled_price or futures_open_price
        pnl = _compute_leg_pnl(
            side=futures_side_str.lower(),
            open_price=futures_open_price,
            close_price=futures_close_price,
            qty=quantity,
        )
        total_fees = pos.get("total_fees", 0.0) + (filled_close.fee or 0.0)

        # Close spot hedge if present
        if pos.get("spot_hedged"):
            spot_ex = self._spot_exchange
            if spot_ex is not None:
                spot_open_price = pos.get("spot_filled_price", futures_open_price)
                # Spot close is opposite of spot open
                spot_side_str = pos.get("side", "")
                if spot_side_str == "long_spot_short_perp":
                    spot_close_side = OrderSide.SELL
                    spot_open_side = "buy"
                else:
                    spot_close_side = OrderSide.BUY
                    spot_open_side = "sell"

                spot_close_order = await self._place_order_with_retry(
                    spot_ex, symbol, spot_close_side, OrderType.MARKET, quantity
                )
                if spot_close_order is not None:
                    filled_spot_close = await self._wait_for_fill(
                        spot_ex, spot_close_order.id, symbol
                    )
                    if filled_spot_close is not None:
                        spot_close_price = (
                            filled_spot_close.filled_price or spot_open_price
                        )
                        pnl += _compute_leg_pnl(
                            side=spot_open_side,
                            open_price=spot_open_price,
                            close_price=spot_close_price,
                            qty=quantity,
                        )
                        total_fees += filled_spot_close.fee or 0.0

        net_pnl = pnl - total_fees
        logger.info(
            "funding_arb_live_position_closed",
            symbol=symbol,
            gross_pnl=round(pnl, 4),
            fees=round(total_fees, 4),
            net_pnl=round(net_pnl, 4),
        )
        return net_pnl

    async def _fetch_funding_rate(self, symbol: str) -> dict[str, Any] | None:
        """Fetch funding rate — uses exchange adapter, ccxt, or public API."""
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

        # Fallback: Binance public API (no auth needed)
        return await self._fetch_funding_rate_public(symbol)

    async def _fetch_funding_rate_public(self, symbol: str) -> dict[str, Any] | None:
        """Fetch funding rate from Binance public API (no API key needed)."""
        import aiohttp

        binance_symbol = symbol.replace("/", "")
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={binance_symbol}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    mark = float(data.get("markPrice", 0))
                    index = float(data.get("indexPrice", 0))
                    spread = (mark - index) / index * 100 if index > 0 else 0
                    return {
                        "symbol": symbol,
                        "funding_rate": float(data.get("lastFundingRate", 0)),
                        "mark_price": mark,
                        "spot_price": index,
                        "spread_pct": spread,
                    }
        except Exception as e:
            logger.debug("funding_rate_public_api_failed", symbol=symbol, error=str(e))
            return None
