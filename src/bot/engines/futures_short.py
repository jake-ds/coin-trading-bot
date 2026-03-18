"""Futures short trading engine — profits from bearish markets.

Opens SHORT futures positions when on-chain signals indicate SELL,
using isolated margin with configurable leverage (default 2x).
Reuses the same signal infrastructure as OnChainTraderEngine.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from bot.engines.base import BaseEngine, DecisionStep, EngineCycleResult
from bot.models.base import OrderSide, OrderType
from bot.onchain.alternative import FearGreedFetcher
from bot.onchain.coingecko import CoinGeckoFetcher
from bot.onchain.coinglass import CoinGlassFetcher
from bot.onchain.defillama import DeFiLlamaFetcher
from bot.onchain.etherscan import EtherscanFetcher
from bot.onchain.models import CompositeSignal, SignalAction
from bot.onchain.signals import compute_composite_signal

if TYPE_CHECKING:
    from bot.config import Settings
    from bot.engines.portfolio_manager import PortfolioManager
    from bot.exchanges.binance_futures import BinanceFuturesAdapter

logger = structlog.get_logger(__name__)


class FuturesShortEngine(BaseEngine):
    """Futures short trading engine driven by on-chain signals.

    Each cycle:
    1. Fetch on-chain data from all APIs in parallel
    2. Compute composite signal for each symbol
    3. Check existing SHORT positions for exit conditions
    4. Open new SHORT positions on strong SELL signals

    Key differences from OnChainTraderEngine (spot longs):
    - Opens SHORT on SELL signals (score below sell_threshold)
    - PnL is inverted: profit when price drops, loss when price rises
    - Uses futures exchange with leverage and isolated margin
    - Stop-loss triggers when price goes UP (adverse for shorts)
    - Take-profit triggers when price goes DOWN (favorable for shorts)
    """

    supports_live: bool = True

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        futures_exchange: BinanceFuturesAdapter | None = None,
        paper_mode: bool = True,
        settings: Settings | None = None,
        signal_source: Any | None = None,
    ):
        s = settings
        loop_interval = (
            getattr(s, "futures_short_loop_interval", 120.0) if s else 120.0
        )
        max_positions = (
            getattr(s, "futures_short_max_positions", 10) if s else 10
        )

        super().__init__(
            portfolio_manager=portfolio_manager,
            exchanges=[],  # We use _futures_exchange directly
            loop_interval=loop_interval,
            max_positions=max_positions,
            paper_mode=paper_mode,
        )

        self._settings = settings
        self._futures_exchange = futures_exchange

        self._symbols: list[str] = (
            getattr(s, "futures_short_symbols", None) if s else None
        ) or [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
            "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
        ]

        # Leverage & margin
        self._leverage = (
            getattr(s, "futures_short_leverage", 2) if s else 2
        )
        self._margin_mode = (
            getattr(s, "futures_short_margin_mode", "isolated") if s else "isolated"
        )

        # Signal thresholds — SHORT on bearish signals
        self._sell_threshold = (
            getattr(s, "futures_short_sell_threshold", -20.0) if s else -20.0
        )
        self._min_confidence = (
            getattr(s, "futures_short_min_confidence", 0.35) if s else 0.35
        )
        # Reuse onchain signal weights
        self._signal_weights = (
            getattr(s, "onchain_signal_weights", None) if s else None
        )
        # Use onchain thresholds for signal computation
        self._buy_threshold = (
            getattr(s, "onchain_buy_threshold", 30.0) if s else 30.0
        )

        # Position sizing
        self._max_position_pct = (
            getattr(s, "futures_short_max_position_pct", 10.0) if s else 10.0
        )
        self._stop_loss_pct = (
            getattr(s, "futures_short_stop_loss_pct", 4.0) if s else 4.0
        )
        self._take_profit_pct = (
            getattr(s, "futures_short_take_profit_pct", 5.0) if s else 5.0
        )
        self._trailing_stop_pct = (
            getattr(s, "futures_short_trailing_stop_pct", 2.5) if s else 2.5
        )
        self._trailing_activate_pct = (
            getattr(s, "futures_short_trailing_activate_pct", 1.5) if s else 1.5
        )

        # Signal source: reuse onchain_trader's signals to avoid duplicate API calls
        self._signal_source = signal_source  # OnChainTraderEngine reference

        # Own fetchers as fallback (only used if no signal_source)
        self._coingecko = CoinGeckoFetcher()
        self._fear_greed = FearGreedFetcher()
        self._defillama = DeFiLlamaFetcher()
        self._coinglass = CoinGlassFetcher(
            api_key=getattr(s, "coinglass_api_key", "") if s else ""
        )
        self._etherscan = EtherscanFetcher(
            api_key=getattr(s, "etherscan_api_key", "") if s else ""
        )

        # Latest signals (for dashboard display)
        self._latest_signals: dict[str, CompositeSignal] = {}

    @property
    def name(self) -> str:
        return "futures_short"

    @property
    def description(self) -> str:
        mode = "paper" if self._paper_mode else "live"
        return f"Futures short trader ({self._leverage}x leverage, {mode})"

    @property
    def latest_signals(self) -> dict[str, dict]:
        return {k: v.to_dict() for k, v in self._latest_signals.items()}

    def _get_shared_signals(self) -> dict[str, CompositeSignal] | None:
        """Try to get signals from the shared signal source (onchain_trader).

        When shared signals are available, re-evaluate action thresholds
        using this engine's own sell_threshold (which is typically less
        negative than onchain_trader's, to trigger shorts more easily).
        """
        if self._signal_source is None:
            return None
        try:
            raw = self._signal_source._latest_signals
            if not raw:
                return None
            # Re-evaluate action using this engine's thresholds
            result: dict[str, CompositeSignal] = {}
            for symbol, sig in raw.items():
                action = SignalAction.HOLD
                if (
                    sig.score >= self._buy_threshold
                    and sig.confidence >= self._min_confidence
                ):
                    action = SignalAction.BUY
                elif (
                    sig.score <= self._sell_threshold
                    and sig.confidence >= self._min_confidence
                ):
                    action = SignalAction.SELL
                result[symbol] = CompositeSignal(
                    symbol=sig.symbol,
                    action=action,
                    score=sig.score,
                    confidence=sig.confidence,
                    signals=sig.signals,
                    timestamp=sig.timestamp,
                )
            return result
        except Exception:
            pass
        return None

    async def _run_cycle(self) -> EngineCycleResult:
        """Execute one trading cycle for futures shorts."""
        decisions: list[DecisionStep] = []
        actions: list[dict] = []
        pnl_update = 0.0
        data_sources: list[str] = []

        # 1. Try to reuse signals from onchain_trader (avoids duplicate API calls)
        shared = self._get_shared_signals()
        if shared:
            # Reuse onchain_trader signals
            for symbol in self._symbols:
                sig = shared.get(symbol)
                if sig is not None:
                    self._latest_signals[symbol] = sig

            decisions.append(DecisionStep(
                label="시그널 공유 (선물숏)",
                observation=f"onchain_trader 시그널 {len(shared)}개 재사용",
                threshold="N/A",
                result="OK",
                category="evaluate",
            ))
            data_sources.append(f"shared({len(shared)})")
        else:
            # Fallback: fetch own data
            decisions.append(DecisionStep(
                label="데이터 수집 (선물숏)",
                observation="모든 API 병렬 요청 시작",
                threshold="N/A",
                result="FETCHING",
                category="evaluate",
            ))

            (
                market_data,
                sentiment_data,
                defi_data,
                derivatives_data,
                flow_data,
            ) = await self._fetch_all_data()

            if market_data:
                data_sources.append(f"CoinGecko({len(market_data)})")
            if sentiment_data:
                data_sources.append(f"F&G({sentiment_data.value})")
            if defi_data:
                data_sources.append("DeFiLlama")
            if derivatives_data:
                data_sources.append(f"CoinGlass({len(derivatives_data)})")
            if flow_data:
                data_sources.append(f"Flow({len(flow_data)})")

            decisions[-1] = DecisionStep(
                label="데이터 수집 (선물숏)",
                observation=f"수집 완료: {', '.join(data_sources) if data_sources else 'None'}",
                threshold="최소 1개 소스",
                result="OK" if data_sources else "NO_DATA",
                category="evaluate",
            )

            # Compute composite signals for each symbol
            for symbol in self._symbols:
                market = market_data.get(symbol) if market_data else None
                deriv = derivatives_data.get(symbol) if derivatives_data else None
                flow = flow_data.get(symbol) if flow_data else None

                signal = compute_composite_signal(
                    symbol=symbol,
                    market=market,
                    sentiment=sentiment_data,
                    defi=defi_data,
                    derivatives=deriv,
                    whale_flow=flow,
                    weights=self._signal_weights,
                    buy_threshold=self._buy_threshold,
                    sell_threshold=self._sell_threshold,
                    min_confidence=self._min_confidence,
                )
                self._latest_signals[symbol] = signal

        # 2. Apply momentum-based sentiment adjustment for shorts
        # The default signal uses contrarian sentiment (Fear=BUY).
        # For short trading, we want momentum: Fear = bearish = SHORT.
        # Flip the sentiment component for short-side evaluation.
        for symbol in list(self._latest_signals.keys()):
            sig = self._latest_signals[symbol]
            sentiment_sig = None
            for s in sig.signals:
                if s.name == "sentiment":
                    sentiment_sig = s
                    break
            if sentiment_sig and sentiment_sig.confidence > 0:
                # Reverse the contrarian sentiment for shorts
                # Original: Fear=+60, Greed=-60
                # Momentum: Fear=-60 (bearish), Greed=+60 (bullish)
                flipped_score = -sentiment_sig.score
                # Recompute weighted score with flipped sentiment
                w = self._signal_weights or {
                    "whale_flow": 0.25, "sentiment": 0.15,
                    "defi_flow": 0.20, "derivatives": 0.25,
                    "market_trend": 0.15,
                }
                total_weight = 0.0
                weighted_score = 0.0
                weighted_conf = 0.0
                for s in sig.signals:
                    weight = w.get(s.name, 0.0)
                    if s.confidence > 0:
                        sc = flipped_score if s.name == "sentiment" else s.score
                        weighted_score += sc * weight
                        weighted_conf += s.confidence * weight
                        total_weight += weight
                if total_weight > 0:
                    new_score = weighted_score / total_weight
                    new_conf = weighted_conf / total_weight
                else:
                    new_score = sig.score
                    new_conf = sig.confidence

                action = SignalAction.HOLD
                if new_score >= self._buy_threshold and new_conf >= self._min_confidence:
                    action = SignalAction.BUY
                elif new_score <= self._sell_threshold and new_conf >= self._min_confidence:
                    action = SignalAction.SELL

                self._latest_signals[symbol] = CompositeSignal(
                    symbol=sig.symbol,
                    action=action,
                    score=new_score,
                    confidence=new_conf,
                    signals=sig.signals,
                    timestamp=sig.timestamp,
                )

        # 3. Log signal decisions
        for symbol in self._symbols:
            signal = self._latest_signals.get(symbol)
            if signal is None:
                continue
            decisions.append(DecisionStep(
                label=f"{symbol} 숏 시그널",
                observation=f"점수={signal.score:+.1f}, 신뢰도={signal.confidence:.2f}",
                threshold=f"숏 진입≤{self._sell_threshold}, 신뢰도≥{self._min_confidence}",
                result=(
                    "SHORT_CANDIDATE" if signal.action == SignalAction.SELL
                    else signal.action.value
                ),
                category="evaluate",
            ))

        # 3. Check existing SHORT positions for exit conditions
        for symbol in list(self._positions.keys()):
            pos = self._positions.get(symbol)
            if pos is None:
                continue

            current_price = await self._get_current_price(symbol)
            if current_price is None:
                continue

            entry_price = pos["entry_price"]
            quantity = pos.get("quantity", 0)

            # SHORT PnL: profit when price goes DOWN
            pnl_pct = ((entry_price - current_price) / entry_price) * 100.0

            # Update position with live data for dashboard
            pos["current_price"] = current_price
            pos["unrealized_pnl"] = round(
                (entry_price - current_price) * quantity, 4
            )
            pos["pnl_pct"] = round(pnl_pct, 2)

            exit_reason = self._check_exit(symbol, pos, current_price, pnl_pct)

            if exit_reason:
                realized_pnl = await self._close_position(
                    symbol, current_price, exit_reason
                )
                pnl_update += realized_pnl
                actions.append({
                    "action": "close_short",
                    "symbol": symbol,
                    "reason": exit_reason,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl": realized_pnl,
                    "pnl_pct": round(pnl_pct, 2),
                })
                decisions.append(DecisionStep(
                    label=f"{symbol} 숏 청산",
                    observation=f"현재가={current_price:.2f}, PnL={pnl_pct:+.1f}%",
                    threshold=f"손절=+{self._stop_loss_pct}%(역행), 익절=+{self._take_profit_pct}%(순행)",
                    result=f"CLOSE_SHORT — {exit_reason}",
                    category="execute",
                ))
            else:
                # Update trailing stop low watermark (for shorts, track the LOW)
                if pnl_pct >= self._trailing_activate_pct:
                    low = pos.get("low_price", entry_price)
                    if current_price < low:
                        pos["low_price"] = current_price
                decisions.append(DecisionStep(
                    label=f"{symbol} 숏 포지션 점검",
                    observation=f"현재가={current_price:.2f}, PnL={pnl_pct:+.1f}%",
                    threshold=f"손절=+{self._stop_loss_pct}%(역행), 익절=+{self._take_profit_pct}%(순행)",
                    result="HOLD_SHORT",
                    category="evaluate",
                ))

        # 4. Open new SHORT positions on strong SELL signals
        regime_adj = self._get_regime_adjustments()
        if regime_adj["size_mult"] == 0.0:
            decisions.append(DecisionStep(
                label="CRISIS 레짐",
                observation="CRISIS 레짐 감지 — 신규 숏 진입 차단",
                threshold="size_mult > 0",
                result="BLOCKED",
                category="decide",
            ))
        else:
            for symbol in self._symbols:
                signal = self._latest_signals.get(symbol)
                if signal is None:
                    continue

                # SHORT on SELL signals (bearish)
                if signal.action != SignalAction.SELL:
                    continue

                if symbol in self._positions:
                    continue

                if not self._has_capacity(symbol):
                    decisions.append(DecisionStep(
                        label=f"{symbol} 숏 진입 시도",
                        observation=f"SELL 시그널 (점수={signal.score:+.1f})",
                        threshold=f"최대 {self._max_positions}포지션",
                        result="SKIP — 용량 초과",
                        category="skip",
                    ))
                    continue

                entry_result, fail_reason = await self._open_short(symbol, signal, regime_adj)
                if entry_result:
                    actions.append(entry_result)
                    decisions.append(DecisionStep(
                        label=f"{symbol} 숏 진입",
                        observation=(
                            f"SELL 시그널 (점수={signal.score:+.1f}, "
                            f"신뢰도={signal.confidence:.2f})"
                        ),
                        threshold=(
                            f"≤{self._sell_threshold}점, "
                            f"≥{self._min_confidence}신뢰도"
                        ),
                        result=f"OPEN_SHORT — ${entry_result.get('notional', 0):.2f} ({self._leverage}x)",
                        category="execute",
                    ))
                elif fail_reason:
                    decisions.append(DecisionStep(
                        label=f"{symbol} 숏 진입 실패",
                        observation=f"SELL 시그널 (점수={signal.score:+.1f})",
                        threshold=fail_reason,
                        result="FAILED",
                        category="skip",
                    ))

        return EngineCycleResult(
            engine_name=self.name,
            cycle_num=self._cycle_count + 1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=0.0,
            actions_taken=actions,
            positions=list(self._positions.values()),
            signals=[s.to_dict() for s in self._latest_signals.values()],
            pnl_update=pnl_update,
            metadata={
                "symbols": self._symbols,
                "data_sources": data_sources,
                "position_count": len(self._positions),
                "leverage": self._leverage,
                "margin_mode": self._margin_mode,
            },
            decisions=decisions,
        )

    async def _fetch_all_data(self) -> tuple:
        """Fetch all on-chain data sources in parallel."""
        results = await asyncio.gather(
            self._safe_fetch(self._coingecko.fetch_market_data, self._symbols),
            self._safe_fetch(self._fear_greed.fetch),
            self._safe_fetch(self._defillama.fetch),
            self._safe_fetch(self._coinglass.fetch_derivatives, self._symbols),
            self._safe_fetch(self._coinglass.fetch_exchange_flow, self._symbols),
            return_exceptions=True,
        )

        processed = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("data_fetch_exception", error=str(r))
                processed.append(None)
            else:
                processed.append(r)

        return tuple(processed)

    @staticmethod
    async def _safe_fetch(coro, *args) -> Any:
        try:
            return await coro(*args)
        except Exception:
            return None

    async def _get_current_price(self, symbol: str) -> float | None:
        """Get current price from futures exchange."""
        if self._futures_exchange is None:
            # Use CoinGecko cached price as fallback
            sig = self._latest_signals.get(symbol)
            if sig and sig.signals:
                for s in sig.signals:
                    if s.name == "market_trend":
                        pass
            return None

        try:
            ticker = await self._futures_exchange.get_ticker(symbol)
            return ticker.get("last", 0.0)
        except Exception as e:
            logger.warning("futures_price_error", symbol=symbol, error=str(e))
            return None

    def _check_exit(
        self, symbol: str, pos: dict, current_price: float, pnl_pct: float
    ) -> str | None:
        """Check if SHORT position should be closed.

        For shorts:
        - pnl_pct > 0 means price went DOWN (profit)
        - pnl_pct < 0 means price went UP (loss)
        """
        # Stop loss — price went UP too much (adverse for shorts)
        if pnl_pct <= -self._stop_loss_pct:
            return f"stop_loss ({pnl_pct:+.1f}%, price rose)"

        # Take profit — price went DOWN enough
        if pnl_pct >= self._take_profit_pct:
            return f"take_profit ({pnl_pct:+.1f}%, price dropped)"

        # Trailing stop for shorts
        entry_price = pos["entry_price"]
        low_price = pos.get("low_price", entry_price)
        if low_price < entry_price:
            low_pnl = ((entry_price - low_price) / entry_price) * 100.0
            if low_pnl >= self._trailing_activate_pct:
                # Price bounced UP from the low
                bounce_from_low = (
                    (current_price - low_price) / low_price
                ) * 100.0
                if bounce_from_low >= self._trailing_stop_pct:
                    return (
                        f"trailing_stop (low={low_price:.2f}, "
                        f"bounce={bounce_from_low:.1f}%)"
                    )

        # Signal reversal — bullish signal while we're short
        signal = self._latest_signals.get(symbol)
        if signal and signal.action == SignalAction.BUY:
            return f"signal_reversal (score={signal.score:+.1f}, now BUY)"

        return None

    async def _open_short(
        self, symbol: str, signal: CompositeSignal, regime_adj: dict
    ) -> tuple[dict | None, str | None]:
        """Open a new SHORT futures position.

        Returns:
            (result_dict, None) on success
            (None, fail_reason) on failure
        """
        # Get current price
        price = None
        if self._futures_exchange:
            try:
                ticker = await self._futures_exchange.get_ticker(symbol)
                price = ticker.get("last", 0.0)
            except Exception as e:
                logger.warning("futures_ticker_error", symbol=symbol, error=str(e))
                return None, f"가격 조회 실패: {e}"

        if not price or price <= 0:
            return None, "가격 없음 (거래소 미연결)"

        # Calculate position size
        capital = self._allocated_capital
        if capital <= 0:
            return None, f"배정 자본 부족 (${capital:.2f})"

        # Base size: max_position_pct of capital
        base_size_usd = capital * (self._max_position_pct / 100.0)

        # Scale by confidence (0.35-1.0 → 50%-100%)
        confidence_scale = 0.5 + (signal.confidence - 0.35) / 1.3
        confidence_scale = max(0.5, min(1.0, confidence_scale))

        # Apply regime adjustment
        size_usd = base_size_usd * confidence_scale * regime_adj["size_mult"]

        # With leverage, the notional value is size_usd * leverage
        # but margin required is just size_usd
        notional = size_usd * self._leverage

        # Minimum order check ($10 on Binance Futures)
        if notional < 10.0:
            logger.debug(
                "short_position_too_small",
                symbol=symbol,
                notional=notional,
            )
            return None, f"최소 주문금액 미달 (${notional:.2f} < $10)"

        quantity = notional / price

        # Execute order
        if self._paper_mode:
            self._add_position(
                symbol=symbol,
                side="short",
                quantity=quantity,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0.0,
                low_price=price,
                margin_usd=size_usd,
                notional_usd=notional,
                leverage=self._leverage,
                signal_score=signal.score,
                signal_confidence=signal.confidence,
            )
            logger.info(
                "paper_short_opened",
                symbol=symbol,
                price=price,
                quantity=round(quantity, 6),
                notional=round(notional, 2),
                margin=round(size_usd, 2),
                leverage=self._leverage,
                signal_score=round(signal.score, 1),
            )
        else:
            if self._futures_exchange is None:
                logger.error("no_futures_exchange_for_live_short")
                return None, "선물 거래소 미연결"

            try:
                # Ensure leverage and margin mode are set
                await self._futures_exchange.ensure_leverage_and_margin(
                    symbol, self._leverage, self._margin_mode
                )

                # SELL to open SHORT
                order = await self._futures_exchange.create_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                )
                filled_price = (
                    order.filled_price
                    if hasattr(order, "filled_price") and order.filled_price
                    else price
                )
                filled_qty = (
                    order.filled_quantity
                    if hasattr(order, "filled_quantity") and order.filled_quantity
                    else quantity
                )
                actual_notional = filled_qty * filled_price
                actual_margin = actual_notional / self._leverage

                self._add_position(
                    symbol=symbol,
                    side="short",
                    quantity=filled_qty,
                    entry_price=filled_price,
                    current_price=filled_price,
                    unrealized_pnl=0.0,
                    low_price=filled_price,
                    margin_usd=actual_margin,
                    notional_usd=actual_notional,
                    leverage=self._leverage,
                    signal_score=signal.score,
                    signal_confidence=signal.confidence,
                    order_id=getattr(order, "id", ""),
                )
                price = filled_price
                quantity = filled_qty
                notional = actual_notional
                size_usd = actual_margin

                logger.info(
                    "live_short_opened",
                    symbol=symbol,
                    price=price,
                    quantity=round(quantity, 6),
                    notional=round(notional, 2),
                    margin=round(size_usd, 2),
                    leverage=self._leverage,
                )
            except Exception as e:
                logger.error(
                    "short_order_failed", symbol=symbol, error=str(e)
                )
                await self._send_alert(f"SHORT ORDER FAILED: {symbol} — {e}")
                return None, f"주문 실패: {e}"

        await self._send_alert(
            f"SHORT opened: {symbol} @ {price:.2f}, "
            f"size=${notional:.2f} ({self._leverage}x), "
            f"signal={signal.score:+.1f}"
        )

        return {
            "action": "open_short",
            "symbol": symbol,
            "side": "short",
            "price": price,
            "quantity": round(quantity, 6),
            "notional": round(notional, 2),
            "margin": round(size_usd, 2),
            "leverage": self._leverage,
            "signal_score": round(signal.score, 1),
            "signal_confidence": round(signal.confidence, 3),
        }, None

    async def _close_position(
        self, symbol: str, current_price: float, reason: str
    ) -> float:
        """Close a SHORT position and return realized PnL."""
        pos = self._positions.get(symbol)
        if pos is None:
            return 0.0

        entry_price = pos["entry_price"]
        quantity = pos["quantity"]
        # SHORT PnL: profit = (entry - exit) * quantity
        pnl = (entry_price - current_price) * quantity

        if self._paper_mode:
            self._remove_position(symbol)
            logger.info(
                "paper_short_closed",
                symbol=symbol,
                entry=entry_price,
                exit=current_price,
                pnl=round(pnl, 4),
                reason=reason,
            )
        else:
            if self._futures_exchange is None:
                logger.error("no_futures_exchange_for_close")
                return 0.0

            try:
                # BUY to close SHORT (reduceOnly)
                await self._futures_exchange.create_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    reduce_only=True,
                )
                self._remove_position(symbol)
                logger.info(
                    "live_short_closed",
                    symbol=symbol,
                    entry=entry_price,
                    exit=current_price,
                    pnl=round(pnl, 4),
                    reason=reason,
                )
            except Exception as e:
                logger.error(
                    "close_short_failed",
                    symbol=symbol,
                    error=str(e),
                )
                await self._send_alert(f"CLOSE SHORT FAILED: {symbol} — {e}")
                return 0.0

        await self._send_alert(
            f"{'📈' if pnl > 0 else '📉'} SHORT {symbol} closed: "
            f"PnL={pnl:+.2f} USDT ({reason})"
        )

        return pnl

    async def stop(self) -> None:
        """Clean up fetcher sessions on stop."""
        await super().stop()
        for fetcher in [
            self._coingecko,
            self._fear_greed,
            self._defillama,
            self._coinglass,
            self._etherscan,
        ]:
            try:
                await fetcher.close()
            except Exception:
                pass
