"""On-chain data driven autonomous trading engine.

Collects data from CoinGecko, Fear&Greed, DeFiLlama, CoinGlass,
computes composite signals, and executes spot trades on Binance.
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
    from bot.exchanges.base import ExchangeAdapter

logger = structlog.get_logger(__name__)


class OnChainTraderEngine(BaseEngine):
    """Autonomous spot trader driven by on-chain/market data signals.

    Each cycle:
    1. Fetch on-chain data from all APIs in parallel
    2. Compute composite signal for each symbol
    3. Check existing positions for exit conditions
    4. Open new positions on strong BUY signals
    """

    supports_live: bool = True

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        exchanges: list[ExchangeAdapter] | None = None,
        paper_mode: bool = True,
        settings: Settings | None = None,
    ):
        s = settings
        loop_interval = getattr(s, "onchain_loop_interval", 300.0) if s else 300.0
        max_positions = getattr(s, "onchain_max_positions", 5) if s else 5

        super().__init__(
            portfolio_manager=portfolio_manager,
            exchanges=exchanges,
            loop_interval=loop_interval,
            max_positions=max_positions,
            paper_mode=paper_mode,
        )

        self._settings = settings
        _default_symbols = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
            "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
            "SUI/USDT", "NEAR/USDT", "ARB/USDT", "OP/USDT", "APT/USDT",
            "PEPE/USDT", "UNI/USDT", "ATOM/USDT", "FIL/USDT", "LTC/USDT",
            "TRX/USDT", "WIF/USDT", "AAVE/USDT", "RENDER/USDT", "MATIC/USDT",
        ]
        self._symbols: list[str] = (
            getattr(s, "onchain_symbols", _default_symbols)
            if s else _default_symbols
        )

        # Signal thresholds
        self._buy_threshold = getattr(s, "onchain_buy_threshold", 30.0) if s else 30.0
        self._sell_threshold = getattr(s, "onchain_sell_threshold", -30.0) if s else -30.0
        self._min_confidence = getattr(s, "onchain_min_confidence", 0.4) if s else 0.4
        self._signal_weights = (
            getattr(s, "onchain_signal_weights", None) if s else None
        )

        # Position sizing
        self._max_position_pct = getattr(s, "onchain_max_position_pct", 20.0) if s else 20.0
        self._stop_loss_pct = getattr(s, "onchain_stop_loss_pct", 5.0) if s else 5.0
        self._take_profit_pct = getattr(s, "onchain_take_profit_pct", 8.0) if s else 8.0
        self._trailing_stop_pct = getattr(s, "onchain_trailing_stop_pct", 3.0) if s else 3.0
        self._trailing_activate_pct = getattr(s, "onchain_trailing_activate_pct", 2.0) if s else 2.0

        # Data fetchers
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
        return "onchain_trader"

    @property
    def description(self) -> str:
        return "On-chain data driven autonomous spot trader"

    @property
    def latest_signals(self) -> dict[str, dict]:
        return {k: v.to_dict() for k, v in self._latest_signals.items()}

    async def _run_cycle(self) -> EngineCycleResult:
        """Execute one trading cycle."""
        decisions: list[DecisionStep] = []
        actions: list[dict] = []
        pnl_update = 0.0

        # 1. Fetch all on-chain data in parallel
        decisions.append(DecisionStep(
            label="데이터 수집",
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

        data_sources = []
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
            label="데이터 수집",
            observation=f"수집 완료: {', '.join(data_sources) if data_sources else 'None'}",
            threshold="최소 1개 소스",
            result="OK" if data_sources else "NO_DATA",
            category="evaluate",
        )

        # 2. Compute composite signals for each symbol
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

            decisions.append(DecisionStep(
                label=f"{symbol} 시그널",
                observation=f"점수={signal.score:+.1f}, 신뢰도={signal.confidence:.2f}",
                threshold=f"매수≥{self._buy_threshold}, 매도≤{self._sell_threshold}, 신뢰도≥{self._min_confidence}",
                result=signal.action.value,
                category="evaluate",
            ))

        # 3. Check existing positions for exit conditions
        for symbol in list(self._positions.keys()):
            pos = self._positions.get(symbol)
            if pos is None:
                continue

            current_price = await self._get_current_price(symbol)
            if current_price is None:
                continue

            entry_price = pos["entry_price"]
            quantity = pos.get("quantity", 0)
            pnl_pct = ((current_price - entry_price) / entry_price) * 100.0

            # Update position with live data for dashboard display
            pos["current_price"] = current_price
            pos["unrealized_pnl"] = round((current_price - entry_price) * quantity, 4)
            pos["pnl_pct"] = round(pnl_pct, 2)

            exit_reason = self._check_exit(symbol, pos, current_price, pnl_pct)

            if exit_reason:
                realized_pnl = await self._close_position(symbol, current_price, exit_reason)
                pnl_update += realized_pnl
                actions.append({
                    "action": "close",
                    "symbol": symbol,
                    "reason": exit_reason,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl": realized_pnl,
                    "pnl_pct": round(pnl_pct, 2),
                })
                decisions.append(DecisionStep(
                    label=f"{symbol} 청산",
                    observation=f"현재가={current_price:.2f}, PnL={pnl_pct:+.1f}%",
                    threshold=f"손절=-{self._stop_loss_pct}%, 익절=+{self._take_profit_pct}%",
                    result=f"CLOSE — {exit_reason}",
                    category="execute",
                ))
            else:
                # Update trailing stop high watermark
                if pnl_pct >= self._trailing_activate_pct:
                    high = pos.get("high_price", entry_price)
                    if current_price > high:
                        pos["high_price"] = current_price
                decisions.append(DecisionStep(
                    label=f"{symbol} 포지션 점검",
                    observation=f"현재가={current_price:.2f}, PnL={pnl_pct:+.1f}%",
                    threshold=f"손절=-{self._stop_loss_pct}%, 익절=+{self._take_profit_pct}%",
                    result="HOLD",
                    category="evaluate",
                ))

        # 4. Open new positions on strong BUY signals
        regime_adj = self._get_regime_adjustments()
        if regime_adj["size_mult"] == 0.0:
            decisions.append(DecisionStep(
                label="CRISIS 레짐",
                observation="CRISIS 레짐 감지 — 신규 진입 차단",
                threshold="size_mult > 0",
                result="BLOCKED",
                category="decide",
            ))
        else:
            for symbol in self._symbols:
                signal = self._latest_signals.get(symbol)
                if signal is None or signal.action != SignalAction.BUY:
                    continue

                if symbol in self._positions:
                    continue

                if not self._has_capacity(symbol):
                    decisions.append(DecisionStep(
                        label=f"{symbol} 진입 시도",
                        observation=f"BUY 시그널 (점수={signal.score:+.1f})",
                        threshold=f"최대 {self._max_positions}포지션",
                        result="SKIP — 용량 초과",
                        category="skip",
                    ))
                    continue

                entry_result = await self._open_position(symbol, signal, regime_adj)
                if entry_result:
                    actions.append(entry_result)
                    decisions.append(DecisionStep(
                        label=f"{symbol} 진입",
                        observation=f"BUY 시그널 (점수={signal.score:+.1f}, 신뢰도={signal.confidence:.2f})",
                        threshold=f"≥{self._buy_threshold}점, ≥{self._min_confidence}신뢰도",
                        result=f"OPEN — ${entry_result.get('cost', 0):.2f}",
                        category="execute",
                    ))

        return EngineCycleResult(
            engine_name=self.name,
            cycle_num=self._cycle_count + 1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=0.0,  # Set by base class
            actions_taken=actions,
            positions=list(self._positions.values()),
            signals=[s.to_dict() for s in self._latest_signals.values()],
            pnl_update=pnl_update,
            metadata={
                "symbols": self._symbols,
                "data_sources": data_sources,
                "position_count": len(self._positions),
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

        # Convert exceptions to None
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
        """Execute a coroutine safely, returning None on error."""
        try:
            return await coro(*args)
        except Exception:
            return None

    async def _get_current_price(self, symbol: str) -> float | None:
        """Get current price from exchange."""
        if not self._exchanges:
            # Use CoinGecko price if no exchange
            sig = self._latest_signals.get(symbol)
            if sig and sig.signals:
                for s in sig.signals:
                    if s.name == "market_trend" and "24h" in s.reason:
                        pass
            # Try cached market data from latest fetch
            return None

        try:
            exchange = self._exchanges[0]
            ticker = await exchange.get_ticker(symbol)
            return ticker.get("last", 0.0)
        except Exception as e:
            logger.warning("price_fetch_error", symbol=symbol, error=str(e))
            return None

    def _check_exit(
        self, symbol: str, pos: dict, current_price: float, pnl_pct: float
    ) -> str | None:
        """Check if position should be closed. Returns exit reason or None."""
        # Stop loss
        if pnl_pct <= -self._stop_loss_pct:
            return f"stop_loss ({pnl_pct:+.1f}%)"

        # Take profit
        if pnl_pct >= self._take_profit_pct:
            return f"take_profit ({pnl_pct:+.1f}%)"

        # Trailing stop
        entry_price = pos["entry_price"]
        high_price = pos.get("high_price", entry_price)
        if high_price > entry_price:
            high_pnl = ((high_price - entry_price) / entry_price) * 100.0
            if high_pnl >= self._trailing_activate_pct:
                drawdown_from_high = ((high_price - current_price) / high_price) * 100.0
                if drawdown_from_high >= self._trailing_stop_pct:
                    return f"trailing_stop (high={high_price:.2f}, drop={drawdown_from_high:.1f}%)"

        # Signal reversal
        signal = self._latest_signals.get(symbol)
        if signal and signal.action == SignalAction.SELL:
            return f"signal_reversal (score={signal.score:+.1f})"

        return None

    async def _open_position(
        self, symbol: str, signal: CompositeSignal, regime_adj: dict
    ) -> dict | None:
        """Open a new spot position."""
        if not self._exchanges:
            logger.warning("no_exchange_available")
            return None

        exchange = self._exchanges[0]

        # Get current price
        try:
            ticker = await exchange.get_ticker(symbol)
            price = ticker.get("last", 0.0)
            if price <= 0:
                return None
        except Exception as e:
            logger.warning("ticker_error", symbol=symbol, error=str(e))
            return None

        # Calculate position size
        capital = self._allocated_capital
        if capital <= 0:
            return None

        # Base size: max_position_pct of capital
        base_size_usd = capital * (self._max_position_pct / 100.0)

        # Scale by confidence (0.4-1.0 → 50%-100%)
        confidence_scale = 0.5 + (signal.confidence - 0.4) / 1.2
        confidence_scale = max(0.5, min(1.0, confidence_scale))

        # Apply regime adjustment
        size_usd = base_size_usd * confidence_scale * regime_adj["size_mult"]

        # Minimum order check ($10 on Binance)
        if size_usd < 10.0:
            logger.debug("position_too_small", symbol=symbol, size_usd=size_usd)
            return None

        quantity = size_usd / price

        # Execute order
        if self._paper_mode:
            # Paper trade — just track internally
            self._add_position(
                symbol=symbol,
                side="buy",
                quantity=quantity,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0.0,
                high_price=price,
                cost_usd=size_usd,
                signal_score=signal.score,
                signal_confidence=signal.confidence,
            )
            logger.info(
                "paper_position_opened",
                symbol=symbol,
                price=price,
                quantity=round(quantity, 6),
                cost_usd=round(size_usd, 2),
                signal_score=round(signal.score, 1),
            )
        else:
            # Live trade
            try:
                order = await exchange.create_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                )
                filled_price = order.filled_price if hasattr(order, "filled_price") else price
                filled_qty = order.filled_quantity if hasattr(order, "filled_quantity") else quantity
                self._add_position(
                    symbol=symbol,
                    side="buy",
                    quantity=filled_qty,
                    entry_price=filled_price,
                    current_price=filled_price,
                    unrealized_pnl=0.0,
                    high_price=filled_price,
                    cost_usd=filled_qty * filled_price,
                    signal_score=signal.score,
                    signal_confidence=signal.confidence,
                    order_id=getattr(order, "id", ""),
                )
                price = filled_price
                quantity = filled_qty
                size_usd = filled_qty * filled_price
                logger.info(
                    "live_position_opened",
                    symbol=symbol,
                    price=price,
                    quantity=round(quantity, 6),
                    cost_usd=round(size_usd, 2),
                )
            except Exception as e:
                logger.error("order_failed", symbol=symbol, error=str(e))
                await self._send_alert(f"ORDER FAILED: {symbol} — {e}")
                return None

        return {
            "action": "open",
            "symbol": symbol,
            "side": "buy",
            "price": price,
            "quantity": round(quantity, 6),
            "cost": round(size_usd, 2),
            "signal_score": round(signal.score, 1),
            "signal_confidence": round(signal.confidence, 3),
        }

    async def _close_position(
        self, symbol: str, current_price: float, reason: str
    ) -> float:
        """Close a position and return realized PnL."""
        pos = self._positions.get(symbol)
        if pos is None:
            return 0.0

        entry_price = pos["entry_price"]
        quantity = pos["quantity"]
        pnl = (current_price - entry_price) * quantity

        if self._paper_mode:
            self._remove_position(symbol)
            logger.info(
                "paper_position_closed",
                symbol=symbol,
                entry=entry_price,
                exit=current_price,
                pnl=round(pnl, 4),
                reason=reason,
            )
        else:
            try:
                exchange = self._exchanges[0]
                await exchange.create_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                )
                self._remove_position(symbol)
                logger.info(
                    "live_position_closed",
                    symbol=symbol,
                    entry=entry_price,
                    exit=current_price,
                    pnl=round(pnl, 4),
                    reason=reason,
                )
            except Exception as e:
                logger.error(
                    "close_order_failed",
                    symbol=symbol,
                    error=str(e),
                )
                await self._send_alert(f"CLOSE FAILED: {symbol} — {e}")
                return 0.0

        await self._send_alert(
            f"{'📈' if pnl > 0 else '📉'} {symbol} closed: "
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
