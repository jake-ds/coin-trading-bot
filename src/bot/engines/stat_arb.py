"""Statistical arbitrage (pairs trading) engine.

Monitors correlated asset pairs.  When the z-score of their price ratio
exceeds a threshold, opens a mean-reversion trade (long underperformer,
short outperformer).  Closes when z-score reverts to exit level.

Reuses cointegration and z-score utilities from bot.quant.statistics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from bot.engines.base import BaseEngine, DecisionStep, EngineCycleResult

if TYPE_CHECKING:
    from bot.config import Settings
    from bot.engines.portfolio_manager import PortfolioManager
    from bot.exchanges.base import ExchangeAdapter

logger = structlog.get_logger(__name__)


class StatisticalArbEngine(BaseEngine):
    """Statistical arbitrage engine using pairs trading."""

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
            loop_interval=60.0,
            max_positions=5,
            paper_mode=paper_mode,
        )
        self._pairs: list[list[str]] = (
            list(s.stat_arb_pairs) if s else [["BTC/USDT", "ETH/USDT"]]
        )
        self._lookback = s.stat_arb_lookback if s else 100
        self._entry_zscore = s.stat_arb_entry_zscore if s else 2.0
        self._exit_zscore = s.stat_arb_exit_zscore if s else 0.5
        self._stop_zscore = s.stat_arb_stop_zscore if s else 4.0
        self._min_correlation = s.stat_arb_min_correlation if s else 0.7

        # Price history cache: symbol -> list of close prices
        self._price_cache: dict[str, list[float]] = {}

    @property
    def name(self) -> str:
        return "stat_arb"

    @property
    def description(self) -> str:
        return "Statistical arbitrage (pairs trading with z-score mean reversion)"

    async def _run_cycle(self) -> EngineCycleResult:
        cycle_start = datetime.now(timezone.utc)
        actions: list[dict] = []
        signals: list[dict] = []
        decisions: list[DecisionStep] = []
        pnl_update = 0.0

        # Update price caches
        await self._update_price_cache()

        for pair in self._pairs:
            if len(pair) != 2:
                continue
            sym_a, sym_b = pair[0], pair[1]
            pair_label = f"{sym_a}/{sym_b}"

            prices_a = self._price_cache.get(sym_a, [])
            prices_b = self._price_cache.get(sym_b, [])

            min_len = min(len(prices_a), len(prices_b))
            if min_len < self._lookback:
                signals.append({
                    "pair": pair_label,
                    "status": "insufficient_data",
                    "available": min_len,
                    "required": self._lookback,
                })
                decisions.append(DecisionStep(
                    label=f"{pair_label} 데이터 충분성",
                    observation=f"가용 데이터 {min_len}개 / 필요 {self._lookback}개",
                    threshold=f"최소 {self._lookback}개 가격 데이터 필요",
                    result=f"SKIP - 데이터 부족 ({min_len}/{self._lookback})",
                    category="skip",
                ))
                continue

            # Use last N prices
            a = np.array(prices_a[-self._lookback:], dtype=float)
            b = np.array(prices_b[-self._lookback:], dtype=float)

            # Correlation check
            corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else 0.0
            if abs(corr) < self._min_correlation:
                signals.append({
                    "pair": pair_label,
                    "status": "low_correlation",
                    "correlation": round(corr, 4),
                })
                decisions.append(DecisionStep(
                    label=f"{pair_label} 상관관계",
                    observation=f"상관계수 {corr:.4f}",
                    threshold=f"최소 상관계수 >= {self._min_correlation}",
                    result=f"SKIP - 상관관계 부족 ({abs(corr):.4f} < {self._min_correlation})",
                    category="skip",
                ))
                continue

            # Calculate ratio and z-score
            ratio = a / np.where(b == 0, 1e-10, b)
            mean = float(np.mean(ratio))
            std = float(np.std(ratio))
            if std == 0:
                continue

            zscore = (ratio[-1] - mean) / std
            pair_key = f"{sym_a}|{sym_b}"

            signals.append({
                "pair": pair_label,
                "status": "active",
                "correlation": round(corr, 4),
                "zscore": round(float(zscore), 4),
                "ratio": round(float(ratio[-1]), 6),
                "mean_ratio": round(mean, 6),
            })

            decisions.append(DecisionStep(
                label=f"{pair_label} Z-Score 분석",
                observation=(
                    f"상관계수 {corr:.4f}, Z-Score {float(zscore):.4f}, "
                    f"비율 {float(ratio[-1]):.6f} (평균 {mean:.6f})"
                ),
                threshold=(
                    f"진입: |z| >= {self._entry_zscore} | "
                    f"청산: |z| <= {self._exit_zscore} | "
                    f"손절: |z| >= {self._stop_zscore}"
                ),
                result="",  # filled below
                category="evaluate",
            ))

            # Check existing positions for exit
            if pair_key in self._positions:
                pos = self._positions[pair_key]
                exit_result = self._check_exit(pair_key, zscore, pos)
                if exit_result:
                    pnl_update += exit_result.get("pnl", 0)
                    actions.append(exit_result)
                    decisions[-1].result = (
                        f"EXIT - {exit_result['reason']}, "
                        f"PnL: {exit_result['pnl']:.4f}"
                    )
                    decisions[-1].category = "execute"
                else:
                    decisions[-1].result = (
                        f"HOLD - 포지션 유지 (진입 z: {pos.get('entry_zscore', 0):.4f})"
                    )
                    decisions[-1].category = "decide"
                continue

            # Check for new entry
            if self._has_capacity():
                entry = self._check_entry(
                    pair_key, sym_a, sym_b, zscore, a[-1], b[-1]
                )
                if entry:
                    actions.append(entry)
                    decisions[-1].result = (
                        f"ENTRY - {entry['side_a']}_{entry['side_b']}, "
                        f"Z-Score {float(zscore):.4f}"
                    )
                    decisions[-1].category = "execute"
                else:
                    decisions[-1].result = (
                        f"HOLD - Z-Score {float(zscore):.4f}, 진입 기준 미달"
                    )
                    decisions[-1].category = "skip"
            else:
                decisions[-1].result = "SKIP - 포지션 한도 초과"
                decisions[-1].category = "skip"

        return EngineCycleResult(
            engine_name=self.name,
            cycle_num=self._cycle_count + 1,
            timestamp=cycle_start.isoformat(),
            duration_ms=0.0,
            actions_taken=actions,
            positions=list(self._positions.values()),
            signals=signals,
            pnl_update=pnl_update,
            metadata={"pairs_monitored": len(self._pairs)},
            decisions=decisions,
        )

    # ------------------------------------------------------------------
    # Signal logic
    # ------------------------------------------------------------------

    def _check_entry(
        self,
        pair_key: str,
        sym_a: str,
        sym_b: str,
        zscore: float,
        price_a: float,
        price_b: float,
    ) -> dict[str, Any] | None:
        """Check if z-score warrants a new pairs trade entry."""
        if abs(zscore) < self._entry_zscore:
            return None

        # z > +entry → A is relatively overpriced: short A, long B
        # z < -entry → A is relatively underpriced: long A, short B
        if zscore > self._entry_zscore:
            side_a, side_b = "short", "long"
        else:
            side_a, side_b = "long", "short"

        position_capital = self._allocated_capital / max(self._max_positions, 1)
        qty_a = position_capital / 2 / price_a if price_a > 0 else 0
        qty_b = position_capital / 2 / price_b if price_b > 0 else 0

        self._add_position(
            symbol=pair_key,
            side=f"{side_a}_{side_b}",
            quantity=0,  # Not a single-asset position
            entry_price=0,
            sym_a=sym_a,
            sym_b=sym_b,
            side_a=side_a,
            side_b=side_b,
            entry_zscore=float(zscore),
            price_a=price_a,
            price_b=price_b,
            qty_a=qty_a,
            qty_b=qty_b,
        )

        logger.info(
            "stat_arb_entry",
            pair=pair_key,
            zscore=round(float(zscore), 4),
            side_a=side_a,
            side_b=side_b,
        )

        return {
            "action": "entry",
            "pair": pair_key,
            "zscore": round(float(zscore), 4),
            "side_a": side_a,
            "side_b": side_b,
            "price_a": price_a,
            "price_b": price_b,
        }

    def _check_exit(
        self, pair_key: str, zscore: float, pos: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Check if an existing position should be closed."""
        entry_z = pos.get("entry_zscore", 0)

        # Exit conditions:
        # 1. Z-score reverted close to zero (profit)
        # 2. Z-score blew through stop (loss)
        should_exit = False
        reason = ""

        if abs(zscore) <= self._exit_zscore:
            should_exit = True
            reason = "mean_reversion"
        elif abs(zscore) >= self._stop_zscore:
            should_exit = True
            reason = "stop_loss"
        # Also exit if zscore crossed zero from entry side
        elif entry_z > 0 and zscore < 0:
            should_exit = True
            reason = "crossed_zero"
        elif entry_z < 0 and zscore > 0:
            should_exit = True
            reason = "crossed_zero"

        if not should_exit:
            return None

        # Estimate PnL (simplified paper mode)
        pnl = self._estimate_pairs_pnl(pos, zscore)
        self._remove_position(pair_key)

        logger.info(
            "stat_arb_exit",
            pair=pair_key,
            reason=reason,
            entry_zscore=round(entry_z, 4),
            exit_zscore=round(float(zscore), 4),
            pnl=round(pnl, 4),
        )

        return {
            "action": "exit",
            "pair": pair_key,
            "reason": reason,
            "entry_zscore": round(entry_z, 4),
            "exit_zscore": round(float(zscore), 4),
            "pnl": round(pnl, 4),
        }

    def _estimate_pairs_pnl(self, pos: dict, exit_zscore: float) -> float:
        """Estimate PnL from z-score movement (paper mode)."""
        entry_z = pos.get("entry_zscore", 0)
        price_a = pos.get("price_a", 0)
        qty_a = pos.get("qty_a", 0)
        side_a = pos.get("side_a", "long")

        # Simplified: PnL proportional to z-score reversion
        z_change = abs(entry_z) - abs(exit_zscore)  # positive if reverted
        notional = qty_a * price_a if qty_a and price_a else 0
        # Rough estimate: 1 unit z-score change ≈ 0.5-1% of notional
        pnl = z_change * notional * 0.005

        if side_a == "short":
            pnl = -pnl  # Invert if we were short the overperformer

        return pnl

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    async def _update_price_cache(self) -> None:
        """Fetch latest prices and update the rolling cache."""
        all_symbols = set()
        for pair in self._pairs:
            all_symbols.update(pair)

        for symbol in all_symbols:
            price = await self._get_price(symbol)
            if price is not None and price > 0:
                if symbol not in self._price_cache:
                    self._price_cache[symbol] = []
                self._price_cache[symbol].append(price)
                # Keep cache bounded
                max_len = self._lookback + 50
                if len(self._price_cache[symbol]) > max_len:
                    self._price_cache[symbol] = self._price_cache[symbol][-max_len:]

    async def _get_price(self, symbol: str) -> float | None:
        for exchange in self._exchanges:
            try:
                ticker = await exchange.get_ticker(symbol)
                return ticker.get("last", 0.0)
            except Exception:
                pass
        return None
