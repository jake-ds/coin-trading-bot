"""Triangular arbitrage strategy within a single exchange.

Exploits price discrepancies in 3-leg trading cycles:
e.g., USDT -> BTC -> ETH -> USDT

Detects profitable cycles after accounting for 3x taker fees.
Uses graph-based cycle search across available trading pairs.
"""

from __future__ import annotations

import itertools
from typing import Any

import structlog

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy

logger = structlog.get_logger()


class TriangularArbStrategy(BaseStrategy):
    """Triangular arbitrage within a single exchange.

    Expects kwargs:
        tickers: dict mapping symbol -> {'bid': float, 'ask': float}
        fee_rate: float (taker fee rate, e.g., 0.001 for 0.1%)
    """

    def __init__(
        self,
        min_profit_pct: float = 0.1,
        default_fee_rate: float = 0.001,
        base_currencies: list[str] | None = None,
    ):
        self._min_profit_pct = min_profit_pct
        self._default_fee_rate = default_fee_rate
        self._base_currencies = base_currencies or ["USDT", "BTC", "ETH", "BNB"]

    @property
    def name(self) -> str:
        return "triangular_arb"

    @property
    def required_history_length(self) -> int:
        return 1  # Only needs current ticker data

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        symbol = kwargs.get("symbol", "ARB/USDT")
        tickers = kwargs.get("tickers", {})
        fee_rate = kwargs.get("fee_rate", self._default_fee_rate)

        if not tickers or len(tickers) < 3:
            return self._hold(symbol, {"reason": "insufficient_tickers"})

        # Build graph: currency -> {currency -> (symbol, rate, side)}
        graph = self._build_graph(tickers)
        if not graph:
            return self._hold(symbol, {"reason": "no_graph_built"})

        # Find best triangular cycle
        best_cycle = self._find_best_cycle(graph, fee_rate)
        if not best_cycle:
            return self._hold(symbol, {"reason": "no_profitable_cycle"})

        profit_pct = best_cycle["profit_pct"]
        if profit_pct < self._min_profit_pct:
            return self._hold(symbol, {
                "reason": "profit_below_threshold",
                "best_profit_pct": round(profit_pct, 6),
                "cycle": best_cycle["path"],
            })

        # Profitable cycle found
        confidence = min(profit_pct / (self._min_profit_pct * 5), 1.0)
        metadata = {
            "cycle": best_cycle["path"],
            "legs": best_cycle["legs"],
            "profit_pct": round(profit_pct, 6),
            "gross_profit_pct": round(best_cycle["gross_profit_pct"], 6),
            "total_fee_pct": round(fee_rate * 3 * 100, 4),
        }

        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=SignalAction.BUY,
            confidence=round(confidence, 4),
            metadata=metadata,
        )

    def _build_graph(
        self, tickers: dict[str, dict]
    ) -> dict[str, dict[str, list[tuple[str, float, str]]]]:
        """Build directed graph of currency conversions.

        Returns:
            graph[from_currency][to_currency] = [(symbol, rate, side), ...]
        """
        graph: dict[str, dict[str, list[tuple[str, float, str]]]] = {}

        for symbol, ticker in tickers.items():
            bid = ticker.get("bid", 0)
            ask = ticker.get("ask", 0)
            if not bid or not ask or bid <= 0 or ask <= 0:
                continue

            # Parse symbol: "BTC/USDT" -> base="BTC", quote="USDT"
            parts = symbol.split("/")
            if len(parts) != 2:
                continue
            base, quote = parts

            # Forward: quote -> base (buy at ask)
            graph.setdefault(quote, {}).setdefault(base, []).append(
                (symbol, 1.0 / ask, "buy")
            )
            # Reverse: base -> quote (sell at bid)
            graph.setdefault(base, {}).setdefault(quote, []).append(
                (symbol, bid, "sell")
            )

        return graph

    def _find_best_cycle(
        self, graph: dict, fee_rate: float
    ) -> dict | None:
        """Find the most profitable 3-leg cycle.

        Returns:
            Dict with 'path', 'legs', 'profit_pct', 'gross_profit_pct' or None.
        """
        best = None
        fee_multiplier = (1 - fee_rate) ** 3

        for start in self._base_currencies:
            if start not in graph:
                continue

            for mid1 in graph[start]:
                if mid1 == start or mid1 not in graph:
                    continue

                for mid2 in graph[mid1]:
                    if mid2 == start or mid2 == mid1 or mid2 not in graph:
                        continue

                    if start not in graph.get(mid2, {}):
                        continue

                    # Find best rate for each leg
                    leg1_options = graph[start][mid1]
                    leg2_options = graph[mid1][mid2]
                    leg3_options = graph[mid2][start]

                    for l1, l2, l3 in itertools.product(
                        leg1_options, leg2_options, leg3_options
                    ):
                        gross_rate = l1[1] * l2[1] * l3[1]
                        net_rate = gross_rate * fee_multiplier
                        profit_pct = (net_rate - 1.0) * 100

                        if best is None or profit_pct > best["profit_pct"]:
                            best = {
                                "path": [start, mid1, mid2, start],
                                "legs": [
                                    {"symbol": l1[0], "side": l1[2], "rate": round(l1[1], 8)},
                                    {"symbol": l2[0], "side": l2[2], "rate": round(l2[1], 8)},
                                    {"symbol": l3[0], "side": l3[2], "rate": round(l3[1], 8)},
                                ],
                                "profit_pct": profit_pct,
                                "gross_profit_pct": (gross_rate - 1.0) * 100,
                            }

        return best

    def _hold(self, symbol: str, metadata: dict) -> TradingSignal:
        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            metadata=metadata,
        )
