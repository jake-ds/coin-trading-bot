"""Cross-exchange arbitrage strategy."""

from typing import Any

from bot.exchanges.base import ExchangeAdapter
from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy, strategy_registry


class ArbitrageStrategy(BaseStrategy):
    """Detect price differences across exchanges for arbitrage opportunities."""

    def __init__(
        self,
        min_spread_pct: float = 0.5,
        fee_pct: float = 0.1,
    ):
        self._min_spread_pct = min_spread_pct
        self._fee_pct = fee_pct
        self._exchanges: list[ExchangeAdapter] = []

    def set_exchanges(self, exchanges: list[ExchangeAdapter]) -> None:
        """Set the exchange adapters to compare prices across."""
        self._exchanges = exchanges

    @property
    def name(self) -> str:
        return "arbitrage"

    @property
    def required_history_length(self) -> int:
        return 1

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        symbol = kwargs.get("symbol", ohlcv_data[-1].symbol if ohlcv_data else "UNKNOWN")
        exchanges = kwargs.get("exchanges", self._exchanges)

        if len(exchanges) < 2:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "need_at_least_2_exchanges"},
            )

        # Fetch prices from all exchanges
        prices: dict[str, dict] = {}
        for exchange in exchanges:
            try:
                ticker = await exchange.get_ticker(symbol)
                prices[exchange.name] = ticker
            except (ValueError, ConnectionError, RuntimeError):
                continue

        if len(prices) < 2:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "insufficient_price_data"},
            )

        # Find best bid (sell) and best ask (buy)
        best_bid_exchange = max(prices, key=lambda e: prices[e].get("bid", 0))
        best_ask_exchange = min(prices, key=lambda e: prices[e].get("ask", float("inf")))

        # Arbitrage requires buying and selling on DIFFERENT exchanges
        if best_bid_exchange == best_ask_exchange:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "same_exchange"},
            )

        best_bid = prices[best_bid_exchange]["bid"]
        best_ask = prices[best_ask_exchange]["ask"]

        if best_ask <= 0:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "invalid_prices"},
            )

        # Calculate spread
        spread_pct = ((best_bid - best_ask) / best_ask) * 100
        total_fees = self._fee_pct * 2  # Fee on both buy and sell
        net_spread = spread_pct - total_fees

        metadata = {
            "buy_exchange": best_ask_exchange,
            "sell_exchange": best_bid_exchange,
            "buy_price": best_ask,
            "sell_price": best_bid,
            "spread_pct": spread_pct,
            "net_spread_pct": net_spread,
            "fees_pct": total_fees,
        }

        if net_spread >= self._min_spread_pct:
            confidence = min(net_spread / 5.0, 1.0)
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=confidence,
                metadata=metadata,
            )

        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            metadata=metadata,
        )


strategy_registry.register(ArbitrageStrategy())
