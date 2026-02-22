"""Order book analysis for detecting imbalances and walls."""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from bot.exchanges.base import ExchangeAdapter

logger = structlog.get_logger()


@dataclass
class WallInfo:
    """Information about a detected order book wall."""

    price: float
    volume: float
    side: str  # "bid" or "ask"
    volume_ratio: float  # how many times avg volume


@dataclass
class OrderBookAnalysis:
    """Result of order book analysis."""

    bid_volume: float = 0.0
    ask_volume: float = 0.0
    imbalance_ratio: float = 1.0  # bid_volume / ask_volume
    walls: list[WallInfo] = field(default_factory=list)
    buy_wall_below: bool = False  # buy wall detected below current price (support)
    sell_wall_above: bool = False  # sell wall detected above current price (resistance)
    confidence_modifier: float = 1.0  # multiplier for signal confidence


class OrderBookAnalyzer:
    """Analyzes order book data to detect imbalances and walls.

    Uses bid/ask volume imbalance and wall detection to provide
    a confidence modifier for trading signals.

    - Imbalance ratio > 2.0 = strong buying pressure (boost BUY confidence)
    - Imbalance ratio < 0.5 = strong selling pressure (boost SELL confidence)
    - Buy wall below price = support (boost BUY confidence)
    - Sell wall above price = resistance (reduce BUY confidence)
    """

    def __init__(
        self,
        wall_threshold: float = 5.0,
        strong_buy_imbalance: float = 2.0,
        strong_sell_imbalance: float = 0.5,
        max_confidence_boost: float = 1.5,
        min_confidence_factor: float = 0.5,
        depth: int = 20,
    ):
        """Initialize OrderBookAnalyzer.

        Args:
            wall_threshold: Multiplier above average level volume to detect a wall.
            strong_buy_imbalance: Imbalance ratio threshold for strong buying pressure.
            strong_sell_imbalance: Imbalance ratio threshold for strong selling pressure.
            max_confidence_boost: Maximum confidence multiplier.
            min_confidence_factor: Minimum confidence multiplier.
            depth: Number of order book levels to fetch.
        """
        self._wall_threshold = wall_threshold
        self._strong_buy_imbalance = strong_buy_imbalance
        self._strong_sell_imbalance = strong_sell_imbalance
        self._max_confidence_boost = max_confidence_boost
        self._min_confidence_factor = min_confidence_factor
        self._depth = depth

    @property
    def wall_threshold(self) -> float:
        return self._wall_threshold

    @property
    def strong_buy_imbalance(self) -> float:
        return self._strong_buy_imbalance

    @property
    def strong_sell_imbalance(self) -> float:
        return self._strong_sell_imbalance

    @property
    def depth(self) -> int:
        return self._depth

    async def fetch_and_analyze(
        self,
        exchange: ExchangeAdapter,
        symbol: str,
    ) -> OrderBookAnalysis:
        """Fetch order book from exchange and analyze it.

        Args:
            exchange: Exchange adapter to fetch order book from.
            symbol: Trading pair symbol.

        Returns:
            OrderBookAnalysis with imbalance, walls, and confidence modifier.
        """
        try:
            order_book = await exchange.get_order_book(
                symbol, limit=self._depth
            )
        except Exception:
            logger.warning(
                "order_book_fetch_failed",
                symbol=symbol,
                exc_info=True,
            )
            return OrderBookAnalysis()

        return self.analyze(order_book, symbol)

    def analyze(
        self,
        order_book: dict,
        symbol: str = "",
    ) -> OrderBookAnalysis:
        """Analyze an order book dict.

        Args:
            order_book: Dict with 'bids' and 'asks' lists of [price, quantity] pairs.
            symbol: Trading pair symbol (for logging).

        Returns:
            OrderBookAnalysis with computed metrics.
        """
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        if not bids and not asks:
            return OrderBookAnalysis()

        # Calculate total volumes
        bid_volume = sum(level[1] for level in bids) if bids else 0.0
        ask_volume = sum(level[1] for level in asks) if asks else 0.0

        # Calculate imbalance ratio
        if ask_volume > 0:
            imbalance_ratio = bid_volume / ask_volume
        elif bid_volume > 0:
            imbalance_ratio = float("inf")
        else:
            imbalance_ratio = 1.0

        # Detect walls
        all_volumes = [level[1] for level in bids] + [level[1] for level in asks]
        avg_volume = sum(all_volumes) / len(all_volumes) if all_volumes else 0.0

        walls: list[WallInfo] = []
        current_price = self._get_current_price(bids, asks)

        buy_wall_below = False
        sell_wall_above = False

        if avg_volume > 0:
            # Check bid walls (buy walls)
            for price, volume in bids:
                ratio = volume / avg_volume
                if ratio >= self._wall_threshold:
                    walls.append(
                        WallInfo(
                            price=price,
                            volume=volume,
                            side="bid",
                            volume_ratio=ratio,
                        )
                    )
                    if current_price is not None and price < current_price:
                        buy_wall_below = True

            # Check ask walls (sell walls)
            for price, volume in asks:
                ratio = volume / avg_volume
                if ratio >= self._wall_threshold:
                    walls.append(
                        WallInfo(
                            price=price,
                            volume=volume,
                            side="ask",
                            volume_ratio=ratio,
                        )
                    )
                    if current_price is not None and price > current_price:
                        sell_wall_above = True

        # Calculate confidence modifier
        confidence_modifier = self._calculate_confidence_modifier(
            imbalance_ratio, buy_wall_below, sell_wall_above
        )

        analysis = OrderBookAnalysis(
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance_ratio=imbalance_ratio,
            walls=walls,
            buy_wall_below=buy_wall_below,
            sell_wall_above=sell_wall_above,
            confidence_modifier=confidence_modifier,
        )

        logger.debug(
            "order_book_analyzed",
            symbol=symbol,
            imbalance_ratio=round(imbalance_ratio, 4),
            walls=len(walls),
            buy_wall_below=buy_wall_below,
            sell_wall_above=sell_wall_above,
            confidence_modifier=round(confidence_modifier, 4),
        )

        return analysis

    def get_buy_confidence_modifier(
        self, analysis: OrderBookAnalysis
    ) -> float:
        """Get a confidence modifier specifically for BUY signals.

        Strong buying pressure (high imbalance) and buy walls increase confidence.
        Sell walls above price decrease confidence.

        Args:
            analysis: OrderBookAnalysis result.

        Returns:
            Multiplier for BUY signal confidence.
        """
        modifier = 1.0

        # Imbalance effect on BUY
        if analysis.imbalance_ratio >= self._strong_buy_imbalance:
            # Strong buying pressure boosts BUY confidence
            boost = min(
                analysis.imbalance_ratio / self._strong_buy_imbalance,
                self._max_confidence_boost,
            )
            modifier *= boost
        elif analysis.imbalance_ratio <= self._strong_sell_imbalance:
            # Strong selling pressure reduces BUY confidence
            modifier *= self._min_confidence_factor

        # Wall effects on BUY
        if analysis.buy_wall_below:
            modifier *= 1.2  # Support below = more confidence for BUY
        if analysis.sell_wall_above:
            modifier *= 0.8  # Resistance above = less confidence for BUY

        return max(
            self._min_confidence_factor,
            min(modifier, self._max_confidence_boost),
        )

    def get_sell_confidence_modifier(
        self, analysis: OrderBookAnalysis
    ) -> float:
        """Get a confidence modifier specifically for SELL signals.

        Strong selling pressure (low imbalance) and sell walls increase SELL confidence.
        Buy walls below price decrease SELL confidence.

        Args:
            analysis: OrderBookAnalysis result.

        Returns:
            Multiplier for SELL signal confidence.
        """
        modifier = 1.0

        # Imbalance effect on SELL
        if analysis.imbalance_ratio <= self._strong_sell_imbalance:
            # Strong selling pressure boosts SELL confidence
            boost = min(
                (1.0 / analysis.imbalance_ratio)
                / (1.0 / self._strong_sell_imbalance)
                if analysis.imbalance_ratio > 0
                else self._max_confidence_boost,
                self._max_confidence_boost,
            )
            modifier *= boost
        elif analysis.imbalance_ratio >= self._strong_buy_imbalance:
            # Strong buying pressure reduces SELL confidence
            modifier *= self._min_confidence_factor

        # Wall effects on SELL
        if analysis.sell_wall_above:
            modifier *= 1.2  # Resistance above = more confidence for SELL
        if analysis.buy_wall_below:
            modifier *= 0.8  # Support below = less confidence for SELL

        return max(
            self._min_confidence_factor,
            min(modifier, self._max_confidence_boost),
        )

    def _calculate_confidence_modifier(
        self,
        imbalance_ratio: float,
        buy_wall_below: bool,
        sell_wall_above: bool,
    ) -> float:
        """Calculate a general confidence modifier from order book analysis.

        This is the default modifier stored in OrderBookAnalysis.
        For direction-specific modifiers, use get_buy/sell_confidence_modifier().
        """
        modifier = 1.0

        # Pure imbalance effect (direction-neutral)
        if imbalance_ratio >= self._strong_buy_imbalance:
            modifier = min(
                imbalance_ratio / self._strong_buy_imbalance,
                self._max_confidence_boost,
            )
        elif imbalance_ratio <= self._strong_sell_imbalance:
            modifier = min(
                (1.0 / imbalance_ratio) / (1.0 / self._strong_sell_imbalance)
                if imbalance_ratio > 0
                else self._max_confidence_boost,
                self._max_confidence_boost,
            )

        return max(
            self._min_confidence_factor,
            min(modifier, self._max_confidence_boost),
        )

    @staticmethod
    def _get_current_price(
        bids: list, asks: list
    ) -> float | None:
        """Estimate current price from best bid and ask."""
        if bids and asks:
            return (bids[0][0] + asks[0][0]) / 2.0
        if bids:
            return bids[0][0]
        if asks:
            return asks[0][0]
        return None
