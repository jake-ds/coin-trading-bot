"""Tests for OrderBookAnalyzer — order book imbalance detection and wall analysis."""

from unittest.mock import AsyncMock

import pytest

from bot.data.order_book import OrderBookAnalysis, OrderBookAnalyzer, WallInfo

# --- Fixtures ---


@pytest.fixture
def analyzer():
    """Default OrderBookAnalyzer."""
    return OrderBookAnalyzer()


@pytest.fixture
def custom_analyzer():
    """OrderBookAnalyzer with custom thresholds."""
    return OrderBookAnalyzer(
        wall_threshold=3.0,
        strong_buy_imbalance=1.5,
        strong_sell_imbalance=0.7,
        max_confidence_boost=2.0,
        min_confidence_factor=0.3,
        depth=10,
    )


def make_order_book(
    bids: list[list[float]] | None = None,
    asks: list[list[float]] | None = None,
) -> dict:
    """Create an order book dict."""
    return {
        "bids": bids or [],
        "asks": asks or [],
    }


def make_balanced_book() -> dict:
    """Create a balanced order book."""
    return make_order_book(
        bids=[
            [100.0, 10.0],
            [99.0, 10.0],
            [98.0, 10.0],
        ],
        asks=[
            [101.0, 10.0],
            [102.0, 10.0],
            [103.0, 10.0],
        ],
    )


def make_buy_heavy_book() -> dict:
    """Create an order book with strong buying pressure (imbalance > 2.0)."""
    return make_order_book(
        bids=[
            [100.0, 50.0],
            [99.0, 40.0],
            [98.0, 30.0],
        ],
        asks=[
            [101.0, 10.0],
            [102.0, 10.0],
            [103.0, 10.0],
        ],
    )


def make_sell_heavy_book() -> dict:
    """Create an order book with strong selling pressure (imbalance < 0.5)."""
    return make_order_book(
        bids=[
            [100.0, 5.0],
            [99.0, 5.0],
            [98.0, 5.0],
        ],
        asks=[
            [101.0, 40.0],
            [102.0, 30.0],
            [103.0, 20.0],
        ],
    )


def make_buy_wall_book() -> dict:
    """Create an order book with a buy wall below current price.

    avg = (10+10+300+10+10+10)/6 = 58.3
    Wall at 98.0: 300/58.3 = 5.14 >= 5.0 threshold
    """
    return make_order_book(
        bids=[
            [100.0, 10.0],
            [99.0, 10.0],
            [98.0, 300.0],  # Buy wall: 300 >> avg
        ],
        asks=[
            [101.0, 10.0],
            [102.0, 10.0],
            [103.0, 10.0],
        ],
    )


def make_sell_wall_book() -> dict:
    """Create an order book with a sell wall above current price.

    avg = (10+10+10+10+300+10)/6 = 58.3
    Wall at 102.0: 300/58.3 = 5.14 >= 5.0 threshold
    """
    return make_order_book(
        bids=[
            [100.0, 10.0],
            [99.0, 10.0],
            [98.0, 10.0],
        ],
        asks=[
            [101.0, 10.0],
            [102.0, 300.0],  # Sell wall: 300 >> avg
            [103.0, 10.0],
        ],
    )


# --- Basic Tests ---


class TestOrderBookAnalyzerInit:
    def test_default_params(self, analyzer):
        assert analyzer.wall_threshold == 5.0
        assert analyzer.strong_buy_imbalance == 2.0
        assert analyzer.strong_sell_imbalance == 0.5
        assert analyzer.depth == 20

    def test_custom_params(self, custom_analyzer):
        assert custom_analyzer.wall_threshold == 3.0
        assert custom_analyzer.strong_buy_imbalance == 1.5
        assert custom_analyzer.strong_sell_imbalance == 0.7
        assert custom_analyzer.depth == 10


# --- Imbalance Calculation Tests ---


class TestImbalanceCalculation:
    def test_balanced_order_book(self, analyzer):
        """Equal bid/ask volume should give imbalance ratio of 1.0."""
        book = make_balanced_book()
        analysis = analyzer.analyze(book)
        assert analysis.bid_volume == 30.0
        assert analysis.ask_volume == 30.0
        assert analysis.imbalance_ratio == 1.0

    def test_buy_heavy_imbalance(self, analyzer):
        """Heavy bids should give imbalance > 2.0."""
        book = make_buy_heavy_book()
        analysis = analyzer.analyze(book)
        assert analysis.bid_volume == 120.0
        assert analysis.ask_volume == 30.0
        assert analysis.imbalance_ratio == pytest.approx(4.0)

    def test_sell_heavy_imbalance(self, analyzer):
        """Heavy asks should give imbalance < 0.5."""
        book = make_sell_heavy_book()
        analysis = analyzer.analyze(book)
        assert analysis.bid_volume == 15.0
        assert analysis.ask_volume == 90.0
        assert analysis.imbalance_ratio == pytest.approx(15.0 / 90.0)
        assert analysis.imbalance_ratio < 0.5

    def test_empty_order_book(self, analyzer):
        """Empty order book should give neutral analysis."""
        book = make_order_book()
        analysis = analyzer.analyze(book)
        assert analysis.bid_volume == 0.0
        assert analysis.ask_volume == 0.0
        assert analysis.imbalance_ratio == 1.0
        assert analysis.confidence_modifier == 1.0

    def test_only_bids(self, analyzer):
        """Only bids, no asks should give infinite imbalance."""
        book = make_order_book(bids=[[100.0, 10.0], [99.0, 10.0]])
        analysis = analyzer.analyze(book)
        assert analysis.bid_volume == 20.0
        assert analysis.ask_volume == 0.0
        assert analysis.imbalance_ratio == float("inf")

    def test_only_asks(self, analyzer):
        """Only asks, no bids should give zero imbalance."""
        book = make_order_book(asks=[[101.0, 10.0], [102.0, 10.0]])
        analysis = analyzer.analyze(book)
        assert analysis.bid_volume == 0.0
        assert analysis.ask_volume == 20.0
        assert analysis.imbalance_ratio == 0.0


# --- Wall Detection Tests ---


class TestWallDetection:
    def test_buy_wall_detected(self, analyzer):
        """Buy wall below current price should be detected."""
        book = make_buy_wall_book()
        analysis = analyzer.analyze(book)
        assert analysis.buy_wall_below is True
        bid_walls = [w for w in analysis.walls if w.side == "bid"]
        assert len(bid_walls) >= 1
        assert bid_walls[0].price == 98.0
        assert bid_walls[0].volume == 300.0
        assert bid_walls[0].volume_ratio >= 5.0

    def test_sell_wall_detected(self, analyzer):
        """Sell wall above current price should be detected."""
        book = make_sell_wall_book()
        analysis = analyzer.analyze(book)
        assert analysis.sell_wall_above is True
        ask_walls = [w for w in analysis.walls if w.side == "ask"]
        assert len(ask_walls) >= 1
        assert ask_walls[0].price == 102.0
        assert ask_walls[0].volume == 300.0

    def test_no_walls_balanced(self, analyzer):
        """Balanced book should have no walls."""
        book = make_balanced_book()
        analysis = analyzer.analyze(book)
        assert len(analysis.walls) == 0
        assert analysis.buy_wall_below is False
        assert analysis.sell_wall_above is False

    def test_wall_threshold_custom(self, custom_analyzer):
        """Custom wall threshold (3.0x) should detect walls more aggressively."""
        # Average volume: (10+10+60+10+10+10) / 6 = ~18.3
        # 60 / 18.3 ≈ 3.27 → above 3.0 threshold
        book = make_order_book(
            bids=[
                [100.0, 10.0],
                [99.0, 10.0],
                [98.0, 60.0],
            ],
            asks=[
                [101.0, 10.0],
                [102.0, 10.0],
                [103.0, 10.0],
            ],
        )
        analysis = custom_analyzer.analyze(book)
        assert analysis.buy_wall_below is True
        assert len(analysis.walls) >= 1

    def test_wall_at_best_bid(self, analyzer):
        """Wall at best bid (closest to mid-price) should be detected below mid.

        Best bid = 100, best ask = 101, mid = 100.5
        Wall at 100.0 is below mid = support.
        avg = (300+10+10+10+10+10)/6 = 58.3, 300/58.3 = 5.14 >= 5.0
        """
        book = make_order_book(
            bids=[
                [100.0, 300.0],  # wall at best bid
                [99.0, 10.0],
                [98.0, 10.0],
            ],
            asks=[
                [101.0, 10.0],
                [102.0, 10.0],
                [103.0, 10.0],
            ],
        )
        analysis = analyzer.analyze(book)
        bid_walls = [w for w in analysis.walls if w.side == "bid"]
        assert len(bid_walls) >= 1
        assert analysis.buy_wall_below is True

    def test_multiple_walls(self, custom_analyzer):
        """Multiple walls can be detected on both sides (using lower threshold=3.0).

        With 7 normal levels at 10 and 3 wall levels at 250:
        avg = (70 + 750) / 10 = 82
        250 / 82 = 3.05 >= 3.0 threshold
        """
        book = make_order_book(
            bids=[
                [100.0, 10.0],
                [99.0, 10.0],
                [98.0, 250.0],  # wall
                [97.0, 10.0],
                [96.0, 250.0],  # wall
            ],
            asks=[
                [101.0, 10.0],
                [102.0, 250.0],  # wall
                [103.0, 10.0],
                [104.0, 10.0],
                [105.0, 10.0],
            ],
        )
        analysis = custom_analyzer.analyze(book)
        assert len(analysis.walls) >= 3
        assert analysis.buy_wall_below is True
        assert analysis.sell_wall_above is True


# --- Confidence Modifier Tests ---


class TestConfidenceModifier:
    def test_balanced_modifier_is_neutral(self, analyzer):
        """Balanced book should give confidence_modifier close to 1.0."""
        book = make_balanced_book()
        analysis = analyzer.analyze(book)
        assert analysis.confidence_modifier == pytest.approx(1.0)

    def test_buy_heavy_modifier_is_boosted(self, analyzer):
        """Strong buying pressure should boost confidence modifier."""
        book = make_buy_heavy_book()
        analysis = analyzer.analyze(book)
        # imbalance = 4.0, strong_buy = 2.0 → modifier = 4.0/2.0 = 2.0 (capped at 1.5)
        assert analysis.confidence_modifier > 1.0
        assert analysis.confidence_modifier <= 1.5

    def test_sell_heavy_modifier_is_boosted(self, analyzer):
        """Strong selling pressure should boost confidence modifier."""
        book = make_sell_heavy_book()
        analysis = analyzer.analyze(book)
        assert analysis.confidence_modifier > 1.0

    def test_modifier_capped_at_max(self, analyzer):
        """Confidence modifier should not exceed max_confidence_boost."""
        book = make_order_book(
            bids=[[100.0, 1000.0]],
            asks=[[101.0, 1.0]],
        )
        analysis = analyzer.analyze(book)
        assert analysis.confidence_modifier <= 1.5

    def test_modifier_floored_at_min(self, analyzer):
        """Confidence modifier should not go below min_confidence_factor."""
        analysis = OrderBookAnalysis()
        modifier = analyzer.get_buy_confidence_modifier(analysis)
        assert modifier >= 0.5


# --- Buy/Sell Specific Modifiers ---


class TestDirectionalModifiers:
    def test_buy_modifier_with_buying_pressure(self, analyzer):
        """Buy modifier should increase with buying pressure."""
        book = make_buy_heavy_book()
        analysis = analyzer.analyze(book)
        modifier = analyzer.get_buy_confidence_modifier(analysis)
        assert modifier > 1.0

    def test_buy_modifier_with_selling_pressure(self, analyzer):
        """Buy modifier should decrease with selling pressure."""
        book = make_sell_heavy_book()
        analysis = analyzer.analyze(book)
        modifier = analyzer.get_buy_confidence_modifier(analysis)
        assert modifier < 1.0

    def test_sell_modifier_with_selling_pressure(self, analyzer):
        """Sell modifier should increase with selling pressure."""
        book = make_sell_heavy_book()
        analysis = analyzer.analyze(book)
        modifier = analyzer.get_sell_confidence_modifier(analysis)
        assert modifier > 1.0

    def test_sell_modifier_with_buying_pressure(self, analyzer):
        """Sell modifier should decrease with buying pressure."""
        book = make_buy_heavy_book()
        analysis = analyzer.analyze(book)
        modifier = analyzer.get_sell_confidence_modifier(analysis)
        assert modifier < 1.0

    def test_buy_modifier_with_buy_wall_support(self, analyzer):
        """Buy wall below price should boost buy confidence."""
        analysis = OrderBookAnalysis(
            bid_volume=30.0,
            ask_volume=30.0,
            imbalance_ratio=1.0,
            buy_wall_below=True,
        )
        modifier = analyzer.get_buy_confidence_modifier(analysis)
        assert modifier > 1.0  # 1.0 * 1.2 = 1.2

    def test_buy_modifier_with_sell_wall_resistance(self, analyzer):
        """Sell wall above price should reduce buy confidence."""
        analysis = OrderBookAnalysis(
            bid_volume=30.0,
            ask_volume=30.0,
            imbalance_ratio=1.0,
            sell_wall_above=True,
        )
        modifier = analyzer.get_buy_confidence_modifier(analysis)
        assert modifier < 1.0  # 1.0 * 0.8 = 0.8

    def test_sell_modifier_with_sell_wall(self, analyzer):
        """Sell wall above price should boost sell confidence."""
        analysis = OrderBookAnalysis(
            bid_volume=30.0,
            ask_volume=30.0,
            imbalance_ratio=1.0,
            sell_wall_above=True,
        )
        modifier = analyzer.get_sell_confidence_modifier(analysis)
        assert modifier > 1.0

    def test_sell_modifier_with_buy_wall(self, analyzer):
        """Buy wall below price should reduce sell confidence."""
        analysis = OrderBookAnalysis(
            bid_volume=30.0,
            ask_volume=30.0,
            imbalance_ratio=1.0,
            buy_wall_below=True,
        )
        modifier = analyzer.get_sell_confidence_modifier(analysis)
        assert modifier < 1.0


# --- Current Price Detection ---


class TestCurrentPrice:
    def test_mid_price_from_bids_and_asks(self, analyzer):
        """Current price should be mid-point of best bid and ask."""
        price = OrderBookAnalyzer._get_current_price(
            [[100.0, 10.0]], [[101.0, 10.0]]
        )
        assert price == pytest.approx(100.5)

    def test_price_from_bids_only(self, analyzer):
        """With only bids, use best bid as price."""
        price = OrderBookAnalyzer._get_current_price(
            [[100.0, 10.0]], []
        )
        assert price == 100.0

    def test_price_from_asks_only(self, analyzer):
        """With only asks, use best ask as price."""
        price = OrderBookAnalyzer._get_current_price(
            [], [[101.0, 10.0]]
        )
        assert price == 101.0

    def test_no_price_empty(self, analyzer):
        """Empty book returns None."""
        price = OrderBookAnalyzer._get_current_price([], [])
        assert price is None


# --- Fetch and Analyze Tests ---


class TestFetchAndAnalyze:
    @pytest.mark.asyncio
    async def test_fetch_and_analyze_success(self, analyzer):
        """Successful fetch and analysis."""
        mock_exchange = AsyncMock()
        mock_exchange.get_order_book.return_value = make_balanced_book()

        analysis = await analyzer.fetch_and_analyze(
            mock_exchange, "BTC/USDT"
        )
        assert analysis.bid_volume == 30.0
        assert analysis.ask_volume == 30.0
        mock_exchange.get_order_book.assert_called_once_with(
            "BTC/USDT", limit=20
        )

    @pytest.mark.asyncio
    async def test_fetch_and_analyze_error_returns_default(self, analyzer):
        """Exchange error should return default neutral analysis."""
        mock_exchange = AsyncMock()
        mock_exchange.get_order_book.side_effect = Exception("connection error")

        analysis = await analyzer.fetch_and_analyze(
            mock_exchange, "BTC/USDT"
        )
        assert analysis.bid_volume == 0.0
        assert analysis.ask_volume == 0.0
        assert analysis.imbalance_ratio == 1.0
        assert analysis.confidence_modifier == 1.0

    @pytest.mark.asyncio
    async def test_fetch_uses_configured_depth(self, custom_analyzer):
        """Fetch should use the configured depth."""
        mock_exchange = AsyncMock()
        mock_exchange.get_order_book.return_value = make_balanced_book()

        await custom_analyzer.fetch_and_analyze(mock_exchange, "ETH/USDT")
        mock_exchange.get_order_book.assert_called_once_with(
            "ETH/USDT", limit=10
        )


# --- Edge Cases ---


class TestEdgeCases:
    def test_single_level_each_side(self, analyzer):
        """Single level on each side should work."""
        book = make_order_book(
            bids=[[100.0, 50.0]],
            asks=[[101.0, 10.0]],
        )
        analysis = analyzer.analyze(book)
        assert analysis.imbalance_ratio == 5.0

    def test_zero_volume_levels(self, analyzer):
        """Zero-volume levels should not affect calculations."""
        book = make_order_book(
            bids=[[100.0, 0.0], [99.0, 10.0]],
            asks=[[101.0, 10.0]],
        )
        analysis = analyzer.analyze(book)
        assert analysis.bid_volume == 10.0

    def test_analysis_with_symbol_logging(self, analyzer):
        """Symbol parameter should be accepted for logging."""
        book = make_balanced_book()
        analysis = analyzer.analyze(book, symbol="BTC/USDT")
        assert analysis.imbalance_ratio == 1.0

    def test_wall_info_dataclass(self):
        """WallInfo should hold correct data."""
        wall = WallInfo(
            price=50000.0,
            volume=100.0,
            side="bid",
            volume_ratio=6.5,
        )
        assert wall.price == 50000.0
        assert wall.volume == 100.0
        assert wall.side == "bid"
        assert wall.volume_ratio == 6.5

    def test_order_book_analysis_defaults(self):
        """OrderBookAnalysis defaults should be neutral."""
        analysis = OrderBookAnalysis()
        assert analysis.bid_volume == 0.0
        assert analysis.ask_volume == 0.0
        assert analysis.imbalance_ratio == 1.0
        assert analysis.walls == []
        assert analysis.buy_wall_below is False
        assert analysis.sell_wall_above is False
        assert analysis.confidence_modifier == 1.0
