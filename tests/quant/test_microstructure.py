"""Tests for order book microstructure analysis."""


from bot.quant.microstructure import (
    compute_orderbook_metrics,
    detect_walls,
    microprice,
    orderbook_imbalance,
    vwap_midprice,
)


class TestVWAPMidprice:
    def test_symmetric_book(self):
        bids = [(100.0, 10.0), (99.0, 20.0), (98.0, 30.0)]
        asks = [(101.0, 10.0), (102.0, 20.0), (103.0, 30.0)]
        vwap = vwap_midprice(bids, asks, depth=3)
        assert 99.0 < vwap < 102.0

    def test_empty_book(self):
        assert vwap_midprice([], []) == 0.0

    def test_depth_limit(self):
        bids = [(100.0, 10.0), (99.0, 20.0)]
        asks = [(101.0, 10.0), (102.0, 20.0)]
        vwap_1 = vwap_midprice(bids, asks, depth=1)
        vwap_all = vwap_midprice(bids, asks, depth=2)
        # With depth=1, only top level used
        assert vwap_1 == (100.0 + 101.0) / 2
        assert vwap_all > 0


class TestMicroprice:
    def test_equal_sizes(self):
        mp = microprice(100.0, 101.0, 10.0, 10.0)
        # Equal sizes -> standard midprice
        assert mp == 100.5

    def test_heavy_bid_side(self):
        mp = microprice(100.0, 101.0, 100.0, 10.0)
        # Large bid -> price should lean towards ask
        assert mp > 100.5

    def test_heavy_ask_side(self):
        mp = microprice(100.0, 101.0, 10.0, 100.0)
        # Large ask -> price should lean towards bid
        assert mp < 100.5

    def test_zero_sizes(self):
        mp = microprice(100.0, 101.0, 0.0, 0.0)
        assert mp == 100.5


class TestOrderbookImbalance:
    def test_balanced(self):
        bids = [(100.0, 10.0)]
        asks = [(101.0, 10.0)]
        assert orderbook_imbalance(bids, asks) == 0.0

    def test_buy_pressure(self):
        bids = [(100.0, 100.0)]
        asks = [(101.0, 10.0)]
        imb = orderbook_imbalance(bids, asks)
        assert imb > 0.5

    def test_sell_pressure(self):
        bids = [(100.0, 10.0)]
        asks = [(101.0, 100.0)]
        imb = orderbook_imbalance(bids, asks)
        assert imb < -0.5

    def test_empty(self):
        assert orderbook_imbalance([], []) == 0.0


class TestDetectWalls:
    def test_wall_detected(self):
        levels = [
            (100.0, 10.0),
            (99.0, 10.0),
            (98.0, 100.0),  # Wall: 100/32.5 ~ 3.08x average
            (97.0, 10.0),
        ]
        walls = detect_walls(levels, threshold_multiplier=3.0)
        assert len(walls) == 1
        assert walls[0]["price"] == 98.0

    def test_no_walls(self):
        levels = [(100.0, 10.0), (99.0, 11.0), (98.0, 9.0)]
        walls = detect_walls(levels, threshold_multiplier=3.0)
        assert len(walls) == 0

    def test_too_few_levels(self):
        walls = detect_walls([(100.0, 10.0)])
        assert walls == []


class TestComputeOrderbookMetrics:
    def test_full_metrics(self):
        bids = [(100.0, 10.0), (99.0, 20.0), (98.0, 50.0)]
        asks = [(101.0, 10.0), (102.0, 20.0), (103.0, 30.0)]
        metrics = compute_orderbook_metrics(bids, asks)
        assert metrics["vwap_mid"] > 0
        assert metrics["microprice"] > 0
        assert -1 <= metrics["imbalance"] <= 1
        assert metrics["spread"] == 1.0
        assert metrics["spread_bps"] > 0

    def test_empty_book(self):
        metrics = compute_orderbook_metrics([], [])
        assert metrics["vwap_mid"] == 0.0
        assert metrics["spread"] == 0.0
