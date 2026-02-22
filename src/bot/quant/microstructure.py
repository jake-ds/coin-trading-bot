"""Order book microstructure analysis.

Provides VWAP mid-price, microprice, order book imbalance, and wall detection
for order-flow based signal confidence adjustment.
"""

from __future__ import annotations

import numpy as np
import structlog

logger = structlog.get_logger()


def vwap_midprice(
    bids: list[tuple[float, float]], asks: list[tuple[float, float]], depth: int = 5
) -> float:
    """Calculate volume-weighted average price mid-price.

    Uses top N levels of the order book weighted by volume.

    Args:
        bids: List of (price, quantity) tuples, best bid first.
        asks: List of (price, quantity) tuples, best ask first.
        depth: Number of levels to use.

    Returns:
        VWAP mid-price.
    """
    if not bids or not asks:
        return 0.0

    top_bids = bids[:depth]
    top_asks = asks[:depth]

    bid_vwap_num = sum(p * q for p, q in top_bids)
    bid_vwap_den = sum(q for _, q in top_bids)
    ask_vwap_num = sum(p * q for p, q in top_asks)
    ask_vwap_den = sum(q for _, q in top_asks)

    if bid_vwap_den < 1e-10 or ask_vwap_den < 1e-10:
        return 0.0

    bid_vwap = bid_vwap_num / bid_vwap_den
    ask_vwap = ask_vwap_num / ask_vwap_den

    return (bid_vwap + ask_vwap) / 2.0


def microprice(
    best_bid: float,
    best_ask: float,
    bid_size: float,
    ask_size: float,
) -> float:
    """Calculate microprice (size-weighted mid-price).

    Microprice adjusts the mid-price towards the side with more volume,
    providing a better estimate of the true price.

    Args:
        best_bid: Best bid price.
        best_ask: Best ask price.
        bid_size: Size at best bid.
        ask_size: Size at best ask.

    Returns:
        Microprice.
    """
    total_size = bid_size + ask_size
    if total_size < 1e-10:
        return (best_bid + best_ask) / 2.0

    # Microprice: weighted towards the side with LESS volume
    # (if asks are thin, price likely moves up -> weight towards ask)
    return (best_bid * ask_size + best_ask * bid_size) / total_size


def orderbook_imbalance(
    bids: list[tuple[float, float]], asks: list[tuple[float, float]], depth: int = 10
) -> float:
    """Calculate order book imbalance ratio.

    Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    Range: [-1, +1]. Positive means buy pressure, negative means sell pressure.

    Args:
        bids: List of (price, quantity) tuples.
        asks: List of (price, quantity) tuples.
        depth: Number of levels to consider.

    Returns:
        Imbalance ratio in [-1, 1].
    """
    bid_vol = sum(q for _, q in bids[:depth])
    ask_vol = sum(q for _, q in asks[:depth])
    total = bid_vol + ask_vol

    if total < 1e-10:
        return 0.0

    return (bid_vol - ask_vol) / total


def detect_walls(
    levels: list[tuple[float, float]],
    threshold_multiplier: float = 3.0,
) -> list[dict]:
    """Detect large walls (support/resistance) in order book levels.

    A wall is a price level with volume significantly above the average.

    Args:
        levels: List of (price, quantity) tuples from one side of the book.
        threshold_multiplier: Volume must be this many times the average.

    Returns:
        List of dicts with 'price', 'quantity', 'ratio' for detected walls.
    """
    if len(levels) < 3:
        return []

    quantities = np.array([q for _, q in levels])
    avg_qty = float(np.mean(quantities))

    if avg_qty < 1e-10:
        return []

    walls = []
    for price, qty in levels:
        ratio = qty / avg_qty
        if ratio >= threshold_multiplier:
            walls.append({
                "price": price,
                "quantity": qty,
                "ratio": round(ratio, 2),
            })

    return walls


def compute_orderbook_metrics(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    depth: int = 10,
) -> dict:
    """Compute comprehensive order book metrics.

    Args:
        bids: List of (price, quantity) tuples, best bid first.
        asks: List of (price, quantity) tuples, best ask first.
        depth: Number of levels to analyze.

    Returns:
        Dict with 'vwap_mid', 'microprice', 'imbalance', 'spread',
        'spread_bps', 'bid_walls', 'ask_walls'.
    """
    if not bids or not asks:
        return {
            "vwap_mid": 0.0,
            "microprice": 0.0,
            "imbalance": 0.0,
            "spread": 0.0,
            "spread_bps": 0.0,
            "bid_walls": [],
            "ask_walls": [],
        }

    best_bid_price, best_bid_size = bids[0]
    best_ask_price, best_ask_size = asks[0]
    mid = (best_bid_price + best_ask_price) / 2.0
    spread = best_ask_price - best_bid_price
    spread_bps = (spread / mid * 10000) if mid > 0 else 0.0

    return {
        "vwap_mid": vwap_midprice(bids, asks, depth),
        "microprice": microprice(best_bid_price, best_ask_price, best_bid_size, best_ask_size),
        "imbalance": orderbook_imbalance(bids, asks, depth),
        "spread": spread,
        "spread_bps": round(spread_bps, 2),
        "bid_walls": detect_walls(bids[:depth]),
        "ask_walls": detect_walls(asks[:depth]),
    }
