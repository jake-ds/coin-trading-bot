"""Signal scoring system — combines on-chain data into composite trading signals."""

from __future__ import annotations

import structlog

from bot.onchain.models import (
    CompositeSignal,
    DefiData,
    DerivativesData,
    MarketData,
    SentimentData,
    SignalAction,
    SignalScore,
    WhaleFlowData,
)

logger = structlog.get_logger(__name__)

# Default signal weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "whale_flow": 0.25,
    "sentiment": 0.15,
    "defi_flow": 0.20,
    "derivatives": 0.25,
    "market_trend": 0.15,
}


def score_whale_flow(flow: WhaleFlowData | None) -> SignalScore:
    """Score exchange flow data.

    Net outflow from exchanges = accumulation = bullish.
    Net inflow to exchanges = distribution = bearish.
    """
    if flow is None:
        return SignalScore(name="whale_flow", score=0.0, confidence=0.0, reason="No data")

    net = flow.net_flow

    if abs(net) < 1e-9:
        return SignalScore(
            name="whale_flow", score=0.0, confidence=0.2,
            reason="Neutral flow",
        )

    # Negative net_flow = outflow = bullish, positive = inflow = bearish
    # Normalize: large flow magnitude increases confidence
    total_volume = flow.inflow + flow.outflow
    if total_volume > 0:
        flow_ratio = net / total_volume  # -1 to +1
        score = -flow_ratio * 100.0  # Invert: outflow = positive score
        score = max(-100.0, min(100.0, score))
        confidence = min(1.0, abs(flow_ratio) * 2.0)  # Higher ratio = higher confidence
    else:
        score = -50.0 if net > 0 else 50.0
        confidence = 0.3

    direction = "outflow (bullish)" if net < 0 else "inflow (bearish)"
    return SignalScore(
        name="whale_flow",
        score=score,
        confidence=confidence,
        reason=f"Net {direction}: {abs(net):.2f}",
    )


def score_sentiment(sentiment: SentimentData | None) -> SignalScore:
    """Score Fear & Greed Index using contrarian logic.

    Extreme Fear (<25) = contrarian BUY signal.
    Extreme Greed (>75) = contrarian SELL signal.
    """
    if sentiment is None:
        return SignalScore(name="sentiment", score=0.0, confidence=0.0, reason="No data")

    value = sentiment.value

    if value <= 10:
        score = 100.0
        confidence = 0.9
        reason = f"Extreme Fear ({value}) — strong contrarian BUY"
    elif value <= 25:
        score = 60.0
        confidence = 0.7
        reason = f"Fear ({value}) — contrarian BUY"
    elif value <= 40:
        score = 20.0
        confidence = 0.4
        reason = f"Mild Fear ({value}) — slight BUY bias"
    elif value <= 60:
        score = 0.0
        confidence = 0.3
        reason = f"Neutral ({value})"
    elif value <= 75:
        score = -20.0
        confidence = 0.4
        reason = f"Mild Greed ({value}) — slight SELL bias"
    elif value <= 90:
        score = -60.0
        confidence = 0.7
        reason = f"Greed ({value}) — contrarian SELL"
    else:
        score = -100.0
        confidence = 0.9
        reason = f"Extreme Greed ({value}) — strong contrarian SELL"

    return SignalScore(name="sentiment", score=score, confidence=confidence, reason=reason)


def score_defi_flow(defi: DefiData | None) -> SignalScore:
    """Score DeFi ecosystem health.

    TVL increase = risk-on = bullish.
    Stablecoin supply increase = liquidity inflow = bullish.
    """
    if defi is None:
        return SignalScore(name="defi_flow", score=0.0, confidence=0.0, reason="No data")

    tvl_score = 0.0
    stable_score = 0.0
    reasons = []

    # TVL change scoring
    tvl_change = defi.tvl_change_24h_pct
    if abs(tvl_change) > 0.5:
        tvl_score = max(-100.0, min(100.0, tvl_change * 20.0))
        reasons.append(f"TVL {tvl_change:+.1f}%")

    # Stablecoin supply scoring
    stable_change = defi.stablecoin_supply_change_24h_pct
    if abs(stable_change) > 0.1:
        stable_score = max(-100.0, min(100.0, stable_change * 30.0))
        reasons.append(f"Stablecoin supply {stable_change:+.2f}%")

    # Combine (equal weight)
    combined_score = (tvl_score + stable_score) / 2.0
    confidence = min(1.0, (abs(tvl_change) + abs(stable_change)) / 5.0)
    confidence = max(0.2, confidence)

    reason = ", ".join(reasons) if reasons else "Neutral DeFi flow"

    return SignalScore(
        name="defi_flow",
        score=max(-100.0, min(100.0, combined_score)),
        confidence=confidence,
        reason=reason,
    )


def score_derivatives(deriv: DerivativesData | None) -> SignalScore:
    """Score derivatives market data.

    - Extreme positive funding = overleveraged longs = bearish
    - Extreme negative funding = overleveraged shorts = bullish
    - Large OI increase = potential squeeze
    - Liquidation cascade = potential reversal
    """
    if deriv is None:
        return SignalScore(name="derivatives", score=0.0, confidence=0.0, reason="No data")

    score = 0.0
    confidence = 0.3
    reasons = []

    # Funding rate scoring (annualized: rate * 3 * 365)
    fr = deriv.funding_rate
    if abs(fr) > 0.0001:
        # High positive funding = bearish (longs paying shorts)
        # High negative funding = bullish (shorts paying longs)
        fr_score = -fr * 100000.0  # Scale to meaningful range
        fr_score = max(-80.0, min(80.0, fr_score))
        score += fr_score * 0.4
        confidence = max(confidence, min(1.0, abs(fr) * 5000.0))
        annualized = fr * 3 * 365 * 100
        reasons.append(f"Funding {fr:.4f} ({annualized:.1f}% ann.)")

    # OI change scoring
    oi_change = deriv.oi_change_24h_pct
    if abs(oi_change) > 5.0:
        # Large OI increase with positive funding = bearish squeeze setup
        # Large OI increase with negative funding = bullish squeeze setup
        oi_direction = -1.0 if fr > 0 else 1.0
        oi_score = oi_direction * min(50.0, abs(oi_change) * 2.0)
        score += oi_score * 0.3
        reasons.append(f"OI {oi_change:+.1f}%")

    # Liquidation scoring
    long_liq = deriv.liquidations_24h_long
    short_liq = deriv.liquidations_24h_short
    total_liq = long_liq + short_liq
    if total_liq > 100_000_000:  # >$100M liquidations = significant
        # More long liquidations = potential bottom (bullish reversal)
        # More short liquidations = potential top (bearish reversal)
        if long_liq > short_liq * 1.5:
            liq_score = 30.0  # Bullish reversal signal
            reasons.append(f"Long liq cascade ${long_liq/1e6:.0f}M")
        elif short_liq > long_liq * 1.5:
            liq_score = -30.0  # Bearish reversal signal
            reasons.append(f"Short liq cascade ${short_liq/1e6:.0f}M")
        else:
            liq_score = 0.0
        score += liq_score * 0.3

    score = max(-100.0, min(100.0, score))
    reason = ", ".join(reasons) if reasons else "Neutral derivatives"

    return SignalScore(
        name="derivatives", score=score, confidence=confidence, reason=reason,
    )


def score_market_trend(market: MarketData | None) -> SignalScore:
    """Score market trend based on price action and dominance.

    - Strong uptrend = bullish momentum
    - BTC dominance rising = risk-off (bearish for alts)
    """
    if market is None:
        return SignalScore(name="market_trend", score=0.0, confidence=0.0, reason="No data")

    score = 0.0
    reasons = []

    # 24h price change scoring
    change = market.price_change_24h_pct
    if abs(change) > 1.0:
        # Moderate momentum following
        price_score = max(-60.0, min(60.0, change * 8.0))
        score += price_score * 0.6
        reasons.append(f"24h: {change:+.1f}%")

    # BTC dominance for alts (higher dominance = bearish for alts)
    if "BTC" not in market.symbol:
        btc_dom = market.btc_dominance
        if btc_dom > 55:
            dom_score = -(btc_dom - 50) * 2.0  # Bearish for alts
            score += dom_score * 0.4
            reasons.append(f"BTC dom {btc_dom:.1f}% (high)")
        elif btc_dom < 45:
            dom_score = (50 - btc_dom) * 2.0  # Bullish for alts
            score += dom_score * 0.4
            reasons.append(f"BTC dom {btc_dom:.1f}% (low)")

    score = max(-100.0, min(100.0, score))
    confidence = min(1.0, abs(change) / 10.0 + 0.2)
    reason = ", ".join(reasons) if reasons else "Neutral trend"

    return SignalScore(
        name="market_trend", score=score, confidence=confidence, reason=reason,
    )


def compute_composite_signal(
    symbol: str,
    market: MarketData | None = None,
    sentiment: SentimentData | None = None,
    defi: DefiData | None = None,
    derivatives: DerivativesData | None = None,
    whale_flow: WhaleFlowData | None = None,
    weights: dict[str, float] | None = None,
    buy_threshold: float = 30.0,
    sell_threshold: float = -30.0,
    min_confidence: float = 0.4,
) -> CompositeSignal:
    """Compute weighted composite signal from all data sources.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        market: Market price/trend data
        sentiment: Fear & Greed data
        defi: DeFi TVL/stablecoin data
        derivatives: OI/funding/liquidation data
        whale_flow: Exchange flow data
        weights: Signal category weights (default: DEFAULT_WEIGHTS)
        buy_threshold: Score above which to signal BUY
        sell_threshold: Score below which to signal SELL
        min_confidence: Minimum confidence to act

    Returns:
        CompositeSignal with action, score, and individual signal details.
    """
    w = weights or DEFAULT_WEIGHTS

    # Score each category
    scores = [
        score_whale_flow(whale_flow),
        score_sentiment(sentiment),
        score_defi_flow(defi),
        score_derivatives(derivatives),
        score_market_trend(market),
    ]

    # Weighted sum
    total_weight = 0.0
    weighted_score = 0.0
    weighted_confidence = 0.0

    for sig in scores:
        weight = w.get(sig.name, 0.0)
        if sig.confidence > 0:
            weighted_score += sig.score * weight
            weighted_confidence += sig.confidence * weight
            total_weight += weight

    if total_weight > 0:
        final_score = weighted_score / total_weight
        final_confidence = weighted_confidence / total_weight
    else:
        final_score = 0.0
        final_confidence = 0.0

    # Determine action
    action = SignalAction.HOLD
    if final_score >= buy_threshold and final_confidence >= min_confidence:
        action = SignalAction.BUY
    elif final_score <= sell_threshold and final_confidence >= min_confidence:
        action = SignalAction.SELL

    return CompositeSignal(
        symbol=symbol,
        action=action,
        score=final_score,
        confidence=final_confidence,
        signals=scores,
    )
