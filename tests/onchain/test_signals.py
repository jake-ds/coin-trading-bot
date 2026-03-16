"""Tests for on-chain signal scoring system."""

import pytest

from bot.onchain.models import (
    CompositeSignal,
    DefiData,
    DerivativesData,
    MarketData,
    SentimentData,
    SignalAction,
    WhaleFlowData,
)
from bot.onchain.signals import (
    compute_composite_signal,
    score_defi_flow,
    score_derivatives,
    score_market_trend,
    score_sentiment,
    score_whale_flow,
)


# ---------------------------------------------------------------------------
# Whale Flow
# ---------------------------------------------------------------------------


def test_whale_flow_outflow_bullish():
    """Net outflow from exchanges should be bullish (positive score)."""
    flow = WhaleFlowData(
        symbol="BTC",
        net_flow=-1000.0,  # outflow
        inflow=500.0,
        outflow=1500.0,
    )
    result = score_whale_flow(flow)
    assert result.score > 0
    assert result.confidence > 0


def test_whale_flow_inflow_bearish():
    """Net inflow to exchanges should be bearish (negative score)."""
    flow = WhaleFlowData(
        symbol="BTC",
        net_flow=1000.0,  # inflow
        inflow=1500.0,
        outflow=500.0,
    )
    result = score_whale_flow(flow)
    assert result.score < 0


def test_whale_flow_none():
    """No data should return 0 score and 0 confidence."""
    result = score_whale_flow(None)
    assert result.score == 0.0
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------


def test_sentiment_extreme_fear_bullish():
    """Extreme Fear (<25) should produce a strong BUY signal."""
    sentiment = SentimentData(value=15, classification="Extreme Fear")
    result = score_sentiment(sentiment)
    assert result.score > 50
    assert result.confidence > 0.5


def test_sentiment_extreme_greed_bearish():
    """Extreme Greed (>90) should produce a strong SELL signal."""
    sentiment = SentimentData(value=95, classification="Extreme Greed")
    result = score_sentiment(sentiment)
    assert result.score < -50
    assert result.confidence > 0.5


def test_sentiment_neutral():
    """Neutral sentiment (40-60) should produce score near 0."""
    sentiment = SentimentData(value=50, classification="Neutral")
    result = score_sentiment(sentiment)
    assert abs(result.score) < 10


def test_sentiment_none():
    result = score_sentiment(None)
    assert result.score == 0.0
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# DeFi Flow
# ---------------------------------------------------------------------------


def test_defi_tvl_increase_bullish():
    """TVL increase should be bullish."""
    defi = DefiData(
        total_tvl=105_000_000_000,
        tvl_change_24h_pct=5.0,
        stablecoin_total_supply=150_000_000_000,
        stablecoin_supply_change_24h_pct=0.5,
    )
    result = score_defi_flow(defi)
    assert result.score > 0


def test_defi_tvl_decrease_bearish():
    """TVL decrease should be bearish."""
    defi = DefiData(
        total_tvl=95_000_000_000,
        tvl_change_24h_pct=-5.0,
        stablecoin_total_supply=150_000_000_000,
        stablecoin_supply_change_24h_pct=-0.5,
    )
    result = score_defi_flow(defi)
    assert result.score < 0


def test_defi_none():
    result = score_defi_flow(None)
    assert result.score == 0.0
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------


def test_derivatives_high_positive_funding_bearish():
    """High positive funding rate should be bearish (longs overleveraged)."""
    deriv = DerivativesData(
        symbol="BTC",
        funding_rate=0.001,
        open_interest=10_000_000_000,
        oi_change_24h_pct=0.0,
    )
    result = score_derivatives(deriv)
    assert result.score < 0


def test_derivatives_high_negative_funding_bullish():
    """High negative funding rate should be bullish (shorts overleveraged)."""
    deriv = DerivativesData(
        symbol="BTC",
        funding_rate=-0.001,
        open_interest=10_000_000_000,
        oi_change_24h_pct=0.0,
    )
    result = score_derivatives(deriv)
    assert result.score > 0


def test_derivatives_none():
    result = score_derivatives(None)
    assert result.score == 0.0
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Market Trend
# ---------------------------------------------------------------------------


def test_market_trend_strong_up():
    """Strong positive price change should be bullish."""
    market = MarketData(
        symbol="BTC/USDT",
        price=85000,
        price_change_24h_pct=8.0,
        btc_dominance=50.0,
    )
    result = score_market_trend(market)
    assert result.score > 0


def test_market_trend_strong_down():
    """Strong negative price change should be bearish."""
    market = MarketData(
        symbol="BTC/USDT",
        price=80000,
        price_change_24h_pct=-8.0,
        btc_dominance=50.0,
    )
    result = score_market_trend(market)
    assert result.score < 0


def test_market_trend_none():
    result = score_market_trend(None)
    assert result.score == 0.0
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Composite Signal
# ---------------------------------------------------------------------------


def test_composite_buy_signal():
    """Strong bullish data should produce a BUY signal."""
    signal = compute_composite_signal(
        symbol="BTC/USDT",
        market=MarketData(symbol="BTC/USDT", price=85000, price_change_24h_pct=5.0),
        sentiment=SentimentData(value=15, classification="Extreme Fear"),
        defi=DefiData(tvl_change_24h_pct=3.0, stablecoin_supply_change_24h_pct=1.0),
        derivatives=DerivativesData(symbol="BTC", funding_rate=-0.0005),
        whale_flow=WhaleFlowData(symbol="BTC", net_flow=-1000, inflow=200, outflow=1200),
        buy_threshold=30.0,
        sell_threshold=-30.0,
        min_confidence=0.3,
    )
    assert signal.action == SignalAction.BUY
    assert signal.score > 30.0
    assert signal.confidence > 0.3


def test_composite_sell_signal():
    """Strong bearish data should produce a SELL signal."""
    signal = compute_composite_signal(
        symbol="BTC/USDT",
        market=MarketData(symbol="BTC/USDT", price=80000, price_change_24h_pct=-5.0),
        sentiment=SentimentData(value=95, classification="Extreme Greed"),
        defi=DefiData(tvl_change_24h_pct=-3.0, stablecoin_supply_change_24h_pct=-1.0),
        derivatives=DerivativesData(symbol="BTC", funding_rate=0.001),
        whale_flow=WhaleFlowData(symbol="BTC", net_flow=1000, inflow=1200, outflow=200),
        buy_threshold=30.0,
        sell_threshold=-30.0,
        min_confidence=0.3,
    )
    assert signal.action == SignalAction.SELL
    assert signal.score < -30.0


def test_composite_hold_when_neutral():
    """Mixed/neutral data should produce HOLD."""
    signal = compute_composite_signal(
        symbol="BTC/USDT",
        market=MarketData(symbol="BTC/USDT", price=85000, price_change_24h_pct=0.5),
        sentiment=SentimentData(value=50, classification="Neutral"),
        defi=DefiData(tvl_change_24h_pct=0.1, stablecoin_supply_change_24h_pct=0.0),
        derivatives=DerivativesData(symbol="BTC", funding_rate=0.00005),
        whale_flow=WhaleFlowData(symbol="BTC", net_flow=0, inflow=100, outflow=100),
    )
    assert signal.action == SignalAction.HOLD


def test_composite_hold_when_low_confidence():
    """Should HOLD even with positive score if confidence is too low."""
    signal = compute_composite_signal(
        symbol="BTC/USDT",
        buy_threshold=30.0,
        min_confidence=0.9,  # Very high threshold
        # With no data, confidence will be 0
    )
    assert signal.action == SignalAction.HOLD


def test_composite_signal_to_dict():
    """CompositeSignal.to_dict() should produce valid dict."""
    signal = compute_composite_signal(
        symbol="BTC/USDT",
        sentiment=SentimentData(value=20, classification="Fear"),
    )
    d = signal.to_dict()
    assert d["symbol"] == "BTC/USDT"
    assert "score" in d
    assert "confidence" in d
    assert "signals" in d
    assert isinstance(d["signals"], list)
    assert len(d["signals"]) == 5


def test_composite_custom_weights():
    """Custom weights should change the final score."""
    # Use a moderate value so score isn't capped at 100
    base = compute_composite_signal(
        symbol="BTC/USDT",
        sentiment=SentimentData(value=30, classification="Fear"),
        market=MarketData(symbol="BTC/USDT", price=85000, price_change_24h_pct=-2.0),
    )

    heavy_sentiment = compute_composite_signal(
        symbol="BTC/USDT",
        sentiment=SentimentData(value=30, classification="Fear"),
        market=MarketData(symbol="BTC/USDT", price=85000, price_change_24h_pct=-2.0),
        weights={
            "whale_flow": 0.0,
            "sentiment": 1.0,
            "defi_flow": 0.0,
            "derivatives": 0.0,
            "market_trend": 0.0,
        },
    )

    # With 100% weight on bullish sentiment, score should be higher than
    # default weights which also include bearish market_trend
    assert heavy_sentiment.score > base.score
