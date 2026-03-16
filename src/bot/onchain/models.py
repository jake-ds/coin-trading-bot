"""Data models for on-chain signal system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class MarketData:
    """Aggregated market data from CoinGecko."""

    symbol: str
    price: float = 0.0
    price_change_24h_pct: float = 0.0
    market_cap: float = 0.0
    volume_24h: float = 0.0
    btc_dominance: float = 0.0
    total_market_cap: float = 0.0
    timestamp: str = ""


@dataclass
class SentimentData:
    """Fear & Greed index data."""

    value: int = 50  # 0-100
    classification: str = "Neutral"  # Extreme Fear, Fear, Neutral, Greed, Extreme Greed
    timestamp: str = ""


@dataclass
class DefiData:
    """DeFi ecosystem data from DeFiLlama."""

    total_tvl: float = 0.0
    tvl_change_24h_pct: float = 0.0
    stablecoin_total_supply: float = 0.0
    stablecoin_supply_change_24h_pct: float = 0.0
    timestamp: str = ""


@dataclass
class DerivativesData:
    """Derivatives market data from CoinGlass."""

    symbol: str = "BTC"
    open_interest: float = 0.0
    oi_change_24h_pct: float = 0.0
    funding_rate: float = 0.0
    liquidations_24h_long: float = 0.0
    liquidations_24h_short: float = 0.0
    timestamp: str = ""


@dataclass
class WhaleFlowData:
    """Exchange flow data (net inflow/outflow)."""

    symbol: str = "BTC"
    net_flow: float = 0.0  # positive = inflow (bearish), negative = outflow (bullish)
    inflow: float = 0.0
    outflow: float = 0.0
    timestamp: str = ""


@dataclass
class SignalScore:
    """Individual signal score from one data category."""

    name: str  # whale_flow, sentiment, defi_flow, derivatives, market_trend
    score: float = 0.0  # -100 to +100
    confidence: float = 0.0  # 0.0 to 1.0
    reason: str = ""


@dataclass
class CompositeSignal:
    """Weighted composite signal from all data sources."""

    symbol: str
    action: SignalAction = SignalAction.HOLD
    score: float = 0.0  # -100 to +100
    confidence: float = 0.0  # 0.0 to 1.0
    signals: list[SignalScore] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "score": round(self.score, 2),
            "confidence": round(self.confidence, 3),
            "signals": [
                {
                    "name": s.name,
                    "score": round(s.score, 2),
                    "confidence": round(s.confidence, 3),
                    "reason": s.reason,
                }
                for s in self.signals
            ],
            "timestamp": self.timestamp,
        }
