"""Tests for on-chain data fetchers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.onchain.alternative import FearGreedFetcher
from bot.onchain.coingecko import CoinGeckoFetcher
from bot.onchain.coinglass import CoinGlassFetcher
from bot.onchain.defillama import DeFiLlamaFetcher
from bot.onchain.etherscan import EtherscanFetcher
from bot.onchain.fetcher_base import BaseFetcher
from bot.onchain.models import MarketData, SentimentData


# ---------------------------------------------------------------------------
# BaseFetcher
# ---------------------------------------------------------------------------

class ConcreteFetcher(BaseFetcher):
    async def fetch(self):
        return await self._get("/test")


@pytest.mark.asyncio
async def test_base_fetcher_cache():
    """Cached response should be returned without making a new request."""
    fetcher = ConcreteFetcher(base_url="https://example.com", cache_ttl=60.0)
    fetcher._set_cached("/test:None", {"result": "cached"})
    result = await fetcher.fetch()
    assert result == {"result": "cached"}
    await fetcher.close()


@pytest.mark.asyncio
async def test_base_fetcher_cache_expiry():
    """Expired cache should not be returned."""
    fetcher = ConcreteFetcher(base_url="https://example.com", cache_ttl=0.0)
    fetcher._set_cached("/test:None", {"result": "old"})
    # Cache should already be expired since TTL is 0
    assert fetcher._get_cached("/test:None") is None
    await fetcher.close()


# ---------------------------------------------------------------------------
# CoinGecko
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_coingecko_fetch_market_data():
    """CoinGecko fetcher should parse market data correctly."""
    fetcher = CoinGeckoFetcher(cache_ttl=0.0)

    mock_global = {
        "data": {
            "market_cap_percentage": {"btc": 52.3},
            "total_market_cap": {"usd": 2500000000000},
        }
    }
    mock_coins = [
        {
            "id": "bitcoin",
            "current_price": 85000,
            "price_change_percentage_24h": 2.5,
            "market_cap": 1700000000000,
            "total_volume": 50000000000,
        },
        {
            "id": "ethereum",
            "current_price": 3200,
            "price_change_percentage_24h": -1.2,
            "market_cap": 400000000000,
            "total_volume": 20000000000,
        },
    ]

    call_count = 0

    async def mock_get(path, params=None):
        nonlocal call_count
        call_count += 1
        if "/global" in path:
            return mock_global
        return mock_coins

    fetcher._get = mock_get

    result = await fetcher.fetch_market_data(["BTC/USDT", "ETH/USDT"])
    assert result is not None
    assert "BTC/USDT" in result
    assert "ETH/USDT" in result
    assert result["BTC/USDT"].price == 85000
    assert result["BTC/USDT"].price_change_24h_pct == 2.5
    assert result["BTC/USDT"].btc_dominance == 52.3
    assert result["ETH/USDT"].price == 3200
    await fetcher.close()


@pytest.mark.asyncio
async def test_coingecko_fetch_returns_none_on_failure():
    """Should return None if API fails."""
    fetcher = CoinGeckoFetcher(cache_ttl=0.0)

    async def mock_get(path, params=None):
        return None

    fetcher._get = mock_get
    result = await fetcher.fetch_market_data(["BTC/USDT"])
    assert result is None
    await fetcher.close()


# ---------------------------------------------------------------------------
# Fear & Greed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fear_greed_fetch():
    """Fear & Greed fetcher should return SentimentData."""
    fetcher = FearGreedFetcher(cache_ttl=0.0)

    mock_data = {
        "data": [
            {
                "value": "23",
                "value_classification": "Extreme Fear",
                "timestamp": "1710547200",
            }
        ]
    }

    async def mock_get(path, params=None):
        return mock_data

    fetcher._get = mock_get
    result = await fetcher.fetch()
    assert result is not None
    assert isinstance(result, SentimentData)
    assert result.value == 23
    assert result.classification == "Extreme Fear"
    await fetcher.close()


@pytest.mark.asyncio
async def test_fear_greed_returns_none_on_empty():
    """Should return None if no data."""
    fetcher = FearGreedFetcher(cache_ttl=0.0)

    async def mock_get(path, params=None):
        return {"data": []}

    fetcher._get = mock_get
    result = await fetcher.fetch()
    assert result is None
    await fetcher.close()


# ---------------------------------------------------------------------------
# DeFiLlama
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_defillama_fetch():
    """DeFiLlama fetcher should return DefiData."""
    fetcher = DeFiLlamaFetcher(cache_ttl=0.0)

    call_map = {}

    async def mock_get(path, params=None):
        if "historicalChainTvl" in path:
            return [
                {"tvl": 100_000_000_000},
                {"tvl": 105_000_000_000},
            ]
        return None

    async def mock_get_absolute(url, params=None):
        if "stablecoins" in url and "charts" not in url:
            return {
                "peggedAssets": [
                    {"circulating": {"peggedUSD": 50_000_000_000}}
                ]
            }
        if "stablecoincharts" in url:
            return None
        return None

    fetcher._get = mock_get
    fetcher._get_absolute = mock_get_absolute
    result = await fetcher.fetch()
    assert result is not None
    assert result.total_tvl == 105_000_000_000
    assert result.tvl_change_24h_pct == pytest.approx(5.0, rel=0.01)
    assert result.stablecoin_total_supply == 50_000_000_000
    await fetcher.close()


# ---------------------------------------------------------------------------
# CoinGlass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_coinglass_no_api_key():
    """CoinGlass should return None without API key."""
    fetcher = CoinGlassFetcher(api_key="", cache_ttl=0.0)
    result = await fetcher.fetch()
    assert result is None
    await fetcher.close()


@pytest.mark.asyncio
async def test_coinglass_fetch_derivatives():
    """CoinGlass fetcher should parse derivatives data."""
    fetcher = CoinGlassFetcher(api_key="test-key", cache_ttl=0.0)

    async def mock_get(path, params=None):
        if "funding-rate" in path:
            return {"code": "0", "data": [{"rate": 0.0003}]}
        if "openInterest" in path:
            return {
                "code": "0",
                "data": [
                    {"openInterest": 10_000_000_000},
                    {"openInterest": 10_500_000_000},
                ],
            }
        if "liquidation" in path:
            return {
                "code": "0",
                "data": {
                    "longLiquidationUsd": 50_000_000,
                    "shortLiquidationUsd": 30_000_000,
                },
            }
        return None

    # Override the parent _get since CoinGlass overrides it
    original_get = fetcher._get

    async def patched_get(path, params=None):
        cache_key = f"{path}:{params}"
        cached = fetcher._get_cached(cache_key)
        if cached is not None:
            return cached
        result = await mock_get(path, params)
        if result:
            fetcher._set_cached(cache_key, result)
        return result

    fetcher._get = patched_get

    result = await fetcher.fetch_derivatives(["BTC/USDT"])
    assert result is not None
    assert "BTC/USDT" in result
    deriv = result["BTC/USDT"]
    assert deriv.funding_rate == 0.0003
    assert deriv.open_interest == 10_500_000_000
    assert deriv.oi_change_24h_pct == pytest.approx(5.0, rel=0.01)
    await fetcher.close()


# ---------------------------------------------------------------------------
# Etherscan
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_etherscan_no_api_key():
    """Etherscan should return None without API key."""
    fetcher = EtherscanFetcher(api_key="", cache_ttl=0.0)
    result = await fetcher.fetch()
    assert result is None
    await fetcher.close()


@pytest.mark.asyncio
async def test_etherscan_fetch_whale_transfers():
    """Etherscan fetcher should return WhaleFlowData."""
    fetcher = EtherscanFetcher(api_key="test-key", cache_ttl=0.0)

    mock_data = {
        "status": "1",
        "result": [
            {"value": "32000000000000000000"},  # 32 ETH
            {"value": "64000000000000000000"},  # 64 ETH
        ],
    }

    async def mock_get(path, params=None):
        return mock_data

    fetcher._get = mock_get
    result = await fetcher.fetch()
    assert result is not None
    assert result.symbol == "ETH"
    assert result.outflow == pytest.approx(96.0, rel=0.01)
    assert result.net_flow < 0  # Outflow is bullish (negative net)
    await fetcher.close()
