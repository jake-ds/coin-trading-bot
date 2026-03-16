"""CoinGecko API fetcher — prices, market cap, trending, global market data."""

from __future__ import annotations

from bot.onchain.fetcher_base import BaseFetcher
from bot.onchain.models import MarketData

# CoinGecko symbol ID mapping
SYMBOL_TO_COINGECKO: dict[str, str] = {
    "BTC/USDT": "bitcoin",
    "ETH/USDT": "ethereum",
    "SOL/USDT": "solana",
    "XRP/USDT": "ripple",
    "DOGE/USDT": "dogecoin",
    "ADA/USDT": "cardano",
    "AVAX/USDT": "avalanche-2",
    "MATIC/USDT": "matic-network",
    "DOT/USDT": "polkadot",
    "LINK/USDT": "chainlink",
    "BNB/USDT": "binancecoin",
}


class CoinGeckoFetcher(BaseFetcher):
    """Fetches market data from CoinGecko free API.

    Rate limit: 30 requests/minute (free tier).
    """

    def __init__(self, cache_ttl: float = 120.0):
        super().__init__(
            base_url="https://api.coingecko.com/api/v3",
            cache_ttl=cache_ttl,
            min_request_interval=2.5,  # ~24 req/min to stay under limit
        )

    async def fetch(self) -> dict[str, MarketData] | None:
        """Fetch market data for all tracked symbols + global data."""
        return await self.fetch_market_data(list(SYMBOL_TO_COINGECKO.keys()))

    async def fetch_market_data(
        self, symbols: list[str]
    ) -> dict[str, MarketData] | None:
        """Fetch price, market cap, and 24h change for given symbols."""
        coin_ids = []
        id_to_symbol: dict[str, str] = {}
        for sym in symbols:
            cg_id = SYMBOL_TO_COINGECKO.get(sym)
            if cg_id:
                coin_ids.append(cg_id)
                id_to_symbol[cg_id] = sym

        if not coin_ids:
            return None

        # Fetch global data first
        global_data = await self._get("/global")
        btc_dominance = 0.0
        total_market_cap = 0.0
        if global_data and "data" in global_data:
            gd = global_data["data"]
            btc_dominance = gd.get("market_cap_percentage", {}).get("btc", 0.0)
            total_market_cap = sum(
                gd.get("total_market_cap", {}).values()
            )

        # Fetch coin data
        data = await self._get(
            "/coins/markets",
            params={
                "vs_currency": "usd",
                "ids": ",".join(coin_ids),
                "order": "market_cap_desc",
                "sparkline": "false",
                "price_change_percentage": "24h",
            },
        )

        if data is None:
            return None

        result: dict[str, MarketData] = {}
        for coin in data:
            cg_id = coin.get("id", "")
            sym = id_to_symbol.get(cg_id)
            if not sym:
                continue
            result[sym] = MarketData(
                symbol=sym,
                price=coin.get("current_price", 0.0) or 0.0,
                price_change_24h_pct=coin.get("price_change_percentage_24h", 0.0) or 0.0,
                market_cap=coin.get("market_cap", 0.0) or 0.0,
                volume_24h=coin.get("total_volume", 0.0) or 0.0,
                btc_dominance=btc_dominance,
                total_market_cap=total_market_cap,
            )

        return result if result else None
