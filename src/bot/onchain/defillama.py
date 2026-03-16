"""DeFiLlama API fetcher — TVL and stablecoin supply data."""

from __future__ import annotations

from bot.onchain.fetcher_base import BaseFetcher
from bot.onchain.models import DefiData


class DeFiLlamaFetcher(BaseFetcher):
    """Fetches DeFi TVL and stablecoin data from DeFiLlama.

    Rate limit: essentially unlimited.
    """

    def __init__(self, cache_ttl: float = 600.0):
        super().__init__(
            base_url="https://api.llama.fi",
            cache_ttl=cache_ttl,
            min_request_interval=1.0,
        )

    async def fetch(self) -> DefiData | None:
        """Fetch aggregated DeFi TVL and stablecoin supply."""
        tvl_data = await self._fetch_tvl()
        stable_data = await self._fetch_stablecoins()

        if tvl_data is None and stable_data is None:
            return None

        result = DefiData()

        if tvl_data is not None:
            result.total_tvl = tvl_data.get("total_tvl", 0.0)
            result.tvl_change_24h_pct = tvl_data.get("change_24h", 0.0)

        if stable_data is not None:
            result.stablecoin_total_supply = stable_data.get("total_supply", 0.0)
            result.stablecoin_supply_change_24h_pct = stable_data.get(
                "change_24h", 0.0
            )

        return result

    async def _fetch_tvl(self) -> dict | None:
        """Fetch total TVL across all protocols."""
        data = await self._get("/v2/historicalChainTvl")
        if data is None or not isinstance(data, list) or len(data) < 2:
            return None

        # Last two data points for 24h change
        current = data[-1]
        previous = data[-2] if len(data) >= 2 else data[-1]

        current_tvl = current.get("tvl", 0.0)
        previous_tvl = previous.get("tvl", 0.0)

        change_24h = 0.0
        if previous_tvl > 0:
            change_24h = ((current_tvl - previous_tvl) / previous_tvl) * 100.0

        return {"total_tvl": current_tvl, "change_24h": change_24h}

    async def _fetch_stablecoins(self) -> dict | None:
        """Fetch stablecoin total supply from stablecoins.llama.fi."""
        data = await self._get_absolute(
            "https://stablecoins.llama.fi/stablecoins",
            params={"includePrices": "false"},
        )
        if data is None or "peggedAssets" not in data:
            return None

        total_supply = 0.0
        for asset in data["peggedAssets"]:
            circ = asset.get("circulating", {})
            total_supply += circ.get("peggedUSD", 0.0)

        # For change, use stablecoin chart data
        chart_data = await self._get_absolute(
            "https://stablecoins.llama.fi/stablecoincharts/all",
            params={"stablecoin": "1"},
        )
        change_24h = 0.0
        if chart_data and isinstance(chart_data, list) and len(chart_data) >= 2:
            current_entry = chart_data[-1]
            prev_entry = chart_data[-2]
            curr_val = sum(
                v for v in current_entry.get("totalCirculating", {}).values()
                if isinstance(v, (int, float))
            )
            prev_val = sum(
                v for v in prev_entry.get("totalCirculating", {}).values()
                if isinstance(v, (int, float))
            )
            if prev_val > 0:
                change_24h = ((curr_val - prev_val) / prev_val) * 100.0

        return {"total_supply": total_supply, "change_24h": change_24h}
