"""Etherscan API fetcher — whale ERC-20 transfers (optional)."""

from __future__ import annotations

from bot.onchain.fetcher_base import BaseFetcher
from bot.onchain.models import WhaleFlowData


class EtherscanFetcher(BaseFetcher):
    """Fetches large ERC-20 transfers from Etherscan.

    Rate limit: 5 req/sec (free tier).
    Optional — returns None if no API key.
    """

    def __init__(self, api_key: str = "", cache_ttl: float = 300.0):
        super().__init__(
            base_url="https://api.etherscan.io/api",
            cache_ttl=cache_ttl,
            min_request_interval=0.25,
        )
        self._api_key = api_key

    async def fetch(self) -> WhaleFlowData | None:
        """Fetch whale transfer data for ETH. Returns None if no API key."""
        if not self._api_key:
            return None
        return await self.fetch_whale_transfers()

    async def fetch_whale_transfers(self) -> WhaleFlowData | None:
        """Fetch recent large ETH transfers as a proxy for whale activity."""
        # Use internal transactions of top exchange wallets as proxy
        # This is a simplified approach — real implementation would track
        # known exchange wallet addresses
        data = await self._get(
            "",
            params={
                "module": "account",
                "action": "txlist",
                "address": "0x00000000219ab540356cBB839Cbe05303d7705Fa",  # ETH2 deposit contract
                "page": "1",
                "offset": "10",
                "sort": "desc",
                "apikey": self._api_key,
            },
        )

        if data is None or data.get("status") != "1":
            return None

        results = data.get("result", [])
        if not results:
            return None

        total_value = sum(
            int(tx.get("value", "0")) / 1e18 for tx in results
        )

        return WhaleFlowData(
            symbol="ETH",
            net_flow=-total_value,  # deposits to staking = outflow from exchange = bullish
            inflow=0.0,
            outflow=total_value,
        )
