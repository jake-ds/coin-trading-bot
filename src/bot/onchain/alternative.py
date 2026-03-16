"""Alternative.me Fear & Greed Index fetcher."""

from __future__ import annotations

from bot.onchain.fetcher_base import BaseFetcher
from bot.onchain.models import SentimentData


class FearGreedFetcher(BaseFetcher):
    """Fetches Fear & Greed Index from Alternative.me.

    Rate limit: essentially unlimited.
    """

    def __init__(self, cache_ttl: float = 3600.0):
        super().__init__(
            base_url="https://api.alternative.me",
            cache_ttl=cache_ttl,
            min_request_interval=1.0,
        )

    async def fetch(self) -> SentimentData | None:
        """Fetch current Fear & Greed Index."""
        data = await self._get("/fng/", params={"limit": "1"})
        if data is None or "data" not in data:
            return None

        entries = data["data"]
        if not entries:
            return None

        entry = entries[0]
        value = int(entry.get("value", 50))
        classification = entry.get("value_classification", "Neutral")

        return SentimentData(
            value=value,
            classification=classification,
            timestamp=entry.get("timestamp", ""),
        )
