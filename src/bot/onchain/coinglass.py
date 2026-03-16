"""CoinGlass API fetcher — open interest, funding rates, liquidations, exchange flow."""

from __future__ import annotations

from bot.onchain.fetcher_base import BaseFetcher
from bot.onchain.models import DerivativesData, WhaleFlowData

# CoinGlass uses base symbols (no /USDT suffix)
SYMBOL_MAP: dict[str, str] = {
    "BTC/USDT": "BTC",
    "ETH/USDT": "ETH",
    "SOL/USDT": "SOL",
    "XRP/USDT": "XRP",
    "DOGE/USDT": "DOGE",
}


class CoinGlassFetcher(BaseFetcher):
    """Fetches derivatives data from CoinGlass API.

    Requires API key (free tier available).
    Rate limit: varies by plan.
    """

    def __init__(self, api_key: str = "", cache_ttl: float = 300.0):
        super().__init__(
            base_url="https://open-api-v3.coinglass.com/api",
            cache_ttl=cache_ttl,
            min_request_interval=3.0,
        )
        self._api_key = api_key

    async def _get(self, path: str, params: dict | None = None) -> dict | None:
        """Override to add API key header."""
        if not self._api_key:
            return None

        cache_key = f"{path}:{params}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        await self._rate_limit()

        try:
            session = await self._get_session()
            url = f"{self._base_url}{path}"
            headers = {"coinglassSecret": self._api_key}
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if data.get("code") != "0" and data.get("success") is not True:
                    return None
                self._set_cached(cache_key, data)
                return data
        except Exception:
            return None

    async def fetch(self) -> dict[str, DerivativesData] | None:
        """Fetch derivatives data for all tracked symbols."""
        if not self._api_key:
            return None
        return await self.fetch_derivatives(list(SYMBOL_MAP.keys()))

    async def fetch_derivatives(
        self, symbols: list[str]
    ) -> dict[str, DerivativesData] | None:
        """Fetch OI, funding rate, liquidation data."""
        result: dict[str, DerivativesData] = {}

        for sym in symbols:
            base = SYMBOL_MAP.get(sym, sym.split("/")[0])
            deriv = DerivativesData(symbol=base)

            # Funding rate
            fr_data = await self._get(
                "/futures/funding-rate-oi-weight",
                params={"symbol": base},
            )
            if fr_data and "data" in fr_data:
                items = fr_data["data"]
                if isinstance(items, list) and items:
                    deriv.funding_rate = items[0].get("rate", 0.0) or 0.0
                elif isinstance(items, dict):
                    deriv.funding_rate = items.get("rate", 0.0) or 0.0

            # Open interest
            oi_data = await self._get(
                "/futures/openInterest/chart",
                params={"symbol": base, "interval": "1d", "limit": "2"},
            )
            if oi_data and "data" in oi_data:
                oi_list = oi_data["data"]
                if isinstance(oi_list, list) and len(oi_list) >= 2:
                    curr_oi = oi_list[-1].get("openInterest", 0.0)
                    prev_oi = oi_list[-2].get("openInterest", 0.0)
                    deriv.open_interest = curr_oi
                    if prev_oi > 0:
                        deriv.oi_change_24h_pct = (
                            (curr_oi - prev_oi) / prev_oi
                        ) * 100.0

            # Liquidations
            liq_data = await self._get(
                "/futures/liquidation/detail",
                params={"symbol": base, "interval": "1d"},
            )
            if liq_data and "data" in liq_data:
                liq = liq_data["data"]
                if isinstance(liq, dict):
                    deriv.liquidations_24h_long = liq.get("longLiquidationUsd", 0.0) or 0.0
                    deriv.liquidations_24h_short = liq.get("shortLiquidationUsd", 0.0) or 0.0

            result[sym] = deriv

        return result if result else None

    async def fetch_exchange_flow(
        self, symbols: list[str]
    ) -> dict[str, WhaleFlowData] | None:
        """Fetch exchange net flow data (inflow/outflow)."""
        result: dict[str, WhaleFlowData] = {}

        for sym in symbols:
            base = SYMBOL_MAP.get(sym, sym.split("/")[0])
            flow_data = await self._get(
                "/indicator/exchange/netflow",
                params={"symbol": base, "interval": "1d"},
            )
            if flow_data and "data" in flow_data:
                entries = flow_data["data"]
                if isinstance(entries, list) and entries:
                    latest = entries[-1]
                    net_flow = latest.get("netflow", 0.0) or 0.0
                    result[sym] = WhaleFlowData(
                        symbol=base,
                        net_flow=net_flow,
                        inflow=latest.get("inflow", 0.0) or 0.0,
                        outflow=latest.get("outflow", 0.0) or 0.0,
                    )

        return result if result else None
