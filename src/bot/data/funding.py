"""Funding rate monitor for perpetual futures markets."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from bot.data.store import DataStore
    from bot.exchanges.base import ExchangeAdapter

logger = structlog.get_logger(__name__)


class FundingRateMonitor:
    """Monitor funding rates from perpetual futures exchanges.

    Fetches funding rates via ccxt's `fetch_funding_rate` and stores them
    in the database. Provides access to current and historical funding
    rate data for strategy use.

    Funding rates are typically settled every 8 hours on most exchanges.
    Positive rates mean longs pay shorts; negative means shorts pay longs.
    """

    def __init__(
        self,
        exchange: ExchangeAdapter,
        store: DataStore | None = None,
        funding_rate_history: int = 50,
    ):
        self._exchange = exchange
        self._store = store
        self._funding_rate_history = funding_rate_history
        self._latest_rates: dict[str, dict[str, Any]] = {}
        self._rate_history: dict[str, list[dict[str, Any]]] = {}

    @property
    def latest_rates(self) -> dict[str, dict[str, Any]]:
        """Get the latest funding rate data for all monitored symbols."""
        return dict(self._latest_rates)

    def get_latest_rate(self, symbol: str) -> dict[str, Any] | None:
        """Get the latest funding rate data for a specific symbol."""
        return self._latest_rates.get(symbol)

    def get_rate_history(self, symbol: str) -> list[dict[str, Any]]:
        """Get historical funding rates for a symbol."""
        return list(self._rate_history.get(symbol, []))

    def get_average_rate(self, symbol: str, periods: int = 10) -> float | None:
        """Get the average funding rate over the last N periods.

        Returns None if no history available.
        """
        history = self._rate_history.get(symbol, [])
        if not history:
            return None
        recent = history[-periods:]
        rates = [h["funding_rate"] for h in recent if h.get("funding_rate") is not None]
        if not rates:
            return None
        return sum(rates) / len(rates)

    async def fetch_funding_rate(self, symbol: str) -> dict[str, Any] | None:
        """Fetch current funding rate for a symbol from the exchange.

        Uses ccxt's fetch_funding_rate method. Returns None if the exchange
        doesn't support funding rates or the symbol is not a perpetual contract.

        Returns dict with: funding_rate, funding_timestamp, mark_price,
        spot_price (if available), spread_pct.
        """
        try:
            ccxt_exchange = self._get_ccxt_exchange()
            if ccxt_exchange is None:
                logger.warning("funding_rate_no_ccxt_exchange")
                return None

            if not hasattr(ccxt_exchange, "fetch_funding_rate"):
                logger.debug(
                    "funding_rate_not_supported",
                    exchange=self._exchange.name,
                )
                return None

            raw = await ccxt_exchange.fetch_funding_rate(symbol)
            if not raw:
                return None

            # Convert to regular dict to avoid AsyncMock issues with .get()
            if not isinstance(raw, dict):
                return None
            data = dict(raw)

            funding_rate = data.get("fundingRate")
            if funding_rate is None:
                return None

            funding_ts = data.get("fundingDatetime") or data.get("datetime")
            if isinstance(funding_ts, str):
                funding_timestamp = datetime.fromisoformat(
                    funding_ts.replace("Z", "+00:00")
                )
            elif isinstance(funding_ts, (int, float)):
                funding_timestamp = datetime.utcfromtimestamp(funding_ts / 1000)
            else:
                funding_timestamp = datetime.utcnow()

            mark_price = float(data.get("markPrice") or 0)
            index_price = float(data.get("indexPrice") or 0)

            # Calculate spread between mark (perpetual) and index (spot-equivalent)
            spread_pct = 0.0
            if mark_price > 0 and index_price > 0:
                spread_pct = (mark_price - index_price) / index_price * 100

            result: dict[str, Any] = {
                "symbol": symbol,
                "funding_rate": float(funding_rate),
                "funding_timestamp": funding_timestamp,
                "mark_price": mark_price,
                "spot_price": index_price,
                "spread_pct": spread_pct,
                "next_funding_time": data.get("fundingTimestamp"),
            }

            # Store in memory
            self._latest_rates[symbol] = result
            if symbol not in self._rate_history:
                self._rate_history[symbol] = []
            self._rate_history[symbol].append(result)
            # Trim history to configured max
            if len(self._rate_history[symbol]) > self._funding_rate_history:
                self._rate_history[symbol] = self._rate_history[symbol][
                    -self._funding_rate_history :
                ]

            # Persist to database if store available
            if self._store is not None:
                await self._store.save_funding_rate(
                    symbol=symbol,
                    funding_rate=float(funding_rate),
                    funding_timestamp=funding_timestamp,
                    mark_price=mark_price,
                    spot_price=index_price,
                    spread_pct=spread_pct,
                )

            return result

        except Exception as e:
            logger.warning(
                "funding_rate_fetch_error",
                symbol=symbol,
                error=str(e),
            )
            return None

    async def fetch_funding_rate_history(
        self,
        symbol: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch historical funding rates from the exchange.

        Returns list of funding rate records, oldest first.
        """
        try:
            ccxt_exchange = self._get_ccxt_exchange()
            if ccxt_exchange is None:
                return []

            if not hasattr(ccxt_exchange, "fetch_funding_rate_history"):
                logger.debug(
                    "funding_rate_history_not_supported",
                    exchange=self._exchange.name,
                )
                return []

            raw_history = await ccxt_exchange.fetch_funding_rate_history(
                symbol, limit=limit
            )
            if not raw_history:
                return []

            results = []
            for raw in raw_history:
                if not isinstance(raw, dict):
                    continue
                data = dict(raw)
                fr = data.get("fundingRate")
                if fr is None:
                    continue

                ts = data.get("datetime") or data.get("timestamp")
                if isinstance(ts, str):
                    timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                elif isinstance(ts, (int, float)):
                    timestamp = datetime.utcfromtimestamp(ts / 1000)
                else:
                    timestamp = datetime.utcnow()

                results.append({
                    "symbol": symbol,
                    "funding_rate": float(fr),
                    "funding_timestamp": timestamp,
                    "mark_price": float(data.get("markPrice") or 0),
                    "spot_price": float(data.get("indexPrice") or 0),
                })

            # Update memory history
            if results:
                self._rate_history[symbol] = results[-self._funding_rate_history :]

            return results

        except Exception as e:
            logger.warning(
                "funding_rate_history_fetch_error",
                symbol=symbol,
                error=str(e),
            )
            return []

    def update_rate(
        self,
        symbol: str,
        funding_rate: float,
        funding_timestamp: datetime | None = None,
        mark_price: float = 0.0,
        spot_price: float = 0.0,
    ) -> None:
        """Manually update funding rate data (for testing or manual feeds).

        Args:
            symbol: Trading pair symbol.
            funding_rate: The funding rate value (e.g., 0.0005 for 0.05%).
            funding_timestamp: When the funding rate applies.
            mark_price: Perpetual contract mark price.
            spot_price: Spot market price.
        """
        if funding_timestamp is None:
            funding_timestamp = datetime.utcnow()

        spread_pct = 0.0
        if mark_price > 0 and spot_price > 0:
            spread_pct = (mark_price - spot_price) / spot_price * 100

        result: dict[str, Any] = {
            "symbol": symbol,
            "funding_rate": funding_rate,
            "funding_timestamp": funding_timestamp,
            "mark_price": mark_price,
            "spot_price": spot_price,
            "spread_pct": spread_pct,
        }

        self._latest_rates[symbol] = result
        if symbol not in self._rate_history:
            self._rate_history[symbol] = []
        self._rate_history[symbol].append(result)
        if len(self._rate_history[symbol]) > self._funding_rate_history:
            self._rate_history[symbol] = self._rate_history[symbol][
                -self._funding_rate_history :
            ]

    def _get_ccxt_exchange(self) -> Any | None:
        """Get the underlying ccxt exchange instance.

        Traverses the adapter chain: ResilientExchange → ExchangeAdapter → ccxt.
        Stops when it finds an object with fetch_funding_rate method or
        when the chain ends.
        """
        exchange = self._exchange
        # Check if current level has what we need
        if hasattr(exchange, "fetch_funding_rate"):
            return exchange

        # Traverse adapter wrappers (max 5 levels)
        for _ in range(5):
            inner = getattr(exchange, "_exchange", None)
            if inner is None or inner is exchange:
                break
            exchange = inner
            if hasattr(exchange, "fetch_funding_rate"):
                return exchange

        return None
