"""Multi-symbol data alignment for pair trading and portfolio analysis.

Provides aligned close prices, spread computation, log returns,
and rolling correlation matrix calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from bot.data.store import DataStore
    from bot.models import OHLCV

logger = structlog.get_logger()


class PairDataProvider:
    """Provides aligned multi-symbol data for quant strategies."""

    @staticmethod
    def get_aligned_closes_from_candles(
        candles_by_symbol: dict[str, list["OHLCV"]],
    ) -> dict[str, np.ndarray]:
        """Align close prices across multiple symbols by timestamp.

        Args:
            candles_by_symbol: Dict mapping symbol -> list of OHLCV candles.

        Returns:
            Dict mapping symbol -> aligned close price array.
            Only timestamps present in ALL symbols are included.
        """
        if not candles_by_symbol:
            return {}

        # Build timestamp -> close for each symbol
        ts_prices: dict[str, dict[int, float]] = {}
        for symbol, candles in candles_by_symbol.items():
            ts_prices[symbol] = {}
            for c in candles:
                ts_key = int(c.timestamp.timestamp())
                ts_prices[symbol][ts_key] = c.close

        # Find common timestamps
        all_ts_sets = [set(v.keys()) for v in ts_prices.values()]
        if not all_ts_sets:
            return {}

        common_ts = sorted(set.intersection(*all_ts_sets))
        if not common_ts:
            return {}

        result = {}
        for symbol in candles_by_symbol:
            result[symbol] = np.array([ts_prices[symbol][ts] for ts in common_ts])

        return result

    @staticmethod
    async def get_aligned_closes(
        symbols: list[str],
        store: "DataStore",
        timeframe: str = "1h",
        limit: int = 500,
    ) -> dict[str, np.ndarray]:
        """Fetch and align close prices from data store.

        Args:
            symbols: List of trading symbols.
            store: DataStore instance.
            timeframe: Candle timeframe.
            limit: Max candles per symbol.

        Returns:
            Dict mapping symbol -> aligned close price array.
        """
        candles_by_symbol: dict[str, list[OHLCV]] = {}
        for symbol in symbols:
            candles = await store.get_candles(
                symbol=symbol, timeframe=timeframe, limit=limit
            )
            if candles:
                candles_by_symbol[symbol] = candles

        return PairDataProvider.get_aligned_closes_from_candles(candles_by_symbol)


def compute_spread(
    prices_a: np.ndarray, prices_b: np.ndarray, hedge_ratio: float | np.ndarray
) -> np.ndarray:
    """Compute the spread between two price series.

    spread = prices_a - hedge_ratio * prices_b

    Args:
        prices_a: First price series.
        prices_b: Second price series.
        hedge_ratio: Scalar or array of hedge ratios.

    Returns:
        Spread array.
    """
    return prices_a - hedge_ratio * prices_b


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from a price series.

    Args:
        prices: Price series.

    Returns:
        Log return array (length = len(prices) - 1).
    """
    prices = np.asarray(prices, dtype=float)
    # Avoid log(0)
    prices = np.maximum(prices, 1e-10)
    return np.diff(np.log(prices))


def compute_correlation_matrix(
    returns: dict[str, np.ndarray], window: int | None = None
) -> dict[str, dict[str, float]]:
    """Compute pairwise correlation matrix from return series.

    Args:
        returns: Dict mapping symbol -> return array.
        window: If provided, use only the last N observations.

    Returns:
        Nested dict: result[symbol_a][symbol_b] = correlation.
    """
    symbols = list(returns.keys())
    if len(symbols) < 2:
        return {s: {s: 1.0} for s in symbols}

    # Align lengths
    min_len = min(len(v) for v in returns.values())
    if window is not None:
        min_len = min(min_len, window)

    if min_len < 5:
        return {s: {s2: 0.0 for s2 in symbols} for s in symbols}

    # Build matrix
    data = np.column_stack([returns[s][-min_len:] for s in symbols])
    corr = np.corrcoef(data, rowvar=False)

    result: dict[str, dict[str, float]] = {}
    for i, s1 in enumerate(symbols):
        result[s1] = {}
        for j, s2 in enumerate(symbols):
            val = float(corr[i, j])
            result[s1][s2] = val if not np.isnan(val) else 0.0

    return result
