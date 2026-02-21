"""Shared indicator utilities used across strategies and risk management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bot.models import OHLCV


def calculate_atr(candles: list[OHLCV], period: int = 14) -> float | None:
    """Calculate the Average True Range (ATR) for a list of candles.

    ATR measures volatility as the average of true ranges over `period` candles.
    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))

    Args:
        candles: List of OHLCV candles (must have at least period + 1 candles).
        period: Number of periods for ATR calculation (default 14).

    Returns:
        The ATR value as a float, or None if insufficient data.
    """
    if len(candles) < period + 1:
        return None

    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    closes = np.array([c.close for c in candles])

    # True Range: max(high - low, |high - prev_close|, |low - prev_close|)
    prev_closes = closes[:-1]
    current_highs = highs[1:]
    current_lows = lows[1:]

    tr1 = current_highs - current_lows
    tr2 = np.abs(current_highs - prev_closes)
    tr3 = np.abs(current_lows - prev_closes)

    true_ranges = np.maximum(tr1, np.maximum(tr2, tr3))

    if len(true_ranges) < period:
        return None

    # Use simple moving average of the last `period` true ranges
    atr = float(np.mean(true_ranges[-period:]))
    return atr


def calculate_atr_series(candles: list[OHLCV], period: int = 14) -> list[float]:
    """Calculate a rolling ATR series for a list of candles.

    Returns a list of ATR values, one for each candle starting from index `period`.
    Earlier indices have insufficient data and are not included.

    Args:
        candles: List of OHLCV candles.
        period: Number of periods for ATR calculation (default 14).

    Returns:
        List of ATR values. Length = len(candles) - period if sufficient data.
    """
    if len(candles) < period + 1:
        return []

    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    closes = np.array([c.close for c in candles])

    prev_closes = closes[:-1]
    current_highs = highs[1:]
    current_lows = lows[1:]

    tr1 = current_highs - current_lows
    tr2 = np.abs(current_highs - prev_closes)
    tr3 = np.abs(current_lows - prev_closes)

    true_ranges = np.maximum(tr1, np.maximum(tr2, tr3))

    # Rolling mean of true ranges
    atr_values = []
    for i in range(period - 1, len(true_ranges)):
        atr = float(np.mean(true_ranges[i - period + 1 : i + 1]))
        atr_values.append(atr)

    return atr_values
