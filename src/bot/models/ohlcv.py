"""OHLCV (candlestick) data model."""

from datetime import datetime

from pydantic import Field, field_validator

from bot.models.base import FrozenModel


class OHLCV(FrozenModel):
    """Represents a single OHLCV candlestick."""

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    symbol: str = ""
    timeframe: str = "1h"

    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v: float, info) -> float:
        if "low" in info.data and v < info.data["low"]:
            raise ValueError("high must be >= low")
        return v

    @field_validator("high")
    @classmethod
    def high_gte_open_close(cls, v: float, info) -> float:
        for field in ("open", "close"):
            if field in info.data and v < info.data[field]:
                raise ValueError(f"high must be >= {field}")
        return v

    @field_validator("low")
    @classmethod
    def low_lte_open_close(cls, v: float, info) -> float:
        for field in ("open", "close"):
            if field in info.data and v > info.data[field]:
                raise ValueError(f"low must be <= {field}")
        return v
