"""Base model and common enums for the trading bot."""

from enum import Enum

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Immutable base model."""

    model_config = ConfigDict(frozen=True)


class OrderSide(str, Enum):
    """Order side: buy or sell."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    """Order lifecycle status."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class SignalAction(str, Enum):
    """Trading signal action."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
