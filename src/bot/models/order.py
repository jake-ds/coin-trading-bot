"""Order data model."""

from datetime import datetime

from pydantic import Field, field_validator

from bot.models.base import FrozenModel, OrderSide, OrderStatus, OrderType


class Order(FrozenModel):
    """Represents a trading order."""

    id: str
    exchange: str
    symbol: str
    side: OrderSide
    type: OrderType
    price: float = Field(ge=0)
    quantity: float = Field(gt=0)
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None
    filled_at: datetime | None = None
    filled_price: float | None = None
    filled_quantity: float | None = None
    fee: float = 0.0

    @field_validator("price")
    @classmethod
    def market_order_price(cls, v: float, info) -> float:
        if info.data.get("type") == OrderType.MARKET and v != 0:
            raise ValueError("Market orders should have price=0 (filled at market)")
        if info.data.get("type") == OrderType.LIMIT and v <= 0:
            raise ValueError("Limit orders must have a positive price")
        return v
