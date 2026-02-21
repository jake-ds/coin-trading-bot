"""Portfolio data model."""

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from bot.models.base import FrozenModel


class Position(FrozenModel):
    """Represents an open position in a single asset."""

    symbol: str
    quantity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    current_price: float = Field(gt=0)
    unrealized_pnl: float = 0.0

    @model_validator(mode="after")
    def calculate_pnl(self) -> "Position":
        expected = (self.current_price - self.entry_price) * self.quantity
        if self.unrealized_pnl == 0.0 and expected != 0.0:
            object.__setattr__(self, "unrealized_pnl", expected)
        return self


class Portfolio(FrozenModel):
    """Represents the current portfolio state."""

    balances: dict[str, float] = Field(default_factory=dict)
    positions: list[Position] = Field(default_factory=list)
    total_value: float = Field(ge=0, default=0.0)
    unrealized_pnl: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("balances")
    @classmethod
    def validate_balances(cls, v: dict[str, float]) -> dict[str, float]:
        for currency, amount in v.items():
            if amount < 0:
                raise ValueError(f"Balance for {currency} cannot be negative")
        return v
