"""Tests for Portfolio and Position models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from bot.models import Portfolio, Position


class TestPosition:
    def test_valid_position(self):
        pos = Position(
            symbol="BTC/USDT",
            quantity=0.5,
            entry_price=50000.0,
            current_price=55000.0,
        )
        assert pos.symbol == "BTC/USDT"
        assert pos.quantity == 0.5
        assert pos.entry_price == 50000.0
        assert pos.current_price == 55000.0
        assert pos.unrealized_pnl == 2500.0

    def test_negative_pnl(self):
        pos = Position(
            symbol="ETH/USDT",
            quantity=10.0,
            entry_price=3000.0,
            current_price=2800.0,
        )
        assert pos.unrealized_pnl == -2000.0

    def test_zero_quantity_rejected(self):
        with pytest.raises(ValidationError):
            Position(
                symbol="BTC/USDT",
                quantity=0,
                entry_price=50000.0,
                current_price=50000.0,
            )

    def test_negative_price_rejected(self):
        with pytest.raises(ValidationError):
            Position(
                symbol="BTC/USDT",
                quantity=1.0,
                entry_price=-100.0,
                current_price=50000.0,
            )


class TestPortfolio:
    def test_valid_portfolio(self):
        portfolio = Portfolio(
            balances={"USDT": 10000.0, "BTC": 0.5},
            positions=[
                Position(
                    symbol="BTC/USDT",
                    quantity=0.5,
                    entry_price=50000.0,
                    current_price=55000.0,
                )
            ],
            total_value=37500.0,
        )
        assert portfolio.balances["USDT"] == 10000.0
        assert len(portfolio.positions) == 1
        assert portfolio.total_value == 37500.0

    def test_empty_portfolio(self):
        portfolio = Portfolio()
        assert portfolio.balances == {}
        assert portfolio.positions == []
        assert portfolio.total_value == 0.0
        assert portfolio.unrealized_pnl == 0.0
        assert isinstance(portfolio.timestamp, datetime)

    def test_negative_balance_rejected(self):
        with pytest.raises(ValidationError):
            Portfolio(balances={"USDT": -100.0})

    def test_negative_total_value_rejected(self):
        with pytest.raises(ValidationError):
            Portfolio(total_value=-1.0)

    def test_multiple_positions(self):
        portfolio = Portfolio(
            positions=[
                Position(
                    symbol="BTC/USDT",
                    quantity=1.0,
                    entry_price=50000.0,
                    current_price=55000.0,
                ),
                Position(
                    symbol="ETH/USDT",
                    quantity=10.0,
                    entry_price=3000.0,
                    current_price=3200.0,
                ),
            ],
            total_value=87000.0,
        )
        assert len(portfolio.positions) == 2
        assert portfolio.positions[0].unrealized_pnl == 5000.0
        assert portfolio.positions[1].unrealized_pnl == 2000.0
