"""Tests for OHLCV model."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from bot.models import OHLCV


class TestOHLCV:
    def test_valid_ohlcv(self):
        candle = OHLCV(
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=1000.0,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        assert candle.open == 100.0
        assert candle.high == 110.0
        assert candle.low == 95.0
        assert candle.close == 105.0
        assert candle.volume == 1000.0
        assert candle.symbol == "BTC/USDT"

    def test_zero_volume_allowed(self):
        candle = OHLCV(
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=0.0,
        )
        assert candle.volume == 0.0

    def test_negative_price_rejected(self):
        with pytest.raises(ValidationError):
            OHLCV(
                timestamp=datetime(2024, 1, 1),
                open=-1.0,
                high=100.0,
                low=50.0,
                close=75.0,
                volume=100.0,
            )

    def test_high_less_than_low_rejected(self):
        with pytest.raises(ValidationError):
            OHLCV(
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=50.0,
                low=90.0,
                close=80.0,
                volume=100.0,
            )

    def test_high_less_than_open_rejected(self):
        with pytest.raises(ValidationError):
            OHLCV(
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=90.0,
                low=80.0,
                close=85.0,
                volume=100.0,
            )

    def test_low_greater_than_close_rejected(self):
        with pytest.raises(ValidationError):
            OHLCV(
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=110.0,
                low=105.0,
                close=95.0,
                volume=100.0,
            )

    def test_frozen(self):
        candle = OHLCV(
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=0.0,
        )
        with pytest.raises(ValidationError):
            candle.open = 200.0

    def test_default_symbol_and_timeframe(self):
        candle = OHLCV(
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=0.0,
        )
        assert candle.symbol == ""
        assert candle.timeframe == "1h"
