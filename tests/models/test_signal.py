"""Tests for TradingSignal model."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from bot.models import SignalAction, TradingSignal


class TestTradingSignal:
    def test_valid_signal(self):
        signal = TradingSignal(
            strategy_name="ma_crossover",
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            confidence=0.85,
        )
        assert signal.strategy_name == "ma_crossover"
        assert signal.symbol == "BTC/USDT"
        assert signal.action == SignalAction.BUY
        assert signal.confidence == 0.85
        assert isinstance(signal.timestamp, datetime)

    def test_hold_signal(self):
        signal = TradingSignal(
            strategy_name="rsi",
            symbol="ETH/USDT",
            action=SignalAction.HOLD,
            confidence=0.5,
        )
        assert signal.action == SignalAction.HOLD

    def test_confidence_bounds(self):
        # Valid bounds
        TradingSignal(
            strategy_name="test", symbol="BTC/USDT", action=SignalAction.BUY, confidence=0.0
        )
        TradingSignal(
            strategy_name="test", symbol="BTC/USDT", action=SignalAction.BUY, confidence=1.0
        )

        # Out of bounds
        with pytest.raises(ValidationError):
            TradingSignal(
                strategy_name="test", symbol="BTC/USDT", action=SignalAction.BUY, confidence=-0.1
            )
        with pytest.raises(ValidationError):
            TradingSignal(
                strategy_name="test", symbol="BTC/USDT", action=SignalAction.BUY, confidence=1.1
            )

    def test_empty_strategy_name_rejected(self):
        with pytest.raises(ValidationError):
            TradingSignal(
                strategy_name="", symbol="BTC/USDT", action=SignalAction.BUY, confidence=0.5
            )

    def test_empty_symbol_rejected(self):
        with pytest.raises(ValidationError):
            TradingSignal(
                strategy_name="test", symbol="", action=SignalAction.BUY, confidence=0.5
            )

    def test_metadata(self):
        signal = TradingSignal(
            strategy_name="ma_crossover",
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            confidence=0.8,
            metadata={"short_ma": 50100.0, "long_ma": 49500.0},
        )
        assert signal.metadata["short_ma"] == 50100.0

    def test_default_metadata_empty(self):
        signal = TradingSignal(
            strategy_name="test", symbol="BTC/USDT", action=SignalAction.HOLD, confidence=0.5
        )
        assert signal.metadata == {}
