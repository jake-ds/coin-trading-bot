"""Tests for V2-009: Signal ensemble voting system."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.config import Settings, TradingMode
from bot.main import TradingBot
from bot.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    SignalAction,
    TradingSignal,
)
from bot.strategies.ensemble import SignalEnsemble


def make_signal(
    strategy_name: str = "test",
    symbol: str = "BTC/USDT",
    action: SignalAction = SignalAction.HOLD,
    confidence: float = 0.5,
) -> TradingSignal:
    return TradingSignal(
        strategy_name=strategy_name,
        symbol=symbol,
        action=action,
        confidence=confidence,
    )


def make_mock_strategy(
    name: str,
    action: SignalAction = SignalAction.HOLD,
    confidence: float = 0.5,
    required_history: int = 1,
):
    """Create a mock strategy that returns a specific signal."""
    strategy = MagicMock()
    strategy.name = name
    strategy.required_history_length = required_history
    signal = make_signal(
        strategy_name=name, action=action, confidence=confidence
    )
    strategy.analyze = AsyncMock(return_value=signal)
    return strategy


def make_settings(**kwargs):
    """Create test settings with safe defaults."""
    defaults = {
        "trading_mode": TradingMode.PAPER,
        "database_url": "sqlite+aiosqlite:///:memory:",
        "binance_api_key": "",
        "upbit_api_key": "",
        "symbols": ["BTC/USDT"],
        "loop_interval_seconds": 1,
        "signal_min_agreement": 2,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


class TestSignalEnsembleVoting:
    """Test the vote() method with various signal combinations."""

    def test_two_buy_one_hold_returns_buy(self):
        """2 BUY + 1 HOLD = BUY when min_agreement=2."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.7),
            make_signal(strategy_name="s2", action=SignalAction.BUY, confidence=0.9),
            make_signal(strategy_name="s3", action=SignalAction.HOLD, confidence=0.0),
        ]
        result = ensemble.vote(signals, "BTC/USDT")
        assert result.action == SignalAction.BUY
        assert result.strategy_name == "ensemble"
        assert result.metadata["ensemble_agreement"] == 2
        assert result.metadata["total_signals"] == 3

    def test_one_buy_one_sell_returns_hold_conflict(self):
        """1 BUY + 1 SELL = HOLD (conflict)."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.8),
            make_signal(strategy_name="s2", action=SignalAction.SELL, confidence=0.6),
        ]
        result = ensemble.vote(signals, "BTC/USDT")
        assert result.action == SignalAction.HOLD
        assert result.metadata["reason"] == "conflict"

    def test_all_hold_returns_hold(self):
        """All HOLD = HOLD."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.HOLD),
            make_signal(strategy_name="s2", action=SignalAction.HOLD),
            make_signal(strategy_name="s3", action=SignalAction.HOLD),
        ]
        result = ensemble.vote(signals, "BTC/USDT")
        assert result.action == SignalAction.HOLD
        assert result.metadata["reason"] == "insufficient_agreement"

    def test_no_signals_returns_hold(self):
        """Empty signals list returns HOLD."""
        ensemble = SignalEnsemble(min_agreement=2)
        result = ensemble.vote([], "BTC/USDT")
        assert result.action == SignalAction.HOLD
        assert result.metadata["reason"] == "no_signals"

    def test_one_buy_insufficient_agreement(self):
        """1 BUY with min_agreement=2 returns HOLD."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.9),
        ]
        result = ensemble.vote(signals, "BTC/USDT")
        assert result.action == SignalAction.HOLD
        assert result.metadata["reason"] == "insufficient_agreement"

    def test_two_sell_returns_sell(self):
        """2 SELL signals with min_agreement=2 returns SELL."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.SELL, confidence=0.7),
            make_signal(strategy_name="s2", action=SignalAction.SELL, confidence=0.8),
        ]
        result = ensemble.vote(signals, "BTC/USDT")
        assert result.action == SignalAction.SELL
        assert result.metadata["ensemble_agreement"] == 2

    def test_three_buy_returns_buy_with_all_strategies(self):
        """3 BUY with min_agreement=2 includes all 3 in agreement count."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.6),
            make_signal(strategy_name="s2", action=SignalAction.BUY, confidence=0.7),
            make_signal(strategy_name="s3", action=SignalAction.BUY, confidence=0.8),
        ]
        result = ensemble.vote(signals, "BTC/USDT")
        assert result.action == SignalAction.BUY
        assert result.metadata["ensemble_agreement"] == 3
        assert sorted(result.metadata["agreeing_strategies"]) == ["s1", "s2", "s3"]

    def test_conflict_with_multiple_buys_and_sells(self):
        """2 BUY + 1 SELL = HOLD (conflict, despite agreement threshold met)."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.8),
            make_signal(strategy_name="s2", action=SignalAction.BUY, confidence=0.7),
            make_signal(strategy_name="s3", action=SignalAction.SELL, confidence=0.6),
        ]
        result = ensemble.vote(signals, "BTC/USDT")
        assert result.action == SignalAction.HOLD
        assert result.metadata["reason"] == "conflict"

    def test_min_agreement_one_single_buy(self):
        """With min_agreement=1, a single BUY is enough."""
        ensemble = SignalEnsemble(min_agreement=1)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.9),
        ]
        result = ensemble.vote(signals, "BTC/USDT")
        assert result.action == SignalAction.BUY


class TestEnsembleWeightedConfidence:
    """Test weighted confidence calculation."""

    def test_equal_weights_average_confidence(self):
        """With equal weights (default), confidence is simple average."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.6),
            make_signal(strategy_name="s2", action=SignalAction.BUY, confidence=0.8),
        ]
        result = ensemble.vote(signals, "BTC/USDT")
        assert result.confidence == pytest.approx(0.7)

    def test_custom_weights_affect_confidence(self):
        """Strategy weights change the resulting confidence."""
        ensemble = SignalEnsemble(
            min_agreement=2,
            strategy_weights={"s1": 2.0, "s2": 1.0},
        )
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.6),
            make_signal(strategy_name="s2", action=SignalAction.BUY, confidence=0.9),
        ]
        result = ensemble.vote(signals, "BTC/USDT")
        # weighted: (0.6*2 + 0.9*1) / (2+1) = (1.2 + 0.9) / 3 = 0.7
        assert result.confidence == pytest.approx(0.7)

    def test_missing_weight_defaults_to_one(self):
        """Strategies without explicit weights default to 1.0."""
        ensemble = SignalEnsemble(
            min_agreement=2,
            strategy_weights={"s1": 3.0},
        )
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.5),
            make_signal(strategy_name="s2", action=SignalAction.BUY, confidence=0.5),
        ]
        result = ensemble.vote(signals, "BTC/USDT")
        # Both have confidence 0.5, so weighted average is 0.5 regardless of weights
        assert result.confidence == pytest.approx(0.5)

    def test_different_confidences_weighted(self):
        """Heavier weight strategy influences confidence more."""
        ensemble = SignalEnsemble(
            min_agreement=2,
            strategy_weights={"s1": 3.0, "s2": 1.0},
        )
        signals = [
            make_signal(strategy_name="s1", action=SignalAction.BUY, confidence=0.9),
            make_signal(strategy_name="s2", action=SignalAction.BUY, confidence=0.1),
        ]
        result = ensemble.vote(signals, "BTC/USDT")
        # (0.9*3 + 0.1*1) / (3+1) = (2.7 + 0.1) / 4 = 0.7
        assert result.confidence == pytest.approx(0.7)


class TestEnsembleCollectSignals:
    """Test the collect_signals() method."""

    @pytest.mark.asyncio
    async def test_collect_signals_from_multiple_strategies(self):
        """collect_signals should run all strategies and return their signals."""
        ensemble = SignalEnsemble(min_agreement=2)
        strategies = [
            make_mock_strategy("s1", SignalAction.BUY, 0.8),
            make_mock_strategy("s2", SignalAction.BUY, 0.7),
            make_mock_strategy("s3", SignalAction.HOLD, 0.0),
        ]

        candles = [MagicMock()] * 10
        signals = await ensemble.collect_signals("BTC/USDT", strategies, candles)

        assert len(signals) == 3
        assert signals[0].action == SignalAction.BUY
        assert signals[1].action == SignalAction.BUY
        assert signals[2].action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_collect_skips_insufficient_history(self):
        """Strategies needing more candles than available are skipped."""
        ensemble = SignalEnsemble(min_agreement=2)
        strategies = [
            make_mock_strategy("s1", SignalAction.BUY, 0.8, required_history=5),
            make_mock_strategy("s2", SignalAction.BUY, 0.7, required_history=100),
        ]

        candles = [MagicMock()] * 10  # only 10 candles
        signals = await ensemble.collect_signals("BTC/USDT", strategies, candles)

        assert len(signals) == 1  # only s1 has enough
        assert signals[0].strategy_name == "s1"

    @pytest.mark.asyncio
    async def test_collect_handles_strategy_error(self):
        """If a strategy raises an exception, it's logged and skipped."""
        ensemble = SignalEnsemble(min_agreement=2)

        good_strategy = make_mock_strategy("good", SignalAction.BUY, 0.8)
        bad_strategy = MagicMock()
        bad_strategy.name = "bad"
        bad_strategy.required_history_length = 1
        bad_strategy.analyze = AsyncMock(side_effect=RuntimeError("strategy error"))

        strategies = [good_strategy, bad_strategy]
        candles = [MagicMock()] * 10

        signals = await ensemble.collect_signals("BTC/USDT", strategies, candles)

        assert len(signals) == 1
        assert signals[0].strategy_name == "good"

    @pytest.mark.asyncio
    async def test_collect_empty_strategies(self):
        """No strategies should produce no signals."""
        ensemble = SignalEnsemble(min_agreement=2)
        signals = await ensemble.collect_signals("BTC/USDT", [], [MagicMock()] * 10)
        assert signals == []


class TestEnsembleProperties:
    """Test ensemble configuration properties."""

    def test_min_agreement_property(self):
        ensemble = SignalEnsemble(min_agreement=3)
        assert ensemble.min_agreement == 3

    def test_strategy_weights_property(self):
        weights = {"s1": 2.0, "s2": 0.5}
        ensemble = SignalEnsemble(strategy_weights=weights)
        assert ensemble.strategy_weights == weights

    def test_default_min_agreement(self):
        ensemble = SignalEnsemble()
        assert ensemble.min_agreement == 2

    def test_default_strategy_weights_empty(self):
        ensemble = SignalEnsemble()
        assert ensemble.strategy_weights == {}


class TestEnsembleConfigIntegration:
    """Test ensemble configuration via Settings."""

    def test_config_default_signal_min_agreement(self):
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
        )
        assert settings.signal_min_agreement == 2

    def test_config_custom_signal_min_agreement(self):
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            signal_min_agreement=3,
        )
        assert settings.signal_min_agreement == 3

    def test_config_default_strategy_weights(self):
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
        )
        assert settings.strategy_weights == {}

    def test_config_custom_strategy_weights(self):
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            strategy_weights={"rsi": 2.0, "macd": 1.5},
        )
        assert settings.strategy_weights == {"rsi": 2.0, "macd": 1.5}


class TestEnsembleMainIntegration:
    """Test ensemble wiring in TradingBot._trading_cycle()."""

    @pytest.mark.asyncio
    async def test_ensemble_created_on_initialize(self):
        """SignalEnsemble should be created during bot initialization."""
        settings = make_settings(signal_min_agreement=3)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        assert bot._signal_ensemble is not None
        assert isinstance(bot._signal_ensemble, SignalEnsemble)
        assert bot._signal_ensemble.min_agreement == 3

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_ensemble_uses_strategy_weights_from_config(self):
        """Ensemble should use strategy weights from config."""
        weights = {"rsi": 2.0, "macd": 1.0}
        settings = make_settings(strategy_weights=weights)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        assert bot._signal_ensemble.strategy_weights == weights

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_ensemble_prevents_triple_position(self):
        """3 strategies BUYing should produce 1 trade, not 3 (the original bug)."""
        settings = make_settings(signal_min_agreement=2)
        bot = TradingBot(settings=settings)
        await bot.initialize()
        bot._risk_manager.update_portfolio_value(10000.0)

        mock_candle = MagicMock()
        mock_candle.close = 50000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle] * 200)

        # 3 strategies all returning BUY
        strategies = [
            make_mock_strategy("s1", SignalAction.BUY, 0.7),
            make_mock_strategy("s2", SignalAction.BUY, 0.8),
            make_mock_strategy("s3", SignalAction.BUY, 0.9),
        ]

        mock_order = Order(
            id="buy-001",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            price=0,
            quantity=0.02,
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            filled_price=50000.0,
            filled_quantity=0.02,
        )
        mock_engine = AsyncMock()
        mock_engine.execute_signal = AsyncMock(return_value=mock_order)
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = strategies
            await bot._trading_cycle()

        # Only 1 trade should be executed (ensemble combines into single signal)
        assert mock_engine.execute_signal.call_count == 1

        # The signal should be from "ensemble" strategy
        call_args = mock_engine.execute_signal.call_args
        signal = call_args[0][0]
        assert signal.strategy_name == "ensemble"
        assert signal.action == SignalAction.BUY

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_ensemble_holds_on_conflict_in_cycle(self):
        """Mixed BUY/SELL signals should produce no trade."""
        settings = make_settings(signal_min_agreement=2)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        mock_candle = MagicMock()
        mock_candle.close = 50000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle] * 200)

        strategies = [
            make_mock_strategy("s1", SignalAction.BUY, 0.8),
            make_mock_strategy("s2", SignalAction.SELL, 0.7),
        ]

        mock_engine = AsyncMock()
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = strategies
            await bot._trading_cycle()

        # No trade should be executed
        mock_engine.execute_signal.assert_not_called()

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_ensemble_holds_on_insufficient_agreement_in_cycle(self):
        """Single BUY with min_agreement=2 should produce no trade."""
        settings = make_settings(signal_min_agreement=2)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        mock_candle = MagicMock()
        mock_candle.close = 50000.0
        bot._store.get_candles = AsyncMock(return_value=[mock_candle] * 200)

        strategies = [
            make_mock_strategy("s1", SignalAction.BUY, 0.9),
            make_mock_strategy("s2", SignalAction.HOLD, 0.0),
        ]

        mock_engine = AsyncMock()
        bot._execution_engines = {"binance": mock_engine}

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = strategies
            await bot._trading_cycle()

        # No trade — only 1 BUY, need 2
        mock_engine.execute_signal.assert_not_called()

        await bot.shutdown()
