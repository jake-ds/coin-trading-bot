"""Integration tests for V2-011: Market regime detection wiring in main.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.config import Settings
from bot.main import TradingBot
from bot.strategies.regime import MarketRegime, MarketRegimeDetector


def make_settings(**overrides):
    defaults = {
        "trading_mode": "paper",
        "binance_api_key": "",
        "upbit_api_key": "",
        "database_url": "sqlite+aiosqlite:///:memory:",
        "symbols": ["BTC/USDT"],
        "loop_interval_seconds": 1,
        "signal_min_agreement": 1,
        "config_file": "/dev/null",
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestRegimeDetectorWiring:
    """Test that MarketRegimeDetector is created and used in TradingBot."""

    def test_regime_detector_initialized(self):
        bot = TradingBot(settings=make_settings())
        assert bot._regime_detector is None  # Before initialize
        # After initialize, it will be set

    @pytest.mark.asyncio
    async def test_regime_detector_created_on_initialize(self):
        bot = TradingBot(settings=make_settings())
        with patch.object(bot, "_init_exchanges"), \
             patch.object(bot, "_init_telegram"), \
             patch.object(bot, "_start_dashboard"), \
             patch.object(bot, "_load_strategies"):
            await bot.initialize()
            assert bot._regime_detector is not None
            assert isinstance(bot._regime_detector, MarketRegimeDetector)

    @pytest.mark.asyncio
    async def test_regime_detection_called_in_cycle(self):
        """Regime detection is called before strategy analysis."""
        bot = TradingBot(settings=make_settings())

        # Setup mocks
        mock_store = AsyncMock()
        mock_candle = MagicMock()
        mock_candle.close = 100.0
        mock_store.get_candles = AsyncMock(return_value=[mock_candle] * 30)

        bot._store = mock_store
        bot._collector = AsyncMock()
        bot._risk_manager = MagicMock()
        bot._risk_manager.check_and_reset_daily = MagicMock()
        bot._risk_manager._current_portfolio_value = 10000.0
        bot._position_manager = None

        mock_regime_detector = MagicMock()
        mock_regime_detector.required_history_length = 25
        mock_regime_detector.detect = MagicMock(
            return_value=MarketRegime.TRENDING_UP
        )
        bot._regime_detector = mock_regime_detector

        mock_ensemble = MagicMock()
        mock_ensemble.collect_signals = AsyncMock(return_value=[])
        mock_ensemble.vote = MagicMock(
            return_value=MagicMock(action=MagicMock(HOLD="HOLD"))
        )
        mock_vote_signal = MagicMock()
        mock_vote_signal.action = MagicMock()
        mock_vote_signal.action.HOLD = mock_vote_signal.action
        mock_ensemble.vote.return_value = mock_vote_signal
        bot._signal_ensemble = mock_ensemble

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_strategy = MagicMock()
            mock_strategy.adapt_to_regime = MagicMock()
            mock_registry.get_active.return_value = [mock_strategy]

            await bot._trading_cycle()

            # Verify regime was detected
            mock_regime_detector.detect.assert_called_once()
            # Verify adapt_to_regime was called on the strategy
            mock_strategy.adapt_to_regime.assert_called_once_with(
                MarketRegime.TRENDING_UP
            )

    @pytest.mark.asyncio
    async def test_regime_detection_error_gracefully_handled(self):
        """Regime detection errors don't crash the trading cycle."""
        bot = TradingBot(settings=make_settings())

        mock_store = AsyncMock()
        mock_candle = MagicMock()
        mock_candle.close = 100.0
        mock_store.get_candles = AsyncMock(return_value=[mock_candle] * 30)

        bot._store = mock_store
        bot._collector = AsyncMock()
        bot._risk_manager = MagicMock()
        bot._risk_manager.check_and_reset_daily = MagicMock()
        bot._risk_manager._current_portfolio_value = 10000.0
        bot._position_manager = None

        mock_regime_detector = MagicMock()
        mock_regime_detector.required_history_length = 25
        mock_regime_detector.detect = MagicMock(
            side_effect=ValueError("detection error")
        )
        bot._regime_detector = mock_regime_detector

        mock_ensemble = MagicMock()
        mock_ensemble.collect_signals = AsyncMock(return_value=[])
        mock_vote_signal = MagicMock()
        mock_vote_signal.action = MagicMock()
        mock_vote_signal.action.HOLD = mock_vote_signal.action
        mock_ensemble.vote.return_value = mock_vote_signal
        bot._signal_ensemble = mock_ensemble

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = []

            # Should not raise despite regime detection error
            await bot._trading_cycle()

    @pytest.mark.asyncio
    async def test_regime_skipped_with_insufficient_candles(self):
        """Regime detection is skipped when not enough candles."""
        bot = TradingBot(settings=make_settings())

        mock_store = AsyncMock()
        mock_candle = MagicMock()
        mock_candle.close = 100.0
        # Only 5 candles — not enough for regime detection (needs 25)
        mock_store.get_candles = AsyncMock(return_value=[mock_candle] * 5)

        bot._store = mock_store
        bot._collector = AsyncMock()
        bot._risk_manager = MagicMock()
        bot._risk_manager.check_and_reset_daily = MagicMock()
        bot._risk_manager._current_portfolio_value = 10000.0
        bot._position_manager = None

        mock_regime_detector = MagicMock()
        mock_regime_detector.required_history_length = 25
        bot._regime_detector = mock_regime_detector

        mock_ensemble = MagicMock()
        mock_ensemble.collect_signals = AsyncMock(return_value=[])
        mock_vote_signal = MagicMock()
        mock_vote_signal.action = MagicMock()
        mock_vote_signal.action.HOLD = mock_vote_signal.action
        mock_ensemble.vote.return_value = mock_vote_signal
        bot._signal_ensemble = mock_ensemble

        with patch("bot.main.strategy_registry") as mock_registry:
            mock_registry.get_active.return_value = []

            await bot._trading_cycle()

            # detect should NOT be called — insufficient candles
            mock_regime_detector.detect.assert_not_called()
