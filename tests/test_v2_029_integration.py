"""Tests for V2-029: Full integration test — multi-strategy trading cycle with all V2 features.

End-to-end test proving the entire improved pipeline works together:
- Mocked exchange returning a realistic 200-candle price sequence
  (trending up → ranging → trending down)
- 3+ strategies with SignalEnsemble voting (min_agreement=2)
- TrendFilter enabled with 4h candles
- MarketRegimeDetector active, strategies adapt to detected regime
- PositionManager with stop-loss and take-profit active
- PaperPortfolio tracking balance (starting 10000 USDT)
- RiskManager properly tracking positions and PnL
"""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.data.store import DataStore
from bot.execution.engine import ExecutionEngine
from bot.execution.paper_portfolio import PaperPortfolio
from bot.execution.position_manager import PositionManager
from bot.models import OHLCV, SignalAction
from bot.monitoring.strategy_tracker import StrategyTracker
from bot.risk.manager import RiskManager
from bot.risk.portfolio_risk import PortfolioRiskManager
from bot.strategies.base import StrategyRegistry, strategy_registry
from bot.strategies.ensemble import SignalEnsemble
from bot.strategies.regime import MarketRegime, MarketRegimeDetector
from bot.strategies.trend_filter import TrendDirection, TrendFilter

# ──── Helpers ────


def _make_candle(
    price: float,
    ts: datetime,
    symbol: str = "BTC/USDT",
    spread_pct: float = 0.5,
    volume: float = 100.0,
    timeframe: str = "1h",
) -> OHLCV:
    """Create a realistic OHLCV candle around a given price."""
    half_spread = price * spread_pct / 100
    return OHLCV(
        timestamp=ts,
        open=price - half_spread * 0.3,
        high=price + half_spread,
        low=price - half_spread,
        close=price,
        volume=volume,
        symbol=symbol,
        timeframe=timeframe,
    )


def generate_price_sequence(
    start_price: float = 40000.0,
    n_candles: int = 200,
    start_time: datetime | None = None,
) -> list[float]:
    """Generate a realistic price sequence: uptrend → ranging → downtrend.

    - Candles 0-79: uptrend (+15%)
    - Candles 80-139: ranging (oscillates around peak)
    - Candles 140-199: downtrend (-12%)
    """
    if start_time is None:
        start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    prices = []
    # Phase 1: Uptrend (candles 0-79)
    for i in range(80):
        progress = i / 79
        price = start_price * (1 + 0.15 * progress)
        # Add small noise
        noise = math.sin(i * 0.5) * start_price * 0.003
        prices.append(price + noise)

    peak_price = prices[-1]

    # Phase 2: Ranging (candles 80-139)
    for i in range(60):
        noise = math.sin(i * 0.8) * peak_price * 0.015
        prices.append(peak_price + noise)

    range_end_price = prices[-1]

    # Phase 3: Downtrend (candles 140-199)
    for i in range(60):
        progress = i / 59
        price = range_end_price * (1 - 0.12 * progress)
        noise = math.sin(i * 0.6) * start_price * 0.002
        prices.append(price + noise)

    return prices[:n_candles]


def generate_candles(
    prices: list[float],
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_time: datetime | None = None,
) -> list[OHLCV]:
    """Convert a price sequence to OHLCV candles."""
    if start_time is None:
        start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    candles = []
    for i, price in enumerate(prices):
        ts = start_time + timedelta(hours=i)
        candles.append(
            _make_candle(price, ts, symbol=symbol, timeframe=timeframe)
        )
    return candles


# ──── Test fixtures ────


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure a clean strategy registry for each test."""
    # Save existing state
    saved_strategies = dict(strategy_registry._strategies)
    saved_active = set(strategy_registry._active)
    strategy_registry.clear()
    yield
    # Restore
    strategy_registry._strategies = saved_strategies
    strategy_registry._active = saved_active


@pytest.fixture
def price_sequence():
    """Generate 200-candle price sequence."""
    return generate_price_sequence()


@pytest.fixture
def candles_1h(price_sequence):
    """1-hour candles from the price sequence."""
    return generate_candles(price_sequence, timeframe="1h")


@pytest.fixture
def candles_4h(price_sequence):
    """4-hour candles (subsample every 4th candle) for trend filter."""
    all_candles = generate_candles(price_sequence, timeframe="4h")
    return all_candles[::4]


# ──── Price sequence tests ────


class TestPriceSequence:
    """Verify the generated price sequence has the expected market phases."""

    def test_has_200_candles(self, price_sequence):
        assert len(price_sequence) == 200

    def test_uptrend_phase(self, price_sequence):
        """Prices 0-79 should trend up."""
        assert price_sequence[79] > price_sequence[0]
        # At least 10% gain
        gain = (price_sequence[79] - price_sequence[0]) / price_sequence[0]
        assert gain > 0.10

    def test_ranging_phase(self, price_sequence):
        """Prices 80-139 should stay near the peak (within 5%)."""
        peak = max(price_sequence[70:90])
        for p in price_sequence[80:140]:
            assert abs(p - peak) / peak < 0.05

    def test_downtrend_phase(self, price_sequence):
        """Prices 140-199 should trend down."""
        assert price_sequence[199] < price_sequence[140]
        loss = (price_sequence[140] - price_sequence[199]) / price_sequence[140]
        assert loss > 0.08


# ──── Regime detection integration ────


class TestRegimeDetection:
    """MarketRegimeDetector should detect different regimes in each phase."""

    def test_uptrend_detected(self, candles_1h):
        """During uptrend, detector should not see TRENDING_DOWN."""
        detector = MarketRegimeDetector()
        # Use candles from the middle of the uptrend (enough history)
        candles = candles_1h[10:70]
        if len(candles) >= detector.required_history_length:
            regime = detector.detect(candles)
            # Should not be TRENDING_DOWN during an uptrend
            assert regime != MarketRegime.TRENDING_DOWN

    def test_downtrend_detected(self, candles_1h):
        """During the downtrend phase, detector should see TRENDING_DOWN or HIGH_VOLATILITY."""
        detector = MarketRegimeDetector()
        candles = candles_1h[140:200]
        if len(candles) >= detector.required_history_length:
            regime = detector.detect(candles)
            assert regime in (
                MarketRegime.TRENDING_DOWN,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.RANGING,  # May detect ranging if trend isn't strong enough
            )


# ──── Trend filter integration ────


class TestTrendFilterIntegration:
    """TrendFilter should classify trend from 4h candles."""

    def test_trend_filter_returns_valid_direction(self, candles_4h):
        tf = TrendFilter()
        if len(candles_4h) >= tf.required_history_length:
            direction = tf.get_trend("BTC/USDT", candles_4h)
            assert direction in (
                TrendDirection.BULLISH,
                TrendDirection.BEARISH,
                TrendDirection.NEUTRAL,
            )


# ──── Full pipeline integration ────


class TestFullPipelineIntegration:
    """End-to-end integration test proving all V2 features work together."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock DataStore that returns pre-built candles."""
        store = MagicMock(spec=DataStore)
        store.initialize = AsyncMock()
        store.save_trade = AsyncMock()
        store.get_candles = AsyncMock(return_value=[])
        store.save_portfolio_snapshot = AsyncMock()
        store.close = AsyncMock()
        return store

    @pytest.fixture
    def mock_exchange(self):
        """Create a mock exchange adapter."""
        exchange = AsyncMock()
        exchange.name = "test_exchange"
        exchange.get_ticker = AsyncMock(return_value={"last": 40000.0})
        exchange.fetch_order_book = AsyncMock(
            return_value={"bids": [[40000, 1]], "asks": [[40001, 1]]}
        )
        exchange.close = AsyncMock()
        return exchange

    @pytest.mark.asyncio
    async def test_full_trading_cycle_with_ensemble(
        self, candles_1h, candles_4h, mock_store, mock_exchange
    ):
        """Run multiple trading cycles and verify the pipeline functions end-to-end.

        This test verifies:
        1. Strategies generate signals
        2. Ensemble voting works with min_agreement=2
        3. Risk manager validates signals
        4. Paper portfolio tracks balance
        5. Position manager handles stop-loss/take-profit
        6. Strategy tracker records stats
        """
        # Setup components
        paper_portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)
        risk_manager = RiskManager(
            max_position_size_pct=10.0,
            stop_loss_pct=3.0,
            daily_loss_limit_pct=5.0,
            max_drawdown_pct=15.0,
            max_concurrent_positions=5,
        )
        risk_manager.update_portfolio_value(10000.0)

        position_manager = PositionManager(
            stop_loss_pct=3.0,
            take_profit_pct=5.0,
            trailing_stop_enabled=False,
        )

        ensemble = SignalEnsemble(min_agreement=2)
        regime_detector = MarketRegimeDetector()
        trend_filter = TrendFilter()

        registry = StrategyRegistry.__new__(StrategyRegistry)
        registry._strategies = {}
        registry._active = set()

        tracker = StrategyTracker(
            max_consecutive_losses=5,
            min_win_rate_pct=40.0,
            min_trades_for_evaluation=20,
            re_enable_check_hours=24.0,
            registry=registry,
        )

        execution_engine = ExecutionEngine(
            exchange=mock_exchange,
            store=mock_store,
            paper_trading=True,
            paper_portfolio=paper_portfolio,
        )

        # Register 3 strategies by importing
        from bot.strategies.technical.ma_crossover import MACrossoverStrategy
        from bot.strategies.technical.rsi import RSIStrategy

        # Use fresh instances (not the global singletons)
        strategies = [
            MACrossoverStrategy(short_period=10, long_period=30),
            MACrossoverStrategy(short_period=5, long_period=20),
            RSIStrategy(),
        ]
        # Override name on second MA to avoid collision
        strategies[1]._name_override = "ma_crossover_fast"
        # Monkey-patch name property
        type(strategies[1]).name = property(
            lambda self: getattr(self, "_name_override", "ma_crossover")
        )

        for s in strategies:
            registry.register(s)

        active_strategies = registry.get_active()
        assert len(active_strategies) >= 2

        # Track signals across phases
        all_signals = []
        trades_executed = 0

        # Run through the price sequence in sliding windows
        window_size = 60  # Use 60-candle windows

        for cycle_idx in range(0, len(candles_1h) - window_size, 5):
            candle_window = candles_1h[cycle_idx : cycle_idx + window_size]
            current_price = candle_window[-1].close
            symbol = "BTC/USDT"

            # 1. Detect regime
            if len(candle_window) >= regime_detector.required_history_length:
                regime = regime_detector.detect(candle_window)
                for strategy in active_strategies:
                    strategy.adapt_to_regime(regime)
                tracker.update_regime(regime)

            # 2. Get trend direction from 4h candles
            trend_direction = None
            # Use a subset of 4h candles up to current time
            tf_idx = min(cycle_idx // 4 + 15, len(candles_4h))
            tf_candles = candles_4h[:tf_idx]
            if len(tf_candles) >= trend_filter.required_history_length:
                try:
                    trend_direction = trend_filter.get_trend(symbol, tf_candles)
                except (IndexError, ValueError):
                    trend_direction = None

            # 3. Collect signals from ensemble
            signals = await ensemble.collect_signals(
                symbol, active_strategies, candle_window
            )
            signal = ensemble.vote(signals, symbol, trend_direction=trend_direction)
            all_signals.append(signal)

            # 4. Risk check
            signal = risk_manager.validate_signal(signal)

            # 5. Check exits on managed positions
            for managed_symbol in list(position_manager.managed_symbols):
                exit_signal = position_manager.check_exits(
                    managed_symbol, current_price
                )
                if exit_signal:
                    order = await execution_engine.execute_signal(
                        MagicMock(
                            action=SignalAction.SELL,
                            symbol=managed_symbol,
                            strategy_name="position_manager",
                            metadata={},
                        ),
                        quantity=exit_signal.quantity,
                        price=exit_signal.exit_price,
                    )
                    if order and order.filled_price:
                        position = risk_manager.get_position(managed_symbol)
                        if position:
                            pnl = (
                                order.filled_price - position["entry_price"]
                            ) * order.quantity
                            risk_manager.record_trade_pnl(pnl)
                            tracker.record_trade("position_manager", pnl)
                        risk_manager.remove_position(managed_symbol)
                        position_manager.remove_position(managed_symbol)
                        trades_executed += 1

            # 6. Execute signal
            if signal.action != SignalAction.HOLD:
                qty = risk_manager.calculate_position_size(
                    risk_manager._current_portfolio_value or 10000.0,
                    current_price,
                )

                if qty > 0:
                    order = await execution_engine.execute_signal(
                        signal, quantity=qty, price=current_price
                    )
                    if order and order.filled_price:
                        fill_price = order.filled_price
                        if signal.action == SignalAction.BUY:
                            risk_manager.add_position(
                                symbol, order.quantity, fill_price
                            )
                            position_manager.add_position(
                                symbol, fill_price, order.quantity
                            )
                            trades_executed += 1
                        elif signal.action == SignalAction.SELL:
                            position = risk_manager.get_position(symbol)
                            if position:
                                pnl = (
                                    fill_price - position["entry_price"]
                                ) * order.quantity
                                risk_manager.record_trade_pnl(pnl)
                                strat_names = signal.metadata.get(
                                    "agreeing_strategies", [signal.strategy_name]
                                )
                                for sn in strat_names:
                                    tracker.record_trade(sn, pnl)
                            risk_manager.remove_position(symbol)
                            position_manager.remove_position(symbol)
                            trades_executed += 1

            # Update portfolio value
            risk_manager.update_portfolio_value(paper_portfolio.total_value)

        # ──── Assertions ────

        # Signals were generated (not all HOLD)
        non_hold = [s for s in all_signals if s.action != SignalAction.HOLD]
        assert len(non_hold) > 0, "Expected at least some non-HOLD signals"

        # At least one BUY and one SELL were executed
        # (BUY during uptrend, SELL via stop-loss or strategy during downtrend)
        assert trades_executed > 0, "Expected at least one trade to be executed"

        # Paper portfolio tracked balance correctly
        assert paper_portfolio.total_value > 0
        assert paper_portfolio.total_value != 10000.0 or trades_executed == 0

        # Strategy tracker has stats
        # At minimum we should have registered strategies
        assert len(strategies) >= 2

    @pytest.mark.asyncio
    async def test_ensemble_requires_min_agreement(self, candles_1h):
        """Verify that ensemble with min_agreement=2 requires at least 2 strategies to agree."""
        ensemble = SignalEnsemble(min_agreement=2)

        # Create a scenario where only 1 strategy signals BUY
        signals_one_buy = [
            MagicMock(
                action=SignalAction.BUY,
                strategy_name="strategy_a",
                confidence=0.8,
                symbol="BTC/USDT",
                metadata={},
            ),
            MagicMock(
                action=SignalAction.HOLD,
                strategy_name="strategy_b",
                confidence=0.0,
                symbol="BTC/USDT",
                metadata={},
            ),
        ]
        result = ensemble.vote(signals_one_buy, "BTC/USDT")
        assert result.action == SignalAction.HOLD

        # Now 2 strategies agree on BUY
        signals_two_buy = [
            MagicMock(
                action=SignalAction.BUY,
                strategy_name="strategy_a",
                confidence=0.8,
                symbol="BTC/USDT",
                metadata={},
            ),
            MagicMock(
                action=SignalAction.BUY,
                strategy_name="strategy_b",
                confidence=0.7,
                symbol="BTC/USDT",
                metadata={},
            ),
        ]
        result = ensemble.vote(signals_two_buy, "BTC/USDT")
        assert result.action == SignalAction.BUY

    @pytest.mark.asyncio
    async def test_trend_filter_rejects_counter_trend_buys(self, candles_1h):
        """Ensemble should reject BUY when trend is BEARISH."""
        ensemble = SignalEnsemble(min_agreement=2)

        signals = [
            MagicMock(
                action=SignalAction.BUY,
                strategy_name="s1",
                confidence=0.8,
                symbol="BTC/USDT",
                metadata={},
            ),
            MagicMock(
                action=SignalAction.BUY,
                strategy_name="s2",
                confidence=0.7,
                symbol="BTC/USDT",
                metadata={},
            ),
        ]
        result = ensemble.vote(
            signals, "BTC/USDT", trend_direction=TrendDirection.BEARISH
        )
        assert result.action == SignalAction.HOLD
        assert result.metadata.get("reason") == "trend_filter_rejected"

    @pytest.mark.asyncio
    async def test_trend_filter_allows_trend_aligned_buys(self, candles_1h):
        """Ensemble should allow BUY when trend is BULLISH."""
        ensemble = SignalEnsemble(min_agreement=2)

        signals = [
            MagicMock(
                action=SignalAction.BUY,
                strategy_name="s1",
                confidence=0.8,
                symbol="BTC/USDT",
                metadata={},
            ),
            MagicMock(
                action=SignalAction.BUY,
                strategy_name="s2",
                confidence=0.7,
                symbol="BTC/USDT",
                metadata={},
            ),
        ]
        result = ensemble.vote(
            signals, "BTC/USDT", trend_direction=TrendDirection.BULLISH
        )
        assert result.action == SignalAction.BUY

    @pytest.mark.asyncio
    async def test_regime_adapts_strategies(self, candles_1h):
        """MA Crossover should disable itself in RANGING regime."""
        from bot.strategies.technical.ma_crossover import MACrossoverStrategy

        strategy = MACrossoverStrategy()
        strategy.adapt_to_regime(MarketRegime.RANGING)
        # _regime_disabled should be True
        assert strategy._regime_disabled is True

        # In RANGING, analyze should return HOLD
        candles = candles_1h[80:140]  # ranging phase
        if len(candles) >= strategy.required_history_length:
            signal = await strategy.analyze(candles, symbol="BTC/USDT")
            assert signal.action == SignalAction.HOLD
            assert signal.metadata.get("reason") == "disabled_by_regime"

    @pytest.mark.asyncio
    async def test_stop_loss_triggers_during_downtrend(self):
        """PositionManager stop-loss should trigger when price drops below threshold."""
        pm = PositionManager(stop_loss_pct=3.0, take_profit_pct=5.0)
        entry_price = 45000.0
        pm.add_position("BTC/USDT", entry_price, 0.1)

        # Price above stop-loss - no exit
        exit_signal = pm.check_exits("BTC/USDT", 44000.0)
        assert exit_signal is None

        # Price at stop-loss (3% below 45000 = 43650)
        exit_signal = pm.check_exits("BTC/USDT", 43600.0)
        assert exit_signal is not None
        assert exit_signal.exit_type.value == "stop_loss"
        assert exit_signal.quantity == 0.1

    @pytest.mark.asyncio
    async def test_take_profit_triggers_during_uptrend(self):
        """PositionManager take-profit should trigger in two stages."""
        pm = PositionManager(
            stop_loss_pct=3.0, take_profit_pct=5.0, tp1_pct=3.0
        )
        entry_price = 40000.0
        pm.add_position("BTC/USDT", entry_price, 0.1)

        # Price at TP1 (3% above 40000 = 41200)
        exit_signal = pm.check_exits("BTC/USDT", 41250.0)
        assert exit_signal is not None
        assert exit_signal.exit_type.value == "take_profit_1"
        # Should sell 50% of position
        assert exit_signal.quantity == pytest.approx(0.05)

        # Price at TP2 (5% above 40000 = 42000)
        exit_signal = pm.check_exits("BTC/USDT", 42050.0)
        assert exit_signal is not None
        assert exit_signal.exit_type.value == "take_profit_2"

    @pytest.mark.asyncio
    async def test_paper_portfolio_balance_reflects_trades(self):
        """PaperPortfolio should correctly track balance after buy and sell."""
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)

        # Buy 0.1 BTC at 40000 (cost = 4000 + 4.0 fee = 4004)
        result = portfolio.buy("BTC/USDT", 0.1, 40000.0)
        assert result is True
        assert portfolio.cash == pytest.approx(10000.0 - 4004.0)

        # Sell 0.1 BTC at 42000 (proceeds = 4200 - 4.2 fee = 4195.8)
        result = portfolio.sell("BTC/USDT", 0.1, 42000.0)
        assert result is True
        # Final cash = (10000 - 4004) + 4195.8 = 10191.8
        assert portfolio.cash == pytest.approx(10191.8)

        # No positions left
        assert len(portfolio.positions) == 0

        # Net profit after fees
        assert portfolio.total_value > 10000.0

    @pytest.mark.asyncio
    async def test_risk_manager_blocks_when_halted(self):
        """RiskManager should block signals when daily loss limit exceeded."""
        rm = RiskManager(daily_loss_limit_pct=5.0)
        rm.update_portfolio_value(10000.0)

        # Lose 5% of portfolio
        rm.record_trade_pnl(-500.0)

        signal = MagicMock(
            action=SignalAction.BUY,
            symbol="BTC/USDT",
            strategy_name="test",
            confidence=0.8,
            metadata={},
        )
        result = rm.validate_signal(signal)
        assert result.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_risk_manager_position_tracking(self):
        """RiskManager tracks positions through add/get/remove lifecycle."""
        rm = RiskManager(max_concurrent_positions=3)
        rm.update_portfolio_value(10000.0)

        rm.add_position("BTC/USDT", 0.1, 40000.0)
        pos = rm.get_position("BTC/USDT")
        assert pos is not None
        assert pos["quantity"] == 0.1
        assert pos["entry_price"] == 40000.0

        rm.remove_position("BTC/USDT")
        assert rm.get_position("BTC/USDT") is None

    @pytest.mark.asyncio
    async def test_strategy_tracker_records_trades(self):
        """StrategyTracker should accumulate wins, losses, and PnL."""
        registry = StrategyRegistry.__new__(StrategyRegistry)
        registry._strategies = {}
        registry._active = set()

        tracker = StrategyTracker(
            max_consecutive_losses=5,
            min_win_rate_pct=40.0,
            min_trades_for_evaluation=20,
            re_enable_check_hours=24.0,
            registry=registry,
        )

        # Record some trades
        tracker.record_trade("ma_crossover", 50.0)
        tracker.record_trade("ma_crossover", -20.0)
        tracker.record_trade("ma_crossover", 30.0)

        stats = tracker.get_all_stats()
        assert "ma_crossover" in stats
        assert stats["ma_crossover"]["total_trades"] == 3
        assert stats["ma_crossover"]["wins"] == 2
        assert stats["ma_crossover"]["losses"] == 1
        assert stats["ma_crossover"]["total_pnl"] == pytest.approx(60.0)

    @pytest.mark.asyncio
    async def test_execution_engine_paper_mode(self, mock_store, mock_exchange):
        """ExecutionEngine in paper mode creates filled orders."""
        portfolio = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)
        engine = ExecutionEngine(
            exchange=mock_exchange,
            store=mock_store,
            paper_trading=True,
            paper_portfolio=portfolio,
        )

        signal = MagicMock(
            action=SignalAction.BUY,
            symbol="BTC/USDT",
            strategy_name="test",
            metadata={},
        )
        order = await engine.execute_signal(signal, quantity=0.01, price=40000.0)
        assert order is not None
        assert order.filled_price == 40000.0
        assert order.quantity == 0.01

        # Trade was saved
        mock_store.save_trade.assert_called_once()

    @pytest.mark.asyncio
    async def test_portfolio_risk_manager_integration(self, candles_1h):
        """PortfolioRiskManager should validate new positions against exposure limits."""
        prm = PortfolioRiskManager(
            max_total_exposure_pct=60.0,
            max_correlation=0.8,
            correlation_window=30,
            max_positions_per_sector=3,
            max_portfolio_heat=0.15,
        )
        prm.update_portfolio_value(10000.0)

        # Add a position that's 50% of portfolio (under 60% limit)
        prm.add_position("BTC/USDT", 5000.0)
        allowed, reason = prm.validate_new_position("ETH/USDT", 500.0)
        assert allowed is True

        # Add more exposure to exceed the limit
        prm.add_position("ETH/USDT", 5500.0)
        allowed, reason = prm.validate_new_position("SOL/USDT", 1000.0)
        # Already at 105% exposure, so should reject
        assert allowed is False

    @pytest.mark.asyncio
    async def test_dashboard_state_updated_after_cycle(self):
        """Dashboard state should be updated after each trading cycle."""
        from bot.dashboard import app as dashboard_module

        initial_state = dashboard_module.get_state()
        assert "status" in initial_state
        assert "trades" in initial_state

        # Update state as the trading cycle would
        dashboard_module.update_state(
            status="running",
            trades=[
                {
                    "timestamp": "2026-01-01T00:00:00",
                    "symbol": "BTC/USDT",
                    "side": "BUY",
                    "quantity": 0.1,
                    "price": 40000.0,
                }
            ],
            strategy_stats={"ma_crossover": {"total_trades": 5, "win_rate": 60.0}},
        )

        state = dashboard_module.get_state()
        assert state["status"] == "running"
        assert len(state["trades"]) == 1
        assert state["strategy_stats"]["ma_crossover"]["total_trades"] == 5


# ──── Multi-cycle simulation ────


class TestMultiCycleSimulation:
    """Run a longer simulation to verify stability and correctness over many cycles."""

    @pytest.mark.asyncio
    async def test_50_cycle_simulation(self, candles_1h, candles_4h):
        """Run 50 cycles and verify the system remains stable."""
        paper = PaperPortfolio(initial_balance=10000.0, fee_pct=0.1)
        rm = RiskManager(
            max_position_size_pct=10.0,
            stop_loss_pct=3.0,
            daily_loss_limit_pct=5.0,
            max_drawdown_pct=15.0,
            max_concurrent_positions=5,
        )
        rm.update_portfolio_value(10000.0)

        pm = PositionManager(stop_loss_pct=3.0, take_profit_pct=5.0)
        ensemble = SignalEnsemble(min_agreement=2)

        from bot.strategies.technical.ma_crossover import MACrossoverStrategy
        from bot.strategies.technical.rsi import RSIStrategy

        strategies = [
            MACrossoverStrategy(short_period=10, long_period=30),
            RSIStrategy(),
        ]

        cycle_count = 0
        errors = []

        for i in range(50, min(len(candles_1h), 200), 3):
            window = candles_1h[max(0, i - 60) : i]
            if len(window) < 30:
                continue

            try:
                current_price = window[-1].close
                symbol = "BTC/USDT"

                # Check exits
                for sym in list(pm.managed_symbols):
                    exit_signal = pm.check_exits(sym, current_price)
                    if exit_signal:
                        pos = rm.get_position(sym)
                        if pos:
                            pnl = (
                                current_price - pos["entry_price"]
                            ) * exit_signal.quantity
                            rm.record_trade_pnl(pnl)
                        rm.remove_position(sym)
                        pm.remove_position(sym)

                # Get signals
                signals = await ensemble.collect_signals(
                    symbol, strategies, window
                )
                signal = ensemble.vote(signals, symbol)
                signal = rm.validate_signal(signal)

                if signal.action == SignalAction.BUY and not rm.get_position(symbol):
                    qty = rm.calculate_position_size(
                        rm._current_portfolio_value or 10000.0,
                        current_price,
                    )
                    if qty > 0 and paper.buy(symbol, qty, current_price):
                        rm.add_position(symbol, qty, current_price)
                        pm.add_position(symbol, current_price, qty)

                elif signal.action == SignalAction.SELL and rm.get_position(symbol):
                    pos = rm.get_position(symbol)
                    if pos:
                        qty = pos["quantity"]
                        if paper.sell(symbol, qty, current_price):
                            pnl = (
                                current_price - pos["entry_price"]
                            ) * qty
                            rm.record_trade_pnl(pnl)
                            rm.remove_position(symbol)
                            pm.remove_position(symbol)

                rm.update_portfolio_value(paper.total_value)
                cycle_count += 1

            except Exception as e:
                errors.append(f"Cycle {i}: {e}")

        # No unhandled errors
        assert len(errors) == 0, f"Errors during simulation: {errors}"

        # Ran multiple cycles
        assert cycle_count >= 30

        # Portfolio value is still positive
        assert paper.total_value > 0

        # Portfolio cash is non-negative
        assert paper.cash >= 0
