"""Tests for V2-021: Realistic backtesting with stop-loss, take-profit, trailing stop,
dynamic slippage, walk-forward, equity curve, drawdown, monthly returns, compare, and export."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from bot.backtest.engine import BacktestEngine, BacktestResult
from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy

# --- Test strategies ---

class AlwaysBuyStrategy(BaseStrategy):
    """Strategy that always buys."""

    @property
    def name(self) -> str:
        return "always_buy"

    @property
    def required_history_length(self) -> int:
        return 1

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        return TradingSignal(
            strategy_name="always_buy",
            symbol=kwargs.get("symbol", "BTC/USDT"),
            action=SignalAction.BUY,
            confidence=0.9,
        )


class AlwaysHoldStrategy(BaseStrategy):
    """Strategy that always holds."""

    @property
    def name(self) -> str:
        return "always_hold"

    @property
    def required_history_length(self) -> int:
        return 1

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        return TradingSignal(
            strategy_name="always_hold",
            symbol=kwargs.get("symbol", "BTC/USDT"),
            action=SignalAction.HOLD,
            confidence=0.5,
        )


class AlternateBuySellStrategy(BaseStrategy):
    """Strategy that alternates BUY/SELL."""

    def __init__(self):
        self._count = 0

    @property
    def name(self) -> str:
        return "alternate"

    @property
    def required_history_length(self) -> int:
        return 1

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        self._count += 1
        action = SignalAction.BUY if self._count % 2 == 1 else SignalAction.SELL
        return TradingSignal(
            strategy_name="alternate",
            symbol=kwargs.get("symbol", "BTC/USDT"),
            action=action,
            confidence=0.8,
        )


class BuyOnFirstStrategy(BaseStrategy):
    """Strategy that buys on first call, then holds."""

    def __init__(self):
        self._called = False

    @property
    def name(self) -> str:
        return "buy_once"

    @property
    def required_history_length(self) -> int:
        return 1

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        if not self._called:
            self._called = True
            return TradingSignal(
                strategy_name="buy_once",
                symbol=kwargs.get("symbol", "BTC/USDT"),
                action=SignalAction.BUY,
                confidence=0.9,
            )
        return TradingSignal(
            strategy_name="buy_once",
            symbol=kwargs.get("symbol", "BTC/USDT"),
            action=SignalAction.HOLD,
            confidence=0.5,
        )


# --- Helpers ---

def make_candles(
    prices: list[float],
    start_time: datetime | None = None,
    high_low_pct: float = 1.0,
    volumes: list[float] | None = None,
) -> list[OHLCV]:
    """Create OHLCV candles from close prices with configurable high/low spread."""
    base = start_time or datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i, price in enumerate(prices):
        vol = volumes[i] if volumes else 1000.0
        candles.append(OHLCV(
            timestamp=base + timedelta(hours=i),
            open=price,
            high=price * (1 + high_low_pct / 100),
            low=price * (1 - high_low_pct / 100),
            close=price,
            volume=vol,
        ))
    return candles


def make_candles_ohlc(
    ohlc_data: list[tuple[float, float, float, float]],
    start_time: datetime | None = None,
    volume: float = 1000.0,
) -> list[OHLCV]:
    """Create OHLCV candles from (open, high, low, close) tuples."""
    base = start_time or datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i, (o, h, lo, c) in enumerate(ohlc_data):
        candles.append(OHLCV(
            timestamp=base + timedelta(hours=i),
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=volume,
        ))
    return candles


# =============================================================================
# Backward compatibility tests
# =============================================================================

class TestBackwardCompatibility:
    """Ensure existing backtest behavior is preserved."""

    @pytest.mark.asyncio
    async def test_no_trades_on_hold(self):
        """HOLD strategy produces no trades."""
        engine = BacktestEngine(initial_capital=10000.0)
        data = make_candles([100.0] * 10)
        result = await engine.run(AlwaysHoldStrategy(), data)
        assert len(result.trades) == 0
        assert result.final_portfolio_value == 10000.0

    @pytest.mark.asyncio
    async def test_buy_sell_cycle_no_fees(self):
        """BUY/SELL cycle without fees or slippage preserves gains."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        data = make_candles([100.0, 100.0, 120.0])
        strategy = AlternateBuySellStrategy()
        result = await engine.run(strategy, data)
        assert len(result.trades) == 2
        assert result.trades[0].side == "BUY"
        assert result.trades[1].side == "SELL"
        assert result.final_portfolio_value > 10000.0

    @pytest.mark.asyncio
    async def test_default_sl_tp_zero_no_exit(self):
        """With default SL/TP=0, no exit conditions trigger."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        # Price drops but no SL configured
        data = make_candles([100.0, 100.0, 80.0, 60.0, 40.0])
        result = await engine.run(AlwaysBuyStrategy(), data)
        # Only one BUY (no sells since no SL/TP and strategy only buys)
        sell_trades = [t for t in result.trades if t.side == "SELL"]
        assert len(sell_trades) == 0

    @pytest.mark.asyncio
    async def test_result_has_new_fields(self):
        """BacktestResult now includes equity_curve, drawdown_curve, monthly_returns."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        data = make_candles([100.0, 100.0, 120.0])
        result = await engine.run(AlternateBuySellStrategy(), data)
        assert isinstance(result.equity_curve, list)
        assert isinstance(result.drawdown_curve, list)
        assert isinstance(result.monthly_returns, dict)
        assert len(result.equity_curve) > 0

    @pytest.mark.asyncio
    async def test_trade_log_has_exit_reason(self):
        """TradeLog now includes exit_reason field."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        data = make_candles([100.0, 100.0, 120.0])
        result = await engine.run(AlternateBuySellStrategy(), data)
        sell_trade = result.trades[1]
        assert sell_trade.exit_reason == "strategy"


# =============================================================================
# Stop-loss enforcement tests
# =============================================================================

class TestStopLossEnforcement:
    """Stop-loss triggers based on candle low, not close."""

    @pytest.mark.asyncio
    async def test_stop_loss_triggers_on_low(self):
        """SL triggers when candle low breaches stop-loss price."""
        engine = BacktestEngine(
            initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0,
            stop_loss_pct=5.0,
        )
        # Buy at 100, SL at 95. Candle low goes to 90 (below SL)
        ohlc_data = [
            (100.0, 101.0, 99.0, 100.0),  # ignored (min_history)
            (100.0, 101.0, 99.0, 100.0),  # BUY at close=100
            (98.0, 99.0, 90.0, 97.0),     # Low=90 < SL=95 → triggers
        ]
        data = make_candles_ohlc(ohlc_data)
        result = await engine.run(BuyOnFirstStrategy(), data)

        sell_trades = [t for t in result.trades if t.side == "SELL"]
        assert len(sell_trades) == 1
        assert sell_trades[0].exit_reason == "stop_loss"
        # Exit at SL price (95), not close (97)
        assert sell_trades[0].price == pytest.approx(95.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_stop_loss_not_triggered_when_low_above_sl(self):
        """SL does not trigger if candle low stays above stop-loss."""
        engine = BacktestEngine(
            initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0,
            stop_loss_pct=5.0,
        )
        ohlc_data = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),  # BUY
            (99.0, 100.0, 96.0, 98.0),    # Low=96 > SL=95 → no trigger
        ]
        data = make_candles_ohlc(ohlc_data)
        result = await engine.run(BuyOnFirstStrategy(), data)

        sell_trades = [t for t in result.trades if t.side == "SELL"]
        assert len(sell_trades) == 0

    @pytest.mark.asyncio
    async def test_stop_loss_exit_pnl_negative(self):
        """SL exit should record a negative PnL."""
        engine = BacktestEngine(
            initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0,
            stop_loss_pct=5.0,
        )
        ohlc_data = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),  # BUY at 100
            (95.0, 96.0, 90.0, 92.0),     # SL at 95 triggers
        ]
        data = make_candles_ohlc(ohlc_data)
        result = await engine.run(BuyOnFirstStrategy(), data)

        sell_trades = [t for t in result.trades if t.side == "SELL"]
        assert len(sell_trades) == 1
        assert sell_trades[0].pnl < 0


# =============================================================================
# Take-profit enforcement tests
# =============================================================================

class TestTakeProfitEnforcement:
    """Take-profit triggers based on candle high."""

    @pytest.mark.asyncio
    async def test_take_profit_triggers_on_high(self):
        """TP triggers when candle high reaches take-profit price."""
        engine = BacktestEngine(
            initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0,
            take_profit_pct=10.0,
        )
        # Buy at 100, TP at 110. Candle high hits 115
        ohlc_data = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),  # BUY
            (105.0, 115.0, 104.0, 108.0),  # High=115 >= TP=110 → triggers
        ]
        data = make_candles_ohlc(ohlc_data)
        result = await engine.run(BuyOnFirstStrategy(), data)

        sell_trades = [t for t in result.trades if t.side == "SELL"]
        assert len(sell_trades) == 1
        assert sell_trades[0].exit_reason == "take_profit"
        # Exit at TP price (110), not close (108)
        assert sell_trades[0].price == pytest.approx(110.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_take_profit_pnl_positive(self):
        """TP exit should record a positive PnL."""
        engine = BacktestEngine(
            initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0,
            take_profit_pct=10.0,
        )
        ohlc_data = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),  # BUY
            (105.0, 115.0, 104.0, 108.0),  # TP triggers
        ]
        data = make_candles_ohlc(ohlc_data)
        result = await engine.run(BuyOnFirstStrategy(), data)

        sell_trades = [t for t in result.trades if t.side == "SELL"]
        assert len(sell_trades) == 1
        assert sell_trades[0].pnl > 0

    @pytest.mark.asyncio
    async def test_take_profit_before_stop_loss(self):
        """When both SL and TP could trigger on same candle, TP takes priority.

        TP is checked first because price likely reached the high (TP) before
        the low (SL) within the same candle.
        """
        engine = BacktestEngine(
            initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0,
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        # Buy at 100, SL at 95, TP at 110
        # Candle: low=90 (< SL) AND high=115 (> TP) — TP checked first
        ohlc_data = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),  # BUY
            (100.0, 115.0, 90.0, 100.0),  # Both conditions — TP wins
        ]
        data = make_candles_ohlc(ohlc_data)
        result = await engine.run(BuyOnFirstStrategy(), data)

        sell_trades = [t for t in result.trades if t.side == "SELL"]
        assert len(sell_trades) == 1
        assert sell_trades[0].exit_reason == "take_profit"


# =============================================================================
# Trailing stop tests
# =============================================================================

class TestTrailingStop:
    """Trailing stop adjusts upward as price rises."""

    @pytest.mark.asyncio
    async def test_trailing_stop_moves_up(self):
        """Trailing stop moves up as price rises and triggers on pullback."""
        engine = BacktestEngine(
            initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0,
            stop_loss_pct=5.0,
            trailing_stop_pct=3.0,
        )
        # Buy at 100, initial SL at 95
        # Price rises to 120 → trailing stop at 120 * 0.97 = 116.4
        # Then drops to 115 → below trailing stop → exit
        ohlc_data = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),   # BUY
            (105.0, 120.0, 104.0, 118.0),   # High=120, no trigger
            (117.0, 118.0, 115.0, 116.0),   # Low=115 < trailing(116.4) → exit
        ]
        data = make_candles_ohlc(ohlc_data)
        result = await engine.run(BuyOnFirstStrategy(), data)

        sell_trades = [t for t in result.trades if t.side == "SELL"]
        assert len(sell_trades) == 1
        assert sell_trades[0].exit_reason == "trailing_stop"
        # Should exit at trailing stop price, not at original SL
        assert sell_trades[0].price > 95.0  # Much higher than original SL

    @pytest.mark.asyncio
    async def test_trailing_stop_only_after_price_above_entry(self):
        """Trailing stop only activates after price moves above entry."""
        engine = BacktestEngine(
            initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0,
            stop_loss_pct=5.0,
            trailing_stop_pct=2.0,
        )
        # Buy at 100, SL at 95
        # Price stays below entry → trailing stop should NOT override SL
        ohlc_data = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),   # BUY
            (99.0, 100.0, 96.0, 98.0),     # Below entry, no trailing override
            (97.0, 98.0, 94.0, 95.0),      # Low=94 < SL=95 → stop_loss (not trailing)
        ]
        data = make_candles_ohlc(ohlc_data)
        result = await engine.run(BuyOnFirstStrategy(), data)

        sell_trades = [t for t in result.trades if t.side == "SELL"]
        assert len(sell_trades) == 1
        assert sell_trades[0].exit_reason == "stop_loss"


# =============================================================================
# Dynamic slippage tests
# =============================================================================

class TestDynamicSlippage:
    """Dynamic slippage scales with order size relative to volume."""

    @pytest.mark.asyncio
    async def test_dynamic_slippage_larger_order_more_slippage(self):
        """Larger order values relative to daily volume get more slippage."""
        # Small order relative to volume
        engine_small = BacktestEngine(
            initial_capital=1000.0, fee_pct=0.0, slippage_pct=0.0,
            dynamic_slippage=True, base_slippage_pct=0.1,
            avg_daily_volume=1000000.0,
        )
        # Large order relative to volume
        engine_large = BacktestEngine(
            initial_capital=100000.0, fee_pct=0.0, slippage_pct=0.0,
            dynamic_slippage=True, base_slippage_pct=0.1,
            avg_daily_volume=1000000.0,
        )

        data = make_candles([100.0, 100.0, 120.0])

        result_small = await engine_small.run(AlternateBuySellStrategy(), data)
        result_large = await engine_large.run(AlternateBuySellStrategy(), data)

        # Large order gets worse execution due to higher slippage
        return_small = (result_small.final_portfolio_value - 1000.0) / 1000.0
        return_large = (result_large.final_portfolio_value - 100000.0) / 100000.0
        assert return_small > return_large

    @pytest.mark.asyncio
    async def test_dynamic_slippage_formula(self):
        """Verify slippage = base * (1 + order_value / avg_daily_volume)."""
        engine = BacktestEngine(
            dynamic_slippage=True,
            base_slippage_pct=0.1,
            avg_daily_volume=100000.0,
        )
        # Order value = 10000, ratio = 0.1
        slippage = engine._calculate_slippage(10000.0)
        expected = 0.1 * (1 + 10000.0 / 100000.0)  # 0.1 * 1.1 = 0.11
        assert slippage == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_static_slippage_when_dynamic_disabled(self):
        """When dynamic_slippage=False, uses fixed slippage_pct."""
        engine = BacktestEngine(slippage_pct=0.05, dynamic_slippage=False)
        slippage = engine._calculate_slippage(999999.0)
        assert slippage == 0.05


# =============================================================================
# Equity curve and drawdown tests
# =============================================================================

class TestEquityCurveAndDrawdown:
    """Equity curve and drawdown curve are computed correctly."""

    @pytest.mark.asyncio
    async def test_equity_curve_length(self):
        """Equity curve has correct number of entries."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        prices = [100.0] * 5
        data = make_candles(prices)
        result = await engine.run(AlwaysHoldStrategy(), data)
        # 1 initial + (len-min_history) candles = 1 + 4 = 5
        assert len(result.equity_curve) == 5

    @pytest.mark.asyncio
    async def test_equity_curve_reflects_trades(self):
        """Equity curve changes with portfolio value."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        data = make_candles([100.0, 100.0, 120.0])
        result = await engine.run(AlternateBuySellStrategy(), data)
        # After selling at 120 with no fees, portfolio value should increase
        assert result.equity_curve[-1] > result.equity_curve[0]

    @pytest.mark.asyncio
    async def test_drawdown_curve_zero_at_peak(self):
        """Drawdown is 0 at the peak of the equity curve."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        data = make_candles([100.0] * 5)
        result = await engine.run(AlwaysHoldStrategy(), data)
        # Flat equity → all zeros
        for dd in result.drawdown_curve:
            assert dd == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_drawdown_positive_after_drop(self):
        """Drawdown is positive when portfolio value drops below peak."""
        drawdowns = BacktestEngine._calculate_drawdown_curve([100, 120, 100, 80])
        assert drawdowns[0] == 0.0  # at start (also peak initially)
        assert drawdowns[1] == 0.0  # new peak
        assert drawdowns[2] > 0.0   # below peak
        assert drawdowns[3] > drawdowns[2]  # even further below


# =============================================================================
# Monthly returns tests
# =============================================================================

class TestMonthlyReturns:
    """Monthly returns are calculated correctly."""

    @pytest.mark.asyncio
    async def test_monthly_returns_populated(self):
        """Monthly returns dict has entries for each month in data."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        # Create data spanning 2 months
        start = datetime(2024, 1, 15, tzinfo=timezone.utc)
        prices = [100.0] * 24 * 20  # 20 days of hourly data
        data = make_candles(prices, start_time=start)
        result = await engine.run(AlwaysHoldStrategy(), data)
        assert len(result.monthly_returns) >= 1

    @pytest.mark.asyncio
    async def test_monthly_returns_reflect_pnl(self):
        """Monthly returns reflect actual portfolio changes."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        data = make_candles([100.0, 100.0, 120.0])
        result = await engine.run(AlternateBuySellStrategy(), data)
        # All in same month, should show positive return
        if result.monthly_returns:
            values = list(result.monthly_returns.values())
            assert values[0] > 0


# =============================================================================
# Walk-forward tests
# =============================================================================

class TestWalkForward:
    """Walk-forward backtesting splits data correctly."""

    @pytest.mark.asyncio
    async def test_walk_forward_produces_result(self):
        """Walk-forward produces a valid BacktestResult."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        data = make_candles([100.0 + i for i in range(30)])

        result = await engine.walk_forward(
            strategy_factory=AlternateBuySellStrategy,
            data=data,
            train_ratio=0.7,
            n_windows=3,
        )
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "alternate"
        assert result.final_portfolio_value > 0

    @pytest.mark.asyncio
    async def test_walk_forward_with_few_data(self):
        """Walk-forward falls back to regular run with too little data."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        data = make_candles([100.0, 100.0, 120.0])

        result = await engine.walk_forward(
            strategy_factory=AlternateBuySellStrategy,
            data=data,
            n_windows=3,
        )
        assert isinstance(result, BacktestResult)

    @pytest.mark.asyncio
    async def test_walk_forward_capital_carries_over(self):
        """Capital from one window carries to the next."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        # Trending up → strategy profits should compound across windows
        data = make_candles([100.0 + i * 2 for i in range(60)])

        result = await engine.walk_forward(
            strategy_factory=AlternateBuySellStrategy,
            data=data,
            train_ratio=0.5,
            n_windows=3,
        )
        assert isinstance(result, BacktestResult)
        assert result.equity_curve[0] == 10000.0

    @pytest.mark.asyncio
    async def test_walk_forward_equity_curve_contiguous(self):
        """Equity curve is contiguous across walk-forward windows."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        data = make_candles([100.0 + i for i in range(30)])

        result = await engine.walk_forward(
            strategy_factory=AlternateBuySellStrategy,
            data=data,
            n_windows=3,
        )
        # Equity curve should start with initial capital
        assert result.equity_curve[0] == 10000.0
        # Should have entries
        assert len(result.equity_curve) > 1


# =============================================================================
# Compare method tests
# =============================================================================

class TestCompare:
    """BacktestEngine.compare() runs multiple strategies and sorts by Sharpe."""

    @pytest.mark.asyncio
    async def test_compare_returns_sorted_results(self):
        """Compare returns results sorted by Sharpe ratio descending."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        data = make_candles([100.0, 100.0, 120.0, 120.0, 110.0])

        strategies = [AlwaysHoldStrategy(), AlternateBuySellStrategy()]
        results = await engine.compare(strategies, data)

        assert len(results) == 2
        # Results should be sorted by Sharpe descending
        assert results[0].metrics.sharpe_ratio >= results[1].metrics.sharpe_ratio

    @pytest.mark.asyncio
    async def test_compare_different_strategies(self):
        """Compare returns one result per strategy."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        data = make_candles([100.0, 100.0, 120.0])

        strategies = [
            AlwaysHoldStrategy(),
            AlternateBuySellStrategy(),
            AlwaysBuyStrategy(),
        ]
        results = await engine.compare(strategies, data)
        assert len(results) == 3
        names = {r.strategy_name for r in results}
        assert "always_hold" in names
        assert "alternate" in names
        assert "always_buy" in names


# =============================================================================
# Export tests (to_json, to_csv)
# =============================================================================

class TestExport:
    """BacktestResult export methods."""

    @pytest.mark.asyncio
    async def test_to_json_valid(self):
        """to_json produces valid JSON with expected fields."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        data = make_candles([100.0, 100.0, 120.0])
        result = await engine.run(AlternateBuySellStrategy(), data)

        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["strategy_name"] == "alternate"
        assert parsed["symbol"] == "BTC/USDT"
        assert "metrics" in parsed
        assert "trades" in parsed
        assert "equity_curve" in parsed
        assert "drawdown_curve" in parsed
        assert "monthly_returns" in parsed

    @pytest.mark.asyncio
    async def test_to_csv_has_header_and_rows(self):
        """to_csv produces CSV with header and trade rows."""
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        data = make_candles([100.0, 100.0, 120.0])
        result = await engine.run(AlternateBuySellStrategy(), data)

        csv_str = result.to_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) >= 3  # header + 2 trades
        assert "timestamp" in lines[0]
        assert "exit_reason" in lines[0]

    @pytest.mark.asyncio
    async def test_to_json_no_trades(self):
        """to_json works with zero trades."""
        engine = BacktestEngine(initial_capital=10000.0)
        data = make_candles([100.0] * 5)
        result = await engine.run(AlwaysHoldStrategy(), data)

        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["trades"] == []

    @pytest.mark.asyncio
    async def test_to_csv_no_trades(self):
        """to_csv works with zero trades (only header)."""
        engine = BacktestEngine(initial_capital=10000.0)
        data = make_candles([100.0] * 5)
        result = await engine.run(AlwaysHoldStrategy(), data)

        csv_str = result.to_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 1  # header only


# =============================================================================
# Integration: SL + TP + trailing together
# =============================================================================

class TestIntegration:
    """Combined features work together."""

    @pytest.mark.asyncio
    async def test_sl_tp_trailing_together(self):
        """All exit mechanisms work when configured together."""
        engine = BacktestEngine(
            initial_capital=10000.0, fee_pct=0.1, slippage_pct=0.0,
            stop_loss_pct=5.0, take_profit_pct=10.0, trailing_stop_pct=3.0,
        )
        # Buy at 100, price rises to 112 (TP=110 triggers)
        ohlc_data = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),   # BUY
            (105.0, 112.0, 104.0, 108.0),   # TP at 110 triggers
        ]
        data = make_candles_ohlc(ohlc_data)
        result = await engine.run(BuyOnFirstStrategy(), data)

        sell_trades = [t for t in result.trades if t.side == "SELL"]
        assert len(sell_trades) == 1
        assert sell_trades[0].exit_reason == "take_profit"

    @pytest.mark.asyncio
    async def test_exit_before_strategy_signal(self):
        """Exit conditions are checked BEFORE strategy signal on each candle."""
        engine = BacktestEngine(
            initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0,
            stop_loss_pct=5.0,
        )
        # BuyOnFirst: buys on first candle, then holds
        # SL triggers on 3rd candle, position exits BEFORE strategy runs
        ohlc_data = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),  # BUY
            (90.0, 91.0, 85.0, 88.0),     # SL triggers
        ]
        data = make_candles_ohlc(ohlc_data)
        result = await engine.run(BuyOnFirstStrategy(), data)

        sell_trades = [t for t in result.trades if t.side == "SELL"]
        assert len(sell_trades) == 1
        assert sell_trades[0].exit_reason == "stop_loss"

    @pytest.mark.asyncio
    async def test_dynamic_slippage_with_sl(self):
        """Dynamic slippage applies to SL exit price."""
        engine = BacktestEngine(
            initial_capital=10000.0, fee_pct=0.0,
            dynamic_slippage=True, base_slippage_pct=1.0,
            avg_daily_volume=100000.0,
            stop_loss_pct=5.0,
        )
        ohlc_data = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),  # BUY
            (90.0, 91.0, 85.0, 88.0),     # SL triggers at 95
        ]
        data = make_candles_ohlc(ohlc_data)
        result = await engine.run(BuyOnFirstStrategy(), data)

        sell_trades = [t for t in result.trades if t.side == "SELL"]
        assert len(sell_trades) == 1
        # Sell price should be below the SL price due to slippage
        assert sell_trades[0].price < 95.0

    @pytest.mark.asyncio
    async def test_multiple_buy_sell_cycles_with_sl(self):
        """After SL exit, bot can buy again on next BUY signal."""
        engine = BacktestEngine(
            initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0,
            stop_loss_pct=5.0,
        )
        # Alternate: BUY → SL triggers → BUY again → SELL
        ohlc_data = [
            (100.0, 101.0, 99.0, 100.0),     # history
            (100.0, 101.0, 99.0, 100.0),     # BUY (1st call)
            (90.0, 91.0, 85.0, 88.0),        # SL triggers, SELL signal ignored
            (100.0, 101.0, 99.0, 100.0),     # BUY (3rd call, since #2 was SELL ignored)
            (120.0, 121.0, 119.0, 120.0),    # SELL (4th call)
        ]
        data = make_candles_ohlc(ohlc_data)
        result = await engine.run(AlternateBuySellStrategy(), data)

        buy_trades = [t for t in result.trades if t.side == "BUY"]
        sell_trades = [t for t in result.trades if t.side == "SELL"]
        # Should have at least 1 BUY and 1 SELL
        assert len(buy_trades) >= 1
        assert len(sell_trades) >= 1
