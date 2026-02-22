"""Tests for V2-027: Comprehensive backtesting CLI with strategy comparison."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.backtest.__main__ import (
    async_main,
    build_parser,
    calculate_profit_factor,
    export_csv,
    export_json,
    format_comparison_table,
    format_single_result,
    get_backtestable_strategies,
    load_data,
    parse_date,
    run_backtest,
)
from bot.backtest.engine import BacktestEngine, BacktestResult, TradeLog
from bot.models import OHLCV, SignalAction, TradingSignal
from bot.monitoring.metrics import PerformanceMetrics
from bot.strategies.base import BaseStrategy, strategy_registry

# --- Test helpers ---

class SimpleBuyStrategy(BaseStrategy):
    """Test strategy that buys once then holds."""

    def __init__(self, name_val: str = "simple_buy"):
        self._name = name_val
        self._bought = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_history_length(self) -> int:
        return 1

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        if not self._bought:
            self._bought = True
            return TradingSignal(
                strategy_name=self._name,
                symbol=kwargs.get("symbol", "BTC/USDT"),
                action=SignalAction.BUY,
                confidence=0.9,
            )
        return TradingSignal(
            strategy_name=self._name,
            symbol=kwargs.get("symbol", "BTC/USDT"),
            action=SignalAction.HOLD,
            confidence=0.5,
        )


class AlternateBuySellStrategy(BaseStrategy):
    """Strategy that alternates BUY/SELL for generating trades."""

    def __init__(self, name_val: str = "alternator"):
        self._name = name_val
        self._count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_history_length(self) -> int:
        return 1

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        self._count += 1
        action = SignalAction.BUY if self._count % 2 == 1 else SignalAction.SELL
        return TradingSignal(
            strategy_name=self._name,
            symbol=kwargs.get("symbol", "BTC/USDT"),
            action=action,
            confidence=0.8,
        )


def make_candles(
    prices: list[float],
    start_time: datetime | None = None,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
) -> list[OHLCV]:
    """Create OHLCV candles from a list of close prices."""
    base = start_time or datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i, price in enumerate(prices):
        candles.append(OHLCV(
            timestamp=base + timedelta(hours=i),
            open=price,
            high=price * 1.01,
            low=price * 0.99,
            close=price,
            volume=1000.0,
            symbol=symbol,
            timeframe=timeframe,
        ))
    return candles


def make_result(
    strategy_name: str = "test_strategy",
    total_return_pct: float = 10.0,
    sharpe_ratio: float = 1.5,
    max_drawdown_pct: float = 5.0,
    win_rate: float = 60.0,
    total_trades: int = 10,
    winning_trades: int = 6,
    losing_trades: int = 4,
    trades: list[TradeLog] | None = None,
    final_value: float = 11000.0,
) -> BacktestResult:
    """Create a BacktestResult for testing."""
    return BacktestResult(
        strategy_name=strategy_name,
        symbol="BTC/USDT",
        timeframe="1h",
        metrics=PerformanceMetrics(
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
        ),
        trades=trades or [],
        final_portfolio_value=final_value,
    )


# --- Tests ---


class TestBuildParser:
    """Tests for CLI argument parser."""

    def test_default_arguments(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.strategy == "all"
        assert args.symbol == "BTC/USDT"
        assert args.timeframe == "1h"
        assert args.start_date is None
        assert args.end_date is None
        assert args.initial_capital == 10000.0
        assert args.fee_pct == 0.1
        assert args.slippage_pct == 0.05
        assert args.stop_loss_pct == 0.0
        assert args.take_profit_pct == 0.0
        assert args.output_json is None
        assert args.output_csv is None

    def test_custom_arguments(self):
        parser = build_parser()
        args = parser.parse_args([
            "--strategy", "rsi",
            "--symbol", "ETH/USDT",
            "--timeframe", "4h",
            "--start-date", "2024-01-01",
            "--end-date", "2024-06-30",
            "--initial-capital", "50000",
            "--fee-pct", "0.05",
            "--slippage-pct", "0.1",
            "--stop-loss-pct", "3.0",
            "--take-profit-pct", "5.0",
            "--output-json", "results.json",
            "--output-csv", "trades.csv",
        ])
        assert args.strategy == "rsi"
        assert args.symbol == "ETH/USDT"
        assert args.timeframe == "4h"
        assert args.start_date == "2024-01-01"
        assert args.end_date == "2024-06-30"
        assert args.initial_capital == 50000.0
        assert args.fee_pct == 0.05
        assert args.slippage_pct == 0.1
        assert args.stop_loss_pct == 3.0
        assert args.take_profit_pct == 5.0
        assert args.output_json == "results.json"
        assert args.output_csv == "trades.csv"

    def test_database_url_argument(self):
        parser = build_parser()
        args = parser.parse_args(["--database-url", "sqlite+aiosqlite:///test.db"])
        assert args.database_url == "sqlite+aiosqlite:///test.db"

    def test_database_url_default(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.database_url == "sqlite+aiosqlite:///data/trading.db"


class TestParseDate:
    """Tests for date parsing."""

    def test_valid_date(self):
        dt = parse_date("2024-06-15")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.day == 15
        assert dt.tzinfo == timezone.utc

    def test_none_returns_none(self):
        assert parse_date(None) is None

    def test_invalid_date_exits(self):
        with pytest.raises(SystemExit):
            parse_date("not-a-date")

    def test_wrong_format_exits(self):
        with pytest.raises(SystemExit):
            parse_date("06/15/2024")


class TestCalculateProfitFactor:
    """Tests for profit factor calculation."""

    def test_profit_factor_with_wins_and_losses(self):
        bt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        trades = [
            TradeLog(
                timestamp=bt, symbol="BTC/USDT",
                side="SELL", price=100, quantity=1, pnl=500,
            ),
            TradeLog(
                timestamp=bt, symbol="BTC/USDT",
                side="SELL", price=100, quantity=1, pnl=300,
            ),
            TradeLog(
                timestamp=bt, symbol="BTC/USDT",
                side="SELL", price=100, quantity=1, pnl=-200,
            ),
        ]
        pf = calculate_profit_factor(trades)
        assert pf == pytest.approx(4.0)  # 800 / 200

    def test_profit_factor_no_losses(self):
        bt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        trades = [
            TradeLog(
                timestamp=bt, symbol="BTC/USDT",
                side="SELL", price=100, quantity=1, pnl=500,
            ),
        ]
        pf = calculate_profit_factor(trades)
        assert pf == float("inf")

    def test_profit_factor_no_profits(self):
        bt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        trades = [
            TradeLog(
                timestamp=bt, symbol="BTC/USDT",
                side="SELL", price=100, quantity=1, pnl=-500,
            ),
        ]
        pf = calculate_profit_factor(trades)
        assert pf == 0.0

    def test_profit_factor_no_trades(self):
        pf = calculate_profit_factor([])
        assert pf == 0.0

    def test_profit_factor_ignores_buy_trades(self):
        bt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        trades = [
            TradeLog(
                timestamp=bt, symbol="BTC/USDT",
                side="BUY", price=100, quantity=1, pnl=0,
            ),
            TradeLog(
                timestamp=bt, symbol="BTC/USDT",
                side="SELL", price=120, quantity=1, pnl=200,
            ),
        ]
        pf = calculate_profit_factor(trades)
        # Only one winning SELL, no losing SELLs
        assert pf == float("inf")


class TestFormatComparisonTable:
    """Tests for comparison table formatting."""

    def test_table_has_headers(self):
        results = [make_result("strategy_a", sharpe_ratio=1.0)]
        table = format_comparison_table(results)
        assert "Strategy" in table
        assert "Total Return %" in table
        assert "Sharpe Ratio" in table
        assert "Max Drawdown %" in table
        assert "Win Rate %" in table
        assert "Total Trades" in table
        assert "Profit Factor" in table

    def test_table_contains_strategy_name(self):
        results = [make_result("my_strat", sharpe_ratio=2.0)]
        table = format_comparison_table(results)
        assert "my_strat" in table

    def test_table_sorted_by_sharpe_desc(self):
        results = [
            make_result("low_sharpe", sharpe_ratio=0.5),
            make_result("high_sharpe", sharpe_ratio=2.0),
            make_result("mid_sharpe", sharpe_ratio=1.0),
        ]
        table = format_comparison_table(results)
        lines = table.strip().split("\n")
        # Data rows: header row is index 1, separator at 0 and 2, data starts at 3
        data_rows = [
            row for row in lines
            if "|" in row and "Strategy" not in row
        ]
        # First data row should be high_sharpe
        assert "high_sharpe" in data_rows[0]
        # Last data row should be low_sharpe
        assert "low_sharpe" in data_rows[-1]

    def test_table_multiple_results(self):
        results = [
            make_result("strat_a", sharpe_ratio=1.5, total_return_pct=15.0),
            make_result("strat_b", sharpe_ratio=0.8, total_return_pct=5.0),
        ]
        table = format_comparison_table(results)
        assert "strat_a" in table
        assert "strat_b" in table
        assert "15.00" in table
        assert "5.00" in table


class TestFormatSingleResult:
    """Tests for single result formatting."""

    def test_contains_all_fields(self):
        result = make_result(
            "rsi",
            total_return_pct=12.5,
            sharpe_ratio=1.2,
            max_drawdown_pct=4.3,
            win_rate=55.0,
            total_trades=20,
            winning_trades=11,
            losing_trades=9,
            final_value=11250.0,
        )
        output = format_single_result(result)
        assert "rsi" in output
        assert "12.50%" in output
        assert "1.2000" in output
        assert "4.30%" in output
        assert "55.00%" in output
        assert "20" in output
        assert "11" in output
        assert "9" in output
        assert "11250.00" in output


class TestExportJson:
    """Tests for JSON export."""

    def test_single_result_json(self):
        result = make_result("test_strat")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            export_json([result], filepath)
            with open(filepath) as f:
                data = json.load(f)
            assert data["strategy_name"] == "test_strat"
            assert "metrics" in data
            assert data["metrics"]["total_return_pct"] == 10.0
        finally:
            os.unlink(filepath)

    def test_multiple_results_json(self):
        results = [
            make_result("strat_a", sharpe_ratio=2.0),
            make_result("strat_b", sharpe_ratio=1.0),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            export_json(results, filepath)
            with open(filepath) as f:
                data = json.load(f)
            assert "comparison" in data
            assert len(data["comparison"]) == 2
            # Sorted by sharpe desc
            assert data["comparison"][0]["strategy_name"] == "strat_a"
            assert data["comparison"][1]["strategy_name"] == "strat_b"
        finally:
            os.unlink(filepath)


class TestExportCsv:
    """Tests for CSV export."""

    def test_single_result_csv(self):
        bt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        trades = [
            TradeLog(
                timestamp=bt, symbol="BTC/USDT",
                side="BUY", price=100, quantity=1,
            ),
            TradeLog(
                timestamp=bt + timedelta(hours=1),
                symbol="BTC/USDT",
                side="SELL", price=110, quantity=1, pnl=10,
            ),
        ]
        result = make_result("test_strat", trades=trades)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name
        try:
            export_csv([result], filepath)
            with open(filepath) as f:
                content = f.read()
            assert "timestamp" in content
            assert "BTC/USDT" in content
            assert "BUY" in content
            assert "SELL" in content
        finally:
            os.unlink(filepath)

    def test_multiple_results_csv_has_strategy_column(self):
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        trades_a = [
            TradeLog(timestamp=base_time, symbol="BTC/USDT", side="BUY", price=100, quantity=1),
        ]
        trades_b = [
            TradeLog(timestamp=base_time, symbol="BTC/USDT", side="BUY", price=200, quantity=0.5),
        ]
        results = [
            make_result("strat_a", trades=trades_a),
            make_result("strat_b", trades=trades_b),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name
        try:
            export_csv(results, filepath)
            with open(filepath) as f:
                content = f.read()
            # Multi-strategy CSV has strategy column
            assert "strategy" in content
            assert "strat_a" in content
            assert "strat_b" in content
        finally:
            os.unlink(filepath)


class TestGetBacktestableStrategies:
    """Tests for strategy lookup."""

    def setup_method(self):
        """Save registry state and populate with test strategies."""
        self._saved_strategies = dict(strategy_registry._strategies)
        self._saved_active = set(strategy_registry._active)

    def teardown_method(self):
        """Restore registry state."""
        strategy_registry._strategies = self._saved_strategies
        strategy_registry._active = self._saved_active

    def test_get_specific_strategy(self):
        strat = SimpleBuyStrategy("my_test_strategy")
        strategy_registry.register(strat)
        result = get_backtestable_strategies("my_test_strategy")
        assert len(result) == 1
        assert result[0].name == "my_test_strategy"

    def test_get_nonexistent_strategy_exits(self):
        with pytest.raises(SystemExit):
            get_backtestable_strategies("nonexistent_xyz_strategy")

    def test_get_all_skips_arbitrage_dca_funding(self):
        strategy_registry.clear()
        # Register a few test strategies
        strategy_registry.register(SimpleBuyStrategy("rsi"))
        strategy_registry.register(SimpleBuyStrategy("macd"))
        # Register strategies that should be skipped
        strategy_registry.register(SimpleBuyStrategy("arbitrage"))
        strategy_registry.register(SimpleBuyStrategy("dca"))
        strategy_registry.register(SimpleBuyStrategy("funding_rate"))

        result = get_backtestable_strategies("all")
        names = [s.name for s in result]
        assert "rsi" in names
        assert "macd" in names
        assert "arbitrage" not in names
        assert "dca" not in names
        assert "funding_rate" not in names

    def test_get_all_with_no_strategies_exits(self):
        strategy_registry.clear()
        with pytest.raises(SystemExit):
            get_backtestable_strategies("all")


class TestLoadData:
    """Tests for data loading from DataStore."""

    @pytest.mark.asyncio
    async def test_load_data_returns_candles(self):
        candles = make_candles([100.0, 101.0, 102.0])
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=candles)
        mock_store.close = AsyncMock()

        with patch("bot.backtest.__main__.DataStore", return_value=mock_store):
            result = await load_data(
                database_url="sqlite+aiosqlite:///test.db",
                symbol="BTC/USDT",
                timeframe="1h",
                start_date=None,
                end_date=None,
            )
        assert len(result) == 3
        mock_store.initialize.assert_called_once()
        mock_store.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_data_passes_date_filters(self):
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=[])
        mock_store.close = AsyncMock()

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 30, tzinfo=timezone.utc)

        with patch("bot.backtest.__main__.DataStore", return_value=mock_store):
            await load_data(
                database_url="sqlite+aiosqlite:///test.db",
                symbol="ETH/USDT",
                timeframe="4h",
                start_date=start,
                end_date=end,
            )
        mock_store.get_candles.assert_called_once_with(
            symbol="ETH/USDT",
            timeframe="4h",
            start=start,
            end=end,
            limit=10000,
        )

    @pytest.mark.asyncio
    async def test_load_data_closes_store_on_error(self):
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.get_candles = AsyncMock(side_effect=Exception("DB error"))
        mock_store.close = AsyncMock()

        with patch("bot.backtest.__main__.DataStore", return_value=mock_store):
            with pytest.raises(Exception, match="DB error"):
                await load_data(
                    database_url="sqlite+aiosqlite:///test.db",
                    symbol="BTC/USDT",
                    timeframe="1h",
                    start_date=None,
                    end_date=None,
                )
        # Store is still closed via finally
        mock_store.close.assert_called_once()


class TestRunBacktest:
    """Tests for the run_backtest function."""

    def setup_method(self):
        """Save registry state."""
        self._saved_strategies = dict(strategy_registry._strategies)
        self._saved_active = set(strategy_registry._active)

    def teardown_method(self):
        """Restore registry state."""
        strategy_registry._strategies = self._saved_strategies
        strategy_registry._active = self._saved_active

    @pytest.mark.asyncio
    async def test_run_single_strategy(self):
        strategy_registry.clear()
        strategy_registry.register(AlternateBuySellStrategy("test_alternator"))

        candles = make_candles([100.0] * 20)
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=candles)
        mock_store.close = AsyncMock()

        parser = build_parser()
        args = parser.parse_args(["--strategy", "test_alternator"])

        with patch("bot.backtest.__main__.DataStore", return_value=mock_store):
            results = await run_backtest(args)

        assert len(results) == 1
        assert results[0].strategy_name == "test_alternator"

    @pytest.mark.asyncio
    async def test_run_all_strategies(self):
        strategy_registry.clear()
        strategy_registry.register(AlternateBuySellStrategy("strat1"))
        strategy_registry.register(AlternateBuySellStrategy("strat2"))

        candles = make_candles([100.0] * 20)
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=candles)
        mock_store.close = AsyncMock()

        parser = build_parser()
        args = parser.parse_args(["--strategy", "all"])

        with patch("bot.backtest.__main__.DataStore", return_value=mock_store):
            results = await run_backtest(args)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_run_with_no_data_exits(self):
        strategy_registry.clear()
        strategy_registry.register(SimpleBuyStrategy("test"))

        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=[])
        mock_store.close = AsyncMock()

        parser = build_parser()
        args = parser.parse_args(["--strategy", "test"])

        with patch("bot.backtest.__main__.DataStore", return_value=mock_store):
            with pytest.raises(SystemExit):
                await run_backtest(args)

    @pytest.mark.asyncio
    async def test_run_with_custom_capital_and_fees(self):
        strategy_registry.clear()
        strategy_registry.register(AlternateBuySellStrategy("test_strat"))

        candles = make_candles([100.0] * 20)
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=candles)
        mock_store.close = AsyncMock()

        parser = build_parser()
        args = parser.parse_args([
            "--strategy", "test_strat",
            "--initial-capital", "50000",
            "--fee-pct", "0.05",
        ])

        with patch("bot.backtest.__main__.DataStore", return_value=mock_store):
            results = await run_backtest(args)

        assert len(results) == 1
        # With fee=0.05% and flat prices, we should get some final value
        assert results[0].final_portfolio_value > 0

    @pytest.mark.asyncio
    async def test_run_handles_strategy_error_gracefully(self):
        """If one strategy errors, others should still run."""
        strategy_registry.clear()

        # Create a strategy that raises
        class ErrorStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "error_strat"

            @property
            def required_history_length(self) -> int:
                return 1

            async def analyze(self, ohlcv_data, **kwargs):
                raise RuntimeError("Strategy exploded")

        strategy_registry.register(ErrorStrategy())
        strategy_registry.register(AlternateBuySellStrategy("good_strat"))

        candles = make_candles([100.0] * 20)
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=candles)
        mock_store.close = AsyncMock()

        parser = build_parser()
        args = parser.parse_args(["--strategy", "all"])

        with patch("bot.backtest.__main__.DataStore", return_value=mock_store):
            results = await run_backtest(args)

        # Only the good strategy should have results
        assert len(results) == 1
        assert results[0].strategy_name == "good_strat"


class TestAsyncMain:
    """Tests for the main CLI entry point."""

    def setup_method(self):
        """Save registry state."""
        self._saved_strategies = dict(strategy_registry._strategies)
        self._saved_active = set(strategy_registry._active)

    def teardown_method(self):
        """Restore registry state."""
        strategy_registry._strategies = self._saved_strategies
        strategy_registry._active = self._saved_active

    @pytest.mark.asyncio
    async def test_main_with_single_strategy(self, capsys):
        strategy_registry.clear()
        strategy_registry.register(AlternateBuySellStrategy("my_strat"))

        candles = make_candles([100.0] * 20)
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=candles)
        mock_store.close = AsyncMock()

        parser = build_parser()
        args = parser.parse_args(["--strategy", "my_strat"])

        with patch("bot.backtest.__main__.DataStore", return_value=mock_store):
            await async_main(args)

        captured = capsys.readouterr()
        assert "my_strat" in captured.out
        assert "Total Return" in captured.out

    @pytest.mark.asyncio
    async def test_main_with_all_strategies_shows_table(self, capsys):
        strategy_registry.clear()
        strategy_registry.register(AlternateBuySellStrategy("strat_a"))
        strategy_registry.register(AlternateBuySellStrategy("strat_b"))

        candles = make_candles([100.0] * 20)
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=candles)
        mock_store.close = AsyncMock()

        parser = build_parser()
        args = parser.parse_args(["--strategy", "all"])

        with patch("bot.backtest.__main__.DataStore", return_value=mock_store):
            await async_main(args)

        captured = capsys.readouterr()
        assert "strat_a" in captured.out
        assert "strat_b" in captured.out
        # Table separators
        assert "+" in captured.out

    @pytest.mark.asyncio
    async def test_main_with_json_export(self):
        strategy_registry.clear()
        strategy_registry.register(AlternateBuySellStrategy("export_strat"))

        candles = make_candles([100.0] * 20)
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=candles)
        mock_store.close = AsyncMock()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        parser = build_parser()
        args = parser.parse_args([
            "--strategy", "export_strat",
            "--output-json", json_path,
        ])

        try:
            with patch("bot.backtest.__main__.DataStore", return_value=mock_store):
                await async_main(args)

            with open(json_path) as f:
                data = json.load(f)
            assert data["strategy_name"] == "export_strat"
        finally:
            os.unlink(json_path)

    @pytest.mark.asyncio
    async def test_main_with_csv_export(self):
        strategy_registry.clear()
        strategy_registry.register(AlternateBuySellStrategy("csv_strat"))

        candles = make_candles([100.0] * 20)
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.get_candles = AsyncMock(return_value=candles)
        mock_store.close = AsyncMock()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name

        parser = build_parser()
        args = parser.parse_args([
            "--strategy", "csv_strat",
            "--output-csv", csv_path,
        ])

        try:
            with patch("bot.backtest.__main__.DataStore", return_value=mock_store):
                await async_main(args)

            with open(csv_path) as f:
                content = f.read()
            assert "timestamp" in content
        finally:
            os.unlink(csv_path)


class TestEquityCurveInResults:
    """Tests that equity curve data is generated."""

    @pytest.mark.asyncio
    async def test_equity_curve_populated(self):
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        candles = make_candles([100.0, 105.0, 110.0, 108.0, 112.0] * 4)
        strategy = AlternateBuySellStrategy("eq_strat")
        result = await engine.run(strategy, candles)
        # Equity curve should have values for every candle after min_history
        assert len(result.equity_curve) > 0
        assert result.equity_curve[0] == 10000.0

    @pytest.mark.asyncio
    async def test_equity_curve_in_json_export(self):
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        candles = make_candles([100.0, 105.0, 110.0] * 4)
        strategy = AlternateBuySellStrategy("eq_json_strat")
        result = await engine.run(strategy, candles)
        json_data = json.loads(result.to_json())
        assert "equity_curve" in json_data
        assert len(json_data["equity_curve"]) > 0


class TestStrategyComparisonViaEngine:
    """Tests for BacktestEngine.compare() used by CLI."""

    @pytest.mark.asyncio
    async def test_compare_returns_sorted_results(self):
        engine = BacktestEngine(initial_capital=10000.0, fee_pct=0.0, slippage_pct=0.0)
        candles = make_candles([100.0, 110.0, 105.0, 115.0, 108.0] * 4)
        strategies = [
            AlternateBuySellStrategy("strat1"),
            AlternateBuySellStrategy("strat2"),
        ]
        results = await engine.compare(strategies, candles)
        assert len(results) == 2
        # Results should be sorted by sharpe ratio descending
        assert results[0].metrics.sharpe_ratio >= results[1].metrics.sharpe_ratio
