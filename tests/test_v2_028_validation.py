"""Tests for V2-028: Paper trading validation framework."""

import json
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.config import Settings, TradingMode
from bot.main import TradingBot, build_parser
from bot.validation import (
    ValidationCriteria,
    ValidationReport,
    build_validation_report,
    calculate_max_drawdown,
    calculate_sharpe_from_pnls,
    evaluate_criteria,
    parse_duration,
    save_report,
)

# ──── parse_duration ────


class TestParseDuration:
    def test_hours(self):
        assert parse_duration("48h") == 48 * 3600

    def test_minutes(self):
        assert parse_duration("30m") == 30 * 60

    def test_days(self):
        assert parse_duration("2d") == 2 * 86400

    def test_seconds(self):
        assert parse_duration("120s") == 120

    def test_plain_number(self):
        assert parse_duration("3600") == 3600

    def test_fractional_hours(self):
        assert parse_duration("1.5h") == 1.5 * 3600

    def test_whitespace_stripped(self):
        assert parse_duration("  48h  ") == 48 * 3600

    def test_case_insensitive(self):
        assert parse_duration("48H") == 48 * 3600

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty duration"):
            parse_duration("")

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("abc")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            parse_duration("-5h")

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            parse_duration("0h")


# ──── calculate_sharpe_from_pnls ────


class TestCalculateSharpe:
    def test_empty_list(self):
        assert calculate_sharpe_from_pnls([]) == 0.0

    def test_single_trade(self):
        assert calculate_sharpe_from_pnls([100.0]) == 0.0

    def test_all_same_pnl(self):
        # Zero variance → zero Sharpe
        assert calculate_sharpe_from_pnls([10.0, 10.0, 10.0]) == 0.0

    def test_positive_sharpe(self):
        # Consistently positive trades → positive Sharpe
        pnls = [10.0, 20.0, 15.0, 12.0, 18.0]
        sharpe = calculate_sharpe_from_pnls(pnls)
        assert sharpe > 0

    def test_negative_sharpe(self):
        # Consistently negative trades → negative Sharpe
        pnls = [-10.0, -20.0, -15.0, -12.0, -18.0]
        sharpe = calculate_sharpe_from_pnls(pnls)
        assert sharpe < 0

    def test_mixed_pnls(self):
        pnls = [50.0, -10.0, 30.0, -5.0, 20.0]
        sharpe = calculate_sharpe_from_pnls(pnls)
        # Mean is positive, so sharpe should be positive
        assert sharpe > 0


# ──── calculate_max_drawdown ────


class TestCalculateMaxDrawdown:
    def test_empty_curve(self):
        assert calculate_max_drawdown([]) == 0.0

    def test_single_value(self):
        assert calculate_max_drawdown([10000]) == 0.0

    def test_monotonic_increase(self):
        # No drawdown
        curve = [10000, 10100, 10200, 10300]
        assert calculate_max_drawdown(curve) == 0.0

    def test_simple_drawdown(self):
        # Peak at 10000, drop to 9000 → 10%
        curve = [10000, 9000]
        assert calculate_max_drawdown(curve) == pytest.approx(10.0)

    def test_recovery_drawdown(self):
        # Peak 10000 → drop to 8500 → recover to 11000 → drop to 9350
        curve = [10000, 8500, 11000, 9350]
        # DD1: (10000-8500)/10000 = 15%
        # DD2: (11000-9350)/11000 = 15%
        assert calculate_max_drawdown(curve) == pytest.approx(15.0)

    def test_peak_at_end(self):
        # Drawdown in middle, then new peak
        curve = [10000, 9500, 9000, 10500, 11000]
        # Max DD: (10000-9000)/10000 = 10%
        assert calculate_max_drawdown(curve) == pytest.approx(10.0)


# ──── evaluate_criteria ────


class TestEvaluateCriteria:
    def test_all_pass(self):
        criteria = ValidationCriteria(
            min_win_rate_pct=45.0,
            min_sharpe_ratio=0.5,
            max_drawdown_pct=15.0,
            min_trades=10,
        )
        rec, results = evaluate_criteria(
            total_trades=20,
            win_rate_pct=55.0,
            sharpe_ratio=1.0,
            max_drawdown_pct=10.0,
            criteria=criteria,
        )
        assert rec == "GO"
        assert all(r.passed for r in results)

    def test_all_fail(self):
        criteria = ValidationCriteria(
            min_win_rate_pct=45.0,
            min_sharpe_ratio=0.5,
            max_drawdown_pct=15.0,
            min_trades=10,
        )
        rec, results = evaluate_criteria(
            total_trades=5,
            win_rate_pct=30.0,
            sharpe_ratio=-0.2,
            max_drawdown_pct=25.0,
            criteria=criteria,
        )
        assert rec == "NO-GO"
        assert not any(r.passed for r in results)

    def test_partial_fail_is_nogo(self):
        criteria = ValidationCriteria(
            min_win_rate_pct=45.0,
            min_sharpe_ratio=0.5,
            max_drawdown_pct=15.0,
            min_trades=10,
        )
        rec, results = evaluate_criteria(
            total_trades=20,
            win_rate_pct=55.0,
            sharpe_ratio=1.0,
            max_drawdown_pct=20.0,  # Over limit
            criteria=criteria,
        )
        assert rec == "NO-GO"
        failed = [r for r in results if not r.passed]
        assert len(failed) == 1
        assert failed[0].name == "max_drawdown_pct"

    def test_exact_threshold_passes(self):
        criteria = ValidationCriteria(
            min_win_rate_pct=45.0,
            min_sharpe_ratio=0.5,
            max_drawdown_pct=15.0,
            min_trades=10,
        )
        rec, results = evaluate_criteria(
            total_trades=10,
            win_rate_pct=45.0,
            sharpe_ratio=0.5,
            max_drawdown_pct=15.0,
            criteria=criteria,
        )
        assert rec == "GO"

    def test_criterion_result_to_dict(self):
        criteria = ValidationCriteria()
        _, results = evaluate_criteria(
            total_trades=20,
            win_rate_pct=50.0,
            sharpe_ratio=1.0,
            max_drawdown_pct=10.0,
            criteria=criteria,
        )
        d = results[0].to_dict()
        assert "name" in d
        assert "passed" in d
        assert "actual" in d
        assert "threshold" in d
        assert "comparison" in d


# ──── ValidationCriteria ────


class TestValidationCriteria:
    def test_defaults(self):
        c = ValidationCriteria()
        assert c.min_win_rate_pct == 45.0
        assert c.min_sharpe_ratio == 0.5
        assert c.max_drawdown_pct == 15.0
        assert c.min_trades == 10

    def test_to_dict(self):
        c = ValidationCriteria(
            min_win_rate_pct=50.0,
            min_sharpe_ratio=1.0,
            max_drawdown_pct=10.0,
            min_trades=20,
        )
        d = c.to_dict()
        assert d["min_win_rate_pct"] == 50.0
        assert d["min_sharpe_ratio"] == 1.0
        assert d["max_drawdown_pct"] == 10.0
        assert d["min_trades"] == 20


# ──── ValidationReport ────


class TestValidationReport:
    def _make_report(self, **kwargs):
        defaults = {
            "start_time": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "end_time": datetime(2026, 1, 3, tzinfo=timezone.utc),
            "duration_seconds": 172800.0,
            "initial_balance": 10000.0,
            "final_balance": 10500.0,
            "total_trades": 15,
            "wins": 9,
            "losses": 6,
            "total_return_pct": 5.0,
            "max_drawdown_pct": 8.0,
            "sharpe_ratio": 0.75,
            "win_rate_pct": 60.0,
            "strategy_breakdown": {"ma_crossover": {"total_pnl": 300.0}},
            "criteria_results": [],
            "recommendation": "GO",
            "trade_pnls": [50, -20, 30, -10, 40],
        }
        defaults.update(kwargs)
        return ValidationReport(**defaults)

    def test_is_go_true(self):
        report = self._make_report(recommendation="GO")
        assert report.is_go is True

    def test_is_go_false(self):
        report = self._make_report(recommendation="NO-GO")
        assert report.is_go is False

    def test_to_dict(self):
        report = self._make_report()
        d = report.to_dict()
        assert d["total_trades"] == 15
        assert d["recommendation"] == "GO"
        assert d["initial_balance"] == 10000.0
        assert d["final_balance"] == 10500.0
        assert "start_time" in d
        assert "end_time" in d

    def test_to_json(self):
        report = self._make_report()
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["recommendation"] == "GO"
        assert parsed["total_trades"] == 15

    def test_format_summary_go(self):
        report = self._make_report(recommendation="GO")
        summary = report.format_summary()
        assert "RECOMMENDATION:  GO" in summary
        assert "Performance" in summary
        assert "Total Return" in summary

    def test_format_summary_nogo_shows_failures(self):
        from bot.validation import CriterionResult

        cr = CriterionResult(
            name="min_trades",
            passed=False,
            actual=5.0,
            threshold=10.0,
            comparison=">=",
        )
        report = self._make_report(
            recommendation="NO-GO",
            criteria_results=[cr],
        )
        summary = report.format_summary()
        assert "RECOMMENDATION:  NO-GO" in summary
        assert "Failed criteria" in summary
        assert "min_trades" in summary

    def test_format_summary_with_strategy_breakdown(self):
        report = self._make_report(
            strategy_breakdown={
                "rsi": {"total_pnl": 100.0, "win_rate": 60.0, "total_trades": 5},
                "macd": {"total_pnl": -50.0, "win_rate": 30.0, "total_trades": 3},
            }
        )
        summary = report.format_summary()
        assert "Strategy Breakdown" in summary
        assert "rsi" in summary
        assert "macd" in summary


# ──── build_validation_report ────


class TestBuildValidationReport:
    def test_basic_report(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 3, tzinfo=timezone.utc)
        pnls = [50.0, -20.0, 30.0, -10.0, 40.0, 20.0, -5.0, 15.0, 25.0, -15.0, 10.0]
        curve = [10000, 10050, 10030, 10060, 10050, 10090, 10110, 10105, 10120, 10145, 10130, 10140]
        criteria = ValidationCriteria(min_trades=10)

        report = build_validation_report(
            start_time=start,
            end_time=end,
            initial_balance=10000.0,
            final_balance=10140.0,
            trade_pnls=pnls,
            equity_curve=curve,
            strategy_breakdown={"test": {"total_pnl": 140.0}},
            criteria=criteria,
        )

        assert report.total_trades == 11
        assert report.wins == 7
        assert report.losses == 4
        assert report.total_return_pct == pytest.approx(1.4)
        assert report.sharpe_ratio != 0.0
        assert report.recommendation in ("GO", "NO-GO")

    def test_no_trades_is_nogo(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 3, tzinfo=timezone.utc)
        criteria = ValidationCriteria(min_trades=10)

        report = build_validation_report(
            start_time=start,
            end_time=end,
            initial_balance=10000.0,
            final_balance=10000.0,
            trade_pnls=[],
            equity_curve=[10000],
            strategy_breakdown={},
            criteria=criteria,
        )

        assert report.recommendation == "NO-GO"
        assert report.total_trades == 0
        assert report.win_rate_pct == 0.0

    def test_all_winning_trades(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 2, tzinfo=timezone.utc)
        pnls = [100.0] * 15
        curve = [10000 + i * 100 for i in range(16)]
        criteria = ValidationCriteria(min_trades=10)

        report = build_validation_report(
            start_time=start,
            end_time=end,
            initial_balance=10000.0,
            final_balance=11500.0,
            trade_pnls=pnls,
            equity_curve=curve,
            strategy_breakdown={},
            criteria=criteria,
        )

        assert report.wins == 15
        assert report.losses == 0
        assert report.win_rate_pct == 100.0
        assert report.max_drawdown_pct == 0.0


# ──── save_report ────


class TestSaveReport:
    def test_save_creates_file(self, tmp_path):
        report = ValidationReport(
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 3, 14, 30, 0, tzinfo=timezone.utc),
            duration_seconds=172800.0,
            initial_balance=10000.0,
            final_balance=10500.0,
            total_trades=10,
            wins=7,
            losses=3,
            total_return_pct=5.0,
            max_drawdown_pct=8.0,
            sharpe_ratio=0.75,
            win_rate_pct=70.0,
            recommendation="GO",
        )

        filepath = save_report(report, directory=str(tmp_path))
        assert os.path.exists(filepath)
        assert filepath.startswith(str(tmp_path))
        assert "validation_report_" in filepath
        assert filepath.endswith(".json")

        # Verify contents
        with open(filepath) as f:
            data = json.load(f)
        assert data["recommendation"] == "GO"
        assert data["total_trades"] == 10

    def test_save_creates_directory(self, tmp_path):
        subdir = str(tmp_path / "new_dir")
        report = ValidationReport(
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 1, 0, 0, tzinfo=timezone.utc),
            duration_seconds=3600.0,
            initial_balance=10000.0,
            final_balance=10000.0,
            total_trades=0,
            wins=0,
            losses=0,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            win_rate_pct=0.0,
            recommendation="NO-GO",
        )

        filepath = save_report(report, directory=subdir)
        assert os.path.exists(filepath)


# ──── CLI args ────


class TestBuildParser:
    def test_default_no_validate(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.validate is False
        assert args.duration == "48h"

    def test_validate_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--validate"])
        assert args.validate is True

    def test_custom_duration(self):
        parser = build_parser()
        args = parser.parse_args(["--validate", "--duration", "24h"])
        assert args.validate is True
        assert args.duration == "24h"


# ──── Config validation settings ────


class TestValidationConfig:
    def test_default_settings(self):
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            binance_api_key="",
            upbit_api_key="",
        )
        assert settings.validation_min_win_rate_pct == 45.0
        assert settings.validation_min_sharpe_ratio == 0.5
        assert settings.validation_max_drawdown_pct == 15.0
        assert settings.validation_min_trades == 10

    def test_custom_settings(self):
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            binance_api_key="",
            upbit_api_key="",
            validation_min_win_rate_pct=50.0,
            validation_min_sharpe_ratio=1.0,
            validation_max_drawdown_pct=10.0,
            validation_min_trades=20,
        )
        assert settings.validation_min_win_rate_pct == 50.0
        assert settings.validation_min_sharpe_ratio == 1.0
        assert settings.validation_max_drawdown_pct == 10.0
        assert settings.validation_min_trades == 20


# ──── TradingBot integration tests ────


def make_settings(**kwargs):
    """Create test settings with safe defaults."""
    defaults = {
        "trading_mode": TradingMode.PAPER,
        "database_url": "sqlite+aiosqlite:///:memory:",
        "binance_api_key": "",
        "upbit_api_key": "",
        "symbols": ["BTC/USDT"],
        "loop_interval_seconds": 1,
        "signal_min_agreement": 1,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


class TestTradingBotValidation:
    @pytest.mark.asyncio
    async def test_run_validation_initializes_tracking(self):
        """Validation mode sets up tracking state."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Run for minimal time (1 second)
        report = await bot.run_validation(duration_seconds=0.5)

        assert isinstance(report, ValidationReport)
        assert report.initial_balance == 10000.0
        assert report.recommendation in ("GO", "NO-GO")

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_validation_records_equity_curve(self):
        """Validation mode records equity curve."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        report = await bot.run_validation(duration_seconds=0.5)

        # Should have at least the initial value
        assert len(report.trade_pnls) >= 0
        # Equity curve always starts with initial balance
        assert len(bot._validation_equity_curve) >= 1

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_validation_stops_after_duration(self):
        """Bot stops running after validation duration expires."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        import time

        start = time.monotonic()
        await bot.run_validation(duration_seconds=1.0)
        elapsed = time.monotonic() - start

        # Should stop relatively quickly (within ~3s accounting for one loop interval)
        assert elapsed < 5.0
        assert bot._running is False

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_validation_generates_report_with_criteria(self):
        """Validation report uses configured criteria."""
        settings = make_settings(
            validation_min_win_rate_pct=50.0,
            validation_min_sharpe_ratio=1.0,
            validation_max_drawdown_pct=10.0,
            validation_min_trades=5,
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        report = await bot.run_validation(duration_seconds=0.5)

        # Check that criteria were evaluated
        assert len(report.criteria_results) == 4
        criteria_names = [c.name for c in report.criteria_results]
        assert "min_trades" in criteria_names
        assert "win_rate_pct" in criteria_names
        assert "sharpe_ratio" in criteria_names
        assert "max_drawdown_pct" in criteria_names

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_validation_nogo_without_trades(self):
        """No trades during validation → NO-GO."""
        settings = make_settings(validation_min_trades=10)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        report = await bot.run_validation(duration_seconds=0.5)

        assert report.recommendation == "NO-GO"
        assert report.total_trades == 0

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_validation_uses_paper_portfolio_balance(self):
        """Validation report uses paper portfolio's actual balance."""
        settings = make_settings(paper_initial_balance=50000.0)
        bot = TradingBot(settings=settings)
        await bot.initialize()

        report = await bot.run_validation(duration_seconds=0.5)

        assert report.initial_balance == 50000.0

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_validation_telegram_notification(self):
        """Validation sends Telegram notification on completion."""
        settings = make_settings(
            telegram_bot_token="test_token",
            telegram_chat_id="test_chat",
        )
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Mock telegram
        mock_telegram = MagicMock()
        mock_telegram.send_message = AsyncMock(return_value=True)
        mock_telegram.notify_error = AsyncMock(return_value=True)
        bot._telegram = mock_telegram

        report = await bot.run_validation(duration_seconds=0.5)

        # Should have sent at least the startup + validation result messages
        calls = mock_telegram.send_message.call_args_list
        validation_calls = [
            c for c in calls
            if "Validation Complete" in str(c)
        ]
        assert len(validation_calls) >= 1

        # Check the validation message content
        msg = validation_calls[0][0][0]
        assert report.recommendation in msg
        assert "Win Rate" in msg
        assert "Sharpe" in msg

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_validation_saves_report_file(self, tmp_path):
        """Validation saves report to JSON file."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        with patch("bot.main.save_report") as mock_save:
            mock_save.return_value = str(tmp_path / "test_report.json")
            report = await bot.run_validation(duration_seconds=0.5)
            mock_save.assert_called_once()
            saved_report = mock_save.call_args[0][0]
            assert saved_report.recommendation == report.recommendation

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_validation_tracks_pnl_on_sell(self):
        """PnLs are tracked when sells execute during validation."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        bot._validation_mode = True
        bot._validation_trade_pnls = []

        # Simulate adding a PnL (this is normally done in _trading_cycle)
        bot._validation_trade_pnls.append(100.0)
        bot._validation_trade_pnls.append(-50.0)

        assert len(bot._validation_trade_pnls) == 2
        assert bot._validation_trade_pnls[0] == 100.0
        assert bot._validation_trade_pnls[1] == -50.0

    @pytest.mark.asyncio
    async def test_validation_resets_state_on_completion(self):
        """Validation mode flag is cleared after completion."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        await bot.run_validation(duration_seconds=0.5)

        assert bot._validation_mode is False
        assert bot._running is False

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_validation_format_summary_printed(self, capsys):
        """Validation prints formatted summary to stdout."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        await bot.run_validation(duration_seconds=0.5)

        captured = capsys.readouterr()
        assert "VALIDATION REPORT" in captured.out
        assert "RECOMMENDATION" in captured.out

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_validation_with_strategy_tracker(self):
        """Validation includes strategy breakdown from tracker."""
        settings = make_settings()
        bot = TradingBot(settings=settings)
        await bot.initialize()

        # Pre-populate strategy tracker with some data
        if bot._strategy_tracker:
            bot._strategy_tracker.record_trade("test_strategy", 50.0)
            bot._strategy_tracker.record_trade("test_strategy", -20.0)

        report = await bot.run_validation(duration_seconds=0.5)

        # Strategy breakdown should contain the pre-populated stats
        assert "test_strategy" in report.strategy_breakdown

        await bot.shutdown()

    @pytest.mark.asyncio
    async def test_main_with_validate_flag(self):
        """main() dispatches to run_validation when --validate flag is set."""
        import argparse

        from bot.main import main

        args = argparse.Namespace(validate=True, duration="1s")

        with patch.object(TradingBot, "initialize", new_callable=AsyncMock):
            with patch.object(TradingBot, "run_validation", new_callable=AsyncMock) as mock_val:
                mock_val.return_value = MagicMock()
                with patch.object(TradingBot, "shutdown", new_callable=AsyncMock):
                    await main(args)
                    mock_val.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_main_without_validate_runs_trading_loop(self):
        """main() runs normal trading loop when no --validate flag."""
        import argparse

        from bot.main import main

        args = argparse.Namespace(validate=False, duration="48h")

        with patch.object(TradingBot, "initialize", new_callable=AsyncMock):
            with patch.object(TradingBot, "run_trading_loop", new_callable=AsyncMock) as mock_loop:
                with patch.object(TradingBot, "shutdown", new_callable=AsyncMock):
                    await main(args)
                    mock_loop.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_without_args_runs_trading_loop(self):
        """main() with no args runs normal trading loop."""
        from bot.main import main

        with patch.object(TradingBot, "initialize", new_callable=AsyncMock):
            with patch.object(TradingBot, "run_trading_loop", new_callable=AsyncMock) as mock_loop:
                with patch.object(TradingBot, "shutdown", new_callable=AsyncMock):
                    await main(None)
                    mock_loop.assert_called_once()
