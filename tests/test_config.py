"""Tests for configuration system."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from bot.config import Settings, TradingMode, load_settings


class TestTradingMode:
    def test_paper_mode(self):
        assert TradingMode.PAPER.value == "paper"

    def test_live_mode(self):
        assert TradingMode.LIVE.value == "live"


class TestSettings:
    def test_defaults(self):
        settings = Settings(config_file="nonexistent.yaml")
        assert settings.trading_mode == TradingMode.PAPER
        assert settings.binance_api_key == ""
        assert settings.binance_testnet is True
        assert settings.log_level == "INFO"
        assert settings.max_position_size_pct == 10.0
        assert settings.daily_loss_limit_pct == 5.0
        assert settings.max_drawdown_pct == 15.0
        assert settings.dashboard_port == 8000
        assert settings.loop_interval_seconds == 60

    def test_paper_mode_default(self):
        settings = Settings(config_file="nonexistent.yaml")
        assert settings.trading_mode == TradingMode.PAPER

    def test_live_mode(self):
        settings = Settings(trading_mode="live", config_file="nonexistent.yaml")
        assert settings.trading_mode == TradingMode.LIVE

    def test_case_insensitive_trading_mode(self):
        settings = Settings(trading_mode="PAPER", config_file="nonexistent.yaml")
        assert settings.trading_mode == TradingMode.PAPER
        settings2 = Settings(trading_mode="Live", config_file="nonexistent.yaml")
        assert settings2.trading_mode == TradingMode.LIVE

    def test_invalid_trading_mode(self):
        with pytest.raises(ValidationError):
            Settings(trading_mode="invalid", config_file="nonexistent.yaml")

    def test_percentage_validation_max_position(self):
        Settings(max_position_size_pct=0, config_file="nonexistent.yaml")
        Settings(max_position_size_pct=100, config_file="nonexistent.yaml")
        with pytest.raises(ValidationError):
            Settings(max_position_size_pct=-1, config_file="nonexistent.yaml")
        with pytest.raises(ValidationError):
            Settings(max_position_size_pct=101, config_file="nonexistent.yaml")

    def test_percentage_validation_daily_loss(self):
        Settings(daily_loss_limit_pct=0, config_file="nonexistent.yaml")
        Settings(daily_loss_limit_pct=100, config_file="nonexistent.yaml")
        with pytest.raises(ValidationError):
            Settings(daily_loss_limit_pct=-1, config_file="nonexistent.yaml")
        with pytest.raises(ValidationError):
            Settings(daily_loss_limit_pct=101, config_file="nonexistent.yaml")

    def test_percentage_validation_max_drawdown(self):
        Settings(max_drawdown_pct=0, config_file="nonexistent.yaml")
        Settings(max_drawdown_pct=100, config_file="nonexistent.yaml")
        with pytest.raises(ValidationError):
            Settings(max_drawdown_pct=-1, config_file="nonexistent.yaml")
        with pytest.raises(ValidationError):
            Settings(max_drawdown_pct=101, config_file="nonexistent.yaml")

    def test_log_level_validation(self):
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            s = Settings(log_level=level, config_file="nonexistent.yaml")
            assert s.log_level == level

    def test_log_level_case_insensitive(self):
        s = Settings(log_level="debug", config_file="nonexistent.yaml")
        assert s.log_level == "DEBUG"

    def test_invalid_log_level(self):
        with pytest.raises(ValidationError):
            Settings(log_level="VERBOSE", config_file="nonexistent.yaml")

    def test_from_env_vars(self):
        env = {
            "TRADING_MODE": "live",
            "BINANCE_API_KEY": "test-key",
            "BINANCE_SECRET_KEY": "test-secret",
            "MAX_POSITION_SIZE_PCT": "25",
            "LOG_LEVEL": "DEBUG",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = Settings(config_file="nonexistent.yaml")
            assert settings.trading_mode == TradingMode.LIVE
            assert settings.binance_api_key == "test-key"
            assert settings.binance_secret_key == "test-secret"
            assert settings.max_position_size_pct == 25.0
            assert settings.log_level == "DEBUG"

    def test_yaml_override(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("log_level: DEBUG\nmax_position_size_pct: 20\n")
        settings = Settings(config_file=str(yaml_file))
        assert settings.log_level == "DEBUG"
        assert settings.max_position_size_pct == 20

    def test_yaml_file_not_found_is_ok(self):
        settings = Settings(config_file="definitely_nonexistent.yaml")
        assert settings.trading_mode == TradingMode.PAPER

    def test_symbols_default(self):
        settings = Settings(config_file="nonexistent.yaml")
        assert settings.symbols == ["BTC/USDT"]

    def test_dashboard_port_bounds(self):
        Settings(dashboard_port=1, config_file="nonexistent.yaml")
        Settings(dashboard_port=65535, config_file="nonexistent.yaml")
        with pytest.raises(ValidationError):
            Settings(dashboard_port=0, config_file="nonexistent.yaml")
        with pytest.raises(ValidationError):
            Settings(dashboard_port=70000, config_file="nonexistent.yaml")


class TestLoadSettings:
    def test_load_with_overrides(self):
        settings = load_settings(
            trading_mode="paper",
            log_level="ERROR",
            config_file="nonexistent.yaml",
        )
        assert settings.trading_mode == TradingMode.PAPER
        assert settings.log_level == "ERROR"
