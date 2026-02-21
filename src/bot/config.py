"""Configuration system using pydantic-settings with .env and optional YAML override."""

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    """Trading mode: paper or live."""

    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    """Application settings loaded from environment variables and optional YAML config."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Trading mode
    trading_mode: TradingMode = TradingMode.PAPER

    # Binance
    binance_api_key: str = ""
    binance_secret_key: str = ""
    binance_testnet: bool = True

    # Upbit
    upbit_api_key: str = ""
    upbit_secret_key: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///data/trading.db"

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Risk Management
    max_position_size_pct: float = Field(default=10.0, ge=0, le=100)
    daily_loss_limit_pct: float = Field(default=5.0, ge=0, le=100)
    max_drawdown_pct: float = Field(default=15.0, ge=0, le=100)
    max_concurrent_positions: int = Field(default=5, ge=1)
    stop_loss_pct: float = Field(default=3.0, ge=0, le=100)

    # Paper trading
    paper_initial_balance: float = Field(default=10000.0, gt=0)
    paper_fee_pct: float = Field(default=0.1, ge=0, le=100)

    # Trading loop
    loop_interval_seconds: int = Field(default=60, ge=1)

    # Logging
    log_level: str = "INFO"

    # Dashboard
    dashboard_port: int = Field(default=8000, ge=1, le=65535)
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost", "http://localhost:8000"]
    )

    # Symbols to trade
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT"])

    # Config file path
    config_file: str = ""

    @field_validator("trading_mode", mode="before")
    @classmethod
    def validate_trading_mode(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @model_validator(mode="after")
    def apply_yaml_overrides(self) -> "Settings":
        """Apply overrides from YAML config file if specified."""
        config_path = Path(self.config_file) if self.config_file else Path("config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config and isinstance(yaml_config, dict):
                for key, value in yaml_config.items():
                    if hasattr(self, key):
                        object.__setattr__(self, key, value)
        return self


def load_settings(**overrides: Any) -> Settings:
    """Create a Settings instance with optional overrides."""
    return Settings(**overrides)
