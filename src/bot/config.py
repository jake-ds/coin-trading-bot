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
    take_profit_pct: float = Field(default=5.0, ge=0, le=100)
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = Field(default=2.0, ge=0, le=100)

    # Dynamic position sizing (ATR-based)
    risk_per_trade_pct: float = Field(default=1.0, ge=0, le=100)
    atr_multiplier: float = Field(default=2.0, gt=0)
    atr_period: int = Field(default=14, ge=1)

    # Paper trading
    paper_initial_balance: float = Field(default=10000.0, gt=0)
    paper_fee_pct: float = Field(default=0.1, ge=0, le=100)

    # Signal ensemble
    signal_min_agreement: int = Field(default=2, ge=1)
    strategy_weights: dict[str, float] = Field(default_factory=dict)

    # Multi-timeframe
    timeframes: list[str] = Field(default_factory=lambda: ["15m", "1h", "4h", "1d"])
    trend_timeframe: str = "4h"

    # Smart execution
    prefer_limit_orders: bool = True
    limit_order_timeout_seconds: float = Field(default=30.0, gt=0)
    twap_chunk_count: int = Field(default=5, ge=1)

    # Strategy auto-disable
    strategy_max_consecutive_losses: int = Field(default=5, ge=1)
    strategy_min_win_rate_pct: float = Field(default=40.0, ge=0, le=100)
    strategy_min_trades_for_eval: int = Field(default=20, ge=1)
    strategy_re_enable_check_hours: float = Field(default=24.0, gt=0)

    # Portfolio-level risk management
    max_total_exposure_pct: float = Field(default=60.0, ge=0, le=100)
    max_correlation: float = Field(default=0.8, ge=0, le=1.0)
    correlation_window: int = Field(default=30, ge=5)
    max_positions_per_sector: int = Field(default=3, ge=1)
    max_portfolio_heat: float = Field(default=0.15, gt=0)
    sector_map: dict[str, str] = Field(default_factory=dict)

    # Funding rate strategy
    funding_extreme_positive_rate: float = Field(default=0.0005, ge=0)
    funding_extreme_negative_rate: float = Field(default=-0.0003, le=0)
    funding_confidence_scale: float = Field(default=10.0, gt=0)
    funding_spread_threshold_pct: float = Field(default=0.5, ge=0)
    funding_rate_history_limit: int = Field(default=50, ge=1)

    # V3 Quant settings
    var_enabled: bool = False
    var_confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    max_portfolio_var_pct: float = Field(default=5.0, ge=0, le=100)
    quant_pairs: list[list[str]] = Field(
        default_factory=list
    )  # e.g., [["BTC/USDT", "ETH/USDT"]]
    triangular_arb_enabled: bool = False
    rebalance_enabled: bool = False
    rebalance_method: str = "risk_parity"
    rebalance_threshold_pct: float = Field(default=5.0, ge=0, le=100)
    garch_enabled: bool = False

    # Validation criteria (go/no-go)
    validation_min_win_rate_pct: float = Field(default=45.0, ge=0, le=100)
    validation_min_sharpe_ratio: float = Field(default=0.5)
    validation_max_drawdown_pct: float = Field(default=15.0, ge=0, le=100)
    validation_min_trades: int = Field(default=10, ge=1)

    # WebSocket feed
    websocket_enabled: bool = False
    websocket_poll_interval: float = Field(default=5.0, gt=0)
    websocket_max_reconnect_delay: float = Field(default=60.0, gt=0)

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
