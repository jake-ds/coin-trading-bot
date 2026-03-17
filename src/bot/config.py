"""Configuration system using pydantic-settings with .env and optional YAML override."""

import os
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

    # Binance (spot only)
    binance_api_key: str = ""
    binance_secret_key: str = ""
    binance_testnet: bool = True

    # Database
    database_url: str = "sqlite+aiosqlite:///data/trading.db"

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Risk Management
    max_position_size_pct: float = Field(default=20.0, ge=0, le=100)
    daily_loss_limit_pct: float = Field(default=5.0, ge=0, le=100)
    max_drawdown_pct: float = Field(default=15.0, ge=0, le=100)
    max_concurrent_positions: int = Field(default=5, ge=1)
    stop_loss_pct: float = Field(default=5.0, ge=0, le=100)
    take_profit_pct: float = Field(default=8.0, ge=0, le=100)

    # Paper trading
    paper_initial_balance: float = Field(default=10000.0, gt=0)
    paper_fee_pct: float = Field(default=0.1, ge=0, le=100)

    # ── Multi-engine system ──
    engine_mode: bool = True
    engine_total_capital: float = Field(default=900.0, gt=0)
    engine_max_drawdown_pct: float = Field(default=20.0, ge=0, le=100)
    engine_allocations: dict[str, float] = Field(
        default_factory=lambda: {
            "onchain_trader": 1.0,
        }
    )

    # ── OnChain Trader ──
    onchain_loop_interval: float = Field(default=300.0, gt=0)
    onchain_max_positions: int = Field(default=10, ge=1)
    onchain_symbols: list[str] = Field(
        default_factory=lambda: [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
            "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
            "SUI/USDT", "NEAR/USDT", "ARB/USDT", "OP/USDT", "APT/USDT",
            "PEPE/USDT", "UNI/USDT", "ATOM/USDT", "FIL/USDT", "LTC/USDT",
            "TRX/USDT", "WIF/USDT", "AAVE/USDT", "RENDER/USDT", "MATIC/USDT",
        ]
    )
    onchain_buy_threshold: float = Field(default=30.0)
    onchain_sell_threshold: float = Field(default=-30.0)
    onchain_min_confidence: float = Field(default=0.4, ge=0, le=1.0)
    onchain_max_position_pct: float = Field(default=20.0, ge=0, le=100)
    onchain_stop_loss_pct: float = Field(default=5.0, ge=0, le=100)
    onchain_take_profit_pct: float = Field(default=8.0, ge=0, le=100)
    onchain_trailing_stop_pct: float = Field(default=3.0, ge=0, le=100)
    onchain_trailing_activate_pct: float = Field(default=2.0, ge=0, le=100)
    onchain_signal_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "whale_flow": 0.25,
            "sentiment": 0.15,
            "defi_flow": 0.20,
            "derivatives": 0.25,
            "market_trend": 0.15,
        }
    )

    # API keys for onchain data (free tier)
    etherscan_api_key: str = ""
    coinglass_api_key: str = ""

    # Rate limiting
    rate_limit_enabled: bool = True
    exchange_rate_limits: dict[str, dict[str, float]] = Field(default_factory=dict)

    # ── Metrics persistence ──
    metrics_persistence_enabled: bool = True
    metrics_snapshot_interval_minutes: float = Field(default=5.0, gt=0)
    metrics_retention_days: int = Field(default=90, ge=1)

    # ── Market regime ──
    regime_detection_enabled: bool = True
    regime_crisis_threshold: float = Field(default=2.5, gt=0)
    regime_detection_interval_seconds: float = Field(default=300.0, gt=0)
    regime_adaptation_enabled: bool = True
    crisis_circuit_breaker_minutes: float = Field(default=30.0, gt=0)

    # ── Shutdown ──
    shutdown_timeout_seconds: float = Field(default=30.0, gt=0)

    # Trading loop
    loop_interval_seconds: int = Field(default=300, ge=1)

    # Logging
    log_level: str = "INFO"
    trading_env: str = "development"

    # Dashboard
    dashboard_port: int = Field(default=8000, ge=1, le=65535)
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost", "http://localhost:8000"]
    )

    # Dashboard authentication (JWT)
    dashboard_username: str = "admin"
    dashboard_password: str = "changeme"
    jwt_secret: str = ""

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

    @field_validator("trading_env", mode="before")
    @classmethod
    def validate_trading_env(cls, v: Any) -> str:
        valid = {"production", "staging", "development"}
        if isinstance(v, str) and v.lower() in valid:
            return v.lower()
        return "development"

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

        # Apply TRADING_ENV-based log_level when TRADING_ENV is set but LOG_LEVEL is not
        if os.environ.get("TRADING_ENV") and not os.environ.get("LOG_LEVEL"):
            env_log_levels = {
                "production": "WARNING",
                "staging": "INFO",
                "development": "DEBUG",
            }
            env_level = env_log_levels.get(self.trading_env)
            if env_level:
                object.__setattr__(self, "log_level", env_level)

        return self

    def reload(self, updates: dict[str, Any]) -> list[str]:
        """Hot-reload safe settings at runtime."""
        changed: list[str] = []
        for key, value in updates.items():
            if key not in SETTINGS_METADATA:
                raise ValueError(f"Unknown setting: {key}")
            meta = SETTINGS_METADATA[key]
            if meta.get("requires_restart", False):
                raise ValueError(
                    f"Setting '{key}' requires restart and cannot be changed at runtime"
                )
            if not hasattr(self, key):
                raise ValueError(f"Unknown setting: {key}")
            old_value = getattr(self, key)
            if old_value != value:
                object.__setattr__(self, key, value)
                changed.append(key)
        return changed


# ---------------------------------------------------------------------------
# Settings metadata — describes each setting for the web UI
# ---------------------------------------------------------------------------

SETTINGS_METADATA: dict[str, dict[str, Any]] = {
    "engine_mode": {
        "section": "Trading",
        "description": "Enable multi-engine autonomous trading system",
        "type": "bool",
        "requires_restart": True,
    },
    "trading_mode": {
        "section": "Trading",
        "description": "Trading mode: paper or live",
        "type": "select",
        "options": ["paper", "live"],
        "requires_restart": True,
    },
    "symbols": {
        "section": "Trading",
        "description": "Symbols to trade",
        "type": "list",
        "requires_restart": True,
    },
    "loop_interval_seconds": {
        "section": "Trading",
        "description": "Seconds between trading cycles",
        "type": "int",
        "requires_restart": False,
    },
    "binance_api_key": {
        "section": "Exchange",
        "description": "Binance API key",
        "type": "secret",
        "requires_restart": True,
    },
    "binance_secret_key": {
        "section": "Exchange",
        "description": "Binance secret key",
        "type": "secret",
        "requires_restart": True,
    },
    "binance_testnet": {
        "section": "Exchange",
        "description": "Use Binance testnet",
        "type": "bool",
        "requires_restart": True,
    },
    "rate_limit_enabled": {
        "section": "Exchange",
        "description": "Enable API rate limiting",
        "type": "bool",
        "requires_restart": False,
    },
    "database_url": {
        "section": "Exchange",
        "description": "Database connection URL",
        "type": "secret",
        "requires_restart": True,
    },
    "max_position_size_pct": {
        "section": "Risk Management",
        "description": "Maximum position size as % of portfolio",
        "type": "float",
        "requires_restart": False,
    },
    "daily_loss_limit_pct": {
        "section": "Risk Management",
        "description": "Maximum daily loss as % of portfolio",
        "type": "float",
        "requires_restart": False,
    },
    "max_drawdown_pct": {
        "section": "Risk Management",
        "description": "Maximum drawdown % before halting",
        "type": "float",
        "requires_restart": False,
    },
    "max_concurrent_positions": {
        "section": "Risk Management",
        "description": "Maximum number of concurrent positions",
        "type": "int",
        "requires_restart": False,
    },
    "stop_loss_pct": {
        "section": "Risk Management",
        "description": "Stop-loss percentage per trade",
        "type": "float",
        "requires_restart": False,
    },
    "take_profit_pct": {
        "section": "Risk Management",
        "description": "Take-profit percentage per trade",
        "type": "float",
        "requires_restart": False,
    },
    "paper_initial_balance": {
        "section": "Trading",
        "description": "Paper trading initial balance",
        "type": "float",
        "requires_restart": True,
    },
    "paper_fee_pct": {
        "section": "Trading",
        "description": "Paper trading fee percentage",
        "type": "float",
        "requires_restart": False,
    },
    "dashboard_port": {
        "section": "Dashboard",
        "description": "Dashboard server port",
        "type": "int",
        "requires_restart": True,
    },
    "dashboard_username": {
        "section": "Dashboard",
        "description": "Dashboard login username",
        "type": "str",
        "requires_restart": True,
    },
    "dashboard_password": {
        "section": "Dashboard",
        "description": "Dashboard login password",
        "type": "secret",
        "requires_restart": True,
    },
    "jwt_secret": {
        "section": "Dashboard",
        "description": "JWT signing secret",
        "type": "secret",
        "requires_restart": True,
    },
    "telegram_bot_token": {
        "section": "Notifications",
        "description": "Telegram bot token",
        "type": "secret",
        "requires_restart": True,
    },
    "telegram_chat_id": {
        "section": "Notifications",
        "description": "Telegram chat ID",
        "type": "secret",
        "requires_restart": True,
    },
    "trading_env": {
        "section": "Dashboard",
        "description": "Deployment environment (production, staging, development)",
        "type": "select",
        "options": ["production", "staging", "development"],
        "requires_restart": True,
    },
    "log_level": {
        "section": "Dashboard",
        "description": "Logging level",
        "type": "select",
        "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        "requires_restart": False,
    },
    # OnChain Trader settings
    "onchain_loop_interval": {
        "section": "OnChain Trader",
        "description": "Seconds between onchain trader cycles",
        "type": "float",
        "requires_restart": False,
    },
    "onchain_max_positions": {
        "section": "OnChain Trader",
        "description": "Maximum concurrent onchain positions",
        "type": "int",
        "requires_restart": False,
    },
    "onchain_symbols": {
        "section": "OnChain Trader",
        "description": "Symbols for onchain trading",
        "type": "list",
        "requires_restart": True,
    },
    "onchain_buy_threshold": {
        "section": "OnChain Trader",
        "description": "Composite score threshold for BUY signal",
        "type": "float",
        "requires_restart": False,
    },
    "onchain_sell_threshold": {
        "section": "OnChain Trader",
        "description": "Composite score threshold for SELL signal",
        "type": "float",
        "requires_restart": False,
    },
    "onchain_min_confidence": {
        "section": "OnChain Trader",
        "description": "Minimum confidence to act on signal",
        "type": "float",
        "requires_restart": False,
    },
    "onchain_max_position_pct": {
        "section": "OnChain Trader",
        "description": "Maximum position size as % of capital",
        "type": "float",
        "requires_restart": False,
    },
    "onchain_stop_loss_pct": {
        "section": "OnChain Trader",
        "description": "Stop-loss percentage",
        "type": "float",
        "requires_restart": False,
    },
    "onchain_take_profit_pct": {
        "section": "OnChain Trader",
        "description": "Take-profit percentage",
        "type": "float",
        "requires_restart": False,
    },
    "onchain_trailing_stop_pct": {
        "section": "OnChain Trader",
        "description": "Trailing stop distance %",
        "type": "float",
        "requires_restart": False,
    },
    "onchain_signal_weights": {
        "section": "OnChain Trader",
        "description": "Signal category weights (JSON)",
        "type": "dict",
        "requires_restart": False,
    },
    "etherscan_api_key": {
        "section": "OnChain Trader",
        "description": "Etherscan API key for whale tracking",
        "type": "secret",
        "requires_restart": True,
    },
    "coinglass_api_key": {
        "section": "OnChain Trader",
        "description": "CoinGlass API key for derivatives data",
        "type": "secret",
        "requires_restart": True,
    },
    "metrics_persistence_enabled": {
        "section": "Metrics",
        "description": "Enable persisting engine trades and metrics to DB",
        "type": "bool",
        "requires_restart": False,
    },
    "metrics_snapshot_interval_minutes": {
        "section": "Metrics",
        "description": "Minutes between metric snapshots",
        "type": "float",
        "requires_restart": False,
    },
    "metrics_retention_days": {
        "section": "Metrics",
        "description": "Days to retain persisted metrics before cleanup",
        "type": "int",
        "requires_restart": False,
    },
    "regime_detection_enabled": {
        "section": "Market Regime",
        "description": "Enable real-time market regime detection",
        "type": "bool",
        "requires_restart": True,
    },
    "regime_detection_interval_seconds": {
        "section": "Market Regime",
        "description": "Seconds between regime detection checks",
        "type": "float",
        "requires_restart": False,
    },
    "regime_crisis_threshold": {
        "section": "Market Regime",
        "description": "Volatility ratio threshold for CRISIS regime",
        "type": "float",
        "requires_restart": False,
    },
    "regime_adaptation_enabled": {
        "section": "Market Regime",
        "description": "Enable engine regime adaptation and circuit breaker",
        "type": "bool",
        "requires_restart": False,
    },
    "crisis_circuit_breaker_minutes": {
        "section": "Market Regime",
        "description": "Minutes of CRISIS regime before circuit breaker pauses all engines",
        "type": "float",
        "requires_restart": False,
    },
    "shutdown_timeout_seconds": {
        "section": "Shutdown",
        "description": "Timeout in seconds for graceful shutdown",
        "type": "float",
        "requires_restart": True,
    },
}


# ---------------------------------------------------------------------------
# Engine description metadata
# ---------------------------------------------------------------------------

ENGINE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "onchain_trader": {
        "role_ko": "온체인 데이터 기반 자율주행 트레이더",
        "role_en": "On-chain data driven autonomous trader",
        "description_ko": (
            "CoinGecko, Fear&Greed, DeFiLlama, CoinGlass 등 무료 온체인/마켓 데이터를 "
            "수집·분석하여 종합 시그널 스코어를 산출하고, "
            "바이낸스 현물에서 자동으로 매수/매도합니다."
        ),
        "key_params": "buy_threshold, sell_threshold, signal_weights, stop_loss, take_profit",
    },
}


def load_settings(**overrides: Any) -> Settings:
    """Create a Settings instance with optional overrides."""
    return Settings(**overrides)
