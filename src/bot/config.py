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

    # ── Multi-engine system ──
    engine_mode: bool = False  # False = legacy ensemble, True = multi-engine
    engine_total_capital: float = Field(default=10000.0, gt=0)
    engine_max_drawdown_pct: float = Field(default=20.0, ge=0, le=100)
    engine_allocations: dict[str, float] = Field(
        default_factory=lambda: {
            "funding_rate_arb": 0.30,
            "grid_trading": 0.25,
            "stat_arb": 0.20,
            "cross_exchange_arb": 0.15,
        }
    )

    # Binance Futures
    binance_futures_api_key: str = ""
    binance_futures_secret_key: str = ""
    binance_futures_testnet: bool = True

    # Funding rate arbitrage engine
    funding_arb_min_rate: float = Field(default=0.0003, ge=0)
    funding_arb_exit_rate: float = Field(default=0.0001, ge=0)
    funding_arb_max_spread_pct: float = Field(default=0.5, ge=0)
    funding_arb_max_positions: int = Field(default=3, ge=1)
    funding_arb_leverage: int = Field(default=1, ge=1)
    funding_arb_symbols: list[str] = Field(
        default_factory=lambda: [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT",
        ]
    )

    # Grid trading engine
    grid_levels: int = Field(default=10, ge=2)
    grid_spacing_pct: float = Field(default=0.5, gt=0)
    grid_auto_range: bool = True
    grid_range_atr_multiplier: float = Field(default=3.0, gt=0)
    grid_max_open_orders: int = Field(default=20, ge=2)
    grid_symbols: list[str] = Field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    )

    # Cross-exchange arbitrage engine
    cross_arb_min_spread_pct: float = Field(default=0.3, ge=0)
    cross_arb_exchanges: list[str] = Field(
        default_factory=lambda: ["binance", "upbit"]
    )
    cross_arb_symbols: list[str] = Field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    )
    cross_arb_max_position_per_symbol: float = Field(default=1000.0, gt=0)
    cross_arb_rebalance_threshold_pct: float = Field(default=20.0, ge=0)

    # Statistical arbitrage engine
    stat_arb_pairs: list[list[str]] = Field(
        default_factory=lambda: [
            ["BTC/USDT", "ETH/USDT"],
            ["SOL/USDT", "ETH/USDT"],
        ]
    )
    stat_arb_lookback: int = Field(default=100, ge=10)
    stat_arb_entry_zscore: float = Field(default=2.0, gt=0)
    stat_arb_exit_zscore: float = Field(default=0.5, ge=0)
    stat_arb_stop_zscore: float = Field(default=4.0, gt=0)
    stat_arb_min_correlation: float = Field(default=0.7, ge=0, le=1.0)

    # ── Auto-tuner & rebalance ──
    tuner_enabled: bool = True
    tuner_interval_hours: int = Field(default=24, ge=1)
    engine_rebalance_enabled: bool = True
    engine_rebalance_interval_hours: int = Field(default=24, ge=1)
    research_enabled: bool = True
    research_interval_hours: int = Field(default=24, ge=1)

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

    # Rate limiting
    rate_limit_enabled: bool = True
    exchange_rate_limits: dict[str, dict[str, float]] = Field(default_factory=dict)

    # Position reconciliation
    reconciliation_enabled: bool = True
    reconciliation_interval_cycles: int = Field(default=10, ge=1)
    reconciliation_auto_fix: bool = False

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


    def reload(self, updates: dict[str, Any]) -> list[str]:
        """Hot-reload safe settings at runtime.

        Args:
            updates: dict of field_name -> new_value to apply.

        Returns:
            List of field names that were actually changed.

        Raises:
            ValueError: If any field is not a safe (hot-reloadable) setting.
        """
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
    # Engine mode (unsafe — require restart)
    "engine_mode": {
        "section": "Trading",
        "description": "Enable multi-engine autonomous trading system",
        "type": "bool",
        "requires_restart": True,
    },
    # Trading (unsafe — require restart)
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
    # Exchange (unsafe)
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
    "upbit_api_key": {
        "section": "Exchange",
        "description": "Upbit API key",
        "type": "secret",
        "requires_restart": True,
    },
    "upbit_secret_key": {
        "section": "Exchange",
        "description": "Upbit secret key",
        "type": "secret",
        "requires_restart": True,
    },
    # Rate limiting (safe)
    "rate_limit_enabled": {
        "section": "Exchange",
        "description": "Enable API rate limiting",
        "type": "bool",
        "requires_restart": False,
    },
    "exchange_rate_limits": {
        "section": "Exchange",
        "description": "Per-exchange rate limit overrides (JSON)",
        "type": "dict",
        "requires_restart": True,
    },
    # Reconciliation (safe — hot-reloadable)
    "reconciliation_enabled": {
        "section": "Risk Management",
        "description": "Enable position reconciliation with exchange",
        "type": "bool",
        "requires_restart": False,
    },
    "reconciliation_interval_cycles": {
        "section": "Risk Management",
        "description": "Run reconciliation every N trading cycles",
        "type": "int",
        "requires_restart": False,
    },
    "reconciliation_auto_fix": {
        "section": "Risk Management",
        "description": "Auto-fix local state to match exchange on discrepancy",
        "type": "bool",
        "requires_restart": False,
    },
    # Database (unsafe)
    "database_url": {
        "section": "Exchange",
        "description": "Database connection URL",
        "type": "secret",
        "requires_restart": True,
    },
    # Risk Management (safe — hot-reloadable)
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
    "trailing_stop_enabled": {
        "section": "Risk Management",
        "description": "Enable trailing stop-loss",
        "type": "bool",
        "requires_restart": False,
    },
    "trailing_stop_pct": {
        "section": "Risk Management",
        "description": "Trailing stop distance as %",
        "type": "float",
        "requires_restart": False,
    },
    "risk_per_trade_pct": {
        "section": "Risk Management",
        "description": "Risk per trade as % of portfolio (ATR-based sizing)",
        "type": "float",
        "requires_restart": False,
    },
    "atr_multiplier": {
        "section": "Risk Management",
        "description": "ATR multiplier for position sizing",
        "type": "float",
        "requires_restart": False,
    },
    "atr_period": {
        "section": "Risk Management",
        "description": "ATR calculation period",
        "type": "int",
        "requires_restart": False,
    },
    # Strategies (safe)
    "signal_min_agreement": {
        "section": "Strategies",
        "description": "Minimum strategy agreement for signal",
        "type": "int",
        "requires_restart": False,
    },
    "strategy_weights": {
        "section": "Strategies",
        "description": "Per-strategy weight overrides (JSON)",
        "type": "dict",
        "requires_restart": False,
    },
    "strategy_max_consecutive_losses": {
        "section": "Strategies",
        "description": "Max consecutive losses before auto-disable",
        "type": "int",
        "requires_restart": False,
    },
    "strategy_min_win_rate_pct": {
        "section": "Strategies",
        "description": "Min win rate % before auto-disable",
        "type": "float",
        "requires_restart": False,
    },
    "strategy_min_trades_for_eval": {
        "section": "Strategies",
        "description": "Min trades before evaluating strategy",
        "type": "int",
        "requires_restart": False,
    },
    "strategy_re_enable_check_hours": {
        "section": "Strategies",
        "description": "Hours before re-checking disabled strategy",
        "type": "float",
        "requires_restart": False,
    },
    # Paper Trading (safe)
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
    # Smart execution (safe)
    "prefer_limit_orders": {
        "section": "Trading",
        "description": "Prefer limit orders over market orders",
        "type": "bool",
        "requires_restart": False,
    },
    "limit_order_timeout_seconds": {
        "section": "Trading",
        "description": "Timeout for limit orders before fallback to market",
        "type": "float",
        "requires_restart": False,
    },
    "twap_chunk_count": {
        "section": "Trading",
        "description": "Number of chunks for TWAP execution",
        "type": "int",
        "requires_restart": False,
    },
    # Multi-timeframe (safe)
    "timeframes": {
        "section": "Strategies",
        "description": "Timeframes for analysis",
        "type": "list",
        "requires_restart": True,
    },
    "trend_timeframe": {
        "section": "Strategies",
        "description": "Higher timeframe for trend filter",
        "type": "str",
        "requires_restart": False,
    },
    # Portfolio risk (safe)
    "max_total_exposure_pct": {
        "section": "Risk Management",
        "description": "Max total portfolio exposure %",
        "type": "float",
        "requires_restart": False,
    },
    "max_correlation": {
        "section": "Risk Management",
        "description": "Max correlation allowed between positions",
        "type": "float",
        "requires_restart": False,
    },
    "max_portfolio_heat": {
        "section": "Risk Management",
        "description": "Max portfolio heat (risk)",
        "type": "float",
        "requires_restart": False,
    },
    # Dashboard (unsafe)
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
    # Notifications (unsafe)
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
    # Logging (safe)
    "log_level": {
        "section": "Dashboard",
        "description": "Logging level",
        "type": "select",
        "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        "requires_restart": False,
    },
    # Funding rate arb engine
    "funding_arb_min_rate": {
        "section": "Engines",
        "description": "Minimum funding rate to enter position",
        "type": "float",
        "requires_restart": False,
    },
    "funding_arb_exit_rate": {
        "section": "Engines",
        "description": "Funding rate below which to exit position",
        "type": "float",
        "requires_restart": False,
    },
    "funding_arb_max_spread_pct": {
        "section": "Engines",
        "description": "Maximum basis spread % to enter position",
        "type": "float",
        "requires_restart": False,
    },
    "funding_arb_max_positions": {
        "section": "Engines",
        "description": "Maximum concurrent funding arb positions",
        "type": "int",
        "requires_restart": False,
    },
    "funding_arb_leverage": {
        "section": "Engines",
        "description": "Leverage for funding arb positions",
        "type": "int",
        "requires_restart": False,
    },
    "funding_arb_symbols": {
        "section": "Engines",
        "description": "Symbols to monitor for funding rate arbitrage",
        "type": "list",
        "requires_restart": True,
    },
    # Grid trading engine
    "grid_levels": {
        "section": "Engines",
        "description": "Number of grid levels per side",
        "type": "int",
        "requires_restart": False,
    },
    "grid_spacing_pct": {
        "section": "Engines",
        "description": "Grid spacing as percentage between levels",
        "type": "float",
        "requires_restart": False,
    },
    "grid_auto_range": {
        "section": "Engines",
        "description": "Auto-adjust grid range based on volatility",
        "type": "bool",
        "requires_restart": False,
    },
    "grid_max_open_orders": {
        "section": "Engines",
        "description": "Maximum open grid orders",
        "type": "int",
        "requires_restart": False,
    },
    "grid_symbols": {
        "section": "Engines",
        "description": "Symbols for grid trading",
        "type": "list",
        "requires_restart": True,
    },
    # Cross-exchange arb engine
    "cross_arb_min_spread_pct": {
        "section": "Engines",
        "description": "Minimum spread % to execute cross-exchange arb",
        "type": "float",
        "requires_restart": False,
    },
    "cross_arb_symbols": {
        "section": "Engines",
        "description": "Symbols for cross-exchange arbitrage",
        "type": "list",
        "requires_restart": True,
    },
    "cross_arb_max_position_per_symbol": {
        "section": "Engines",
        "description": "Max USDT notional per symbol for cross-exchange arb",
        "type": "float",
        "requires_restart": False,
    },
    # Stat arb engine
    "stat_arb_pairs": {
        "section": "Engines",
        "description": "Symbol pairs for statistical arbitrage",
        "type": "list",
        "requires_restart": True,
    },
    "stat_arb_lookback": {
        "section": "Engines",
        "description": "Lookback period (candles) for z-score calculation",
        "type": "int",
        "requires_restart": False,
    },
    "stat_arb_entry_zscore": {
        "section": "Engines",
        "description": "Z-score threshold to enter pairs trade",
        "type": "float",
        "requires_restart": False,
    },
    "stat_arb_exit_zscore": {
        "section": "Engines",
        "description": "Z-score threshold to exit pairs trade (profit)",
        "type": "float",
        "requires_restart": False,
    },
    "stat_arb_stop_zscore": {
        "section": "Engines",
        "description": "Z-score threshold for stop-loss exit",
        "type": "float",
        "requires_restart": False,
    },
    "stat_arb_min_correlation": {
        "section": "Engines",
        "description": "Minimum correlation required between pairs",
        "type": "float",
        "requires_restart": False,
    },
    "tuner_enabled": {
        "section": "Engines",
        "description": "Enable automatic parameter tuning based on performance",
        "type": "bool",
        "requires_restart": False,
    },
    "tuner_interval_hours": {
        "section": "Engines",
        "description": "Hours between auto-tuner runs",
        "type": "int",
        "requires_restart": False,
    },
    "engine_rebalance_enabled": {
        "section": "Engines",
        "description": "Enable Sharpe-weighted capital rebalancing across engines",
        "type": "bool",
        "requires_restart": False,
    },
    "engine_rebalance_interval_hours": {
        "section": "Engines",
        "description": "Hours between capital rebalancing",
        "type": "int",
        "requires_restart": False,
    },
    "research_enabled": {
        "section": "Research",
        "description": "Enable automated research experiments",
        "type": "bool",
        "requires_restart": False,
    },
    "research_interval_hours": {
        "section": "Research",
        "description": "Hours between research experiment runs",
        "type": "int",
        "requires_restart": False,
    },
}


# ---------------------------------------------------------------------------
# Engine description metadata
# ---------------------------------------------------------------------------

ENGINE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "funding_rate_arb": {
        "role_ko": "델타중립 펀딩비 차익거래",
        "role_en": "Delta-neutral funding rate arbitrage",
        "description_ko": (
            "선물 펀딩비가 양수일 때 현물 매수 + 선물 매도로 "
            "시장 중립 포지션을 구성하여 펀딩비 수익을 추구합니다."
        ),
        "key_params": "min_rate, max_spread_pct, leverage",
    },
    "grid_trading": {
        "role_ko": "그리드 자동매매",
        "role_en": "Automated grid trading",
        "description_ko": (
            "현재 가격 위아래에 일정 간격으로 매수/매도 주문을 배치하여 "
            "횡보장에서 가격 진동으로 수익을 추구합니다."
        ),
        "key_params": "grid_spacing_pct, grid_levels, auto_range",
    },
    "cross_exchange_arb": {
        "role_ko": "거래소간 차익거래",
        "role_en": "Cross-exchange spot arbitrage",
        "description_ko": (
            "동일 심볼의 거래소 간 가격 차이를 감시하여, "
            "스프레드가 비용을 초과할 때 동시 매수/매도로 차익을 실현합니다."
        ),
        "key_params": "min_spread_pct, max_position_per_symbol",
    },
    "stat_arb": {
        "role_ko": "통계적 차익거래",
        "role_en": "Statistical pairs arbitrage",
        "description_ko": (
            "상관관계가 높은 자산 쌍의 가격 비율 Z-Score를 추적하여, "
            "평균 회귀를 이용한 롱/숏 페어트레이딩을 수행합니다."
        ),
        "key_params": "entry_zscore, exit_zscore, lookback, min_correlation",
    },
}


def load_settings(**overrides: Any) -> Settings:
    """Create a Settings instance with optional overrides."""
    return Settings(**overrides)
