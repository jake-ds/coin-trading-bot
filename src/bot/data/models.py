"""SQLAlchemy async models for persistent storage."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class OHLCVRecord(Base):
    """OHLCV candlestick record."""

    __tablename__ = "ohlcv"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_ohlcv_symbol_tf_ts"),
        Index("ix_ohlcv_symbol_timeframe_timestamp", "symbol", "timeframe", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    timeframe: Mapped[str] = mapped_column(String(10))
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)


class TradeRecord(Base):
    """Executed trade record."""

    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_symbol_created_at", "symbol", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(100), index=True)
    exchange: Mapped[str] = mapped_column(String(50))
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    side: Mapped[str] = mapped_column(String(10))
    order_type: Mapped[str] = mapped_column(String(10))
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class FundingRateRecord(Base):
    """Funding rate record for perpetual futures."""

    __tablename__ = "funding_rates"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_funding_symbol_ts"),
        Index("ix_funding_rates_symbol_timestamp", "symbol", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    funding_rate: Mapped[float] = mapped_column(Float)
    funding_timestamp: Mapped[datetime] = mapped_column(DateTime)
    mark_price: Mapped[float] = mapped_column(Float, default=0.0)
    spot_price: Mapped[float] = mapped_column(Float, default=0.0)
    spread_pct: Mapped[float] = mapped_column(Float, default=0.0)


class PortfolioSnapshot(Base):
    """Portfolio state snapshot."""

    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, default=datetime.utcnow)
    total_value: Mapped[float] = mapped_column(Float)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    balances_json: Mapped[str] = mapped_column(String(2000), default="{}")
    positions_json: Mapped[str] = mapped_column(String(5000), default="[]")


class EngineTradeRecord(Base):
    """Persisted engine trade record (from EngineTracker)."""

    __tablename__ = "engine_trades"
    __table_args__ = (
        Index(
            "ix_engine_trades_engine_exit",
            "engine_name", "exit_time",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True,
    )
    engine_name: Mapped[str] = mapped_column(String(50), index=True)
    symbol: Mapped[str] = mapped_column(String(50))
    side: Mapped[str] = mapped_column(String(20))
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    pnl: Mapped[float] = mapped_column(Float)
    cost: Mapped[float] = mapped_column(Float)
    net_pnl: Mapped[float] = mapped_column(Float)
    entry_time: Mapped[str] = mapped_column(String(50))
    exit_time: Mapped[str] = mapped_column(String(50))
    hold_time_seconds: Mapped[float] = mapped_column(
        Float, default=0.0,
    )


class EngineMetricSnapshot(Base):
    """Periodic snapshot of engine performance metrics."""

    __tablename__ = "engine_metric_snapshots"
    __table_args__ = (
        Index(
            "ix_engine_metrics_engine_ts",
            "engine_name", "timestamp",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True,
    )
    engine_name: Mapped[str] = mapped_column(String(50), index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow,
    )
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    profit_factor: Mapped[float] = mapped_column(Float, default=0.0)
    cost_ratio: Mapped[float] = mapped_column(Float, default=0.0)


class AuditLogRecord(Base):
    """Immutable audit log entry for tracking all significant bot actions."""

    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_event_type", "event_type"),
        Index("ix_audit_log_timestamp", "timestamp"),
        Index("ix_audit_log_severity", "severity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    event_type: Mapped[str] = mapped_column(String(50))
    actor: Mapped[str] = mapped_column(String(50), default="system")
    details: Mapped[str] = mapped_column(Text, default="{}")
    severity: Mapped[str] = mapped_column(String(20), default="info")
