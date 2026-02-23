"""DataStore - async CRUD operations for persistent storage."""

import json
from datetime import datetime

from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bot.data.models import (
    AuditLogRecord,
    Base,
    EngineMetricSnapshot,
    EngineTradeRecord,
    FundingRateRecord,
    OHLCVRecord,
    PortfolioSnapshot,
    TradeRecord,
)
from bot.models import OHLCV, Order


class DataStore:
    """Async data store using SQLAlchemy + aiosqlite."""

    def __init__(self, database_url: str = "sqlite+aiosqlite:///data/trading.db"):
        self._engine = create_async_engine(database_url, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self) -> None:
        """Create all tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Close the database engine."""
        await self._engine.dispose()

    # --- OHLCV Operations ---

    async def save_candles(self, candles: list[OHLCV]) -> None:
        """Save a list of OHLCV candles to the database.

        Uses INSERT OR IGNORE to silently skip duplicate candles
        (same symbol + timeframe + timestamp).
        """
        if not candles:
            return
        async with self._session_factory() as session:
            for candle in candles:
                stmt = (
                    insert(OHLCVRecord)
                    .values(
                        symbol=candle.symbol,
                        timeframe=candle.timeframe,
                        timestamp=candle.timestamp,
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume,
                    )
                    .prefix_with("OR IGNORE")
                )
                await session.execute(stmt)
            await session.commit()

    async def get_candles(
        self,
        symbol: str,
        timeframe: str = "1h",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[OHLCV]:
        """Query OHLCV candles by symbol, timeframe, and optional date range."""
        async with self._session_factory() as session:
            stmt = (
                select(OHLCVRecord)
                .where(OHLCVRecord.symbol == symbol)
                .where(OHLCVRecord.timeframe == timeframe)
            )
            if start:
                stmt = stmt.where(OHLCVRecord.timestamp >= start)
            if end:
                stmt = stmt.where(OHLCVRecord.timestamp <= end)
            stmt = stmt.order_by(OHLCVRecord.timestamp.desc()).limit(limit)

            result = await session.execute(stmt)
            records = result.scalars().all()
            return [
                OHLCV(
                    timestamp=r.timestamp,
                    open=r.open,
                    high=r.high,
                    low=r.low,
                    close=r.close,
                    volume=r.volume,
                    symbol=r.symbol,
                    timeframe=r.timeframe,
                )
                for r in reversed(records)
            ]

    async def get_available_symbols(
        self,
        timeframe: str = "1h",
        min_count: int = 100,
    ) -> list[str]:
        """Get symbols that have at least min_count candles for the given timeframe."""
        async with self._session_factory() as session:
            stmt = (
                select(OHLCVRecord.symbol)
                .where(OHLCVRecord.timeframe == timeframe)
                .group_by(OHLCVRecord.symbol)
                .having(func.count() >= min_count)
                .order_by(OHLCVRecord.symbol)
            )
            result = await session.execute(stmt)
            return [row[0] for row in result.all()]

    # --- Trade Operations ---

    async def save_trade(self, order: Order) -> None:
        """Save a trade record from an Order."""
        async with self._session_factory() as session:
            record = TradeRecord(
                order_id=order.id,
                exchange=order.exchange,
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.type.value,
                price=order.filled_price or order.price,
                quantity=order.filled_quantity or order.quantity,
                fee=order.fee,
                status=order.status.value,
                created_at=order.created_at,
                filled_at=order.filled_at,
            )
            session.add(record)
            await session.commit()

    async def get_trades(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query trades by optional date range and symbol."""
        async with self._session_factory() as session:
            stmt = select(TradeRecord)
            if symbol:
                stmt = stmt.where(TradeRecord.symbol == symbol)
            if start:
                stmt = stmt.where(TradeRecord.created_at >= start)
            if end:
                stmt = stmt.where(TradeRecord.created_at <= end)
            stmt = stmt.order_by(TradeRecord.created_at.desc()).limit(limit)

            result = await session.execute(stmt)
            records = result.scalars().all()
            return [
                {
                    "order_id": r.order_id,
                    "exchange": r.exchange,
                    "symbol": r.symbol,
                    "side": r.side,
                    "order_type": r.order_type,
                    "price": r.price,
                    "quantity": r.quantity,
                    "fee": r.fee,
                    "status": r.status,
                    "created_at": r.created_at,
                    "filled_at": r.filled_at,
                }
                for r in records
            ]

    # --- Funding Rate Operations ---

    async def save_funding_rate(
        self,
        symbol: str,
        funding_rate: float,
        funding_timestamp: datetime,
        mark_price: float = 0.0,
        spot_price: float = 0.0,
        spread_pct: float = 0.0,
    ) -> None:
        """Save a funding rate record.

        Uses INSERT OR IGNORE to skip duplicates (same symbol + timestamp).
        """
        async with self._session_factory() as session:
            stmt = (
                insert(FundingRateRecord)
                .values(
                    symbol=symbol,
                    timestamp=funding_timestamp,
                    funding_rate=funding_rate,
                    funding_timestamp=funding_timestamp,
                    mark_price=mark_price,
                    spot_price=spot_price,
                    spread_pct=spread_pct,
                )
                .prefix_with("OR IGNORE")
            )
            await session.execute(stmt)
            await session.commit()

    async def get_funding_rates(
        self,
        symbol: str,
        limit: int = 100,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict]:
        """Query funding rates by symbol with optional date range."""
        async with self._session_factory() as session:
            stmt = select(FundingRateRecord).where(
                FundingRateRecord.symbol == symbol
            )
            if start:
                stmt = stmt.where(FundingRateRecord.timestamp >= start)
            if end:
                stmt = stmt.where(FundingRateRecord.timestamp <= end)
            stmt = stmt.order_by(FundingRateRecord.timestamp.desc()).limit(limit)

            result = await session.execute(stmt)
            records = result.scalars().all()
            return [
                {
                    "symbol": r.symbol,
                    "timestamp": r.timestamp,
                    "funding_rate": r.funding_rate,
                    "funding_timestamp": r.funding_timestamp,
                    "mark_price": r.mark_price,
                    "spot_price": r.spot_price,
                    "spread_pct": r.spread_pct,
                }
                for r in reversed(records)
            ]

    # --- Portfolio Snapshot Operations ---

    async def save_portfolio_snapshot(
        self,
        total_value: float,
        unrealized_pnl: float = 0.0,
        balances: dict[str, float] | None = None,
        positions: list[dict] | None = None,
    ) -> None:
        """Save a portfolio snapshot."""
        async with self._session_factory() as session:
            record = PortfolioSnapshot(
                total_value=total_value,
                unrealized_pnl=unrealized_pnl,
                balances_json=json.dumps(balances or {}),
                positions_json=json.dumps(positions or []),
            )
            session.add(record)
            await session.commit()

    async def get_latest_portfolio_snapshot(self) -> dict | None:
        """Get the most recent portfolio snapshot."""
        async with self._session_factory() as session:
            stmt = (
                select(PortfolioSnapshot)
                .order_by(PortfolioSnapshot.timestamp.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            record = result.scalar_one_or_none()
            if record is None:
                return None
            return {
                "timestamp": record.timestamp,
                "total_value": record.total_value,
                "unrealized_pnl": record.unrealized_pnl,
                "balances": json.loads(record.balances_json),
                "positions": json.loads(record.positions_json),
            }

    async def get_portfolio_snapshots(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Get portfolio snapshots as time-series data for equity curve charting."""
        async with self._session_factory() as session:
            stmt = select(PortfolioSnapshot)
            if start:
                stmt = stmt.where(PortfolioSnapshot.timestamp >= start)
            if end:
                stmt = stmt.where(PortfolioSnapshot.timestamp <= end)
            stmt = stmt.order_by(PortfolioSnapshot.timestamp.desc()).limit(limit)

            result = await session.execute(stmt)
            records = result.scalars().all()
            return [
                {
                    "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                    "total_value": r.total_value,
                    "unrealized_pnl": r.unrealized_pnl,
                }
                for r in reversed(records)
            ]

    # --- Audit Log Operations ---

    async def save_audit_log(
        self,
        event_type: str,
        actor: str = "system",
        details: dict | None = None,
        severity: str = "info",
        timestamp: datetime | None = None,
    ) -> None:
        """Save an immutable audit log entry."""
        async with self._session_factory() as session:
            record = AuditLogRecord(
                timestamp=timestamp or datetime.utcnow(),
                event_type=event_type,
                actor=actor,
                details=json.dumps(details or {}),
                severity=severity,
            )
            session.add(record)
            await session.commit()

    async def get_audit_logs(
        self,
        event_type: str | None = None,
        severity: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        page: int = 1,
        limit: int = 50,
    ) -> dict:
        """Query audit logs with filters and pagination.

        Returns dict with 'logs', 'total', 'page', 'limit', 'total_pages'.
        """
        async with self._session_factory() as session:
            # Build base query
            stmt = select(AuditLogRecord)
            count_stmt = select(func.count(AuditLogRecord.id))

            if event_type:
                stmt = stmt.where(AuditLogRecord.event_type == event_type)
                count_stmt = count_stmt.where(AuditLogRecord.event_type == event_type)
            if severity:
                stmt = stmt.where(AuditLogRecord.severity == severity)
                count_stmt = count_stmt.where(AuditLogRecord.severity == severity)
            if start:
                stmt = stmt.where(AuditLogRecord.timestamp >= start)
                count_stmt = count_stmt.where(AuditLogRecord.timestamp >= start)
            if end:
                stmt = stmt.where(AuditLogRecord.timestamp <= end)
                count_stmt = count_stmt.where(AuditLogRecord.timestamp <= end)

            # Get total count
            total_result = await session.execute(count_stmt)
            total = total_result.scalar() or 0

            # Paginate (newest first)
            offset = (page - 1) * limit
            stmt = stmt.order_by(AuditLogRecord.timestamp.desc())
            stmt = stmt.offset(offset).limit(limit)

            result = await session.execute(stmt)
            records = result.scalars().all()

            logs = [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                    "event_type": r.event_type,
                    "actor": r.actor,
                    "details": json.loads(r.details) if r.details else {},
                    "severity": r.severity,
                }
                for r in records
            ]

            total_pages = max(1, (total + limit - 1) // limit)
            return {
                "logs": logs,
                "total": total,
                "page": page,
                "limit": limit,
                "total_pages": total_pages,
            }

    async def cleanup_old_audit_logs(self, max_age_days: int = 90) -> int:
        """Delete audit log entries older than max_age_days. Returns count deleted."""
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        async with self._session_factory() as session:
            stmt = delete(AuditLogRecord).where(AuditLogRecord.timestamp < cutoff)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount

    # --- Engine Metric Snapshot Operations ---

    async def get_engine_metric_snapshots(
        self,
        engine_name: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Query engine metric snapshots with optional engine/date filters."""
        async with self._session_factory() as session:
            stmt = select(EngineMetricSnapshot)
            if engine_name:
                stmt = stmt.where(
                    EngineMetricSnapshot.engine_name == engine_name
                )
            if start:
                stmt = stmt.where(EngineMetricSnapshot.timestamp >= start)
            if end:
                stmt = stmt.where(EngineMetricSnapshot.timestamp <= end)
            stmt = (
                stmt.order_by(EngineMetricSnapshot.timestamp.desc())
                .limit(limit)
            )

            result = await session.execute(stmt)
            records = result.scalars().all()
            return [
                {
                    "engine_name": r.engine_name,
                    "timestamp": (
                        r.timestamp.isoformat() if r.timestamp else ""
                    ),
                    "total_trades": r.total_trades,
                    "winning_trades": r.winning_trades,
                    "losing_trades": r.losing_trades,
                    "win_rate": r.win_rate,
                    "total_pnl": r.total_pnl,
                    "sharpe_ratio": r.sharpe_ratio,
                    "max_drawdown": r.max_drawdown,
                    "profit_factor": r.profit_factor,
                    "cost_ratio": r.cost_ratio,
                }
                for r in reversed(records)
            ]

    async def get_engine_trades(
        self,
        engine_name: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Query engine trade records with optional engine/date filters."""
        async with self._session_factory() as session:
            stmt = select(EngineTradeRecord)
            if engine_name:
                stmt = stmt.where(
                    EngineTradeRecord.engine_name == engine_name
                )
            if start:
                stmt = stmt.where(EngineTradeRecord.exit_time >= start.isoformat())
            if end:
                stmt = stmt.where(EngineTradeRecord.exit_time <= end.isoformat())
            stmt = (
                stmt.order_by(EngineTradeRecord.exit_time.desc())
                .limit(limit)
            )

            result = await session.execute(stmt)
            records = result.scalars().all()
            return [
                {
                    "engine_name": r.engine_name,
                    "symbol": r.symbol,
                    "side": r.side,
                    "entry_price": r.entry_price,
                    "exit_price": r.exit_price,
                    "quantity": r.quantity,
                    "pnl": r.pnl,
                    "cost": r.cost,
                    "net_pnl": r.net_pnl,
                    "entry_time": r.entry_time,
                    "exit_time": r.exit_time,
                    "hold_time_seconds": r.hold_time_seconds,
                }
                for r in reversed(records)
            ]
