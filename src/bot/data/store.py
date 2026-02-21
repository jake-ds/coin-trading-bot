"""DataStore - async CRUD operations for persistent storage."""

import json
from datetime import datetime

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bot.data.models import Base, OHLCVRecord, PortfolioSnapshot, TradeRecord
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
