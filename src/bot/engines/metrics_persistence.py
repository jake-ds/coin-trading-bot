"""Metrics persistence layer — save/restore engine trades and metrics to DB."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import structlog

from bot.data.models import EngineMetricSnapshot, EngineTradeRecord
from bot.engines.tracker import EngineTracker, TradeRecord

logger = structlog.get_logger(__name__)


class MetricsPersistence:
    """Persists EngineTracker data to SQLite via DataStore's session factory.

    Saves trades as they happen and periodically snapshots engine metrics.
    On startup, restores recent trades back into the tracker.
    """

    def __init__(self, data_store, tracker: EngineTracker):
        self._data_store = data_store
        self._tracker = tracker

    # ------------------------------------------------------------------
    # Save operations
    # ------------------------------------------------------------------

    async def save_trade(
        self, engine_name: str, trade: TradeRecord
    ) -> None:
        """Save a single trade to the database."""
        try:
            async with self._data_store._session_factory() as session:
                record = EngineTradeRecord(
                    engine_name=engine_name,
                    symbol=trade.symbol,
                    side=trade.side,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    quantity=trade.quantity,
                    pnl=trade.pnl,
                    cost=trade.cost,
                    net_pnl=trade.net_pnl,
                    entry_time=trade.entry_time,
                    exit_time=trade.exit_time,
                    hold_time_seconds=trade.hold_time_seconds,
                )
                session.add(record)
                await session.commit()
        except Exception as e:
            logger.error("save_trade_failed", engine=engine_name, error=str(e))

    async def save_metrics_snapshot(self) -> None:
        """Snapshot current metrics for all engines to the database."""
        try:
            all_metrics = self._tracker.get_all_metrics(window_hours=24)
            now = datetime.now(timezone.utc)

            async with self._data_store._session_factory() as session:
                for engine_name, metrics in all_metrics.items():
                    snapshot = EngineMetricSnapshot(
                        engine_name=engine_name,
                        timestamp=now,
                        total_trades=metrics.total_trades,
                        winning_trades=metrics.winning_trades,
                        losing_trades=metrics.losing_trades,
                        win_rate=round(metrics.win_rate, 4),
                        total_pnl=round(metrics.total_pnl, 4),
                        sharpe_ratio=round(metrics.sharpe_ratio, 4),
                        max_drawdown=round(metrics.max_drawdown, 4),
                        profit_factor=round(metrics.profit_factor, 4),
                        cost_ratio=round(metrics.cost_ratio, 4),
                    )
                    session.add(snapshot)
                await session.commit()

            logger.info(
                "metrics_snapshot_saved",
                engines=len(all_metrics),
            )
        except Exception as e:
            logger.error("save_metrics_snapshot_failed", error=str(e))

    # ------------------------------------------------------------------
    # Load / restore operations
    # ------------------------------------------------------------------

    async def load_trades(
        self, since_hours: float = 24
    ) -> dict[str, list[TradeRecord]]:
        """Load recent trades from the database."""
        from sqlalchemy import select

        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        cutoff_str = cutoff.isoformat()

        result: dict[str, list[TradeRecord]] = {}
        try:
            async with self._data_store._session_factory() as session:
                stmt = (
                    select(EngineTradeRecord)
                    .where(EngineTradeRecord.exit_time >= cutoff_str)
                    .order_by(EngineTradeRecord.exit_time)
                )
                rows = await session.execute(stmt)
                for row in rows.scalars().all():
                    trade = TradeRecord(
                        engine_name=row.engine_name,
                        symbol=row.symbol,
                        side=row.side,
                        entry_price=row.entry_price,
                        exit_price=row.exit_price,
                        quantity=row.quantity,
                        pnl=row.pnl,
                        cost=row.cost,
                        net_pnl=row.net_pnl,
                        entry_time=row.entry_time,
                        exit_time=row.exit_time,
                        hold_time_seconds=row.hold_time_seconds,
                    )
                    result.setdefault(row.engine_name, []).append(trade)
        except Exception as e:
            logger.error("load_trades_failed", error=str(e))

        return result

    async def restore_tracker(self) -> None:
        """Restore recent trades into the tracker on startup."""
        trades_by_engine = await self.load_trades(since_hours=24)
        total = 0
        for engine_name, trades in trades_by_engine.items():
            self._tracker.bulk_load_trades(engine_name, trades)
            total += len(trades)

        if total > 0:
            logger.info(
                "tracker_restored",
                engines=len(trades_by_engine),
                total_trades=total,
            )

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _snapshot_loop(
        self, interval_minutes: float = 5.0
    ) -> None:
        """Periodically save metrics snapshots."""
        await asyncio.sleep(60)  # Initial delay: 1 minute
        while True:
            try:
                await self.save_metrics_snapshot()
            except Exception as e:
                logger.error("snapshot_loop_error", error=str(e))
            await asyncio.sleep(interval_minutes * 60)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self, max_days: int = 90) -> None:
        """Remove old records beyond retention period."""
        from sqlalchemy import delete

        cutoff = datetime.now(timezone.utc) - timedelta(days=max_days)
        cutoff_str = cutoff.isoformat()

        try:
            async with self._data_store._session_factory() as session:
                await session.execute(
                    delete(EngineTradeRecord).where(
                        EngineTradeRecord.exit_time < cutoff_str
                    )
                )
                await session.execute(
                    delete(EngineMetricSnapshot).where(
                        EngineMetricSnapshot.timestamp < cutoff
                    )
                )
                await session.commit()
            logger.info("metrics_cleanup_completed", max_days=max_days)
        except Exception as e:
            logger.error("cleanup_failed", error=str(e))
