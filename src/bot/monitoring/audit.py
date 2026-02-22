"""Audit trail logger for recording all significant bot actions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from bot.data.store import DataStore

logger = structlog.get_logger()


class AuditLogger:
    """Records significant events to a persistent, queryable audit log.

    Events are immutable (append-only). Each event has:
    - event_type: categorized action (e.g., 'trade_executed', 'emergency_stop')
    - actor: who triggered it ('system', 'user', 'telegram')
    - details: JSON-serializable dict with event-specific data
    - severity: 'info', 'warning', or 'critical'
    """

    def __init__(self, store: DataStore | None = None):
        self._store = store

    @property
    def store(self) -> DataStore | None:
        return self._store

    @store.setter
    def store(self, value: DataStore | None) -> None:
        self._store = value

    async def log_event(
        self,
        event_type: str,
        actor: str = "system",
        details: dict | None = None,
        severity: str = "info",
    ) -> None:
        """Record an audit event.

        Args:
            event_type: Type of event (e.g., 'trade_executed', 'emergency_stop').
            actor: Who triggered the event ('system', 'user', 'telegram').
            details: Event-specific data as a dict.
            severity: Event severity ('info', 'warning', 'critical').
        """
        now = datetime.now(timezone.utc)

        # Always log to structlog
        logger.info(
            "audit_event",
            event_type=event_type,
            actor=actor,
            severity=severity,
            details=details,
        )

        # Persist to database if store is available
        if self._store is not None:
            try:
                await self._store.save_audit_log(
                    event_type=event_type,
                    actor=actor,
                    details=details,
                    severity=severity,
                    timestamp=now,
                )
            except Exception:
                logger.warning(
                    "audit_log_save_failed",
                    event_type=event_type,
                    exc_info=True,
                )

    # --- Convenience methods for common events ---

    async def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy: str = "",
        actor: str = "system",
    ) -> None:
        """Log a trade execution."""
        await self.log_event(
            event_type="trade_executed",
            actor=actor,
            details={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "strategy": strategy,
            },
        )

    async def log_order_cancelled(
        self,
        order_id: str,
        symbol: str,
        reason: str = "",
        actor: str = "system",
    ) -> None:
        """Log an order cancellation."""
        await self.log_event(
            event_type="order_cancelled",
            actor=actor,
            details={
                "order_id": order_id,
                "symbol": symbol,
                "reason": reason,
            },
        )

    async def log_position_closed(
        self,
        symbol: str,
        quantity: float,
        pnl: float,
        exit_type: str = "",
        actor: str = "system",
    ) -> None:
        """Log a position closure."""
        await self.log_event(
            event_type="position_closed",
            actor=actor,
            details={
                "symbol": symbol,
                "quantity": quantity,
                "pnl": round(pnl, 2),
                "exit_type": exit_type,
            },
        )

    async def log_strategy_toggled(
        self,
        name: str,
        active: bool,
        actor: str = "user",
    ) -> None:
        """Log a strategy toggle."""
        await self.log_event(
            event_type="strategy_toggled",
            actor=actor,
            details={
                "strategy": name,
                "active": active,
            },
        )

    async def log_config_changed(
        self,
        changed: list[str],
        previous: dict | None = None,
        actor: str = "user",
    ) -> None:
        """Log a configuration change."""
        await self.log_event(
            event_type="config_changed",
            actor=actor,
            details={
                "changed_keys": changed,
                "previous": previous,
            },
        )

    async def log_emergency_stop(
        self,
        reason: str,
        cancelled_orders: int = 0,
        actor: str = "system",
    ) -> None:
        """Log an emergency stop activation."""
        await self.log_event(
            event_type="emergency_stop",
            actor=actor,
            severity="critical",
            details={
                "reason": reason,
                "cancelled_orders": cancelled_orders,
            },
        )

    async def log_emergency_resume(
        self,
        previous_reason: str | None = None,
        actor: str = "system",
    ) -> None:
        """Log trading resumption after emergency stop."""
        await self.log_event(
            event_type="emergency_resume",
            actor=actor,
            severity="warning",
            details={"previous_reason": previous_reason},
        )

    async def log_bot_started(self, mode: str, symbols: list[str]) -> None:
        """Log bot startup."""
        await self.log_event(
            event_type="bot_started",
            details={"mode": mode, "symbols": symbols},
        )

    async def log_bot_stopped(self) -> None:
        """Log bot shutdown."""
        await self.log_event(event_type="bot_stopped")

    async def log_auth_login(self, username: str, success: bool) -> None:
        """Log an authentication attempt."""
        event_type = "auth_login" if success else "auth_failed"
        severity = "info" if success else "warning"
        await self.log_event(
            event_type=event_type,
            actor="user",
            severity=severity,
            details={"username": username},
        )

    async def log_preflight_result(
        self,
        overall: str,
        failures: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        """Log pre-flight check results."""
        severity = "critical" if failures else ("warning" if warnings else "info")
        await self.log_event(
            event_type="preflight_result",
            severity=severity,
            details={
                "overall": overall,
                "failures": failures or [],
                "warnings": warnings or [],
            },
        )

    async def log_reconciliation_result(
        self,
        has_discrepancies: bool,
        matched: int = 0,
        discrepancies: int = 0,
    ) -> None:
        """Log reconciliation result."""
        severity = "warning" if has_discrepancies else "info"
        await self.log_event(
            event_type="reconciliation_result",
            severity=severity,
            details={
                "has_discrepancies": has_discrepancies,
                "matched": matched,
                "discrepancies": discrepancies,
            },
        )
