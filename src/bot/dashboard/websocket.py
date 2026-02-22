"""WebSocket connection manager for real-time dashboard updates."""

import asyncio
import json
import time
from typing import Any

import structlog
from fastapi import WebSocket

logger = structlog.get_logger()


class ConnectionManager:
    """Manages WebSocket connections and broadcasts messages with rate limiting.

    Rate limits broadcasts to max 1 per second by debouncing rapid updates.
    """

    def __init__(self, min_broadcast_interval: float = 1.0):
        self._connections: list[WebSocket] = []
        self._min_interval = min_broadcast_interval
        self._last_broadcast_time: float = 0.0
        self._pending_message: dict[str, Any] | None = None
        self._debounce_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @property
    def active_connections(self) -> int:
        """Number of active WebSocket connections."""
        return len(self._connections)

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self._connections.append(websocket)
        logger.info("ws_client_connected", total=len(self._connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket connection."""
        if websocket in self._connections:
            self._connections.remove(websocket)
        logger.info("ws_client_disconnected", total=len(self._connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients, rate-limited.

        If called more frequently than min_broadcast_interval, the message
        is queued and sent after the interval elapses (debounce).
        """
        if not self._connections:
            return

        now = time.monotonic()
        elapsed = now - self._last_broadcast_time

        if elapsed >= self._min_interval:
            await self._send_to_all(message)
        else:
            # Queue the message and schedule a delayed send
            async with self._lock:
                self._pending_message = message
                if self._debounce_task is None or self._debounce_task.done():
                    delay = self._min_interval - elapsed
                    self._debounce_task = asyncio.ensure_future(
                        self._delayed_send(delay)
                    )

    async def broadcast_immediate(self, message: dict[str, Any]) -> None:
        """Broadcast a message immediately, bypassing rate limiting.

        Use for critical events like trade execution or emergency actions.
        """
        if not self._connections:
            return
        await self._send_to_all(message)

    async def _delayed_send(self, delay: float) -> None:
        """Send the pending message after a delay."""
        await asyncio.sleep(delay)
        async with self._lock:
            if self._pending_message is not None:
                await self._send_to_all(self._pending_message)
                self._pending_message = None

    async def _send_to_all(self, message: dict[str, Any]) -> None:
        """Send a JSON message to all connected clients."""
        self._last_broadcast_time = time.monotonic()
        data = json.dumps(message)
        disconnected = []
        for ws in self._connections:
            try:
                await ws.send_text(data)
            except Exception:
                disconnected.append(ws)
        # Clean up disconnected clients
        for ws in disconnected:
            if ws in self._connections:
                self._connections.remove(ws)

    async def send_personal(
        self, websocket: WebSocket, message: dict[str, Any]
    ) -> None:
        """Send a message to a single client."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(websocket)


# Module-level singleton — shared across the app
ws_manager = ConnectionManager()
