"""Telegram notification service with command handling."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

import structlog
from telegram import Bot, Update

logger = structlog.get_logger()

# Type alias for command callbacks
CommandCallback = Callable[[], Coroutine[Any, Any, str]]


class TelegramNotifier:
    """Sends alerts and handles commands via Telegram bot."""

    def __init__(self, bot_token: str, chat_id: str):
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._bot: Bot | None = None
        self._commands: dict[str, CommandCallback] = {}
        self._polling_task: asyncio.Task | None = None
        self._polling: bool = False
        self._last_update_id: int = 0

    def _get_bot(self) -> Bot:
        if self._bot is None:
            self._bot = Bot(token=self._bot_token)
        return self._bot

    def register_command(self, command: str, callback: CommandCallback) -> None:
        """Register a command handler.

        Args:
            command: Command name without leading slash (e.g., 'stop').
            callback: Async callable that returns a response message string.
        """
        self._commands[command.lower()] = callback

    async def start_command_polling(self, interval: float = 2.0) -> None:
        """Start polling for incoming Telegram commands in a background task."""
        if self._polling:
            return
        self._polling = True
        self._polling_task = asyncio.ensure_future(
            self._poll_loop(interval)
        )
        logger.info("telegram_command_polling_started", interval=interval)

    async def stop_command_polling(self) -> None:
        """Stop the command polling task."""
        self._polling = False
        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
        self._polling_task = None
        logger.info("telegram_command_polling_stopped")

    async def _poll_loop(self, interval: float) -> None:
        """Internal polling loop for incoming messages."""
        while self._polling:
            try:
                await self._process_updates()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.warning("telegram_poll_error", exc_info=True)
            await asyncio.sleep(interval)

    async def _process_updates(self) -> None:
        """Fetch and process new updates from Telegram."""
        bot = self._get_bot()
        try:
            updates = await bot.get_updates(
                offset=self._last_update_id + 1,
                timeout=1,
            )
        except Exception:
            logger.debug("telegram_get_updates_error", exc_info=True)
            return

        for update in updates:
            self._last_update_id = update.update_id
            await self._handle_update(update)

    async def _handle_update(self, update: Update) -> None:
        """Handle a single Telegram update (message with command)."""
        if not update.message or not update.message.text:
            return

        # Only process messages from the configured chat
        chat_id = str(update.message.chat_id)
        if chat_id != str(self._chat_id):
            logger.debug(
                "telegram_ignored_chat",
                chat_id=chat_id,
                expected=self._chat_id,
            )
            return

        text = update.message.text.strip()
        if not text.startswith("/"):
            return

        # Parse command (e.g., "/stop" -> "stop", "/stop@botname" -> "stop")
        parts = text[1:].split()
        command = parts[0].split("@")[0].lower() if parts else ""

        if command in self._commands:
            logger.info("telegram_command_received", command=command)
            try:
                response = await self._commands[command]()
                await self.send_message(response)
            except Exception:
                logger.error(
                    "telegram_command_error",
                    command=command,
                    exc_info=True,
                )
                await self.send_message(f"Error executing /{command}")
        else:
            available = ", ".join(f"/{c}" for c in sorted(self._commands))
            await self.send_message(
                f"Unknown command: /{command}\n"
                f"Available: {available}"
            )

    async def send_message(self, message: str) -> bool:
        """Send a message to the configured chat."""
        if not self._bot_token or not self._chat_id:
            logger.debug("telegram_not_configured")
            return False

        try:
            bot = self._get_bot()
            await bot.send_message(chat_id=self._chat_id, text=message)
            return True
        except Exception as e:
            logger.error("telegram_send_failed", error=str(e))
            return False

    async def notify_trade(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> bool:
        """Send trade notification."""
        msg = f"Trade Executed\nSymbol: {symbol}\nSide: {side}\nQty: {quantity}\nPrice: {price}"
        return await self.send_message(msg)

    async def notify_daily_summary(self, metrics: dict) -> bool:
        """Send daily performance summary."""
        msg = (
            f"Daily Summary\n"
            f"Return: {metrics.get('total_return_pct', 0)}%\n"
            f"Trades: {metrics.get('total_trades', 0)}\n"
            f"Win Rate: {metrics.get('win_rate', 0)}%"
        )
        return await self.send_message(msg)

    async def notify_error(self, error: str) -> bool:
        """Send error alert."""
        return await self.send_message(f"ERROR: {error}")
