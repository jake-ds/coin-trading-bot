"""Telegram notification service."""

import structlog
from telegram import Bot

logger = structlog.get_logger()


class TelegramNotifier:
    """Sends alerts via Telegram bot."""

    def __init__(self, bot_token: str, chat_id: str):
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._bot: Bot | None = None

    def _get_bot(self) -> Bot:
        if self._bot is None:
            self._bot = Bot(token=self._bot_token)
        return self._bot

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
