"""Tests for Telegram notification service."""

from unittest.mock import AsyncMock, patch

import pytest

from bot.monitoring.telegram import TelegramNotifier


class TestTelegramNotifier:
    def test_init(self):
        tn = TelegramNotifier(bot_token="token", chat_id="123")
        assert tn._bot_token == "token"
        assert tn._chat_id == "123"

    @pytest.mark.asyncio
    async def test_send_message_not_configured(self):
        tn = TelegramNotifier(bot_token="", chat_id="")
        result = await tn.send_message("test")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        tn = TelegramNotifier(bot_token="token", chat_id="123")
        mock_bot = AsyncMock()
        tn._bot = mock_bot

        result = await tn.send_message("hello")
        assert result is True
        mock_bot.send_message.assert_called_once_with(chat_id="123", text="hello")

    @pytest.mark.asyncio
    async def test_send_message_failure(self):
        tn = TelegramNotifier(bot_token="token", chat_id="123")
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = Exception("API error")
        tn._bot = mock_bot

        result = await tn.send_message("hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_notify_trade(self):
        tn = TelegramNotifier(bot_token="token", chat_id="123")
        mock_bot = AsyncMock()
        tn._bot = mock_bot

        result = await tn.notify_trade("BTC/USDT", "BUY", 0.1, 50000.0)
        assert result is True
        call_args = mock_bot.send_message.call_args
        assert "BTC/USDT" in call_args.kwargs["text"]
        assert "BUY" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_notify_daily_summary(self):
        tn = TelegramNotifier(bot_token="token", chat_id="123")
        mock_bot = AsyncMock()
        tn._bot = mock_bot

        metrics = {"total_return_pct": 5.0, "total_trades": 10, "win_rate": 60.0}
        result = await tn.notify_daily_summary(metrics)
        assert result is True
        call_args = mock_bot.send_message.call_args
        assert "5.0%" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_notify_error(self):
        tn = TelegramNotifier(bot_token="token", chat_id="123")
        mock_bot = AsyncMock()
        tn._bot = mock_bot

        result = await tn.notify_error("Something broke")
        assert result is True
        call_args = mock_bot.send_message.call_args
        assert "ERROR" in call_args.kwargs["text"]

    def test_get_bot_lazy_init(self):
        tn = TelegramNotifier(bot_token="test-token", chat_id="123")
        assert tn._bot is None
        with patch("bot.monitoring.telegram.Bot") as mock_bot_cls:
            bot = tn._get_bot()
            mock_bot_cls.assert_called_once_with(token="test-token")
            assert bot is not None
