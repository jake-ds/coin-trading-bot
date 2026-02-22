"""Tests for structured logging setup."""

from bot.monitoring.logger import setup_logging


class TestSetupLogging:
    def test_setup_info_level(self):
        setup_logging("INFO")

    def test_setup_debug_level(self):
        setup_logging("DEBUG")

    def test_setup_warning_level(self):
        setup_logging("WARNING")
