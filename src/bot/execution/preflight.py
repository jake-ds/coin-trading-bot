"""Pre-flight safety checks for live trading mode.

Runs comprehensive checks before the bot starts live trading to prevent
accidents and ensure proper configuration.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class CheckStatus(str, Enum):
    """Result status for a single pre-flight check."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""

    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class PreFlightResult:
    """Aggregate result of all pre-flight checks."""

    checks: list[CheckResult] = field(default_factory=list)
    overall: CheckStatus = CheckStatus.PASS

    @property
    def has_failures(self) -> bool:
        return any(c.status == CheckStatus.FAIL for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.status == CheckStatus.WARN for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.value,
            "checks": [c.to_dict() for c in self.checks],
            "has_failures": self.has_failures,
            "has_warnings": self.has_warnings,
        }

    def format_summary(self) -> str:
        """Format a human-readable summary of pre-flight results."""
        lines = ["Pre-flight Check Results:", "=" * 40]
        for check in self.checks:
            status_icon = {
                CheckStatus.PASS: "PASS",
                CheckStatus.WARN: "WARN",
                CheckStatus.FAIL: "FAIL",
            }[check.status]
            lines.append(f"  [{status_icon}] {check.name}: {check.message}")
        lines.append("=" * 40)
        lines.append(f"Overall: {self.overall.value}")
        return "\n".join(lines)


class PreFlightChecker:
    """Runs pre-flight safety checks before live trading starts.

    Checks include:
    1. API key validity (exchange connection test)
    2. Sufficient balance
    3. Symbol availability
    4. Rate limit configuration
    5. Stop-loss configured
    6. Daily loss limit configured
    7. Dashboard password changed
    8. Paper validation report with GO recommendation
    """

    def __init__(
        self,
        min_balance_usd: float = 100.0,
        validation_report_dir: str = "data",
    ):
        self._min_balance_usd = min_balance_usd
        self._validation_report_dir = validation_report_dir
        self._last_result: PreFlightResult | None = None

    @property
    def last_result(self) -> PreFlightResult | None:
        return self._last_result

    async def run_all_checks(
        self,
        settings: Any,
        exchanges: list | None = None,
        symbols: list[str] | None = None,
        rate_limit_enabled: bool = True,
    ) -> PreFlightResult:
        """Run all pre-flight checks and return the aggregate result.

        Args:
            settings: Bot Settings object.
            exchanges: List of exchange adapters to test connectivity.
            symbols: Symbols the bot is configured to trade.
            rate_limit_enabled: Whether rate limiting is enabled.

        Returns:
            PreFlightResult with status of all checks.
        """
        result = PreFlightResult()

        # 1. API key validity
        result.checks.append(
            await self._check_api_keys(exchanges)
        )

        # 2. Sufficient balance
        result.checks.append(
            await self._check_balance(exchanges)
        )

        # 3. Symbol availability
        result.checks.append(
            await self._check_symbols(exchanges, symbols or [])
        )

        # 4. Rate limit headroom
        result.checks.append(
            self._check_rate_limit(rate_limit_enabled)
        )

        # 5. Stop-loss configured
        result.checks.append(
            self._check_stop_loss(settings)
        )

        # 6. Daily loss limit configured
        result.checks.append(
            self._check_daily_loss_limit(settings)
        )

        # 7. Dashboard password changed
        result.checks.append(
            self._check_password_changed(settings)
        )

        # 8. Paper validation passed
        result.checks.append(
            self._check_validation_report()
        )

        # Determine overall status
        if any(c.status == CheckStatus.FAIL for c in result.checks):
            result.overall = CheckStatus.FAIL
        elif any(c.status == CheckStatus.WARN for c in result.checks):
            result.overall = CheckStatus.WARN
        else:
            result.overall = CheckStatus.PASS

        self._last_result = result
        return result

    async def _check_api_keys(
        self, exchanges: list | None
    ) -> CheckResult:
        """Check 1: Verify exchange API keys work by fetching balance."""
        if not exchanges:
            return CheckResult(
                name="api_key_validity",
                status=CheckStatus.FAIL,
                message="No exchange adapters configured",
            )

        for exchange in exchanges:
            try:
                await exchange.get_balance()
            except Exception as e:
                return CheckResult(
                    name="api_key_validity",
                    status=CheckStatus.FAIL,
                    message=f"Exchange {exchange.name} connection failed: {e}",
                    details={"exchange": exchange.name, "error": str(e)},
                )

        names = [e.name for e in exchanges]
        return CheckResult(
            name="api_key_validity",
            status=CheckStatus.PASS,
            message=f"Exchange connection verified: {', '.join(names)}",
            details={"exchanges": names},
        )

    async def _check_balance(
        self, exchanges: list | None
    ) -> CheckResult:
        """Check 2: Verify sufficient balance on at least one exchange."""
        if not exchanges:
            return CheckResult(
                name="sufficient_balance",
                status=CheckStatus.FAIL,
                message="No exchanges available to check balance",
            )

        for exchange in exchanges:
            try:
                balances = await exchange.get_balance()
                # Sum up all balances in USD-equivalent currencies
                usd_currencies = {"USDT", "USD", "BUSD", "USDC", "TUSD", "DAI"}
                total_usd = sum(
                    amount
                    for currency, amount in balances.items()
                    if currency.upper() in usd_currencies
                )
                if total_usd >= self._min_balance_usd:
                    return CheckResult(
                        name="sufficient_balance",
                        status=CheckStatus.PASS,
                        message=f"Balance ${total_usd:.2f} >= minimum ${self._min_balance_usd:.2f}",
                        details={
                            "balance_usd": total_usd,
                            "minimum_usd": self._min_balance_usd,
                            "exchange": exchange.name,
                        },
                    )
            except Exception:
                continue

        return CheckResult(
            name="sufficient_balance",
            status=CheckStatus.FAIL,
            message=f"Insufficient balance. Minimum required: ${self._min_balance_usd:.2f}",
            details={"minimum_usd": self._min_balance_usd},
        )

    async def _check_symbols(
        self,
        exchanges: list | None,
        symbols: list[str],
    ) -> CheckResult:
        """Check 3: Verify all configured symbols exist on the exchange."""
        if not symbols:
            return CheckResult(
                name="symbol_availability",
                status=CheckStatus.WARN,
                message="No symbols configured",
            )

        if not exchanges:
            return CheckResult(
                name="symbol_availability",
                status=CheckStatus.FAIL,
                message="No exchanges available to verify symbols",
            )

        # Try to fetch ticker for each symbol on the first exchange
        exchange = exchanges[0]
        missing_symbols = []
        verified_symbols = []

        for symbol in symbols:
            try:
                await exchange.get_ticker(symbol)
                verified_symbols.append(symbol)
            except Exception:
                missing_symbols.append(symbol)

        if missing_symbols:
            return CheckResult(
                name="symbol_availability",
                status=CheckStatus.FAIL,
                message=f"Symbols not available on {exchange.name}: {', '.join(missing_symbols)}",
                details={
                    "missing": missing_symbols,
                    "verified": verified_symbols,
                    "exchange": exchange.name,
                },
            )

        return CheckResult(
            name="symbol_availability",
            status=CheckStatus.PASS,
            message=f"All {len(symbols)} symbols verified on {exchange.name}",
            details={
                "verified": verified_symbols,
                "exchange": exchange.name,
            },
        )

    def _check_rate_limit(self, rate_limit_enabled: bool) -> CheckResult:
        """Check 4: Verify rate limiter is configured."""
        if rate_limit_enabled:
            return CheckResult(
                name="rate_limit_configured",
                status=CheckStatus.PASS,
                message="Rate limiting is enabled",
            )

        return CheckResult(
            name="rate_limit_configured",
            status=CheckStatus.WARN,
            message="Rate limiting is disabled. Exchange may ban your IP.",
            details={"rate_limit_enabled": False},
        )

    def _check_stop_loss(self, settings: Any) -> CheckResult:
        """Check 5: Verify stop-loss is configured (required for live trading)."""
        stop_loss_pct = getattr(settings, "stop_loss_pct", 0)
        if stop_loss_pct > 0:
            return CheckResult(
                name="stop_loss_configured",
                status=CheckStatus.PASS,
                message=f"Stop-loss set to {stop_loss_pct}%",
                details={"stop_loss_pct": stop_loss_pct},
            )

        return CheckResult(
            name="stop_loss_configured",
            status=CheckStatus.FAIL,
            message="Stop-loss not configured. Live trading without stop-loss is not allowed.",
            details={"stop_loss_pct": stop_loss_pct},
        )

    def _check_daily_loss_limit(self, settings: Any) -> CheckResult:
        """Check 6: Verify daily loss limit is configured."""
        daily_loss_limit_pct = getattr(settings, "daily_loss_limit_pct", 0)
        if daily_loss_limit_pct > 0:
            return CheckResult(
                name="daily_loss_limit_configured",
                status=CheckStatus.PASS,
                message=f"Daily loss limit set to {daily_loss_limit_pct}%",
                details={"daily_loss_limit_pct": daily_loss_limit_pct},
            )

        return CheckResult(
            name="daily_loss_limit_configured",
            status=CheckStatus.FAIL,
            message=(
                "Daily loss limit not configured. "
                "Live trading without daily loss limit is not allowed."
            ),
            details={"daily_loss_limit_pct": daily_loss_limit_pct},
        )

    def _check_password_changed(self, settings: Any) -> CheckResult:
        """Check 7: Verify dashboard password has been changed from default."""
        password = getattr(settings, "dashboard_password", "changeme")
        if password != "changeme":
            return CheckResult(
                name="password_changed",
                status=CheckStatus.PASS,
                message="Dashboard password has been changed from default",
            )

        return CheckResult(
            name="password_changed",
            status=CheckStatus.WARN,
            message="Dashboard password is still 'changeme'. Auth is disabled.",
            details={"auth_disabled": True},
        )

    def _check_validation_report(self) -> CheckResult:
        """Check 8: Check for recent validation report with GO recommendation."""
        report_dir = self._validation_report_dir
        if not os.path.isdir(report_dir):
            return CheckResult(
                name="paper_validation",
                status=CheckStatus.WARN,
                message="No validation report directory found. Run --validate first.",
                details={"report_dir": report_dir},
            )

        # Find most recent validation report
        pattern = os.path.join(report_dir, "validation_report_*.json")
        reports = sorted(glob.glob(pattern), reverse=True)

        if not reports:
            return CheckResult(
                name="paper_validation",
                status=CheckStatus.WARN,
                message="No validation reports found. Run --validate first.",
                details={"report_dir": report_dir},
            )

        latest_report_path = reports[0]
        try:
            with open(latest_report_path) as f:
                report_data = json.load(f)

            recommendation = report_data.get("recommendation", "")
            if recommendation == "GO":
                return CheckResult(
                    name="paper_validation",
                    status=CheckStatus.PASS,
                    message=(
                        f"Validation passed with GO recommendation: "
                        f"{os.path.basename(latest_report_path)}"
                    ),
                    details={
                        "report_file": os.path.basename(latest_report_path),
                        "recommendation": recommendation,
                        "total_return_pct": report_data.get("total_return_pct"),
                        "win_rate_pct": report_data.get("win_rate_pct"),
                        "sharpe_ratio": report_data.get("sharpe_ratio"),
                    },
                )
            else:
                return CheckResult(
                    name="paper_validation",
                    status=CheckStatus.WARN,
                    message=(
                        f"Latest validation report recommends NO-GO: "
                        f"{os.path.basename(latest_report_path)}"
                    ),
                    details={
                        "report_file": os.path.basename(latest_report_path),
                        "recommendation": recommendation,
                    },
                )
        except (json.JSONDecodeError, OSError) as e:
            return CheckResult(
                name="paper_validation",
                status=CheckStatus.WARN,
                message=f"Could not read validation report: {e}",
                details={"report_file": os.path.basename(latest_report_path)},
            )
