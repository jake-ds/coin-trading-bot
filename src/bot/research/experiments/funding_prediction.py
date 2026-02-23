"""Funding rate prediction experiment.

Analyzes historical funding rate patterns (time-of-day, day-of-week)
to predict direction for the funding_arb engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import structlog

from bot.engines.tuner import ParamChange
from bot.research.base import ResearchTask
from bot.research.report import ResearchReport

if TYPE_CHECKING:
    from bot.research.data_provider import HistoricalDataProvider

logger = structlog.get_logger(__name__)


class FundingPredictionExperiment(ResearchTask):
    """Analyze funding rate patterns to predict direction."""

    @property
    def target_engine(self) -> str:
        return "funding_rate_arb"

    def __init__(self, data_provider: HistoricalDataProvider | None = None) -> None:
        super().__init__(data_provider=data_provider)
        self._last_report: ResearchReport | None = None
        self._real_funding_data: list[dict] | None = None

    def _fetch_real_funding_rates(self) -> list[float] | None:
        """Fetch real funding rates and store raw data for pattern analysis."""
        if not self.data_provider:
            return None
        try:
            records = self._run_async(
                self.data_provider.get_funding_rates("BTC/USDT", lookback_days=30)
            )
            if len(records) >= 24:
                self._real_funding_data = records
                return [r["funding_rate"] for r in records]
        except Exception:
            logger.warning("funding_prediction_real_data_fetch_failed", exc_info=True)
        return None

    def _analyze_time_patterns(self) -> dict:
        """Analyze time-of-day and day-of-week patterns from real funding data."""
        if not self._real_funding_data:
            return {}
        hourly_rates: dict[int, list[float]] = {}
        daily_rates: dict[int, list[float]] = {}
        for record in self._real_funding_data:
            ts = record.get("timestamp")
            rate = record.get("funding_rate", 0.0)
            if ts is None:
                continue
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            elif isinstance(ts, datetime):
                dt = ts
            else:
                continue
            hour = dt.hour
            dow = dt.weekday()
            hourly_rates.setdefault(hour, []).append(rate)
            daily_rates.setdefault(dow, []).append(rate)

        # Average by settlement hours (0, 8, 16 UTC)
        settlement_hours = [0, 8, 16]
        hourly_avg = {}
        for h in settlement_hours:
            rates_at_h = hourly_rates.get(h, [])
            if rates_at_h:
                hourly_avg[f"hour_{h}"] = round(float(np.mean(rates_at_h)), 6)

        # Average by day of week
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily_avg = {}
        for dow_idx in range(7):
            rates_at_d = daily_rates.get(dow_idx, [])
            if rates_at_d:
                daily_avg[day_names[dow_idx]] = round(float(np.mean(rates_at_d)), 6)

        # Best entry hour
        best_hour = None
        best_avg = -float("inf")
        for h in settlement_hours:
            rates_at_h = hourly_rates.get(h, [])
            if rates_at_h:
                avg = float(np.mean(rates_at_h))
                if avg > best_avg:
                    best_avg = avg
                    best_hour = h

        # Average positive rate
        positive_rates = [
            r["funding_rate"]
            for r in self._real_funding_data
            if r.get("funding_rate", 0) > 0
        ]
        avg_positive_rate = round(float(np.mean(positive_rates)), 6) if positive_rates else 0.0

        return {
            "hourly_pattern": hourly_avg,
            "daily_pattern": daily_avg,
            "best_entry_hour": best_hour,
            "avg_positive_rate": avg_positive_rate,
        }

    def run_experiment(self, **kwargs: object) -> ResearchReport:
        # Priority: kwargs > data_provider > synthetic
        funding_rates = kwargs.get("funding_rates")
        data_source = "kwargs"
        self._real_funding_data = None
        if funding_rates is None:
            real_rates = self._fetch_real_funding_rates()
            if real_rates is not None:
                funding_rates = real_rates
                data_source = "real"
            else:
                funding_rates = self._generate_synthetic_rates()
                data_source = "synthetic"
        rates = list(funding_rates)  # type: ignore[arg-type]

        if len(rates) < 24:
            self._last_report = ResearchReport(
                experiment_name="funding_prediction",
                hypothesis="Funding rates have exploitable time-of-day patterns",
                methodology="Time-series analysis of funding rates",
                data_period=f"{len(rates)} data points",
                results={"error": "Insufficient data"},
                conclusion="Insufficient data for analysis",
            )
            return self._last_report

        arr = np.array(rates)

        # Analyze 8-hour cycle patterns (funding settles every 8h)
        cycle_len = 8
        n_cycles = len(arr) // cycle_len
        if n_cycles < 3:
            cycle_means = []
        else:
            trimmed = arr[: n_cycles * cycle_len]
            cycles = trimmed.reshape(n_cycles, cycle_len)
            cycle_means = [float(np.mean(cycles[:, i])) for i in range(cycle_len)]

        # Overall statistics
        mean_rate = float(np.mean(arr))
        std_rate = float(np.std(arr))
        positive_pct = float(np.mean(arr > 0))

        # Directional persistence (autocorrelation)
        if len(arr) > 1:
            autocorr = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
        else:
            autocorr = 0.0

        # Is there a predictable pattern?
        predictable = abs(autocorr) > 0.3 or (positive_pct > 0.7 or positive_pct < 0.3)

        # Time-of-day / day-of-week pattern analysis (real data only)
        time_patterns = self._analyze_time_patterns()

        results: dict[str, object] = {
            "mean_rate": round(mean_rate, 6),
            "std_rate": round(std_rate, 6),
            "positive_pct": round(positive_pct, 4),
            "autocorrelation": round(autocorr, 4),
            "n_data_points": len(rates),
            "cycle_pattern": [round(m, 6) for m in cycle_means],
            "data_source": data_source,
        }
        if time_patterns:
            results.update(time_patterns)

        # Recommendation: if rates are persistently positive, lower the min_rate threshold
        recommended: list[ParamChange] = []
        if predictable and positive_pct > 0.6 and mean_rate > 0.0001:
            new_min_rate = max(0.0001, round(mean_rate * 0.5, 6))
            recommended.append(
                ParamChange(
                    engine_name="funding_rate_arb",
                    param_name="funding_arb_min_rate",
                    old_value=0.0003,
                    new_value=new_min_rate,
                    reason=(
                        f"Funding prediction: {positive_pct*100:.0f}% positive rates, "
                        f"mean={mean_rate:.6f}, autocorr={autocorr:.2f}"
                    ),
                ),
            )

        self._last_report = ResearchReport(
            experiment_name="funding_prediction",
            hypothesis="Funding rates have exploitable time-of-day patterns",
            methodology=(
                f"Analyze {len(rates)} funding rate observations. "
                f"Check 8-hour cycle patterns, autocorrelation, directional bias."
            ),
            data_period=f"{len(rates)} observations ({n_cycles} complete 8h cycles)",
            results=results,
            conclusion=(
                f"Mean rate: {mean_rate:.6f}, {positive_pct*100:.0f}% positive. "
                f"Autocorrelation: {autocorr:.2f}. "
                + (
                    "Pattern detected — rates are predictable."
                    if predictable
                    else "No strong pattern detected."
                )
            ),
            improvement_significant=predictable,
            recommended_changes=recommended,
        )

        return self._last_report

    def apply_findings(self) -> list[ParamChange]:
        if self._last_report and self._last_report.recommended_changes:
            return self._last_report.recommended_changes
        return []

    @staticmethod
    def _generate_synthetic_rates(n: int = 200) -> list[float]:
        rng = np.random.default_rng(99)
        # Simulate funding rates with positive bias and autocorrelation
        rates = [0.0003]
        for _ in range(n - 1):
            # Mean-reverting with positive drift
            rate = 0.7 * rates[-1] + 0.3 * 0.0003 + rng.normal(0, 0.0001)
            rates.append(rate)
        return rates
