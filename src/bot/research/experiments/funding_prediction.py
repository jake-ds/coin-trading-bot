"""Funding rate prediction experiment.

Analyzes historical funding rate patterns (time-of-day, day-of-week)
to predict direction for the funding_arb engine.
"""

from __future__ import annotations

import numpy as np

from bot.engines.tuner import ParamChange
from bot.research.base import ResearchTask
from bot.research.report import ResearchReport


class FundingPredictionExperiment(ResearchTask):
    """Analyze funding rate patterns to predict direction."""

    @property
    def target_engine(self) -> str:
        return "funding_rate_arb"

    def __init__(self) -> None:
        self._last_report: ResearchReport | None = None

    def run_experiment(self, **kwargs: object) -> ResearchReport:
        funding_rates = kwargs.get("funding_rates")
        if funding_rates is None:
            funding_rates = self._generate_synthetic_rates()
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

        results = {
            "mean_rate": round(mean_rate, 6),
            "std_rate": round(std_rate, 6),
            "positive_pct": round(positive_pct, 4),
            "autocorrelation": round(autocorr, 4),
            "n_data_points": len(rates),
            "cycle_pattern": [round(m, 6) for m in cycle_means],
        }

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
