"""Cointegration experiment for stat_arb pairs.

Runs Engle-Granger cointegration test to identify truly cointegrated pairs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog
from statsmodels.tsa.stattools import coint

from bot.engines.tuner import ParamChange
from bot.research.base import ResearchTask
from bot.research.report import ResearchReport

if TYPE_CHECKING:
    from bot.research.data_provider import HistoricalDataProvider

logger = structlog.get_logger(__name__)


class CointegrationExperiment(ResearchTask):
    """Test price pairs for cointegration to validate stat_arb assumptions."""

    @property
    def target_engine(self) -> str:
        return "stat_arb"

    def __init__(
        self,
        data_provider: HistoricalDataProvider | None = None,
        stat_arb_pairs: list[list[str]] | None = None,
    ) -> None:
        super().__init__(data_provider=data_provider)
        self._last_report: ResearchReport | None = None
        self._stat_arb_pairs = stat_arb_pairs or [
            ["BTC/USDT", "ETH/USDT"],
            ["SOL/USDT", "ETH/USDT"],
        ]

    def _fetch_real_pairs(self) -> dict[str, tuple[list[float], list[float]]] | None:
        """Fetch real price data for configured stat_arb pairs."""
        if not self.data_provider:
            return None
        try:
            # Collect all unique symbols
            all_symbols = list({s for pair in self._stat_arb_pairs for s in pair})
            prices_map = self._run_async(
                self.data_provider.get_multi_prices(all_symbols, "1h", lookback_days=60)
            )
            # Build pairs dict
            pairs: dict[str, tuple[list[float], list[float]]] = {}
            for pair_cfg in self._stat_arb_pairs:
                if len(pair_cfg) != 2:
                    continue
                sym_a, sym_b = pair_cfg
                prices_a = prices_map.get(sym_a, [])
                prices_b = prices_map.get(sym_b, [])
                if len(prices_a) >= 30 and len(prices_b) >= 30:
                    pair_name = f"{sym_a}-{sym_b}"
                    pairs[pair_name] = (prices_a, prices_b)
            if pairs:
                return pairs
        except Exception:
            logger.warning("cointegration_real_data_fetch_failed", exc_info=True)
        return None

    def run_experiment(self, **kwargs: object) -> ResearchReport:
        # Priority: kwargs > data_provider > synthetic
        pairs = kwargs.get("pairs")
        data_source = "kwargs"
        if pairs is None:
            real_pairs = self._fetch_real_pairs()
            if real_pairs is not None:
                pairs = real_pairs
                data_source = "real"
            else:
                pairs = self._generate_synthetic_pairs()
                data_source = "synthetic"

        results: dict[str, object] = {}
        cointegrated_pairs: list[str] = []

        for pair_name, (series_a, series_b) in pairs.items():  # type: ignore[union-attr]
            a = np.array(series_a, dtype=float)
            b = np.array(series_b, dtype=float)

            if len(a) < 30 or len(b) < 30:
                results[pair_name] = {"error": "Insufficient data"}
                continue

            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]

            try:
                t_stat, p_value, _ = coint(a, b)
                is_coint = bool(p_value < 0.05)
                correlation = float(np.corrcoef(a, b)[0, 1])

                results[pair_name] = {
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": round(float(p_value), 4),
                    "cointegrated": is_coint,
                    "correlation": round(correlation, 4),
                    "data_points": min_len,
                }

                if is_coint:
                    cointegrated_pairs.append(pair_name)
            except Exception as e:
                logger.warning("coint_test_failed", pair=pair_name, error=str(e))
                results[pair_name] = {"error": str(e)}

        total = len(results)
        n_coint = len(cointegrated_pairs)

        results["data_source"] = data_source

        self._last_report = ResearchReport(
            experiment_name="cointegration",
            hypothesis="Price pairs used in stat_arb are truly cointegrated",
            methodology=(
                f"Engle-Granger cointegration test (p<0.05) on {total} pairs."
            ),
            data_period=f"{total} pairs analyzed",
            results=results,
            conclusion=(
                f"{n_coint}/{total} pairs are cointegrated. "
                + (
                    "Good pair selection."
                    if n_coint > 0
                    else "No cointegrated pairs found."
                )
            ),
            improvement_significant=n_coint > 0,
        )

        if n_coint > 0:
            # Recommend tightening min_correlation based on findings
            correlations = []
            for pair_name in cointegrated_pairs:
                r = results[pair_name]
                if isinstance(r, dict) and "correlation" in r:
                    correlations.append(r["correlation"])

            if correlations:
                min_corr = min(correlations)
                if min_corr > 0.7:
                    self._last_report.recommended_changes = [
                        ParamChange(
                            engine_name="stat_arb",
                            param_name="stat_arb_min_correlation",
                            old_value=0.7,
                            new_value=round(max(0.5, min_corr - 0.05), 2),
                            reason=(
                                "Cointegration experiment: "
                                f"lowest cointegrated pair corr={min_corr:.2f}"
                            ),
                        ),
                    ]

        return self._last_report

    def apply_findings(self) -> list[ParamChange]:
        if self._last_report and self._last_report.recommended_changes:
            return self._last_report.recommended_changes
        return []

    @staticmethod
    def _generate_synthetic_pairs() -> dict[str, tuple[list[float], list[float]]]:
        rng = np.random.default_rng(42)
        n = 200

        # Pair 1: cointegrated (common factor)
        common = np.cumsum(rng.normal(0, 1, n))
        a1 = common + rng.normal(0, 0.5, n)
        b1 = 0.8 * common + rng.normal(0, 0.5, n) + 10

        # Pair 2: not cointegrated (independent random walks)
        a2 = np.cumsum(rng.normal(0, 1, n)) + 100
        b2 = np.cumsum(rng.normal(0, 1, n)) + 50

        return {
            "BTC/USDT-ETH/USDT": (a1.tolist(), b1.tolist()),
            "SOL/USDT-ETH/USDT": (a2.tolist(), b2.tolist()),
        }
