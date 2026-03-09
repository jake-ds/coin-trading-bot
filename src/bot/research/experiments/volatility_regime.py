"""Volatility regime detection experiment.

Tests whether dynamic grid spacing (wider in high vol) improves Sharpe
for grid_trading and stat_arb engines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

from bot.engines.tuner import ParamChange
from bot.research.backtest_runner import SimpleBacktestRunner
from bot.research.base import ResearchTask
from bot.research.report import ResearchReport

if TYPE_CHECKING:
    from bot.research.data_provider import HistoricalDataProvider

logger = structlog.get_logger(__name__)


class VolatilityRegimeExperiment(ResearchTask):
    """Detect volatility regimes using ATR and test adaptive grid spacing."""

    @property
    def target_engine(self) -> str:
        return "grid_trading"

    def __init__(self, data_provider: HistoricalDataProvider | None = None) -> None:
        super().__init__(data_provider=data_provider)
        self._last_report: ResearchReport | None = None

    def _fetch_real_prices(self) -> list[float] | None:
        """Attempt to fetch real BTC/USDT prices via data_provider."""
        if not self.data_provider:
            return None
        try:
            prices = self._run_async(
                self.data_provider.get_prices("BTC/USDT", "1h", lookback_days=60)
            )
            if len(prices) >= 30:
                return prices
        except Exception:
            logger.warning("volatility_regime_real_data_fetch_failed", exc_info=True)
        return None

    def run_experiment(self, **kwargs: object) -> ResearchReport:
        # Priority: kwargs > data_provider > synthetic
        prices = kwargs.get("prices")
        data_source = "kwargs"
        if prices is None:
            real_prices = self._fetch_real_prices()
            if real_prices is not None:
                prices = real_prices
                data_source = "real"
            else:
                prices = self._generate_synthetic_prices()
                data_source = "synthetic"
        prices = list(prices)  # type: ignore[arg-type]

        atr_period = 14
        atr_values = self._compute_atr(prices, atr_period)

        if len(atr_values) == 0:
            self._last_report = ResearchReport(
                experiment_name="volatility_regime",
                hypothesis="Dynamic grid spacing based on ATR improves Sharpe",
                methodology="ATR-based volatility regime detection",
                data_period=f"{len(prices)} bars",
                results={"error": "Insufficient data", "data_source": data_source},
                conclusion="Insufficient data for analysis",
            )
            return self._last_report

        median_atr = float(np.median(atr_values))

        runner = SimpleBacktestRunner()

        # Fixed spacing strategy
        fixed_spacing = 0.01  # 1%
        fixed_result = runner.run(
            prices[atr_period:],
            lambda p, i: self._grid_strategy(p, i, fixed_spacing),
        )

        # Dynamic spacing: wider when ATR > median
        def dynamic_strategy(p: list[float], i: int) -> float:
            atr_idx = min(i, len(atr_values) - 1)
            high_vol = atr_values[atr_idx] > median_atr
            spacing = fixed_spacing * 1.5 if high_vol else fixed_spacing * 0.8
            return self._grid_strategy(p, i, spacing)

        dynamic_result = runner.run(prices[atr_period:], dynamic_strategy)

        improvement = dynamic_result.sharpe - fixed_result.sharpe
        significant = improvement > 0.1

        self._last_report = ResearchReport(
            experiment_name="volatility_regime",
            hypothesis="Dynamic grid spacing based on ATR improves Sharpe",
            methodology=(
                f"Compare fixed spacing ({fixed_spacing*100:.1f}%) vs ATR-adaptive spacing "
                f"over {len(prices)} price bars. ATR period={atr_period}, "
                f"median ATR={median_atr:.6f}."
            ),
            data_period=f"{len(prices)} bars",
            results={
                "fixed_sharpe": round(fixed_result.sharpe, 4),
                "dynamic_sharpe": round(dynamic_result.sharpe, 4),
                "improvement": round(improvement, 4),
                "fixed_trades": fixed_result.num_trades,
                "dynamic_trades": dynamic_result.num_trades,
                "median_atr": round(median_atr, 6),
                "data_source": data_source,
            },
            conclusion=(
                f"Dynamic spacing "
                f"{'improves' if significant else 'does not improve'} "
                f"Sharpe by {improvement:+.4f} "
                f"({fixed_result.sharpe:.4f} -> {dynamic_result.sharpe:.4f})"
            ),
            improvement_significant=significant,
        )

        if significant:
            self._last_report.recommended_changes = [
                ParamChange(
                    engine_name="grid_trading",
                    param_name="grid_spacing_pct",
                    old_value=1.0,
                    new_value=round(1.0 * 1.2, 2),
                    reason=(
                        "Volatility regime experiment: "
                        f"dynamic spacing improved Sharpe by {improvement:+.4f}"
                    ),
                ),
            ]

        return self._last_report

    def apply_findings(self) -> list[ParamChange]:
        if self._last_report and self._last_report.recommended_changes:
            return self._last_report.recommended_changes
        return []

    @staticmethod
    def _compute_atr(prices: list[float], period: int) -> list[float]:
        if len(prices) < period + 1:
            return []
        tr = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        atr: list[float] = []
        for i in range(len(tr)):
            start = max(0, i - period + 1)
            atr.append(float(np.mean(tr[start : i + 1])))
        return atr

    @staticmethod
    def _grid_strategy(prices: list[float], idx: int, spacing: float) -> float:
        if idx < 1:
            return 0.0
        change = (prices[idx] - prices[idx - 1]) / prices[idx - 1]
        if change < -spacing:
            return 1.0  # buy on dip
        elif change > spacing:
            return -1.0  # sell on rise
        return 0.0

    @staticmethod
    def _generate_synthetic_prices(n: int = 500) -> list[float]:
        rng = np.random.default_rng(42)
        # Two regime model: low vol then high vol
        returns = np.concatenate([
            rng.normal(0.0001, 0.005, n // 2),   # low vol
            rng.normal(0.0001, 0.02, n - n // 2),  # high vol
        ])
        prices = [100.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        return prices
