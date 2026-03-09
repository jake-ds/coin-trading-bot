"""Optimal grid parameter experiment.

Simulates different grid_spacing/levels combinations on historical data
to find optimal parameters for grid_trading.
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


class OptimalGridExperiment(ResearchTask):
    """Find optimal grid spacing and levels via simulation."""

    @property
    def target_engine(self) -> str:
        return "grid_trading"

    def __init__(
        self,
        data_provider: HistoricalDataProvider | None = None,
        grid_symbols: list[str] | None = None,
    ) -> None:
        super().__init__(data_provider=data_provider)
        self._last_report: ResearchReport | None = None
        self._grid_symbols = grid_symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def _fetch_real_prices(self) -> list[float] | None:
        """Fetch real prices for the first grid symbol."""
        if not self.data_provider:
            return None
        try:
            symbol = self._grid_symbols[0] if self._grid_symbols else "BTC/USDT"
            prices = self._run_async(
                self.data_provider.get_prices(symbol, "1h", lookback_days=60)
            )
            if len(prices) >= 30:
                return prices
        except Exception:
            logger.warning("optimal_grid_real_data_fetch_failed", exc_info=True)
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

        spacing_options = [0.005, 0.01, 0.015, 0.02, 0.03]
        runner = SimpleBacktestRunner()
        grid_results: dict[str, dict[str, float]] = {}
        best_sharpe = -999.0
        best_spacing = spacing_options[0]

        for spacing in spacing_options:
            result = runner.run(
                prices,
                lambda p, i, s=spacing: self._grid_strategy(p, i, s),
            )
            key = f"spacing_{spacing*100:.1f}pct"
            grid_results[key] = {
                "sharpe": round(result.sharpe, 4),
                "total_return": round(result.total_return, 6),
                "max_drawdown": round(result.max_drawdown, 6),
                "num_trades": result.num_trades,
            }
            if result.sharpe > best_sharpe:
                best_sharpe = result.sharpe
                best_spacing = spacing

        current_spacing_pct = 1.0  # default config value
        optimal_spacing_pct = best_spacing * 100
        improvement = best_sharpe - grid_results.get(
            f"spacing_{current_spacing_pct:.1f}pct", {"sharpe": 0.0}
        )["sharpe"]

        significant = improvement > 0.1 and abs(optimal_spacing_pct - current_spacing_pct) > 0.1

        grid_results["data_source"] = data_source  # type: ignore[assignment]

        self._last_report = ResearchReport(
            experiment_name="optimal_grid",
            hypothesis="There exists an optimal grid spacing that maximizes Sharpe ratio",
            methodology=(
                f"Simulate {len(spacing_options)} grid spacing values "
                f"({', '.join(f'{s*100:.1f}%' for s in spacing_options)}) "
                f"over {len(prices)} price bars."
            ),
            data_period=f"{len(prices)} bars",
            results=grid_results,
            conclusion=(
                f"Optimal spacing: {optimal_spacing_pct:.1f}% (Sharpe={best_sharpe:.4f}). "
                f"{'Significant improvement' if significant else 'No significant improvement'} "
                f"over current {current_spacing_pct:.1f}%."
            ),
            improvement_significant=significant,
        )

        if significant:
            self._last_report.recommended_changes = [
                ParamChange(
                    engine_name="grid_trading",
                    param_name="grid_spacing_pct",
                    old_value=current_spacing_pct,
                    new_value=round(optimal_spacing_pct, 2),
                    reason=(
                        f"Optimal grid experiment: "
                        f"spacing {optimal_spacing_pct:.1f}% "
                        f"has Sharpe {best_sharpe:.4f}"
                    ),
                ),
            ]

        return self._last_report

    def apply_findings(self) -> list[ParamChange]:
        if self._last_report and self._last_report.recommended_changes:
            return self._last_report.recommended_changes
        return []

    @staticmethod
    def _grid_strategy(prices: list[float], idx: int, spacing: float) -> float:
        if idx < 1:
            return 0.0
        change = (prices[idx] - prices[idx - 1]) / prices[idx - 1]
        if change < -spacing:
            return 1.0
        elif change > spacing:
            return -1.0
        return 0.0

    @staticmethod
    def _generate_synthetic_prices(n: int = 500) -> list[float]:
        rng = np.random.default_rng(123)
        returns = rng.normal(0.0001, 0.01, n)
        prices = [100.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        return prices
