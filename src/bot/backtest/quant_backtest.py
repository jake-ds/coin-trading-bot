"""Quant-specific backtesting extensions.

Supports pairs trading (2-leg) backtest and portfolio optimization backtest
while reusing the existing BacktestResult format.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import structlog

from bot.backtest.engine import BacktestResult, TradeLog
from bot.monitoring.metrics import MetricsCollector, PerformanceMetrics
from bot.quant.portfolio import risk_parity_portfolio
from bot.quant.statistics import (
    calculate_zscore,
    rolling_ols_hedge_ratio,
)

logger = structlog.get_logger()


class PairsBacktestEngine:
    """Backtest pairs trading strategy on two price series.

    Simulates long/short spread trading with proper hedge ratios.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_pct: float = 0.1,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        zscore_stop: float = 3.5,
        hedge_window: int = 60,
        zscore_window: int = 20,
    ):
        self._initial_capital = initial_capital
        self._fee_pct = fee_pct
        self._zscore_entry = zscore_entry
        self._zscore_exit = zscore_exit
        self._zscore_stop = zscore_stop
        self._hedge_window = hedge_window
        self._zscore_window = zscore_window

    async def run(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        timestamps: list[datetime] | None = None,
        symbol_a: str = "A",
        symbol_b: str = "B",
    ) -> BacktestResult:
        """Run pairs trading backtest.

        Args:
            prices_a: Price series for asset A.
            prices_b: Price series for asset B.
            timestamps: Optional timestamp for each price point.
            symbol_a: Name of asset A.
            symbol_b: Name of asset B.

        Returns:
            BacktestResult with pairs trading metrics.
        """
        min_len = min(len(prices_a), len(prices_b))
        prices_a = prices_a[:min_len]
        prices_b = prices_b[:min_len]

        if timestamps is None:
            timestamps = [datetime(2024, 1, 1)] * min_len

        # Compute hedge ratios and spread
        hedge_ratios = rolling_ols_hedge_ratio(
            prices_a, prices_b, window=self._hedge_window
        )
        start_idx = self._hedge_window + self._zscore_window

        cash = self._initial_capital
        position = 0  # +1 = long spread, -1 = short spread
        entry_spread = 0.0
        entry_cost = 0.0
        trades: list[TradeLog] = []
        metrics = MetricsCollector(initial_capital=self._initial_capital)
        equity_curve = [self._initial_capital]

        for i in range(start_idx, min_len):
            hedge = hedge_ratios[i]
            if np.isnan(hedge):
                equity_curve.append(cash)
                continue

            spread = prices_a[i] - hedge * prices_b[i]
            spread_series = prices_a[start_idx:i + 1] - hedge * prices_b[start_idx:i + 1]
            zscores = calculate_zscore(spread_series, window=self._zscore_window)
            z = zscores[-1] if len(zscores) > 0 else float("nan")

            if np.isnan(z):
                equity_curve.append(cash)
                continue

            ts = timestamps[i] if i < len(timestamps) else datetime(2024, 1, 1)

            if position == 0:
                # Entry
                if z < -self._zscore_entry:
                    # Long spread: buy A, sell B
                    position = 1
                    entry_spread = spread
                    entry_cost = cash
                    fee = cash * (self._fee_pct / 100) * 2  # 2 legs
                    cash -= fee
                    trades.append(TradeLog(
                        timestamp=ts,
                        symbol=f"{symbol_a}/{symbol_b}",
                        side="BUY",
                        price=spread,
                        quantity=1.0,
                    ))
                elif z > self._zscore_entry:
                    # Short spread: sell A, buy B
                    position = -1
                    entry_spread = spread
                    entry_cost = cash
                    fee = cash * (self._fee_pct / 100) * 2
                    cash -= fee
                    trades.append(TradeLog(
                        timestamp=ts,
                        symbol=f"{symbol_a}/{symbol_b}",
                        side="SELL",
                        price=spread,
                        quantity=1.0,
                    ))
            else:
                # Exit conditions
                should_exit = False
                exit_reason = ""

                if position == 1 and (z > -self._zscore_exit or z > self._zscore_stop):
                    should_exit = True
                    exit_reason = "reversion" if z > -self._zscore_exit else "stop_loss"
                elif position == -1 and (z < self._zscore_exit or z < -self._zscore_stop):
                    should_exit = True
                    exit_reason = "reversion" if z < self._zscore_exit else "stop_loss"

                if should_exit:
                    spread_pnl = position * (spread - entry_spread)
                    # Approximate: spread_pnl as fraction of entry_spread
                    pnl_pct = (
                        spread_pnl / abs(entry_spread) if abs(entry_spread) > 1e-10 else 0
                    )
                    pnl = entry_cost * pnl_pct
                    fee = abs(pnl) * (self._fee_pct / 100) * 2
                    cash = entry_cost + pnl - fee
                    metrics.record_trade(pnl - fee)
                    trades.append(TradeLog(
                        timestamp=ts,
                        symbol=f"{symbol_a}/{symbol_b}",
                        side="SELL" if position == 1 else "BUY",
                        price=spread,
                        quantity=1.0,
                        pnl=pnl - fee,
                        exit_reason=exit_reason,
                    ))
                    position = 0

            metrics.record_portfolio_value(cash)
            equity_curve.append(cash)

        return BacktestResult(
            strategy_name="pairs_trading",
            symbol=f"{symbol_a}/{symbol_b}",
            timeframe="1h",
            metrics=metrics.calculate(),
            trades=trades,
            final_portfolio_value=round(cash, 2),
            equity_curve=equity_curve,
            drawdown_curve=self._calc_drawdown(equity_curve),
            monthly_returns={},
        )

    @staticmethod
    def _calc_drawdown(equity: list[float]) -> list[float]:
        if not equity:
            return []
        drawdowns = []
        peak = equity[0]
        for v in equity:
            if v > peak:
                peak = v
            dd = ((peak - v) / peak * 100) if peak > 0 else 0.0
            drawdowns.append(round(dd, 4))
        return drawdowns


class PortfolioBacktestEngine:
    """Backtest portfolio optimization strategies.

    Rebalances periodically using risk parity or other methods.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_pct: float = 0.1,
        rebalance_interval: int = 24,
        lookback: int = 60,
        method: str = "risk_parity",
    ):
        self._initial_capital = initial_capital
        self._fee_pct = fee_pct
        self._rebalance_interval = rebalance_interval
        self._lookback = lookback
        self._method = method

    async def run(
        self,
        prices: dict[str, np.ndarray],
        timestamps: list[datetime] | None = None,
    ) -> BacktestResult:
        """Run portfolio optimization backtest.

        Args:
            prices: Dict mapping symbol -> price array (all same length).
            timestamps: Optional timestamp array.

        Returns:
            BacktestResult with portfolio metrics.
        """
        symbols = list(prices.keys())
        n_assets = len(symbols)
        if n_assets < 2:
            return self._empty_result()

        arrays = [np.asarray(prices[s], dtype=float) for s in symbols]
        n_periods = min(len(a) for a in arrays)
        arrays = [a[:n_periods] for a in arrays]

        if timestamps is None:
            timestamps = [datetime(2024, 1, 1)] * n_periods

        # Initialize equal weights
        weights = np.ones(n_assets) / n_assets
        cash = self._initial_capital
        holdings = weights * cash  # Dollar value per asset
        trades: list[TradeLog] = []
        metrics = MetricsCollector(initial_capital=self._initial_capital)
        equity_curve = [self._initial_capital]
        periods_since_rebalance = 0

        for t in range(1, n_periods):
            # Update holdings with price changes
            for i in range(n_assets):
                if arrays[i][t - 1] > 0:
                    holdings[i] *= arrays[i][t] / arrays[i][t - 1]

            portfolio_value = float(np.sum(holdings))
            periods_since_rebalance += 1

            # Rebalance
            if periods_since_rebalance >= self._rebalance_interval and t >= self._lookback:
                # Compute returns for optimization
                ret_matrix = []
                for i in range(n_assets):
                    p = arrays[i][max(0, t - self._lookback):t + 1]
                    r = np.diff(np.log(np.maximum(p, 1e-10)))
                    ret_matrix.append(r)

                min_ret_len = min(len(r) for r in ret_matrix)
                if min_ret_len >= 20:
                    ret_2d = np.column_stack([r[-min_ret_len:] for r in ret_matrix])
                    result = risk_parity_portfolio(ret_2d, annualization=365 * 24)
                    new_weights = np.array(result["weights"])

                    # Apply rebalance with fees
                    target_holdings = new_weights * portfolio_value
                    turnover = float(np.sum(np.abs(target_holdings - holdings)))
                    fee = turnover * (self._fee_pct / 100)
                    portfolio_value -= fee
                    holdings = new_weights * portfolio_value
                    weights = new_weights
                    periods_since_rebalance = 0

                    ts = timestamps[t] if t < len(timestamps) else datetime(2024, 1, 1)
                    trades.append(TradeLog(
                        timestamp=ts,
                        symbol=",".join(symbols),
                        side="REBALANCE",
                        price=portfolio_value,
                        quantity=turnover,
                        pnl=-fee,
                        exit_reason="rebalance",
                    ))

            metrics.record_portfolio_value(portfolio_value)
            equity_curve.append(portfolio_value)

        final_value = float(np.sum(holdings))
        pnl = final_value - self._initial_capital
        if abs(pnl) > 0:
            metrics.record_trade(pnl)

        return BacktestResult(
            strategy_name=f"portfolio_{self._method}",
            symbol=",".join(symbols),
            timeframe="1h",
            metrics=metrics.calculate(),
            trades=trades,
            final_portfolio_value=round(final_value, 2),
            equity_curve=equity_curve,
            drawdown_curve=self._calc_drawdown(equity_curve),
            monthly_returns={},
        )

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            strategy_name=f"portfolio_{self._method}",
            symbol="",
            timeframe="1h",
            metrics=PerformanceMetrics(
                total_return_pct=0,
                sharpe_ratio=0,
                win_rate=0,
                max_drawdown_pct=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
            ),
            final_portfolio_value=self._initial_capital,
        )

    @staticmethod
    def _calc_drawdown(equity: list[float]) -> list[float]:
        if not equity:
            return []
        drawdowns = []
        peak = equity[0]
        for v in equity:
            if v > peak:
                peak = v
            dd = ((peak - v) / peak * 100) if peak > 0 else 0.0
            drawdowns.append(round(dd, 4))
        return drawdowns
