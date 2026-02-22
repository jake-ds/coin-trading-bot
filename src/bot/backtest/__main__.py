"""CLI entry point for backtesting: python -m bot.backtest.

Usage:
    python -m bot.backtest --strategy ma_crossover --symbol BTC/USDT --timeframe 1h
    python -m bot.backtest --strategy all --symbol ETH/USDT --output-json results.json
    python -m bot.backtest --strategy rsi --output-csv trades.csv --initial-capital 50000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from typing import Any

import structlog

from bot.backtest.engine import BacktestEngine, BacktestResult
from bot.data.store import DataStore
from bot.strategies.base import BaseStrategy, strategy_registry

logger = structlog.get_logger()

# Strategies that need special data beyond OHLCV candles and cannot be
# backtested with the standard BacktestEngine.
SKIP_STRATEGIES = {"arbitrage", "dca", "funding_rate"}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m bot.backtest",
        description="Run backtests on trading strategies with historical data.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        help="Strategy name to backtest, or 'all' to compare all strategies (default: all)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Trading symbol (default: BTC/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candle timeframe (default: 1h)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date in YYYY-MM-DD format (optional)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (optional)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital in quote currency (default: 10000.0)",
    )
    parser.add_argument(
        "--fee-pct",
        type=float,
        default=0.1,
        help="Trading fee percentage (default: 0.1)",
    )
    parser.add_argument(
        "--slippage-pct",
        type=float,
        default=0.05,
        help="Slippage percentage (default: 0.05)",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.0,
        help="Stop-loss percentage, 0 to disable (default: 0.0)",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=0.0,
        help="Take-profit percentage, 0 to disable (default: 0.0)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write JSON results file",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to write CSV trade log file",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default="sqlite+aiosqlite:///data/trading.db",
        help="Database URL for historical data (default: sqlite+aiosqlite:///data/trading.db)",
    )
    return parser


def parse_date(date_str: str | None) -> datetime | None:
    """Parse a YYYY-MM-DD date string into a timezone-aware datetime."""
    if date_str is None:
        return None
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"Error: Invalid date format '{date_str}'. Expected YYYY-MM-DD.")
        sys.exit(1)


def get_backtestable_strategies(strategy_name: str) -> list[BaseStrategy]:
    """Get strategies to backtest based on the --strategy argument.

    When strategy_name is 'all', returns all registered strategies that
    can run with standard OHLCV data (skips arbitrage, DCA, funding_rate).

    Returns:
        List of strategy instances to backtest.
    """
    # Ensure strategy modules are imported so registry is populated
    _import_strategies()

    if strategy_name.lower() == "all":
        strategies = []
        for s in strategy_registry.get_all():
            if s.name.lower() not in SKIP_STRATEGIES:
                strategies.append(s)
        if not strategies:
            print("Error: No backtestable strategies found in registry.")
            sys.exit(1)
        return strategies
    else:
        strategy = strategy_registry.get(strategy_name)
        if strategy is None:
            available = [s.name for s in strategy_registry.get_all()]
            print(f"Error: Strategy '{strategy_name}' not found.")
            print(f"Available strategies: {', '.join(sorted(available))}")
            sys.exit(1)
        return [strategy]


def _import_strategies() -> None:
    """Import all strategy modules to populate the registry."""
    try:
        import bot.strategies.arbitrage.arbitrage_strategy  # noqa: F401
        import bot.strategies.dca.dca_strategy  # noqa: F401
        import bot.strategies.technical.bollinger  # noqa: F401
        import bot.strategies.technical.composite  # noqa: F401
        import bot.strategies.technical.funding_rate  # noqa: F401
        import bot.strategies.technical.ma_crossover  # noqa: F401
        import bot.strategies.technical.macd  # noqa: F401
        import bot.strategies.technical.rsi  # noqa: F401
        import bot.strategies.technical.vwap  # noqa: F401
    except ImportError:
        pass
    try:
        import bot.strategies.ml.prediction  # noqa: F401
    except ImportError:
        pass


def calculate_profit_factor(trades: list[Any]) -> float:
    """Calculate profit factor from trade list.

    Profit factor = gross_profit / gross_loss.
    Returns float('inf') if no losses, 0.0 if no profits.
    """
    gross_profit = sum(t.pnl for t in trades if t.side == "SELL" and t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.side == "SELL" and t.pnl < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return round(gross_profit / gross_loss, 2)


def format_comparison_table(results: list[BacktestResult]) -> str:
    """Format backtest results as a comparison table sorted by Sharpe ratio.

    Columns: strategy, total_return, sharpe_ratio, max_drawdown, win_rate,
             total_trades, profit_factor.
    """
    # Sort by Sharpe ratio descending
    sorted_results = sorted(
        results, key=lambda r: r.metrics.sharpe_ratio, reverse=True,
    )

    # Column headers and widths
    headers = [
        "Strategy", "Total Return %", "Sharpe Ratio", "Max Drawdown %",
        "Win Rate %", "Total Trades", "Profit Factor",
    ]
    rows: list[list[str]] = []
    for r in sorted_results:
        pf = calculate_profit_factor(r.trades)
        pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
        rows.append([
            r.strategy_name,
            f"{r.metrics.total_return_pct:.2f}",
            f"{r.metrics.sharpe_ratio:.4f}",
            f"{r.metrics.max_drawdown_pct:.2f}",
            f"{r.metrics.win_rate:.2f}",
            str(r.metrics.total_trades),
            pf_str,
        ])

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Build table
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_line = "|" + "|".join(
        f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)
    ) + "|"

    lines = [sep, header_line, sep]
    for row in rows:
        line = "|" + "|".join(
            f" {cell:<{col_widths[i]}} " for i, cell in enumerate(row)
        ) + "|"
        lines.append(line)
    lines.append(sep)

    return "\n".join(lines)


def format_single_result(result: BacktestResult) -> str:
    """Format a single backtest result as a summary."""
    pf = calculate_profit_factor(result.trades)
    pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"

    lines = [
        f"Backtest Results: {result.strategy_name}",
        f"  Symbol:         {result.symbol}",
        f"  Timeframe:      {result.timeframe}",
        f"  Total Return:   {result.metrics.total_return_pct:.2f}%",
        f"  Sharpe Ratio:   {result.metrics.sharpe_ratio:.4f}",
        f"  Max Drawdown:   {result.metrics.max_drawdown_pct:.2f}%",
        f"  Win Rate:       {result.metrics.win_rate:.2f}%",
        f"  Total Trades:   {result.metrics.total_trades}",
        f"  Winning Trades: {result.metrics.winning_trades}",
        f"  Losing Trades:  {result.metrics.losing_trades}",
        f"  Profit Factor:  {pf_str}",
        f"  Final Value:    {result.final_portfolio_value:.2f}",
    ]
    return "\n".join(lines)


def export_json(results: list[BacktestResult], filepath: str) -> None:
    """Export results to a JSON file."""
    if len(results) == 1:
        data = json.loads(results[0].to_json())
    else:
        data = {
            "comparison": [
                json.loads(r.to_json()) for r in sorted(
                    results, key=lambda r: r.metrics.sharpe_ratio, reverse=True,
                )
            ],
        }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON results written to {filepath}")


def export_csv(results: list[BacktestResult], filepath: str) -> None:
    """Export trade logs to a CSV file."""
    if len(results) == 1:
        csv_data = results[0].to_csv()
    else:
        # Combine all trade logs with strategy name column
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "strategy", "timestamp", "symbol", "side", "price",
            "quantity", "pnl", "exit_reason",
        ])
        for r in results:
            for t in r.trades:
                writer.writerow([
                    r.strategy_name,
                    t.timestamp.isoformat(),
                    t.symbol,
                    t.side,
                    t.price,
                    t.quantity,
                    t.pnl,
                    t.exit_reason,
                ])
        csv_data = output.getvalue()

    with open(filepath, "w") as f:
        f.write(csv_data)
    print(f"CSV trade log written to {filepath}")


async def load_data(
    database_url: str,
    symbol: str,
    timeframe: str,
    start_date: datetime | None,
    end_date: datetime | None,
) -> list:
    """Load historical data from the DataStore.

    Returns:
        List of OHLCV candles from the database.
    """
    store = DataStore(database_url=database_url)
    try:
        await store.initialize()
        candles = await store.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            limit=10000,
        )
        return candles
    finally:
        await store.close()


async def run_backtest(args: argparse.Namespace) -> list[BacktestResult]:
    """Run the backtest(s) based on parsed CLI arguments.

    Returns:
        List of BacktestResult objects.
    """
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)

    # Load historical data
    candles = await load_data(
        database_url=args.database_url,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    if not candles:
        print(f"No historical data found for {args.symbol} ({args.timeframe}).")
        print("Please collect data first using the bot's data collector.")
        sys.exit(1)

    print(f"Loaded {len(candles)} candles for {args.symbol} ({args.timeframe})")
    if candles:
        first_ts = candles[0].timestamp
        last_ts = candles[-1].timestamp
        print(f"Date range: {first_ts} to {last_ts}")

    # Get strategies
    strategies = get_backtestable_strategies(args.strategy)

    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=args.initial_capital,
        fee_pct=args.fee_pct,
        slippage_pct=args.slippage_pct,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
    )

    # Run backtests
    results: list[BacktestResult] = []
    for strategy in strategies:
        print(f"Running backtest: {strategy.name}...")
        try:
            result = await engine.run(
                strategy=strategy,
                data=candles,
                symbol=args.symbol,
                timeframe=args.timeframe,
            )
            results.append(result)
        except Exception as e:
            print(f"  Error running {strategy.name}: {e}")

    return results


async def async_main(args: argparse.Namespace | None = None) -> None:
    """Async entry point for the backtest CLI."""
    if args is None:
        parser = build_parser()
        args = parser.parse_args()

    results = await run_backtest(args)

    if not results:
        print("No backtest results generated.")
        sys.exit(1)

    # Output results
    print()
    if len(results) == 1:
        print(format_single_result(results[0]))
    else:
        print(format_comparison_table(results))

    # Export if requested
    if args.output_json:
        export_json(results, args.output_json)
    if args.output_csv:
        export_csv(results, args.output_csv)


def main() -> None:
    """Synchronous CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
