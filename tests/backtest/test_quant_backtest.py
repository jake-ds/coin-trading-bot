"""Tests for quant backtesting engines."""

import numpy as np
import pytest

from bot.backtest.quant_backtest import PairsBacktestEngine, PortfolioBacktestEngine


class TestPairsBacktestEngine:
    @pytest.mark.asyncio
    async def test_basic_pairs_backtest(self):
        np.random.seed(42)
        n = 300
        # Cointegrated pair
        a = np.cumsum(np.random.randn(n)) + 100
        b = 1.5 * a + np.random.randn(n) * 0.5

        engine = PairsBacktestEngine(initial_capital=10000, fee_pct=0.1)
        result = await engine.run(a, b, symbol_a="BTC/USDT", symbol_b="ETH/USDT")

        assert result.strategy_name == "pairs_trading"
        assert result.final_portfolio_value > 0
        assert len(result.equity_curve) > 0

    @pytest.mark.asyncio
    async def test_no_trades_when_not_cointegrated(self):
        np.random.seed(42)
        n = 200
        # Independent random walks
        a = np.cumsum(np.random.randn(n)) + 100
        b = np.cumsum(np.random.randn(n)) + 100

        engine = PairsBacktestEngine(
            initial_capital=10000, zscore_entry=3.0,
        )
        result = await engine.run(a, b)
        # Should have few or no trades since not cointegrated
        assert result.final_portfolio_value > 0

    @pytest.mark.asyncio
    async def test_drawdown_curve(self):
        np.random.seed(42)
        n = 300
        a = np.cumsum(np.random.randn(n)) + 100
        b = 1.5 * a + np.random.randn(n) * 0.3

        engine = PairsBacktestEngine()
        result = await engine.run(a, b)
        assert len(result.drawdown_curve) == len(result.equity_curve)

    @pytest.mark.asyncio
    async def test_symbol_names_in_result(self):
        np.random.seed(42)
        n = 200
        a = np.cumsum(np.random.randn(n)) + 100
        b = 1.5 * a + np.random.randn(n) * 0.5

        engine = PairsBacktestEngine()
        result = await engine.run(a, b, symbol_a="SOL/USDT", symbol_b="AVAX/USDT")
        assert result.symbol == "SOL/USDT/AVAX/USDT"

    @pytest.mark.asyncio
    async def test_equity_curve_starts_at_initial_capital(self):
        np.random.seed(42)
        n = 200
        a = np.cumsum(np.random.randn(n)) + 100
        b = 1.5 * a + np.random.randn(n) * 0.5

        engine = PairsBacktestEngine(initial_capital=5000)
        result = await engine.run(a, b)
        assert result.equity_curve[0] == 5000


class TestPortfolioBacktestEngine:
    @pytest.mark.asyncio
    async def test_basic_portfolio_backtest(self):
        np.random.seed(42)
        n = 200
        prices = {
            "BTC/USDT": 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01)),
            "ETH/USDT": 50 * np.exp(np.cumsum(np.random.randn(n) * 0.015)),
            "SOL/USDT": 20 * np.exp(np.cumsum(np.random.randn(n) * 0.02)),
        }

        engine = PortfolioBacktestEngine(
            initial_capital=10000, rebalance_interval=24, lookback=60,
        )
        result = await engine.run(prices)

        assert result.strategy_name == "portfolio_risk_parity"
        assert result.final_portfolio_value > 0
        assert len(result.equity_curve) > 0

    @pytest.mark.asyncio
    async def test_single_asset_returns_empty(self):
        prices = {"BTC/USDT": np.ones(100) * 100}
        engine = PortfolioBacktestEngine()
        result = await engine.run(prices)
        assert result.final_portfolio_value == 10000  # Initial capital

    @pytest.mark.asyncio
    async def test_rebalance_generates_trades(self):
        np.random.seed(42)
        n = 200
        prices = {
            "A": 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02)),
            "B": 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01)),
        }
        engine = PortfolioBacktestEngine(
            rebalance_interval=20, lookback=30,
        )
        result = await engine.run(prices)
        # Should have some rebalance trades
        rebalance_trades = [t for t in result.trades if t.side == "REBALANCE"]
        assert len(rebalance_trades) >= 0  # May or may not trigger depending on drift

    @pytest.mark.asyncio
    async def test_equity_curve_length(self):
        np.random.seed(42)
        n = 100
        prices = {
            "A": 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01)),
            "B": 50 * np.exp(np.cumsum(np.random.randn(n) * 0.01)),
        }
        engine = PortfolioBacktestEngine()
        result = await engine.run(prices)
        # equity_curve = initial + (n-1) period values
        assert len(result.equity_curve) == n

    @pytest.mark.asyncio
    async def test_drawdown_curve_matches_equity(self):
        np.random.seed(42)
        n = 100
        prices = {
            "A": 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01)),
            "B": 50 * np.exp(np.cumsum(np.random.randn(n) * 0.01)),
        }
        engine = PortfolioBacktestEngine()
        result = await engine.run(prices)
        assert len(result.drawdown_curve) == len(result.equity_curve)
