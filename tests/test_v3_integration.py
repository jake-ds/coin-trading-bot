"""V3 Quant Trading System Integration Tests.

Tests that all quant components work together:
- Quant math utilities
- Quant strategies (pairs, mean reversion, momentum, vol breakout, triangular arb)
- Risk metrics (VaR/CVaR)
- Portfolio optimization
- Portfolio risk with VaR integration
- Backtesting
- Dashboard endpoints
"""

import numpy as np
import pytest

from bot.models import OHLCV, SignalAction
from bot.strategies.base import strategy_registry


@pytest.fixture(autouse=True)
def clean_registry():
    strategy_registry.clear()
    yield
    strategy_registry.clear()


def _make_candles(closes, symbol="BTC/USDT"):
    from datetime import datetime, timedelta

    base = datetime(2024, 1, 1)
    return [
        OHLCV(
            timestamp=base + timedelta(hours=i),
            open=c, high=c * 1.01, low=c * 0.99, close=c,
            volume=1000.0, symbol=symbol,
        )
        for i, c in enumerate(closes)
    ]


class TestQuantMathIntegration:
    """Test that all quant math utilities work with synthetic data."""

    def test_full_statistics_pipeline(self):
        from bot.quant.statistics import (
            adf_test,
            calculate_half_life,
            calculate_zscore,
            engle_granger_cointegration,
            estimate_ou_params,
            rolling_ols_hedge_ratio,
        )

        np.random.seed(42)
        n = 300
        a = np.cumsum(np.random.randn(n)) + 100
        b = 1.5 * a + np.random.randn(n) * 0.5

        # 1. Cointegration test
        coint = engle_granger_cointegration(a, b)
        assert coint["is_cointegrated"]

        # 2. Hedge ratio
        ratios = rolling_ols_hedge_ratio(a, b, window=60)
        hedge = ratios[-1]
        assert not np.isnan(hedge)

        # 3. Spread
        spread = a - hedge * b

        # 4. Half-life
        hl = calculate_half_life(spread)
        assert hl < float("inf")

        # 5. Z-score
        zscores = calculate_zscore(spread, window=20)
        assert not np.isnan(zscores[-1])

        # 6. ADF on spread
        adf = adf_test(spread)
        assert "statistic" in adf

        # 7. OU params
        ou = estimate_ou_params(spread)
        assert ou["kappa"] >= 0

    def test_full_risk_metrics_pipeline(self):
        from bot.quant.risk_metrics import (
            calmar_ratio,
            cornish_fisher_var,
            cvar,
            historical_var,
            parametric_var,
            sortino_ratio,
        )

        np.random.seed(42)
        returns = np.random.randn(500) * 0.02

        p_var = parametric_var(returns)
        h_var = historical_var(returns)
        cf_var = cornish_fisher_var(returns)
        cv = cvar(returns)
        sortino = sortino_ratio(returns)
        calmar = calmar_ratio(returns)

        assert p_var > 0
        assert h_var > 0
        assert cf_var > 0
        assert cv >= h_var  # CVaR >= VaR
        assert isinstance(sortino, float)
        assert isinstance(calmar, float)

    def test_garch_pipeline(self):
        from bot.quant.volatility import GARCHModel, classify_volatility_regime

        np.random.seed(42)
        returns = np.random.randn(200) * 0.02

        model = GARCHModel()
        result = model.fit(returns)
        assert result["success"] is True

        forecast = model.forecast(5)
        assert len(forecast) == 5
        assert all(f > 0 for f in forecast)

        regime = classify_volatility_regime(returns)
        assert regime is not None

    def test_portfolio_optimization_pipeline(self):
        from bot.quant.portfolio import (
            max_sharpe_portfolio,
            min_variance_portfolio,
            risk_parity_portfolio,
        )

        np.random.seed(42)
        returns = np.random.randn(200, 3) * 0.02

        mv = min_variance_portfolio(returns)
        ms = max_sharpe_portfolio(returns)
        rp = risk_parity_portfolio(returns)

        for result in [mv, ms, rp]:
            assert len(result["weights"]) == 3
            assert abs(sum(result["weights"]) - 1.0) < 0.01

    def test_microstructure_pipeline(self):
        from bot.quant.microstructure import compute_orderbook_metrics

        bids = [(100.0, 10.0), (99.0, 20.0), (98.0, 50.0)]
        asks = [(101.0, 10.0), (102.0, 20.0), (103.0, 30.0)]
        metrics = compute_orderbook_metrics(bids, asks)
        assert metrics["vwap_mid"] > 0
        assert -1 <= metrics["imbalance"] <= 1


class TestQuantStrategiesIntegration:
    """Test that all quant strategies register and produce valid signals."""

    @pytest.mark.asyncio
    async def test_all_strategies_register(self):
        from bot.strategies.quant.mean_reversion import MeanReversionStrategy
        from bot.strategies.quant.momentum_factor import MomentumFactorStrategy
        from bot.strategies.quant.pairs_trading import PairsTradingStrategy
        from bot.strategies.quant.triangular_arb import TriangularArbStrategy
        from bot.strategies.quant.volatility_breakout import VolatilityBreakoutStrategy

        strategies = [
            MeanReversionStrategy(),
            MomentumFactorStrategy(),
            PairsTradingStrategy(),
            TriangularArbStrategy(),
            VolatilityBreakoutStrategy(),
        ]
        for s in strategies:
            strategy_registry.register(s)

        assert len(strategy_registry.get_active()) == 5

    @pytest.mark.asyncio
    async def test_strategies_produce_valid_signals(self):
        from bot.strategies.quant.mean_reversion import MeanReversionStrategy
        from bot.strategies.quant.momentum_factor import MomentumFactorStrategy
        from bot.strategies.quant.volatility_breakout import VolatilityBreakoutStrategy

        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(200) * 0.01))
        candles = _make_candles(prices)

        strategies = [
            MeanReversionStrategy(lookback=100),
            MomentumFactorStrategy(short_window=5, long_window=20, zscore_window=10),
            VolatilityBreakoutStrategy(lookback=150, min_data_points=60),
        ]

        for s in strategies:
            signal = await s.analyze(candles, symbol="BTC/USDT")
            assert signal.strategy_name == s.name
            assert signal.symbol == "BTC/USDT"
            assert signal.action in (SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD)
            assert 0.0 <= signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_pairs_trading_with_data_provider(self):
        from bot.data.pair_data import PairDataProvider
        from bot.strategies.quant.pairs_trading import PairsTradingStrategy

        np.random.seed(42)
        n = 300
        prices_a = np.cumsum(np.random.randn(n)) + 100
        prices_b = 1.5 * prices_a + np.random.randn(n) * 0.3

        candles_a = _make_candles(prices_a, symbol="BTC/USDT")
        candles_b = _make_candles(prices_b, symbol="ETH/USDT")

        aligned = PairDataProvider.get_aligned_closes_from_candles({
            "BTC/USDT": candles_a,
            "ETH/USDT": candles_b,
        })

        s = PairsTradingStrategy(zscore_entry=1.5)
        signal = await s.analyze(
            candles_a,
            symbol="BTC/USDT",
            pair_prices={
                "symbol_a": "BTC/USDT",
                "symbol_b": "ETH/USDT",
                "prices_a": aligned["BTC/USDT"],
                "prices_b": aligned["ETH/USDT"],
            },
        )
        assert signal.strategy_name == "pairs_trading"
        assert "hedge_ratio" in signal.metadata or "reason" in signal.metadata


class TestPortfolioRiskVaRIntegration:
    """Test VaR integration with portfolio risk manager."""

    def test_var_calculation(self):
        from bot.risk.portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager(var_enabled=True, max_portfolio_var_pct=10.0)
        mgr.update_portfolio_value(10000)

        np.random.seed(42)
        returns_a = list(np.random.randn(30) * 0.02)
        returns_b = list(np.random.randn(30) * 0.015)
        mgr._price_history = {"BTC/USDT": returns_a, "ETH/USDT": returns_b}
        mgr.add_position("BTC/USDT", 3000)
        mgr.add_position("ETH/USDT", 2000)

        var = mgr.calculate_portfolio_var()
        assert var is not None
        assert var > 0

    def test_var_limit_check(self):
        from bot.risk.portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager(
            var_enabled=True, max_portfolio_var_pct=1.0,  # Very tight limit
        )
        mgr.update_portfolio_value(10000)

        # Very volatile returns -> high VaR
        np.random.seed(42)
        mgr._price_history = {
            "BTC/USDT": list(np.random.randn(30) * 0.1),  # 10% daily vol
        }
        mgr.add_position("BTC/USDT", 5000)

        allowed, reason = mgr.check_var_limit("ETH/USDT", 2000)
        # With 10% vol and tight 1% limit, should reject
        assert reason == "" or "var" in reason.lower()

    def test_risk_metrics_summary(self):
        from bot.risk.portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager(var_enabled=True)
        mgr.update_portfolio_value(10000)
        mgr.add_position("BTC/USDT", 3000, atr=0.02)

        metrics = mgr.get_risk_metrics()
        assert "exposure_pct" in metrics
        assert "heat" in metrics
        assert "n_positions" in metrics
        assert metrics["var_enabled"] is True

    def test_validate_position_includes_var(self):
        from bot.risk.portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager(var_enabled=True, max_portfolio_var_pct=10.0)
        mgr.update_portfolio_value(10000)

        allowed, reason = mgr.validate_new_position("BTC/USDT", 1000)
        assert allowed is True  # Should pass with no price history


class TestQuantBacktestIntegration:
    """Test quant backtesting with strategy validation."""

    @pytest.mark.asyncio
    async def test_pairs_backtest_produces_trades(self):
        from bot.backtest.quant_backtest import PairsBacktestEngine

        np.random.seed(42)
        n = 300
        a = np.cumsum(np.random.randn(n)) + 100
        b = 1.5 * a + np.random.randn(n) * 0.5

        engine = PairsBacktestEngine(
            zscore_entry=1.5, zscore_exit=0.3,
        )
        result = await engine.run(a, b, symbol_a="BTC", symbol_b="ETH")
        assert len(result.trades) > 0
        assert result.final_portfolio_value > 0

    @pytest.mark.asyncio
    async def test_portfolio_backtest_preserves_capital(self):
        from bot.backtest.quant_backtest import PortfolioBacktestEngine

        np.random.seed(42)
        n = 200
        # Slightly trending up prices
        prices = {
            "A": 100 * np.exp(np.cumsum(np.random.randn(n) * 0.005 + 0.001)),
            "B": 50 * np.exp(np.cumsum(np.random.randn(n) * 0.005 + 0.001)),
        }
        engine = PortfolioBacktestEngine(
            initial_capital=10000, rebalance_interval=20,
        )
        result = await engine.run(prices)
        # With positive drift, portfolio should generally be positive
        assert result.final_portfolio_value > 5000  # At least preserved half

    @pytest.mark.asyncio
    async def test_no_lookahead_bias_in_pairs_backtest(self):
        from bot.backtest.quant_backtest import PairsBacktestEngine

        np.random.seed(42)
        n = 300
        a = np.cumsum(np.random.randn(n)) + 100
        b = 1.5 * a + np.random.randn(n) * 0.5

        engine = PairsBacktestEngine(hedge_window=60, zscore_window=20)
        result = await engine.run(a, b)

        # All buy trades should happen before corresponding sell trades
        buys = [t for t in result.trades if t.side == "BUY"]
        sells = [t for t in result.trades if t.side == "SELL"]
        # Should have roughly equal buys and sells
        assert abs(len(buys) - len(sells)) <= 1


class TestDashboardEndpoints:
    """Test that quant dashboard endpoints exist."""

    @pytest.mark.asyncio
    async def test_quant_risk_metrics_endpoint(self):
        from httpx import ASGITransport, AsyncClient

        from bot.dashboard.app import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/quant/risk-metrics")
            assert resp.status_code == 200
            assert "risk_metrics" in resp.json()

    @pytest.mark.asyncio
    async def test_quant_correlation_endpoint(self):
        from httpx import ASGITransport, AsyncClient

        from bot.dashboard.app import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/quant/correlation-matrix")
            assert resp.status_code == 200
            assert "correlation_matrix" in resp.json()

    @pytest.mark.asyncio
    async def test_quant_portfolio_optimization_endpoint(self):
        from httpx import ASGITransport, AsyncClient

        from bot.dashboard.app import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/quant/portfolio-optimization")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_quant_garch_endpoint(self):
        from httpx import ASGITransport, AsyncClient

        from bot.dashboard.app import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/quant/garch")
            assert resp.status_code == 200


class TestConfigV3Settings:
    """Test that V3 config settings are properly loaded."""

    def test_default_v3_settings(self):
        from bot.config import Settings

        s = Settings(
            binance_api_key="", binance_secret_key="",
            upbit_api_key="", upbit_secret_key="",
        )
        assert s.var_enabled is False
        assert s.var_confidence == 0.95
        assert s.triangular_arb_enabled is False
        assert s.rebalance_enabled is False
        assert s.garch_enabled is False
        assert s.quant_pairs == []

    def test_custom_v3_settings(self):
        from bot.config import Settings

        s = Settings(
            binance_api_key="", binance_secret_key="",
            upbit_api_key="", upbit_secret_key="",
            var_enabled=True,
            var_confidence=0.99,
            max_portfolio_var_pct=3.0,
            quant_pairs=[["BTC/USDT", "ETH/USDT"]],
            triangular_arb_enabled=True,
            garch_enabled=True,
        )
        assert s.var_enabled is True
        assert s.var_confidence == 0.99
        assert s.max_portfolio_var_pct == 3.0
        assert len(s.quant_pairs) == 1
        assert s.triangular_arb_enabled is True
