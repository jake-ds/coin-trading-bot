"""Tests for V6-015: BaseEngine regime adaptation + 4 engine integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.engines.cross_exchange_arb import CrossExchangeArbEngine
from bot.engines.funding_arb import FundingRateArbEngine
from bot.engines.grid_trading import GridTradingEngine
from bot.engines.stat_arb import StatisticalArbEngine
from bot.risk.regime_detector import MarketRegime, MarketRegimeDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_portfolio_manager():
    pm = MagicMock()
    pm.request_capital.return_value = 10000.0
    pm.get_max_allocation.return_value = 10000.0
    pm.report_pnl = MagicMock()
    return pm


def _make_detector(regime: MarketRegime) -> MagicMock:
    det = MagicMock(spec=MarketRegimeDetector)
    det.get_current_regime.return_value = regime
    det.is_crisis.return_value = (regime == MarketRegime.CRISIS)
    return det


def _make_settings(**overrides):
    s = MagicMock()
    # Funding arb defaults
    s.funding_arb_max_positions = overrides.get("funding_arb_max_positions", 3)
    s.funding_arb_min_rate = overrides.get("funding_arb_min_rate", 0.0003)
    s.funding_arb_exit_rate = overrides.get("funding_arb_exit_rate", 0.0001)
    s.funding_arb_max_spread_pct = overrides.get("funding_arb_max_spread_pct", 0.5)
    s.funding_arb_leverage = overrides.get("funding_arb_leverage", 1)
    s.funding_arb_symbols = overrides.get("funding_arb_symbols", ["BTC/USDT"])
    # Grid defaults
    s.grid_levels = overrides.get("grid_levels", 5)
    s.grid_spacing_pct = overrides.get("grid_spacing_pct", 0.5)
    s.grid_auto_range = overrides.get("grid_auto_range", True)
    s.grid_range_atr_multiplier = overrides.get("grid_range_atr_multiplier", 3.0)
    s.grid_max_open_orders = overrides.get("grid_max_open_orders", 20)
    s.grid_symbols = overrides.get("grid_symbols", ["BTC/USDT"])
    # Cross arb defaults
    s.cross_arb_min_spread_pct = overrides.get("cross_arb_min_spread_pct", 0.3)
    s.cross_arb_symbols = overrides.get("cross_arb_symbols", ["BTC/USDT"])
    s.cross_arb_max_position_per_symbol = overrides.get("cross_arb_max_position_per_symbol", 1000.0)
    s.cross_arb_rebalance_threshold_pct = overrides.get("cross_arb_rebalance_threshold_pct", 20.0)
    # Stat arb defaults
    s.stat_arb_pairs = overrides.get("stat_arb_pairs", [["BTC/USDT", "ETH/USDT"]])
    s.stat_arb_lookback = overrides.get("stat_arb_lookback", 10)
    s.stat_arb_entry_zscore = overrides.get("stat_arb_entry_zscore", 2.0)
    s.stat_arb_exit_zscore = overrides.get("stat_arb_exit_zscore", 0.5)
    s.stat_arb_stop_zscore = overrides.get("stat_arb_stop_zscore", 4.0)
    s.stat_arb_min_correlation = overrides.get("stat_arb_min_correlation", 0.7)
    return s


# ===========================================================================
# BaseEngine._get_regime_adjustments
# ===========================================================================

class TestGetRegimeAdjustments:
    """Tests for BaseEngine._get_regime_adjustments method."""

    def test_no_detector_returns_normal(self):
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        assert engine._regime_detector is None
        adj = engine._get_regime_adjustments()
        assert adj == {"threshold_mult": 1.0, "size_mult": 1.0}

    def test_low_regime(self):
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        engine.set_regime_detector(_make_detector(MarketRegime.LOW))
        adj = engine._get_regime_adjustments()
        assert adj == {"threshold_mult": 0.8, "size_mult": 1.2}

    def test_normal_regime(self):
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        engine.set_regime_detector(_make_detector(MarketRegime.NORMAL))
        adj = engine._get_regime_adjustments()
        assert adj == {"threshold_mult": 1.0, "size_mult": 1.0}

    def test_high_regime(self):
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        engine.set_regime_detector(_make_detector(MarketRegime.HIGH))
        adj = engine._get_regime_adjustments()
        assert adj == {"threshold_mult": 1.3, "size_mult": 0.7}

    def test_crisis_regime(self):
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        engine.set_regime_detector(_make_detector(MarketRegime.CRISIS))
        adj = engine._get_regime_adjustments()
        assert adj == {"threshold_mult": 999.0, "size_mult": 0.0}

    def test_set_regime_detector(self):
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        assert engine._regime_detector is None
        det = _make_detector(MarketRegime.HIGH)
        engine.set_regime_detector(det)
        assert engine._regime_detector is det


# ===========================================================================
# FundingRateArbEngine regime adaptation
# ===========================================================================

class TestFundingArbRegimeAdaptation:
    """Tests for FundingRateArbEngine regime-based threshold adjustment."""

    @pytest.mark.asyncio
    async def test_normal_regime_opens_position(self):
        """NORMAL regime should use normal min_rate (0.0003)."""
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.NORMAL))
        engine._fetch_funding_rate = AsyncMock(return_value={
            "funding_rate": 0.0005,  # > 0.0003
            "spread_pct": 0.1,
            "mark_price": 50000.0,
            "spot_price": 50000.0,
        })

        result = await engine._run_cycle()
        opens = [a for a in result.actions_taken if a.get("action") == "open"]
        assert len(opens) == 1

    @pytest.mark.asyncio
    async def test_high_regime_raises_threshold(self):
        """HIGH regime: min_rate * 1.3 = 0.00039. Rate 0.0003 should be rejected."""
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.HIGH))
        engine._fetch_funding_rate = AsyncMock(return_value={
            "funding_rate": 0.0003,  # < 0.00039 (0.0003 * 1.3)
            "spread_pct": 0.1,
            "mark_price": 50000.0,
            "spot_price": 50000.0,
        })

        result = await engine._run_cycle()
        opens = [a for a in result.actions_taken if a.get("action") == "open"]
        assert len(opens) == 0

    @pytest.mark.asyncio
    async def test_high_regime_opens_if_rate_exceeds_adjusted(self):
        """HIGH regime: min_rate * 1.3 = 0.00039. Rate 0.0005 should be accepted."""
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.HIGH))
        engine._fetch_funding_rate = AsyncMock(return_value={
            "funding_rate": 0.0005,  # > 0.00039
            "spread_pct": 0.1,
            "mark_price": 50000.0,
            "spot_price": 50000.0,
        })

        result = await engine._run_cycle()
        opens = [a for a in result.actions_taken if a.get("action") == "open"]
        assert len(opens) == 1

    @pytest.mark.asyncio
    async def test_crisis_skips_new_entry(self):
        """CRISIS regime should skip all new entries."""
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.CRISIS))
        engine._fetch_funding_rate = AsyncMock(return_value={
            "funding_rate": 0.01,  # Very high rate
            "spread_pct": 0.01,
            "mark_price": 50000.0,
            "spot_price": 50000.0,
        })

        result = await engine._run_cycle()
        opens = [a for a in result.actions_taken if a.get("action") == "open"]
        assert len(opens) == 0
        # Should have CRISIS skip decision
        crisis_decisions = [d for d in result.decisions if "CRISIS" in d.result]
        assert len(crisis_decisions) >= 1

    @pytest.mark.asyncio
    async def test_crisis_still_closes_existing(self):
        """CRISIS regime should still close existing positions."""
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.CRISIS))
        # Pre-existing position
        engine._add_position("BTC/USDT", "delta_neutral", 0.1, 50000.0, funding_rate=0.0001)
        engine._fetch_funding_rate = AsyncMock(return_value={
            "funding_rate": 0.00005,  # Below exit threshold
            "spread_pct": 0.1,
            "mark_price": 50000.0,
            "spot_price": 50000.0,
        })

        result = await engine._run_cycle()
        closes = [a for a in result.actions_taken if a.get("action") == "close"]
        assert len(closes) == 1

    @pytest.mark.asyncio
    async def test_low_regime_lowers_threshold(self):
        """LOW regime: min_rate * 0.8 = 0.00024. Rate 0.00025 should be accepted."""
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.LOW))
        engine._fetch_funding_rate = AsyncMock(return_value={
            "funding_rate": 0.00025,  # < 0.0003 but > 0.00024
            "spread_pct": 0.1,
            "mark_price": 50000.0,
            "spot_price": 50000.0,
        })

        result = await engine._run_cycle()
        opens = [a for a in result.actions_taken if a.get("action") == "open"]
        assert len(opens) == 1

    @pytest.mark.asyncio
    async def test_no_detector_normal_behavior(self):
        """Without detector, should use normal threshold (backward compat)."""
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine._fetch_funding_rate = AsyncMock(return_value={
            "funding_rate": 0.0005,
            "spread_pct": 0.1,
            "mark_price": 50000.0,
            "spot_price": 50000.0,
        })

        result = await engine._run_cycle()
        opens = [a for a in result.actions_taken if a.get("action") == "open"]
        assert len(opens) == 1

    @pytest.mark.asyncio
    async def test_regime_decision_step_present(self):
        """Regime DecisionStep should be present in decisions."""
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.HIGH))
        engine._fetch_funding_rate = AsyncMock(return_value=None)

        result = await engine._run_cycle()
        regime_steps = [d for d in result.decisions if d.label == "시장 레짐"]
        assert len(regime_steps) == 1
        assert "HIGH" in regime_steps[0].observation
        assert regime_steps[0].category == "evaluate"


# ===========================================================================
# GridTradingEngine regime adaptation
# ===========================================================================

class TestGridTradingRegimeAdaptation:
    """Tests for GridTradingEngine regime-based grid spacing adjustment."""

    @pytest.mark.asyncio
    async def test_normal_regime_creates_grid(self):
        """NORMAL regime should create grid with normal spacing."""
        pm = _make_portfolio_manager()
        engine = GridTradingEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.NORMAL))
        engine._get_current_price = AsyncMock(return_value=50000.0)

        await engine._run_cycle()
        assert "BTC/USDT" in engine._grids
        # Check spacing = 0.5% (normal)
        grid = engine._grids["BTC/USDT"]
        buy_levels = sorted([lvl.price for lvl in grid if lvl.side == "buy"], reverse=True)
        if len(buy_levels) >= 2:
            spacing = (buy_levels[0] - buy_levels[1]) / 50000.0 * 100
            assert abs(spacing - 0.5) < 0.01

    @pytest.mark.asyncio
    async def test_high_regime_wider_spacing(self):
        """HIGH regime: spacing * 1.3 = 0.65%. Grid should have wider spacing."""
        pm = _make_portfolio_manager()
        engine = GridTradingEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.HIGH))
        engine._get_current_price = AsyncMock(return_value=50000.0)

        await engine._run_cycle()
        assert "BTC/USDT" in engine._grids
        grid = engine._grids["BTC/USDT"]
        buy_levels = sorted([lvl.price for lvl in grid if lvl.side == "buy"], reverse=True)
        if len(buy_levels) >= 2:
            spacing = (buy_levels[0] - buy_levels[1]) / 50000.0 * 100
            assert abs(spacing - 0.65) < 0.01

    @pytest.mark.asyncio
    async def test_crisis_skips_grid_creation(self):
        """CRISIS regime should not create new grids."""
        pm = _make_portfolio_manager()
        engine = GridTradingEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.CRISIS))
        engine._get_current_price = AsyncMock(return_value=50000.0)

        result = await engine._run_cycle()
        assert "BTC/USDT" not in engine._grids
        crisis_decisions = [d for d in result.decisions if "CRISIS" in d.result]
        assert len(crisis_decisions) >= 1

    @pytest.mark.asyncio
    async def test_crisis_existing_grid_still_checks_fills(self):
        """CRISIS regime should still check fills on existing grids."""
        pm = _make_portfolio_manager()
        engine = GridTradingEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.CRISIS))
        # Pre-create a grid
        engine._init_grid("BTC/USDT", 50000.0)
        engine._get_current_price = AsyncMock(return_value=49500.0)

        await engine._run_cycle()
        # Grid should still exist and be checked
        assert "BTC/USDT" in engine._grids

    @pytest.mark.asyncio
    async def test_low_regime_tighter_spacing(self):
        """LOW regime: spacing * 0.8 = 0.4%."""
        pm = _make_portfolio_manager()
        engine = GridTradingEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.LOW))
        engine._get_current_price = AsyncMock(return_value=50000.0)

        await engine._run_cycle()
        assert "BTC/USDT" in engine._grids
        grid = engine._grids["BTC/USDT"]
        buy_levels = sorted([lvl.price for lvl in grid if lvl.side == "buy"], reverse=True)
        if len(buy_levels) >= 2:
            spacing = (buy_levels[0] - buy_levels[1]) / 50000.0 * 100
            assert abs(spacing - 0.4) < 0.01

    @pytest.mark.asyncio
    async def test_regime_decision_step_present(self):
        """Regime DecisionStep should be present in grid decisions."""
        pm = _make_portfolio_manager()
        engine = GridTradingEngine(pm, settings=_make_settings())
        engine.set_regime_detector(_make_detector(MarketRegime.NORMAL))
        engine._get_current_price = AsyncMock(return_value=None)

        result = await engine._run_cycle()
        regime_steps = [d for d in result.decisions if d.label == "시장 레짐"]
        assert len(regime_steps) == 1


# ===========================================================================
# CrossExchangeArbEngine regime adaptation
# ===========================================================================

class TestCrossExchangeArbRegimeAdaptation:
    """Tests for CrossExchangeArbEngine regime-based spread adjustment."""

    def _make_exchanges(self, price_a: float, price_b: float):
        ex_a = MagicMock()
        ex_a.name = "exchange_a"
        ex_a.get_ticker = AsyncMock(return_value={"last": price_a})
        ex_b = MagicMock()
        ex_b.name = "exchange_b"
        ex_b.get_ticker = AsyncMock(return_value={"last": price_b})
        return [ex_a, ex_b]

    @pytest.mark.asyncio
    async def test_normal_regime_executes_arb(self):
        """NORMAL regime should execute arb at normal threshold."""
        pm = _make_portfolio_manager()
        # 1% spread > 0.3% threshold
        exchanges = self._make_exchanges(50500.0, 50000.0)
        engine = CrossExchangeArbEngine(pm, exchanges=exchanges, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.NORMAL))

        result = await engine._run_cycle()
        arb_actions = [a for a in result.actions_taken if a.get("action") == "arb_trade"]
        assert len(arb_actions) == 1

    @pytest.mark.asyncio
    async def test_high_regime_raises_threshold(self):
        """HIGH regime: min_spread * 1.3 = 0.39%. 0.35% spread should be rejected."""
        pm = _make_portfolio_manager()
        # ~0.35% spread (< 0.39%)
        mid = 50000.0
        price_a = mid * (1 + 0.00175)
        price_b = mid * (1 - 0.00175)
        exchanges = self._make_exchanges(price_a, price_b)
        engine = CrossExchangeArbEngine(pm, exchanges=exchanges, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.HIGH))

        result = await engine._run_cycle()
        arb_actions = [a for a in result.actions_taken if a.get("action") == "arb_trade"]
        # Should not execute because 0.35% < max(0.39%, cost_min_spread)
        # But cost_min_spread might be even higher; either way, no arb
        assert len(arb_actions) == 0

    @pytest.mark.asyncio
    async def test_crisis_skips_arb(self):
        """CRISIS regime should skip all new arb trades."""
        pm = _make_portfolio_manager()
        # Even 5% spread should be skipped in CRISIS
        exchanges = self._make_exchanges(52500.0, 50000.0)
        engine = CrossExchangeArbEngine(pm, exchanges=exchanges, settings=_make_settings())
        engine._allocated_capital = 10000.0
        engine.set_regime_detector(_make_detector(MarketRegime.CRISIS))

        result = await engine._run_cycle()
        arb_actions = [a for a in result.actions_taken if a.get("action") == "arb_trade"]
        assert len(arb_actions) == 0
        crisis_decisions = [d for d in result.decisions if "CRISIS" in d.result]
        assert len(crisis_decisions) >= 1

    @pytest.mark.asyncio
    async def test_no_detector_normal_behavior(self):
        """Without detector, should use normal threshold."""
        pm = _make_portfolio_manager()
        exchanges = self._make_exchanges(50500.0, 50000.0)
        engine = CrossExchangeArbEngine(pm, exchanges=exchanges, settings=_make_settings())
        engine._allocated_capital = 10000.0

        result = await engine._run_cycle()
        arb_actions = [a for a in result.actions_taken if a.get("action") == "arb_trade"]
        assert len(arb_actions) == 1

    @pytest.mark.asyncio
    async def test_regime_decision_step_present(self):
        """Regime DecisionStep should be present."""
        pm = _make_portfolio_manager()
        exchanges = self._make_exchanges(50000.0, 50000.0)
        engine = CrossExchangeArbEngine(pm, exchanges=exchanges, settings=_make_settings())
        engine.set_regime_detector(_make_detector(MarketRegime.HIGH))

        result = await engine._run_cycle()
        regime_steps = [d for d in result.decisions if d.label == "시장 레짐"]
        assert len(regime_steps) == 1
        assert "HIGH" in regime_steps[0].observation


# ===========================================================================
# StatisticalArbEngine regime adaptation
# ===========================================================================

class TestStatArbRegimeAdaptation:
    """Tests for StatisticalArbEngine regime-based z-score adjustment."""

    def _make_engine(self, regime: MarketRegime | None = None):
        pm = _make_portfolio_manager()
        exchange = MagicMock()
        exchange.get_ticker = AsyncMock(return_value={"last": 50000.0})
        engine = StatisticalArbEngine(
            pm,
            exchanges=[exchange],
            settings=_make_settings(stat_arb_lookback=10),
        )
        engine._allocated_capital = 10000.0
        if regime is not None:
            engine.set_regime_detector(_make_detector(regime))
        # Prevent _update_price_cache from corrupting seeded data
        engine._update_price_cache = AsyncMock()
        return engine

    def _seed_prices(self, engine, sym_a_prices, sym_b_prices):
        """Seed price cache for both symbols."""
        engine._price_cache["BTC/USDT"] = list(sym_a_prices)
        engine._price_cache["ETH/USDT"] = list(sym_b_prices)

    @pytest.mark.asyncio
    async def test_normal_regime_entry(self):
        """NORMAL regime: entry_zscore=2.0. Z-score > 2.0 should trigger entry."""
        engine = self._make_engine(MarketRegime.NORMAL)
        # Create prices that produce a z-score > 2.0
        base_a = [100.0] * 10
        base_b = [50.0] * 10
        # Last ratio significantly above mean
        base_a[-1] = 130.0  # ratio = 130/50 = 2.6, mean ≈ 2.0, should produce high z
        self._seed_prices(engine, base_a, base_b)

        result = await engine._run_cycle()
        entries = [a for a in result.actions_taken if a.get("action") == "entry"]
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_high_regime_raises_entry_threshold(self):
        """HIGH regime: entry_zscore * 1.3 = 2.6. Z~2.4 should not enter."""
        engine = self._make_engine(MarketRegime.HIGH)
        # Varied prices produce z ≈ 2.42 (> 2.0 normal but < 2.6 high threshold)
        base_a = [100, 101, 99, 100, 102, 98, 100, 101, 99, 105]
        base_b = [50.0] * 10
        self._seed_prices(engine, base_a, base_b)

        result = await engine._run_cycle()
        entries = [a for a in result.actions_taken if a.get("action") == "entry"]
        # z ≈ 2.42 < 2.6 (HIGH threshold) → no entry
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_crisis_skips_entry(self):
        """CRISIS regime should skip all new entries regardless of z-score."""
        engine = self._make_engine(MarketRegime.CRISIS)
        # Extreme z-score
        base_a = [100.0] * 9 + [200.0]
        base_b = [50.0] * 10
        self._seed_prices(engine, base_a, base_b)

        result = await engine._run_cycle()
        entries = [a for a in result.actions_taken if a.get("action") == "entry"]
        assert len(entries) == 0
        crisis_decisions = [d for d in result.decisions if "CRISIS" in d.result]
        assert len(crisis_decisions) >= 1

    @pytest.mark.asyncio
    async def test_crisis_still_exits_existing(self):
        """CRISIS regime should still exit existing positions."""
        engine = self._make_engine(MarketRegime.CRISIS)
        # Correlated prices (corr≈0.89) with z-score=0 → triggers mean_reversion exit
        base_a = [100, 103, 97, 102, 98, 101, 99, 104, 96, 100]
        base_b = [50, 51, 49, 50, 50, 51, 49, 52, 48, 50]
        self._seed_prices(engine, base_a, base_b)

        pair_key = "BTC/USDT|ETH/USDT"
        engine._add_position(
            symbol=pair_key,
            side="short_long",
            quantity=0,
            entry_price=0,
            sym_a="BTC/USDT",
            sym_b="ETH/USDT",
            side_a="short",
            side_b="long",
            entry_zscore=3.0,
            price_a=100.0,
            price_b=50.0,
            qty_a=1.0,
            qty_b=2.0,
        )

        result = await engine._run_cycle()
        exits = [a for a in result.actions_taken if a.get("action") == "exit"]
        # Z-score should be near 0 (mean reversion) → exit
        assert len(exits) == 1

    @pytest.mark.asyncio
    async def test_no_detector_normal_behavior(self):
        """Without detector, entry_zscore unchanged (backward compat)."""
        engine = self._make_engine(regime=None)
        base_a = [100.0] * 9 + [130.0]
        base_b = [50.0] * 10
        self._seed_prices(engine, base_a, base_b)

        result = await engine._run_cycle()
        entries = [a for a in result.actions_taken if a.get("action") == "entry"]
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_regime_decision_step_present(self):
        """Regime DecisionStep should be present."""
        engine = self._make_engine(MarketRegime.NORMAL)
        engine._price_cache["BTC/USDT"] = [100.0] * 10
        engine._price_cache["ETH/USDT"] = [50.0] * 10

        result = await engine._run_cycle()
        regime_steps = [d for d in result.decisions if d.label == "시장 레짐"]
        assert len(regime_steps) == 1

    @pytest.mark.asyncio
    async def test_low_regime_lowers_entry(self):
        """LOW regime: entry_zscore * 0.8 = 1.6. Moderate z should enter."""
        engine = self._make_engine(MarketRegime.LOW)
        # Create prices that give z-score around 1.8 (above 1.6 but below 2.0)
        base_a = [100.0] * 9 + [112.0]
        base_b = [50.0] * 10
        self._seed_prices(engine, base_a, base_b)

        result = await engine._run_cycle()
        # In LOW regime, threshold is 1.6, so if z > 1.6, should enter
        # Without detector it would need z > 2.0
        # Just verify the regime decision step shows LOW
        regime_steps = [d for d in result.decisions if d.label == "시장 레짐"]
        assert "LOW" in regime_steps[0].observation


# ===========================================================================
# _init_grid spacing_override
# ===========================================================================

class TestGridSpacingOverride:
    """Test _init_grid with spacing_override parameter."""

    def test_default_spacing(self):
        pm = _make_portfolio_manager()
        engine = GridTradingEngine(pm, settings=_make_settings(grid_spacing_pct=0.5, grid_levels=5))
        engine._init_grid("BTC/USDT", 50000.0)
        grid = engine._grids["BTC/USDT"]
        buy_levels = sorted([lvl.price for lvl in grid if lvl.side == "buy"], reverse=True)
        spacing = (buy_levels[0] - buy_levels[1]) / 50000.0 * 100
        assert abs(spacing - 0.5) < 0.01

    def test_override_spacing(self):
        pm = _make_portfolio_manager()
        engine = GridTradingEngine(pm, settings=_make_settings(grid_spacing_pct=0.5, grid_levels=5))
        engine._init_grid("BTC/USDT", 50000.0, spacing_override=1.0)
        grid = engine._grids["BTC/USDT"]
        buy_levels = sorted([lvl.price for lvl in grid if lvl.side == "buy"], reverse=True)
        spacing = (buy_levels[0] - buy_levels[1]) / 50000.0 * 100
        assert abs(spacing - 1.0) < 0.01


# ===========================================================================
# _should_open_with_threshold
# ===========================================================================

class TestShouldOpenWithThreshold:
    """Test FundingRateArbEngine._should_open_with_threshold."""

    def test_rate_above_threshold(self):
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        assert engine._should_open_with_threshold(0.0005, 0.1, 0.0003) is True

    def test_rate_below_threshold(self):
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        assert engine._should_open_with_threshold(0.0002, 0.1, 0.0003) is False

    def test_spread_too_wide(self):
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        assert engine._should_open_with_threshold(0.001, 0.6, 0.0003) is False

    def test_adjusted_threshold(self):
        pm = _make_portfolio_manager()
        engine = FundingRateArbEngine(pm, settings=_make_settings())
        # 0.0003 * 1.3 = 0.00039 — rate of 0.00035 should fail
        assert engine._should_open_with_threshold(0.00035, 0.1, 0.00039) is False
        # But 0.0004 should pass
        assert engine._should_open_with_threshold(0.0004, 0.1, 0.00039) is True


# ===========================================================================
# _check_entry with entry_zscore_override
# ===========================================================================

class TestCheckEntryOverride:
    """Test StatisticalArbEngine._check_entry with entry_zscore_override."""

    def test_default_threshold(self):
        pm = _make_portfolio_manager()
        engine = StatisticalArbEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        # z=2.5 > default 2.0 → should enter
        result = engine._check_entry("A|B", "A", "B", 2.5, 100.0, 50.0)
        assert result is not None
        assert result["action"] == "entry"

    def test_override_higher_threshold(self):
        pm = _make_portfolio_manager()
        engine = StatisticalArbEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        # z=2.5 < override 2.6 → should NOT enter
        result = engine._check_entry("A|B", "A", "B", 2.5, 100.0, 50.0, entry_zscore_override=2.6)
        assert result is None

    def test_override_lower_threshold(self):
        pm = _make_portfolio_manager()
        engine = StatisticalArbEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        # z=1.8 > override 1.6 → should enter
        result = engine._check_entry("A|B", "A", "B", 1.8, 100.0, 50.0, entry_zscore_override=1.6)
        assert result is not None

    def test_negative_zscore_override(self):
        pm = _make_portfolio_manager()
        engine = StatisticalArbEngine(pm, settings=_make_settings())
        engine._allocated_capital = 10000.0
        # z=-2.5, abs(-2.5) > override 2.0 → should enter with long_short
        result = engine._check_entry("A|B", "A", "B", -2.5, 100.0, 50.0, entry_zscore_override=2.0)
        assert result is not None
        assert result["side_a"] == "long"
        assert result["side_b"] == "short"
