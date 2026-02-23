"""Tests for TokenScannerEngine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from bot.engines.opportunity_registry import OpportunityRegistry, OpportunityType
from bot.engines.portfolio_manager import PortfolioManager
from bot.engines.scanner import TokenScannerEngine


@pytest.fixture
def pm():
    return PortfolioManager(
        total_capital=10000.0,
        engine_allocations={"token_scanner": 0.0},
    )


@pytest.fixture
def registry():
    return OpportunityRegistry()


def _mock_exchange(name: str = "binance") -> MagicMock:
    """Create a mock exchange adapter."""
    ex = MagicMock()
    ex.name = name
    type(ex).name = PropertyMock(return_value=name)
    # Give it a ccxt-like _exchange attribute
    ex._exchange = MagicMock()
    return ex


def make_scanner(
    pm, registry, exchanges=None, **overrides
) -> TokenScannerEngine:
    engine = TokenScannerEngine(
        portfolio_manager=pm,
        exchanges=exchanges or [],
        paper_mode=True,
        settings=None,
        registry=registry,
    )
    for k, v in overrides.items():
        setattr(engine, k, v)
    return engine


class TestScannerProperties:
    def test_name(self, pm, registry):
        s = make_scanner(pm, registry)
        assert s.name == "token_scanner"

    def test_description(self, pm, registry):
        s = make_scanner(pm, registry)
        assert "scanner" in s.description.lower()

    def test_registry_property(self, pm, registry):
        s = make_scanner(pm, registry)
        assert s.registry is registry

    def test_max_positions_zero(self, pm, registry):
        s = make_scanner(pm, registry)
        assert s._max_positions == 0


class TestFetchAllTickers:
    @pytest.mark.asyncio
    async def test_fetch_from_ccxt(self, pm, registry):
        ex = _mock_exchange()
        ticker_data = {
            "BTC/USDT": {"last": 50000, "quoteVolume": 1_000_000, "percentage": 2.5},
            "ETH/USDT": {"last": 3000, "quoteVolume": 500_000, "percentage": -1.2},
        }
        ex._exchange.fetch_tickers = AsyncMock(return_value=ticker_data)

        s = make_scanner(pm, registry, exchanges=[ex])
        result = await s._fetch_all_tickers()
        assert len(result) == 2
        assert "BTC/USDT" in result

    @pytest.mark.asyncio
    async def test_fetch_empty_on_failure(self, pm, registry):
        ex = _mock_exchange()
        ex._exchange.fetch_tickers = AsyncMock(side_effect=Exception("API error"))

        s = make_scanner(pm, registry, exchanges=[ex])
        # Disable public client fallback for this test
        s._public_exchange = MagicMock()
        s._public_exchange.fetch_tickers = AsyncMock(
            side_effect=Exception("no fallback"),
        )
        result = await s._fetch_all_tickers()
        assert result == {}


class TestFundingRateScan:
    @pytest.mark.asyncio
    async def test_scan_funding_rates(self, pm, registry):
        ex = _mock_exchange()
        ex._exchange.fetch_funding_rates = AsyncMock(return_value={
            "BTC/USDT": {"fundingRate": 0.001, "symbol": "BTC/USDT"},
            "ETH/USDT": {"fundingRate": 0.0005, "symbol": "ETH/USDT"},
            "DOGE/USDT": {"fundingRate": -0.0001, "symbol": "DOGE/USDT"},
        })

        s = make_scanner(pm, registry, exchanges=[ex])
        # Pre-populate ticker cache
        s._ticker_cache = {
            "BTC/USDT": {"last": 50000},
            "ETH/USDT": {"last": 3000},
            "DOGE/USDT": {"last": 0.1},
        }

        decisions = []
        count = await s._scan_funding_rates(s._ticker_cache, decisions)
        # Only positive funding rates should be published
        assert count == 2
        opps = registry.get_top(OpportunityType.FUNDING_RATE, n=10)
        symbols = [o.symbol for o in opps]
        assert "BTC/USDT" in symbols
        assert "ETH/USDT" in symbols
        assert "DOGE/USDT" not in symbols

    @pytest.mark.asyncio
    async def test_scan_funding_rates_no_data(self, pm, registry):
        ex = _mock_exchange()
        ex._exchange.fetch_funding_rates = AsyncMock(return_value={})

        s = make_scanner(pm, registry, exchanges=[ex])
        decisions = []
        count = await s._scan_funding_rates({}, decisions)
        assert count == 0


class TestVolatilityScan:
    def test_scan_volatility(self, pm, registry):
        s = make_scanner(pm, registry)
        tickers = {
            "BTC/USDT": {"last": 50000, "quoteVolume": 5_000_000, "percentage": 8.5},
            "ETH/USDT": {"last": 3000, "quoteVolume": 2_000_000, "percentage": 3.2},
            "STABLE/USDT": {"last": 1, "quoteVolume": 1_000_000, "percentage": 0.01},
        }

        decisions = []
        count = s._scan_volatility(tickers, decisions)
        assert count >= 2
        opps = registry.get_top(OpportunityType.VOLATILITY, n=10)
        # BTC should rank higher (more volatile + higher volume)
        assert opps[0].symbol == "BTC/USDT"


class TestSpreadScan:
    @pytest.mark.asyncio
    async def test_scan_spreads(self, pm, registry):
        ex_a = _mock_exchange("binance")
        ex_b = _mock_exchange("upbit")
        ex_b._exchange.fetch_tickers = AsyncMock(return_value={
            "BTC/USDT": {"last": 50500},
            "ETH/USDT": {"last": 3001},
        })

        s = make_scanner(pm, registry, exchanges=[ex_a, ex_b])
        tickers_a = {
            "BTC/USDT": {"last": 50000, "quoteVolume": 1_000_000},
            "ETH/USDT": {"last": 3000, "quoteVolume": 500_000},
        }

        decisions = []
        count = await s._scan_spreads(tickers_a, decisions)
        # BTC has ~1% spread → should be published
        assert count >= 1
        opps = registry.get_top(OpportunityType.CROSS_EXCHANGE_SPREAD, n=10)
        assert any(o.symbol == "BTC/USDT" for o in opps)

    @pytest.mark.asyncio
    async def test_scan_spreads_single_exchange(self, pm, registry):
        ex = _mock_exchange()
        s = make_scanner(pm, registry, exchanges=[ex])
        decisions = []
        count = await s._scan_spreads({}, decisions)
        assert count == 0


class TestCorrelationScan:
    def test_scan_correlations(self, pm, registry):
        s = make_scanner(pm, registry)
        # Build up price history (30+ points of correlated data)
        for i in range(35):
            s._price_history.setdefault("A/USDT", []).append(100 + i * 0.5)
            s._price_history.setdefault("B/USDT", []).append(200 + i * 1.0)
            s._price_history.setdefault("C/USDT", []).append(50 + ((-1) ** i) * i)

        tickers = {
            "A/USDT": {"last": 117, "quoteVolume": 2_000_000},
            "B/USDT": {"last": 234, "quoteVolume": 1_000_000},
            "C/USDT": {"last": 50, "quoteVolume": 500_000},
        }

        decisions = []
        count = s._scan_correlations(tickers, decisions)
        # A and B should be highly correlated
        assert count >= 1
        opps = registry.get_top(OpportunityType.CORRELATION, n=10)
        assert len(opps) >= 1

    def test_scan_correlations_insufficient_data(self, pm, registry):
        s = make_scanner(pm, registry)
        tickers = {"A/USDT": {"last": 100, "quoteVolume": 1_000_000}}
        decisions = []
        count = s._scan_correlations(tickers, decisions)
        assert count == 0


class TestFullCycle:
    @pytest.mark.asyncio
    async def test_run_cycle_empty_exchange(self, pm, registry):
        """Scanner handles no-data gracefully."""
        s = make_scanner(pm, registry, exchanges=[])
        await s.start()
        result = await s._run_cycle()
        assert result.engine_name == "token_scanner"
        assert result.pnl_update == 0.0
        assert len(result.decisions) >= 1

    @pytest.mark.asyncio
    async def test_run_cycle_with_data(self, pm, registry):
        ex = _mock_exchange()
        ex._exchange.fetch_tickers = AsyncMock(return_value={
            "BTC/USDT": {
                "last": 50000, "quoteVolume": 5_000_000,
                "percentage": 5.0, "baseVolume": 100,
            },
            "ETH/USDT": {
                "last": 3000, "quoteVolume": 2_000_000,
                "percentage": 3.0, "baseVolume": 666,
            },
        })

        s = make_scanner(pm, registry, exchanges=[ex])
        s._enabled_scans = ["volatility"]
        await s.start()
        result = await s._run_cycle()
        assert result.engine_name == "token_scanner"
        assert result.pnl_update == 0.0
        assert result.metadata["opportunities_published"] >= 1

        # Check registry was populated
        vol_opps = registry.get_top(OpportunityType.VOLATILITY, n=10)
        assert len(vol_opps) >= 1


class TestDecisionSteps:
    @pytest.mark.asyncio
    async def test_decisions_contain_scan_summary(self, pm, registry):
        ex = _mock_exchange()
        ex._exchange.fetch_tickers = AsyncMock(return_value={
            "BTC/USDT": {"last": 50000, "quoteVolume": 5_000_000, "percentage": 5.0},
        })

        s = make_scanner(pm, registry, exchanges=[ex])
        s._enabled_scans = ["volatility"]
        await s.start()
        result = await s._run_cycle()
        labels = [d.label for d in result.decisions]
        assert "배치 티커 조회" in labels
        assert "거래량 필터링" in labels
        assert "변동성 스캔" in labels
        assert "스캔 완료 요약" in labels


class TestVolumeHelper:
    def test_quote_volume(self):
        assert TokenScannerEngine._get_volume_usdt({"quoteVolume": 1_000_000}) == 1_000_000

    def test_fallback_base_volume(self):
        assert TokenScannerEngine._get_volume_usdt(
            {"baseVolume": 10, "last": 50000}
        ) == 500_000

    def test_zero_volume(self):
        assert TokenScannerEngine._get_volume_usdt({}) == 0
