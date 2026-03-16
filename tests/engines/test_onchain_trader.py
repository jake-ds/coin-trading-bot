"""Tests for the OnChainTraderEngine."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.engines.base import EngineCycleResult
from bot.engines.onchain_trader import OnChainTraderEngine
from bot.engines.portfolio_manager import PortfolioManager
from bot.onchain.models import (
    CompositeSignal,
    DefiData,
    DerivativesData,
    MarketData,
    SentimentData,
    SignalAction,
    WhaleFlowData,
)


@pytest.fixture
def portfolio_manager():
    pm = PortfolioManager(
        total_capital=1000.0,
        engine_allocations={"onchain_trader": 1.0},
        max_drawdown_pct=20.0,
    )
    return pm


@pytest.fixture
def mock_exchange():
    exchange = AsyncMock()
    exchange.name = "binance"
    exchange.get_ticker = AsyncMock(return_value={
        "bid": 84900,
        "ask": 85100,
        "last": 85000,
        "volume": 50000,
    })
    exchange.create_order = AsyncMock(return_value=MagicMock(
        filled_price=85000,
        filled_quantity=0.001,
        id="test-order-123",
    ))
    exchange.close = AsyncMock()
    return exchange


@pytest.fixture
def engine(portfolio_manager, mock_exchange):
    eng = OnChainTraderEngine(
        portfolio_manager=portfolio_manager,
        exchanges=[mock_exchange],
        paper_mode=True,
        settings=None,
    )
    return eng


def test_engine_properties(engine):
    """Test engine name and description."""
    assert engine.name == "onchain_trader"
    assert "on-chain" in engine.description.lower() or "On-chain" in engine.description


def test_engine_supports_live(engine):
    assert engine.supports_live is True


@pytest.mark.asyncio
async def test_engine_run_cycle_no_data(engine, portfolio_manager):
    """Engine should handle cycle when all data fetchers return None."""
    # Mock all fetchers to return None
    engine._coingecko.fetch_market_data = AsyncMock(return_value=None)
    engine._fear_greed.fetch = AsyncMock(return_value=None)
    engine._defillama.fetch = AsyncMock(return_value=None)
    engine._coinglass.fetch_derivatives = AsyncMock(return_value=None)
    engine._coinglass.fetch_exchange_flow = AsyncMock(return_value=None)

    # Start engine to allocate capital
    engine._allocated_capital = portfolio_manager.request_capital(
        "onchain_trader", 1000.0
    )

    result = await engine._run_cycle()

    assert isinstance(result, EngineCycleResult)
    assert result.engine_name == "onchain_trader"
    assert len(result.actions_taken) == 0
    # All signals should be HOLD with no data
    for sig in engine._latest_signals.values():
        assert sig.action == SignalAction.HOLD


@pytest.mark.asyncio
async def test_engine_opens_position_on_buy_signal(engine, portfolio_manager, mock_exchange):
    """Engine should open a position when BUY signal is strong enough."""
    # Mock data that produces a BUY signal
    engine._coingecko.fetch_market_data = AsyncMock(return_value={
        "BTC/USDT": MarketData(
            symbol="BTC/USDT", price=85000, price_change_24h_pct=5.0
        ),
        "ETH/USDT": MarketData(
            symbol="ETH/USDT", price=3200, price_change_24h_pct=3.0
        ),
        "SOL/USDT": MarketData(
            symbol="SOL/USDT", price=150, price_change_24h_pct=2.0
        ),
    })
    engine._fear_greed.fetch = AsyncMock(return_value=SentimentData(
        value=15, classification="Extreme Fear"
    ))
    engine._defillama.fetch = AsyncMock(return_value=DefiData(
        tvl_change_24h_pct=3.0, stablecoin_supply_change_24h_pct=1.0
    ))
    engine._coinglass.fetch_derivatives = AsyncMock(return_value={
        "BTC/USDT": DerivativesData(symbol="BTC", funding_rate=-0.0005),
    })
    engine._coinglass.fetch_exchange_flow = AsyncMock(return_value={
        "BTC/USDT": WhaleFlowData(
            symbol="BTC", net_flow=-1000, inflow=200, outflow=1200
        ),
    })

    engine._allocated_capital = portfolio_manager.request_capital(
        "onchain_trader", 1000.0
    )

    result = await engine._run_cycle()

    assert isinstance(result, EngineCycleResult)
    # At least BTC should have a BUY signal and trigger an open
    buy_actions = [a for a in result.actions_taken if a.get("action") == "open"]
    assert len(buy_actions) >= 1
    assert engine.position_count >= 1


@pytest.mark.asyncio
async def test_engine_closes_position_on_stop_loss(engine, portfolio_manager, mock_exchange):
    """Engine should close position when price drops below stop loss."""
    engine._allocated_capital = portfolio_manager.request_capital(
        "onchain_trader", 1000.0
    )

    # Manually add a position
    engine._add_position(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.01,
        entry_price=85000.0,
        high_price=85000.0,
    )
    assert engine.position_count == 1

    # Mock price dropped 6% (below 5% stop loss)
    mock_exchange.get_ticker.return_value = {
        "bid": 79800,
        "ask": 79900,
        "last": 79900,
        "volume": 50000,
    }

    # Mock all data fetchers
    engine._coingecko.fetch_market_data = AsyncMock(return_value=None)
    engine._fear_greed.fetch = AsyncMock(return_value=None)
    engine._defillama.fetch = AsyncMock(return_value=None)
    engine._coinglass.fetch_derivatives = AsyncMock(return_value=None)
    engine._coinglass.fetch_exchange_flow = AsyncMock(return_value=None)

    result = await engine._run_cycle()

    close_actions = [a for a in result.actions_taken if a.get("action") == "close"]
    assert len(close_actions) == 1
    assert "stop_loss" in close_actions[0]["reason"]
    assert engine.position_count == 0


@pytest.mark.asyncio
async def test_engine_closes_position_on_take_profit(engine, portfolio_manager, mock_exchange):
    """Engine should close position when price rises above take profit."""
    engine._allocated_capital = portfolio_manager.request_capital(
        "onchain_trader", 1000.0
    )

    engine._add_position(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.01,
        entry_price=85000.0,
        high_price=85000.0,
    )

    # Mock price rose 9% (above 8% take profit)
    mock_exchange.get_ticker.return_value = {
        "bid": 92600,
        "ask": 92700,
        "last": 92650,
        "volume": 50000,
    }

    engine._coingecko.fetch_market_data = AsyncMock(return_value=None)
    engine._fear_greed.fetch = AsyncMock(return_value=None)
    engine._defillama.fetch = AsyncMock(return_value=None)
    engine._coinglass.fetch_derivatives = AsyncMock(return_value=None)
    engine._coinglass.fetch_exchange_flow = AsyncMock(return_value=None)

    result = await engine._run_cycle()

    close_actions = [a for a in result.actions_taken if a.get("action") == "close"]
    assert len(close_actions) == 1
    assert "take_profit" in close_actions[0]["reason"]
    assert result.pnl_update > 0


@pytest.mark.asyncio
async def test_engine_closes_on_signal_reversal(engine, portfolio_manager, mock_exchange):
    """Engine should close position when signal reverses to SELL."""
    engine._allocated_capital = portfolio_manager.request_capital(
        "onchain_trader", 1000.0
    )

    engine._add_position(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.01,
        entry_price=85000.0,
        high_price=85000.0,
    )

    # Price slightly up (no stop/take profit trigger)
    mock_exchange.get_ticker.return_value = {
        "bid": 85400,
        "ask": 85500,
        "last": 85500,
        "volume": 50000,
    }

    # Mock strong SELL signal data
    engine._coingecko.fetch_market_data = AsyncMock(return_value={
        "BTC/USDT": MarketData(
            symbol="BTC/USDT", price=85500, price_change_24h_pct=-5.0
        ),
    })
    engine._fear_greed.fetch = AsyncMock(return_value=SentimentData(
        value=95, classification="Extreme Greed"
    ))
    engine._defillama.fetch = AsyncMock(return_value=DefiData(
        tvl_change_24h_pct=-3.0, stablecoin_supply_change_24h_pct=-1.0
    ))
    engine._coinglass.fetch_derivatives = AsyncMock(return_value={
        "BTC/USDT": DerivativesData(symbol="BTC", funding_rate=0.001),
    })
    engine._coinglass.fetch_exchange_flow = AsyncMock(return_value={
        "BTC/USDT": WhaleFlowData(
            symbol="BTC", net_flow=1000, inflow=1200, outflow=200
        ),
    })

    result = await engine._run_cycle()

    close_actions = [a for a in result.actions_taken if a.get("action") == "close"]
    assert len(close_actions) == 1
    assert "signal_reversal" in close_actions[0]["reason"]


@pytest.mark.asyncio
async def test_engine_respects_max_positions(engine, portfolio_manager, mock_exchange):
    """Engine should not open more positions than max_positions."""
    engine._max_positions = 2
    engine._allocated_capital = portfolio_manager.request_capital(
        "onchain_trader", 1000.0
    )

    # Add 2 existing positions with prices matching ticker
    engine._add_position(symbol="BTC/USDT", side="buy", quantity=0.01, entry_price=85000, high_price=85000)
    engine._add_position(symbol="ETH/USDT", side="buy", quantity=0.1, entry_price=85000, high_price=85000)

    # Mock data fetchers (neutral — no sell signals to close existing)
    engine._coingecko.fetch_market_data = AsyncMock(return_value=None)
    engine._fear_greed.fetch = AsyncMock(return_value=None)
    engine._defillama.fetch = AsyncMock(return_value=None)
    engine._coinglass.fetch_derivatives = AsyncMock(return_value=None)
    engine._coinglass.fetch_exchange_flow = AsyncMock(return_value=None)

    # Mock prices so existing positions don't get closed (same as entry)
    mock_exchange.get_ticker.return_value = {
        "bid": 85000, "ask": 85100, "last": 85000, "volume": 50000,
    }

    result = await engine._run_cycle()

    # Should not have opened any new positions (at capacity)
    open_actions = [a for a in result.actions_taken if a.get("action") == "open"]
    assert len(open_actions) == 0
    # Both positions should remain
    assert engine.position_count == 2


@pytest.mark.asyncio
async def test_engine_crisis_blocks_new_entries(engine, portfolio_manager):
    """CRISIS regime should block new position entries."""
    engine._allocated_capital = portfolio_manager.request_capital(
        "onchain_trader", 1000.0
    )

    # Mock regime detector returning CRISIS
    mock_regime = MagicMock()
    mock_regime.get_current_regime.return_value = MagicMock(value="CRISIS")
    engine._regime_detector = mock_regime

    # Mock strong buy data
    engine._coingecko.fetch_market_data = AsyncMock(return_value={
        "BTC/USDT": MarketData(
            symbol="BTC/USDT", price=85000, price_change_24h_pct=5.0
        ),
    })
    engine._fear_greed.fetch = AsyncMock(return_value=SentimentData(value=15))
    engine._defillama.fetch = AsyncMock(return_value=None)
    engine._coinglass.fetch_derivatives = AsyncMock(return_value=None)
    engine._coinglass.fetch_exchange_flow = AsyncMock(return_value=None)

    result = await engine._run_cycle()

    # No new positions opened
    open_actions = [a for a in result.actions_taken if a.get("action") == "open"]
    assert len(open_actions) == 0


@pytest.mark.asyncio
async def test_engine_latest_signals_property(engine, portfolio_manager):
    """latest_signals property should return dict of signal dicts."""
    engine._allocated_capital = portfolio_manager.request_capital(
        "onchain_trader", 1000.0
    )

    engine._coingecko.fetch_market_data = AsyncMock(return_value=None)
    engine._fear_greed.fetch = AsyncMock(return_value=None)
    engine._defillama.fetch = AsyncMock(return_value=None)
    engine._coinglass.fetch_derivatives = AsyncMock(return_value=None)
    engine._coinglass.fetch_exchange_flow = AsyncMock(return_value=None)

    await engine._run_cycle()

    signals = engine.latest_signals
    assert isinstance(signals, dict)
    # Should have signals for all configured symbols
    for sym in engine._symbols:
        assert sym in signals
        assert "score" in signals[sym]
        assert "action" in signals[sym]


@pytest.mark.asyncio
async def test_engine_stop_closes_sessions(engine):
    """Engine stop should close fetcher sessions."""
    engine._coingecko.close = AsyncMock()
    engine._fear_greed.close = AsyncMock()
    engine._defillama.close = AsyncMock()
    engine._coinglass.close = AsyncMock()
    engine._etherscan.close = AsyncMock()

    await engine.stop()

    engine._coingecko.close.assert_called_once()
    engine._fear_greed.close.assert_called_once()
    engine._defillama.close.assert_called_once()
    engine._coinglass.close.assert_called_once()
    engine._etherscan.close.assert_called_once()
