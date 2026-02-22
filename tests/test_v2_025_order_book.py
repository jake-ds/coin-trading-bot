"""Integration tests for V2-025: Order book imbalance detection in SignalEnsemble."""

import pytest  # noqa: I001 - ruff wants no gap, but pytest is third-party

from bot.data.order_book import OrderBookAnalysis, OrderBookAnalyzer
from bot.models import SignalAction, TradingSignal
from bot.strategies.ensemble import SignalEnsemble


# --- Fixtures ---


@pytest.fixture
def analyzer():
    return OrderBookAnalyzer()


@pytest.fixture
def ensemble_with_ob(analyzer):
    """SignalEnsemble with order book analyzer."""
    return SignalEnsemble(
        min_agreement=1,
        order_book_analyzer=analyzer,
    )


@pytest.fixture
def ensemble_without_ob():
    """SignalEnsemble without order book analyzer (backward compat)."""
    return SignalEnsemble(min_agreement=1)


def make_buy_signal(name="strategy_a", confidence=0.7):
    return TradingSignal(
        strategy_name=name,
        symbol="BTC/USDT",
        action=SignalAction.BUY,
        confidence=confidence,
    )


def make_sell_signal(name="strategy_a", confidence=0.7):
    return TradingSignal(
        strategy_name=name,
        symbol="BTC/USDT",
        action=SignalAction.SELL,
        confidence=confidence,
    )


def make_hold_signal(name="strategy_a"):
    return TradingSignal(
        strategy_name=name,
        symbol="BTC/USDT",
        action=SignalAction.HOLD,
        confidence=0.0,
    )


# --- Ensemble with Order Book Modifier ---


class TestEnsembleOrderBookIntegration:
    def test_buy_confidence_boosted_by_buying_pressure(
        self, ensemble_with_ob
    ):
        """Strong buying pressure should boost BUY confidence via order book."""
        signals = [make_buy_signal(confidence=0.6)]
        ob_analysis = OrderBookAnalysis(
            bid_volume=100.0,
            ask_volume=30.0,
            imbalance_ratio=100.0 / 30.0,  # ~3.33 > 2.0
        )
        result = ensemble_with_ob.vote(
            signals, "BTC/USDT", order_book_analysis=ob_analysis
        )
        assert result.action == SignalAction.BUY
        assert result.confidence > 0.6  # Boosted
        assert "order_book_modifier" in result.metadata
        assert result.metadata["order_book_modifier"] > 1.0
        assert "imbalance_ratio" in result.metadata

    def test_buy_confidence_reduced_by_selling_pressure(
        self, ensemble_with_ob
    ):
        """Strong selling pressure should reduce BUY confidence."""
        signals = [make_buy_signal(confidence=0.8)]
        ob_analysis = OrderBookAnalysis(
            bid_volume=10.0,
            ask_volume=60.0,
            imbalance_ratio=10.0 / 60.0,  # ~0.167 < 0.5
        )
        result = ensemble_with_ob.vote(
            signals, "BTC/USDT", order_book_analysis=ob_analysis
        )
        assert result.action == SignalAction.BUY
        assert result.confidence < 0.8  # Reduced
        assert result.metadata["order_book_modifier"] < 1.0

    def test_sell_confidence_boosted_by_selling_pressure(
        self, ensemble_with_ob
    ):
        """Strong selling pressure should boost SELL confidence."""
        signals = [make_sell_signal(confidence=0.6)]
        ob_analysis = OrderBookAnalysis(
            bid_volume=10.0,
            ask_volume=60.0,
            imbalance_ratio=10.0 / 60.0,
        )
        result = ensemble_with_ob.vote(
            signals, "BTC/USDT", order_book_analysis=ob_analysis
        )
        assert result.action == SignalAction.SELL
        assert result.confidence > 0.6
        assert result.metadata["order_book_modifier"] > 1.0

    def test_sell_confidence_reduced_by_buying_pressure(
        self, ensemble_with_ob
    ):
        """Strong buying pressure should reduce SELL confidence."""
        signals = [make_sell_signal(confidence=0.8)]
        ob_analysis = OrderBookAnalysis(
            bid_volume=100.0,
            ask_volume=30.0,
            imbalance_ratio=100.0 / 30.0,
        )
        result = ensemble_with_ob.vote(
            signals, "BTC/USDT", order_book_analysis=ob_analysis
        )
        assert result.action == SignalAction.SELL
        assert result.confidence < 0.8
        assert result.metadata["order_book_modifier"] < 1.0

    def test_confidence_capped_at_1(self, ensemble_with_ob):
        """Boosted confidence should never exceed 1.0."""
        signals = [make_buy_signal(confidence=0.9)]
        ob_analysis = OrderBookAnalysis(
            bid_volume=500.0,
            ask_volume=10.0,
            imbalance_ratio=50.0,  # Extreme imbalance
        )
        result = ensemble_with_ob.vote(
            signals, "BTC/USDT", order_book_analysis=ob_analysis
        )
        assert result.action == SignalAction.BUY
        assert result.confidence <= 1.0

    def test_hold_not_modified(self, ensemble_with_ob):
        """HOLD signals should not be modified by order book."""
        signals = [make_hold_signal()]
        ob_analysis = OrderBookAnalysis(
            bid_volume=500.0,
            ask_volume=10.0,
            imbalance_ratio=50.0,
        )
        result = ensemble_with_ob.vote(
            signals, "BTC/USDT", order_book_analysis=ob_analysis
        )
        assert result.action == SignalAction.HOLD
        # HOLD has no order_book_modifier since it goes through insufficient_agreement
        assert "order_book_modifier" not in result.metadata

    def test_buy_wall_boosts_buy(self, ensemble_with_ob):
        """Buy wall below price should boost BUY confidence."""
        signals = [make_buy_signal(confidence=0.5)]
        ob_analysis = OrderBookAnalysis(
            bid_volume=30.0,
            ask_volume=30.0,
            imbalance_ratio=1.0,
            buy_wall_below=True,
        )
        result = ensemble_with_ob.vote(
            signals, "BTC/USDT", order_book_analysis=ob_analysis
        )
        assert result.action == SignalAction.BUY
        # modifier = 1.0 * 1.2 (buy wall) = 1.2
        assert result.confidence == pytest.approx(0.5 * 1.2)
        assert result.metadata["order_book_modifier"] == pytest.approx(1.2)

    def test_sell_wall_reduces_buy(self, ensemble_with_ob):
        """Sell wall above price should reduce BUY confidence."""
        signals = [make_buy_signal(confidence=0.5)]
        ob_analysis = OrderBookAnalysis(
            bid_volume=30.0,
            ask_volume=30.0,
            imbalance_ratio=1.0,
            sell_wall_above=True,
        )
        result = ensemble_with_ob.vote(
            signals, "BTC/USDT", order_book_analysis=ob_analysis
        )
        assert result.action == SignalAction.BUY
        # modifier = 1.0 * 0.8 (sell wall) = 0.8
        assert result.confidence == pytest.approx(0.5 * 0.8)
        assert result.metadata["order_book_modifier"] == pytest.approx(0.8)


# --- Backward Compatibility ---


class TestBackwardCompatibility:
    def test_no_ob_analysis_no_change(self, ensemble_with_ob):
        """No order_book_analysis should leave confidence unchanged."""
        signals = [make_buy_signal(confidence=0.7)]
        result = ensemble_with_ob.vote(signals, "BTC/USDT")
        assert result.action == SignalAction.BUY
        assert result.confidence == 0.7
        assert "order_book_modifier" not in result.metadata

    def test_no_ob_analyzer_no_change(self, ensemble_without_ob):
        """Ensemble without analyzer should ignore order_book_analysis."""
        signals = [make_buy_signal(confidence=0.7)]
        ob_analysis = OrderBookAnalysis(
            bid_volume=500.0,
            ask_volume=10.0,
            imbalance_ratio=50.0,
        )
        result = ensemble_without_ob.vote(
            signals, "BTC/USDT", order_book_analysis=ob_analysis
        )
        assert result.action == SignalAction.BUY
        assert result.confidence == 0.7
        assert "order_book_modifier" not in result.metadata

    def test_conflict_not_affected_by_ob(self, ensemble_with_ob):
        """Conflict detection should run before order book modification."""
        signals = [make_buy_signal(), make_sell_signal(name="strategy_b")]
        ob_analysis = OrderBookAnalysis(
            bid_volume=500.0,
            ask_volume=10.0,
            imbalance_ratio=50.0,
        )
        result = ensemble_with_ob.vote(
            signals, "BTC/USDT", order_book_analysis=ob_analysis
        )
        assert result.action == SignalAction.HOLD
        assert result.metadata.get("reason") == "conflict"

    def test_ensemble_constructor_backward_compat(self):
        """Creating ensemble without order_book_analyzer should work."""
        ensemble = SignalEnsemble(min_agreement=2)
        assert ensemble._order_book_analyzer is None
        signals = [make_buy_signal(), make_buy_signal(name="b")]
        result = ensemble.vote(signals, "BTC/USDT")
        assert result.action == SignalAction.BUY

    def test_vote_backward_compat_no_kwargs(self, ensemble_without_ob):
        """Vote should work without order_book_analysis keyword."""
        signals = [make_buy_signal()]
        result = ensemble_without_ob.vote(signals, "BTC/USDT")
        assert result.action == SignalAction.BUY

    def test_balanced_ob_neutral_effect(self, ensemble_with_ob):
        """Balanced order book (ratio=1.0) should not change confidence."""
        signals = [make_buy_signal(confidence=0.6)]
        ob_analysis = OrderBookAnalysis(
            bid_volume=30.0,
            ask_volume=30.0,
            imbalance_ratio=1.0,
        )
        result = ensemble_with_ob.vote(
            signals, "BTC/USDT", order_book_analysis=ob_analysis
        )
        assert result.action == SignalAction.BUY
        # Modifier is 1.0, so confidence unchanged
        assert result.confidence == pytest.approx(0.6)
        assert result.metadata["order_book_modifier"] == pytest.approx(1.0)
