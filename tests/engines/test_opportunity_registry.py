"""Tests for OpportunityRegistry."""

from datetime import datetime, timedelta, timezone

import pytest

from bot.engines.opportunity_registry import (
    Opportunity,
    OpportunityRegistry,
    OpportunityType,
)


@pytest.fixture
def registry():
    return OpportunityRegistry()


def _make_opp(
    symbol: str = "BTC/USDT",
    otype: OpportunityType = OpportunityType.FUNDING_RATE,
    score: float = 50.0,
    ttl_minutes: int = 60,
    **extra_metrics,
) -> Opportunity:
    now = datetime.now(timezone.utc)
    return Opportunity(
        symbol=symbol,
        type=otype,
        score=score,
        metrics=extra_metrics,
        discovered_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=ttl_minutes)).isoformat(),
        source_exchange="binance",
    )


def _make_expired_opp(
    symbol: str = "OLD/USDT",
    otype: OpportunityType = OpportunityType.FUNDING_RATE,
    score: float = 90.0,
) -> Opportunity:
    past = datetime.now(timezone.utc) - timedelta(hours=2)
    return Opportunity(
        symbol=symbol,
        type=otype,
        score=score,
        metrics={},
        discovered_at=(past - timedelta(hours=1)).isoformat(),
        expires_at=past.isoformat(),
        source_exchange="binance",
    )


class TestPublishAndGet:
    def test_publish_replaces_all(self, registry):
        opps1 = [_make_opp("A", score=10), _make_opp("B", score=20)]
        registry.publish(OpportunityType.FUNDING_RATE, opps1)
        assert len(registry.get_top(OpportunityType.FUNDING_RATE)) == 2

        opps2 = [_make_opp("C", score=30)]
        registry.publish(OpportunityType.FUNDING_RATE, opps2)
        result = registry.get_top(OpportunityType.FUNDING_RATE)
        assert len(result) == 1
        assert result[0].symbol == "C"

    def test_get_top_returns_sorted_by_score(self, registry):
        opps = [
            _make_opp("LOW", score=10),
            _make_opp("HIGH", score=90),
            _make_opp("MID", score=50),
        ]
        registry.publish(OpportunityType.FUNDING_RATE, opps)
        result = registry.get_top(OpportunityType.FUNDING_RATE, n=3)
        assert [o.symbol for o in result] == ["HIGH", "MID", "LOW"]

    def test_get_top_respects_n(self, registry):
        opps = [_make_opp(f"SYM{i}", score=float(i)) for i in range(10)]
        registry.publish(OpportunityType.VOLATILITY, opps)
        result = registry.get_top(OpportunityType.VOLATILITY, n=3)
        assert len(result) == 3

    def test_get_top_respects_min_score(self, registry):
        opps = [
            _make_opp("LOW", score=10),
            _make_opp("HIGH", score=80),
        ]
        registry.publish(OpportunityType.FUNDING_RATE, opps)
        result = registry.get_top(OpportunityType.FUNDING_RATE, min_score=50.0)
        assert len(result) == 1
        assert result[0].symbol == "HIGH"


class TestExpiry:
    def test_expired_items_filtered_from_get(self, registry):
        opps = [
            _make_opp("FRESH", score=50, ttl_minutes=60),
            _make_expired_opp("STALE", score=90),
        ]
        registry.publish(OpportunityType.FUNDING_RATE, opps)
        result = registry.get_top(OpportunityType.FUNDING_RATE)
        assert len(result) == 1
        assert result[0].symbol == "FRESH"

    def test_clear_expired(self, registry):
        opps = [
            _make_opp("FRESH", score=50),
            _make_expired_opp("STALE1"),
            _make_expired_opp("STALE2"),
        ]
        registry.publish(OpportunityType.FUNDING_RATE, opps)
        removed = registry.clear_expired()
        assert removed == 2
        result = registry.get_top(OpportunityType.FUNDING_RATE)
        assert len(result) == 1

    def test_is_expired_method(self):
        opp = _make_expired_opp()
        assert opp.is_expired()

        fresh = _make_opp(ttl_minutes=60)
        assert not fresh.is_expired()

    def test_no_expiry_never_expires(self):
        opp = Opportunity(
            symbol="X", type=OpportunityType.FUNDING_RATE, score=50,
        )
        assert not opp.is_expired()


class TestGetSymbols:
    def test_get_symbols_returns_strings(self, registry):
        opps = [_make_opp("A", score=80), _make_opp("B", score=60)]
        registry.publish(OpportunityType.FUNDING_RATE, opps)
        syms = registry.get_symbols(OpportunityType.FUNDING_RATE)
        assert syms == ["A", "B"]

    def test_get_symbols_empty(self, registry):
        assert registry.get_symbols(OpportunityType.VOLATILITY) == []


class TestGetPairs:
    def test_get_pairs(self, registry):
        opps = [
            Opportunity(
                symbol="A|B",
                type=OpportunityType.CORRELATION,
                score=95,
                metrics={"pair": ["A", "B"], "correlation": 0.95},
                expires_at=(
                    datetime.now(timezone.utc) + timedelta(hours=24)
                ).isoformat(),
            ),
            Opportunity(
                symbol="C|D",
                type=OpportunityType.CORRELATION,
                score=80,
                metrics={"pair": ["C", "D"], "correlation": 0.80},
                expires_at=(
                    datetime.now(timezone.utc) + timedelta(hours=24)
                ).isoformat(),
            ),
        ]
        registry.publish(OpportunityType.CORRELATION, opps)
        pairs = registry.get_pairs(n=10, min_score=0)
        assert pairs == [["A", "B"], ["C", "D"]]

    def test_get_pairs_empty(self, registry):
        assert registry.get_pairs() == []


class TestSummaryAndAll:
    def test_get_summary(self, registry):
        opps = [_make_opp("A", score=80), _make_opp("B", score=60)]
        registry.publish(OpportunityType.FUNDING_RATE, opps)
        summary = registry.get_summary()
        assert summary["funding_rate"]["count"] == 2
        assert summary["funding_rate"]["top_score"] == 80.0
        assert "A" in summary["funding_rate"]["symbols"]

    def test_get_all_opportunities(self, registry):
        opps = [_make_opp("X", otype=OpportunityType.VOLATILITY, score=70)]
        registry.publish(OpportunityType.VOLATILITY, opps)
        all_opps = registry.get_all_opportunities()
        assert len(all_opps["volatility"]) == 1
        assert all_opps["volatility"][0]["symbol"] == "X"
        assert all_opps["funding_rate"] == []

    def test_to_dict(self):
        opp = _make_opp("BTC/USDT", score=75.123)
        d = opp.to_dict()
        assert d["symbol"] == "BTC/USDT"
        assert d["type"] == "funding_rate"
        assert d["score"] == 75.12


class TestThreadSafety:
    def test_concurrent_publish_and_get(self, registry):
        """Basic concurrency: interleave publish and get calls."""
        import threading

        errors = []

        def publisher():
            try:
                for i in range(50):
                    opps = [_make_opp(f"SYM{i}", score=float(i))]
                    registry.publish(OpportunityType.FUNDING_RATE, opps)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    registry.get_top(OpportunityType.FUNDING_RATE)
                    registry.get_symbols(OpportunityType.FUNDING_RATE)
                    registry.get_summary()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=publisher),
            threading.Thread(target=reader),
            threading.Thread(target=publisher),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent access errors: {errors}"
