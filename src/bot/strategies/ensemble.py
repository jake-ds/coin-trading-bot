"""Signal ensemble voting system for combining multiple strategy signals."""

from typing import Any

import structlog

from bot.models import SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy

logger = structlog.get_logger()


class SignalEnsemble:
    """Combines signals from multiple strategies into a single voting decision.

    Instead of executing every strategy signal independently, this collects
    all signals for a symbol and produces a single weighted vote.
    """

    def __init__(
        self,
        min_agreement: int = 2,
        strategy_weights: dict[str, float] | None = None,
    ):
        self._min_agreement = min_agreement
        self._strategy_weights = strategy_weights or {}

    @property
    def min_agreement(self) -> int:
        return self._min_agreement

    @property
    def strategy_weights(self) -> dict[str, float]:
        return self._strategy_weights

    async def collect_signals(
        self,
        symbol: str,
        strategies: list[BaseStrategy],
        candles: list,
        **kwargs: Any,
    ) -> list[TradingSignal]:
        """Run all active strategies and collect their signals for a symbol.

        Args:
            symbol: Trading pair symbol.
            strategies: List of strategy instances to run.
            candles: OHLCV candle data.
            **kwargs: Additional context passed to each strategy.

        Returns:
            List of TradingSignal from each strategy that had enough data.
        """
        signals: list[TradingSignal] = []

        for strategy in strategies:
            if len(candles) < strategy.required_history_length:
                continue

            try:
                signal = await strategy.analyze(candles, symbol=symbol, **kwargs)
                signals.append(signal)
            except Exception:
                logger.warning(
                    "strategy_analysis_failed",
                    strategy=strategy.name,
                    symbol=symbol,
                    exc_info=True,
                )

        return signals

    def vote(self, signals: list[TradingSignal], symbol: str) -> TradingSignal:
        """Combine signals into a single decision via weighted voting.

        Voting logic:
        - BUY requires at least min_agreement strategies agreeing on BUY.
        - Same for SELL.
        - If both BUY and SELL signals are present, return HOLD (conflict).
        - If no agreement threshold met, return HOLD.
        - Final confidence = weighted average of agreeing strategies' confidence.

        Args:
            signals: List of signals from different strategies.
            symbol: Trading pair symbol.

        Returns:
            A single TradingSignal representing the ensemble decision.
        """
        if not signals:
            return self._make_hold(symbol, reason="no_signals")

        buy_signals = [s for s in signals if s.action == SignalAction.BUY]
        sell_signals = [s for s in signals if s.action == SignalAction.SELL]

        # CONFLICT: if both BUY and SELL present, return HOLD
        if buy_signals and sell_signals:
            logger.info(
                "ensemble_conflict",
                symbol=symbol,
                buy_count=len(buy_signals),
                sell_count=len(sell_signals),
            )
            return self._make_hold(
                symbol,
                reason="conflict",
                buy_count=len(buy_signals),
                sell_count=len(sell_signals),
            )

        # Check BUY agreement
        if len(buy_signals) >= self._min_agreement:
            confidence = self._weighted_confidence(buy_signals)
            strategy_names = [s.strategy_name for s in buy_signals]
            logger.info(
                "ensemble_buy",
                symbol=symbol,
                agreement=len(buy_signals),
                confidence=round(confidence, 4),
                strategies=strategy_names,
            )
            return TradingSignal(
                strategy_name="ensemble",
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=confidence,
                metadata={
                    "ensemble_agreement": len(buy_signals),
                    "total_signals": len(signals),
                    "agreeing_strategies": strategy_names,
                },
            )

        # Check SELL agreement
        if len(sell_signals) >= self._min_agreement:
            confidence = self._weighted_confidence(sell_signals)
            strategy_names = [s.strategy_name for s in sell_signals]
            logger.info(
                "ensemble_sell",
                symbol=symbol,
                agreement=len(sell_signals),
                confidence=round(confidence, 4),
                strategies=strategy_names,
            )
            return TradingSignal(
                strategy_name="ensemble",
                symbol=symbol,
                action=SignalAction.SELL,
                confidence=confidence,
                metadata={
                    "ensemble_agreement": len(sell_signals),
                    "total_signals": len(signals),
                    "agreeing_strategies": strategy_names,
                },
            )

        # No agreement threshold met
        logger.debug(
            "ensemble_hold",
            symbol=symbol,
            buy_count=len(buy_signals),
            sell_count=len(sell_signals),
            hold_count=len(signals) - len(buy_signals) - len(sell_signals),
            min_agreement=self._min_agreement,
        )
        return self._make_hold(
            symbol,
            reason="insufficient_agreement",
            buy_count=len(buy_signals),
            sell_count=len(sell_signals),
        )

    def _weighted_confidence(self, signals: list[TradingSignal]) -> float:
        """Calculate weighted average confidence for a set of signals."""
        total_weight = 0.0
        weighted_sum = 0.0

        for signal in signals:
            weight = self._strategy_weights.get(signal.strategy_name, 1.0)
            weighted_sum += signal.confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _make_hold(self, symbol: str, reason: str, **metadata: Any) -> TradingSignal:
        """Create a HOLD signal with metadata."""
        return TradingSignal(
            strategy_name="ensemble",
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            metadata={"reason": reason, **metadata},
        )
