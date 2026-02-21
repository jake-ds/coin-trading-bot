"""ML-based price prediction strategy using scikit-learn."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from sklearn.ensemble import RandomForestClassifier

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy

logger = structlog.get_logger()

# Prediction labels
LABEL_UP = 1
LABEL_DOWN = -1
LABEL_NEUTRAL = 0


class MLPredictionStrategy(BaseStrategy):
    """Predicts short-term price movements using a Random Forest classifier.

    Features: RSI, MACD histogram, Bollinger Band width, volume change,
    price momentum (multiple periods).
    """

    def __init__(
        self,
        lookback_period: int = 14,
        min_training_samples: int = 500,
        prediction_threshold: float = 0.6,
        model_path: str | None = None,
    ):
        self._lookback = lookback_period
        self._min_training_samples = min_training_samples
        self._prediction_threshold = prediction_threshold
        self._model_path = model_path
        self._model: RandomForestClassifier | None = None
        self._is_trained = False

        if model_path:
            self._load_model(model_path)

    @property
    def name(self) -> str:
        return "ml_prediction"

    @property
    def required_history_length(self) -> int:
        return max(self._min_training_samples, 50)

    def _compute_features(self, candles: list[OHLCV]) -> np.ndarray:
        """Compute feature vector from OHLCV data.

        Features per candle:
        0. RSI (14-period approximation)
        1. Price momentum (5-period)
        2. Price momentum (10-period)
        3. Volume change ratio
        4. Bollinger Band width approximation
        5. MACD histogram approximation
        """
        closes = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])

        features_list = []
        start = max(self._lookback, 14)

        for i in range(start, len(closes)):
            window = closes[i - self._lookback:i + 1]

            # RSI approximation
            deltas = np.diff(window)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))

            # Price momentum
            mom_5 = (closes[i] - closes[i - 5]) / closes[i - 5] if closes[i - 5] > 0 else 0
            if i >= 10 and closes[i - 10] > 0:
                mom_10 = (closes[i] - closes[i - 10]) / closes[i - 10]
            else:
                mom_10 = 0

            # Volume change
            vol_change = (volumes[i] - volumes[i - 1]) / volumes[i - 1] if volumes[i - 1] > 0 else 0

            # Bollinger Band width (20-period)
            bb_window = closes[max(0, i - 20):i + 1]
            bb_std = np.std(bb_window)
            bb_mean = np.mean(bb_window)
            bb_width = (2 * bb_std) / bb_mean if bb_mean > 0 else 0

            # MACD histogram approximation (12-period EMA - 26-period EMA simplified)
            ema_12 = np.mean(closes[max(0, i - 12):i + 1])
            ema_26 = np.mean(closes[max(0, i - 26):i + 1])
            macd_hist = (ema_12 - ema_26) / closes[i] if closes[i] > 0 else 0

            features_list.append([rsi, mom_5, mom_10, vol_change, bb_width, macd_hist])

        return np.array(features_list) if features_list else np.empty((0, 6))

    def _compute_labels(self, candles: list[OHLCV], threshold: float = 0.005) -> np.ndarray:
        """Compute labels: UP/DOWN/NEUTRAL based on next candle's close."""
        closes = np.array([c.close for c in candles])
        start = max(self._lookback, 14)
        labels = []

        for i in range(start, len(closes)):
            if i + 1 < len(closes):
                pct_change = (closes[i + 1] - closes[i]) / closes[i]
                if pct_change > threshold:
                    labels.append(LABEL_UP)
                elif pct_change < -threshold:
                    labels.append(LABEL_DOWN)
                else:
                    labels.append(LABEL_NEUTRAL)
            else:
                labels.append(LABEL_NEUTRAL)

        return np.array(labels)

    def train(self, candles: list[OHLCV]) -> bool:
        """Train the model on historical data.

        Returns True if training was successful.
        """
        if len(candles) < self._min_training_samples:
            logger.warning(
                "ml_insufficient_data",
                available=len(candles),
                required=self._min_training_samples,
            )
            return False

        features = self._compute_features(candles)
        labels = self._compute_labels(candles)

        # Align: labels has one more entry than needed if we're at the last candle
        min_len = min(len(features), len(labels))
        if min_len < 10:
            return False

        features = features[:min_len]
        labels = labels[:min_len]

        self._model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(features, labels)
        self._is_trained = True

        logger.info(
            "ml_model_trained",
            samples=min_len,
            classes=list(np.unique(labels)),
        )
        return True

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        """Analyze data and predict price direction."""
        symbol = kwargs.get("symbol", "BTC/USDT")

        # Auto-train if not trained yet
        if not self._is_trained:
            if len(ohlcv_data) >= self._min_training_samples:
                self.train(ohlcv_data)
            else:
                return TradingSignal(
                    strategy_name=self.name,
                    symbol=symbol,
                    action=SignalAction.HOLD,
                    confidence=0.0,
                    metadata={"reason": "insufficient_data_for_training"},
                )

        if not self._is_trained or self._model is None:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "model_not_trained"},
            )

        features = self._compute_features(ohlcv_data)
        if len(features) == 0:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
            )

        # Use the last feature vector for prediction
        last_features = features[-1:].reshape(1, -1)
        prediction = self._model.predict(last_features)[0]
        probabilities = self._model.predict_proba(last_features)[0]
        confidence = float(max(probabilities))

        if prediction == LABEL_UP and confidence >= self._prediction_threshold:
            action = SignalAction.BUY
        elif prediction == LABEL_DOWN and confidence >= self._prediction_threshold:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD

        return TradingSignal(
            strategy_name=self.name,
            symbol=symbol,
            action=action,
            confidence=confidence,
            metadata={
                "prediction": int(prediction),
                "probabilities": [float(p) for p in probabilities],
            },
        )

    def save_model(self, path: str) -> None:
        """Save trained model to disk."""
        if self._model is not None:
            with open(path, "wb") as f:
                pickle.dump(self._model, f)
            logger.info("ml_model_saved", path=path)

    def _load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        p = Path(path)
        if p.exists():
            with open(p, "rb") as f:
                self._model = pickle.load(f)  # noqa: S301
            self._is_trained = True
            logger.info("ml_model_loaded", path=path)
