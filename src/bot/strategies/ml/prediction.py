"""ML-based price prediction strategy using scikit-learn.

V2: GradientBoosting with proper cross-validation, feature engineering,
walk-forward training, and model versioning.
"""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

from bot.models import OHLCV, SignalAction, TradingSignal
from bot.strategies.base import BaseStrategy

logger = structlog.get_logger()

# Prediction labels
LABEL_UP = 1
LABEL_DOWN = -1
LABEL_NEUTRAL = 0


class MLPredictionStrategy(BaseStrategy):
    """Predicts short-term price movements using a Gradient Boosting classifier.

    V2 improvements:
    - GradientBoosting instead of RandomForest (better for financial data)
    - 20+ engineered features (lag returns, rolling stats, time features)
    - TimeSeriesSplit cross-validation (no future data leakage)
    - Walk-forward retraining every N candles
    - Calibrated probabilities for better confidence scores
    - Minimum CV accuracy threshold (HOLD if model isn't good enough)
    - Model versioning with timestamp (keeps last 3)
    """

    def __init__(
        self,
        lookback_period: int = 14,
        min_training_samples: int = 500,
        prediction_threshold: float = 0.6,
        model_path: str | None = None,
        # V2 parameters
        n_cv_splits: int = 5,
        min_cv_accuracy: float = 0.55,
        retrain_interval: int = 500,
        max_model_versions: int = 3,
        label_threshold: float = 0.005,
    ):
        self._lookback = lookback_period
        self._min_training_samples = min_training_samples
        self._prediction_threshold = prediction_threshold
        self._model_path = model_path
        self._model: CalibratedClassifierCV | GradientBoostingClassifier | None = None
        self._is_trained = False

        # V2 attributes
        self._n_cv_splits = n_cv_splits
        self._min_cv_accuracy = min_cv_accuracy
        self._retrain_interval = retrain_interval
        self._max_model_versions = max_model_versions
        self._label_threshold = label_threshold
        self._candles_since_train = 0
        self._last_cv_accuracy: float | None = None
        self._feature_importances: dict[str, float] | None = None
        self._feature_names: list[str] = self._get_feature_names()

        # Regime adaptation
        self._regime_disabled = False

        if model_path:
            self._load_model(model_path)

    @property
    def name(self) -> str:
        return "ml_prediction"

    @property
    def required_history_length(self) -> int:
        return max(self._min_training_samples, 50)

    @staticmethod
    def _get_feature_names() -> list[str]:
        """Return ordered list of feature names."""
        return [
            # Original features (0-5)
            "rsi_14",
            "momentum_5",
            "momentum_10",
            "volume_change",
            "bb_width",
            "macd_hist",
            # Lag return features (6-10)
            "return_1",
            "return_2",
            "return_3",
            "return_5",
            "return_10",
            # Rolling statistics (11-16)
            "rolling_std_10",
            "rolling_std_20",
            "rolling_skew_10",
            "rolling_mean_ratio_10",
            "rolling_mean_ratio_20",
            "high_low_range",
            # Volume features (17-19)
            "volume_ratio_10",
            "volume_std_10",
            "volume_trend",
            # Time features (20-21)
            "hour_sin",
            "hour_cos",
            # Additional technical (22-24)
            "close_to_high_ratio",
            "close_to_low_ratio",
            "body_ratio",
        ]

    def _compute_features(self, candles: list[OHLCV]) -> np.ndarray:
        """Compute 25-feature vector from OHLCV data.

        Features:
        - Original 6: RSI, momentum (5,10), volume change, BB width, MACD hist
        - Lag returns: returns at t-1, t-2, t-3, t-5, t-10
        - Rolling stats: std (10,20), skew (10), mean ratio (10,20), high-low range
        - Volume: ratio to 10-period avg, std, trend
        - Time: hour of day (sin/cos encoding)
        - Additional: close/high ratio, close/low ratio, candle body ratio
        """
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        opens = np.array([c.open for c in candles])
        volumes = np.array([c.volume for c in candles])
        timestamps = [c.timestamp for c in candles]

        features_list = []
        start = max(self._lookback, 20)  # Need at least 20 candles for rolling stats

        for i in range(start, len(closes)):
            feat = []

            window = closes[i - self._lookback:i + 1]

            # 0. RSI approximation
            deltas = np.diff(window)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))
            feat.append(rsi)

            # 1. Price momentum (5-period)
            mom_5 = (closes[i] - closes[i - 5]) / closes[i - 5] if closes[i - 5] > 0 else 0
            feat.append(mom_5)

            # 2. Price momentum (10-period)
            if i >= 10 and closes[i - 10] > 0:
                mom_10 = (closes[i] - closes[i - 10]) / closes[i - 10]
            else:
                mom_10 = 0
            feat.append(mom_10)

            # 3. Volume change
            vol_change = (
                (volumes[i] - volumes[i - 1]) / volumes[i - 1] if volumes[i - 1] > 0 else 0
            )
            feat.append(vol_change)

            # 4. Bollinger Band width (20-period)
            bb_window = closes[max(0, i - 20):i + 1]
            bb_std = np.std(bb_window)
            bb_mean = np.mean(bb_window)
            bb_width = (2 * bb_std) / bb_mean if bb_mean > 0 else 0
            feat.append(bb_width)

            # 5. MACD histogram approximation
            ema_12 = np.mean(closes[max(0, i - 12):i + 1])
            ema_26 = np.mean(closes[max(0, i - 26):i + 1])
            macd_hist = (ema_12 - ema_26) / closes[i] if closes[i] > 0 else 0
            feat.append(macd_hist)

            # 6-10. Lag returns
            for lag in [1, 2, 3, 5, 10]:
                if i >= lag and closes[i - lag] > 0:
                    ret = (closes[i] - closes[i - lag]) / closes[i - lag]
                else:
                    ret = 0.0
                feat.append(ret)

            # 11. Rolling std (10-period)
            window_10 = closes[max(0, i - 10):i + 1]
            mean_10 = np.mean(window_10)
            feat.append(np.std(window_10) / mean_10 if mean_10 > 0 else 0)

            # 12. Rolling std (20-period)
            window_20 = closes[max(0, i - 20):i + 1]
            mean_20 = np.mean(window_20)
            feat.append(np.std(window_20) / mean_20 if mean_20 > 0 else 0)

            # 13. Rolling skewness (10-period) — simplified
            if len(window_10) >= 3:
                centered = window_10 - np.mean(window_10)
                std_val = np.std(window_10)
                if std_val > 0:
                    skew = np.mean(centered**3) / (std_val**3)
                else:
                    skew = 0.0
            else:
                skew = 0.0
            feat.append(skew)

            # 14. Rolling mean ratio (10-period) — close / SMA10
            feat.append(closes[i] / mean_10 if mean_10 > 0 else 1.0)

            # 15. Rolling mean ratio (20-period) — close / SMA20
            feat.append(closes[i] / mean_20 if mean_20 > 0 else 1.0)

            # 16. High-low range (normalized)
            feat.append(
                (highs[i] - lows[i]) / closes[i] if closes[i] > 0 else 0
            )

            # 17. Volume ratio (current / 10-period average)
            vol_window = volumes[max(0, i - 10):i + 1]
            vol_avg = np.mean(vol_window)
            feat.append(volumes[i] / vol_avg if vol_avg > 0 else 1.0)

            # 18. Volume std (10-period, normalized)
            feat.append(np.std(vol_window) / vol_avg if vol_avg > 0 else 0)

            # 19. Volume trend (linear slope over 10 periods, normalized)
            if len(vol_window) >= 3:
                x = np.arange(len(vol_window))
                coeffs = np.polyfit(x, vol_window, 1)
                feat.append(coeffs[0] / vol_avg if vol_avg > 0 else 0)
            else:
                feat.append(0.0)

            # 20-21. Hour of day (sin/cos encoding for cyclical time)
            try:
                hour = timestamps[i].hour
            except (AttributeError, TypeError):
                hour = 0
            feat.append(np.sin(2 * np.pi * hour / 24))
            feat.append(np.cos(2 * np.pi * hour / 24))

            # 22. Close-to-high ratio
            feat.append(
                (closes[i] - lows[i]) / (highs[i] - lows[i])
                if (highs[i] - lows[i]) > 0
                else 0.5
            )

            # 23. Close-to-low ratio
            feat.append(
                (highs[i] - closes[i]) / (highs[i] - lows[i])
                if (highs[i] - lows[i]) > 0
                else 0.5
            )

            # 24. Candle body ratio (|close - open| / (high - low))
            feat.append(
                abs(closes[i] - opens[i]) / (highs[i] - lows[i])
                if (highs[i] - lows[i]) > 0
                else 0
            )

            features_list.append(feat)

        return np.array(features_list) if features_list else np.empty((0, 25))

    def _compute_labels(self, candles: list[OHLCV], threshold: float | None = None) -> np.ndarray:
        """Compute labels: UP/DOWN/NEUTRAL based on next candle's close."""
        if threshold is None:
            threshold = self._label_threshold
        closes = np.array([c.close for c in candles])
        start = max(self._lookback, 20)
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

    def _cross_validate(
        self, features: np.ndarray, labels: np.ndarray
    ) -> float:
        """Run TimeSeriesSplit cross-validation and return mean accuracy.

        Uses TimeSeriesSplit to prevent future data leakage.
        """
        n_splits = min(self._n_cv_splits, max(2, len(features) // 20))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        scores = []
        for train_idx, test_idx in tscv.split(features):
            x_train, x_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # Need at least 2 classes in training data
            if len(np.unique(y_train)) < 2:
                continue

            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            scores.append(score)

        return float(np.mean(scores)) if scores else 0.0

    def train(self, candles: list[OHLCV]) -> bool:
        """Train the model on historical data with cross-validation.

        Returns True if training was successful and CV accuracy meets threshold.
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

        # Align features and labels
        min_len = min(len(features), len(labels))
        if min_len < 10:
            return False

        features = features[:min_len]
        labels = labels[:min_len]

        # Need at least 2 classes for classification
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            logger.warning("ml_single_class", classes=list(unique_labels))
            return False

        # Cross-validate to check model quality
        cv_accuracy = self._cross_validate(features, labels)
        self._last_cv_accuracy = cv_accuracy

        logger.info(
            "ml_cross_validation",
            cv_accuracy=round(cv_accuracy, 4),
            threshold=self._min_cv_accuracy,
            samples=min_len,
            n_splits=min(self._n_cv_splits, max(2, len(features) // 20)),
        )

        if cv_accuracy < self._min_cv_accuracy:
            logger.warning(
                "ml_low_accuracy",
                cv_accuracy=round(cv_accuracy, 4),
                threshold=self._min_cv_accuracy,
            )
            # Still train but mark as low accuracy
            self._is_trained = False

        # Train final model on all data using GradientBoosting
        base_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

        # Calibrate probabilities for better confidence scores
        try:
            calibrated = CalibratedClassifierCV(
                estimator=base_model,
                cv=TimeSeriesSplit(n_splits=min(3, max(2, min_len // 20))),
                method="sigmoid",
            )
            calibrated.fit(features, labels)
            self._model = calibrated
        except (ValueError, IndexError):
            # Fall back to uncalibrated if calibration fails
            base_model.fit(features, labels)
            self._model = base_model

        # Log feature importances (available from base GradientBoosting)
        try:
            if hasattr(self._model, "estimator"):
                # CalibratedClassifierCV — get from base estimator of first calibrated classifier
                base = self._model.calibrated_classifiers_[0].estimator
            else:
                base = self._model
            importances = base.feature_importances_
            names = self._feature_names[:len(importances)]
            importance_dict = dict(zip(names, [float(v) for v in importances]))
            sorted_imp = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            self._feature_importances = sorted_imp
            logger.info(
                "ml_feature_importances",
                top_5=dict(list(sorted_imp.items())[:5]),
            )
        except (AttributeError, IndexError):
            self._feature_importances = None

        self._candles_since_train = 0

        if cv_accuracy >= self._min_cv_accuracy:
            self._is_trained = True
            logger.info(
                "ml_model_trained",
                samples=min_len,
                classes=list(unique_labels),
                cv_accuracy=round(cv_accuracy, 4),
            )
            return True
        else:
            return False

    def needs_retrain(self, new_candles: int = 1) -> bool:
        """Check if the model should be retrained based on candle count."""
        self._candles_since_train += new_candles
        return self._candles_since_train >= self._retrain_interval

    async def analyze(self, ohlcv_data: list[OHLCV], **kwargs: Any) -> TradingSignal:
        """Analyze data and predict price direction."""
        symbol = kwargs.get("symbol", "BTC/USDT")

        # Regime check
        if self._regime_disabled:
            return TradingSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                metadata={"reason": "regime_disabled"},
            )

        # Walk-forward: retrain if enough new candles
        if self._is_trained and self.needs_retrain(0):
            if len(ohlcv_data) >= self._min_training_samples:
                logger.info(
                    "ml_walk_forward_retrain",
                    candles_since_train=self._candles_since_train,
                )
                self.train(ohlcv_data)

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
                metadata={
                    "reason": "model_not_trained",
                    "cv_accuracy": self._last_cv_accuracy,
                },
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

        # Increment candle counter for walk-forward
        self._candles_since_train += 1

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
                "cv_accuracy": self._last_cv_accuracy,
                "candles_since_train": self._candles_since_train,
                "feature_importances": self._feature_importances,
            },
        )

    def adapt_to_regime(self, regime: Any) -> None:
        """Adapt to market regime. Disable in HIGH_VOLATILITY (noisy predictions)."""
        try:
            regime_name = regime.value if hasattr(regime, "value") else str(regime)
        except (AttributeError, TypeError):
            return

        if regime_name == "HIGH_VOLATILITY":
            self._regime_disabled = True
        else:
            self._regime_disabled = False

    def save_model(self, path: str) -> None:
        """Save trained model to disk with versioning.

        Creates timestamped versions and keeps the last N versions.
        """
        if self._model is not None:
            # Save at the requested path
            joblib.dump(self._model, path)

            # Also save a versioned copy
            p = Path(path)
            versioned_name = f"{p.stem}_{int(time.time())}{p.suffix}"
            versioned_path = p.parent / versioned_name
            joblib.dump(self._model, versioned_path)

            # Clean up old versions (keep last N)
            self._cleanup_old_versions(p)

            logger.info(
                "ml_model_saved",
                path=path,
                versioned_path=str(versioned_path),
            )

    def _cleanup_old_versions(self, base_path: Path) -> None:
        """Remove old model versions, keeping only the last N."""
        pattern = str(base_path.parent / f"{base_path.stem}_*{base_path.suffix}")
        versions = sorted(glob.glob(pattern))
        while len(versions) > self._max_model_versions:
            old = versions.pop(0)
            try:
                os.remove(old)
                logger.info("ml_old_model_removed", path=old)
            except OSError:
                pass

    def _load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        p = Path(path)
        if p.exists():
            self._model = joblib.load(p)
            self._is_trained = True
            logger.info("ml_model_loaded", path=path)
