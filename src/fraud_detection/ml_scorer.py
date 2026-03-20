"""
ML-based fraud scoring using gradient boosted trees.

Provides a primary fraud scorer that wraps an XGBoost/gradient-boosted-trees
model behind a scikit-learn-compatible interface.  A simulated scorer is
included for demo and testing purposes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.fraud_detection.transaction_processor import EnrichedTransaction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = (
    ["amount", "is_recurring", "hour_of_day", "day_of_week",
     "days_since_last_txn", "avg_amount_30d", "txn_count_30d",
     "amount_deviation_ratio", "is_new_merchant", "is_high_risk_country"]
    + [f"mcc_{mcc}" for mcc in ["5411", "5812", "5912", "5999", "7995",
                                  "4829", "6011", "5944", "5732", "5691"]]
    + [f"channel_{ch}" for ch in ["online", "in_store", "atm", "mobile", "phone"]]
)

# Known categorical values for one-hot encoding.
# Unknown values get all-zero encoding (no false ordinality imposed).
_KNOWN_MCCS: list[str] = [
    "5411", "5812", "5912", "5999", "7995",
    "4829", "6011", "5944", "5732", "5691",
]
_KNOWN_CHANNELS: list[str] = ["online", "in_store", "atm", "mobile", "phone"]


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class MLScoringResult(BaseModel):
    """Output of the ML fraud scorer."""

    score: float = Field(ge=0.0, le=1.0, description="Fraud probability from the ML model.")
    feature_importances: dict[str, float] = Field(
        default_factory=dict,
        description="Per-feature contribution to this prediction.",
    )
    top_risk_factors: list[str] = Field(
        default_factory=list,
        description="Human-readable names of the top contributing risk factors.",
    )
    scored_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Base scorer
# ---------------------------------------------------------------------------


class MLFraudScorer:
    """Wraps an XGBoost / gradient-boosted-trees model behind a
    scikit-learn-compatible ``predict_proba`` interface.

    Parameters
    ----------
    feature_columns:
        Ordered list of feature names the model expects.
    """

    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self._feature_columns = feature_columns or FEATURE_COLUMNS
        self._model: Any | None = None

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def load_model(self, path: str | Path) -> None:
        """Load a persisted model from *path*.

        Supports joblib (``.joblib``), pickle (``.pkl``), and native
        XGBoost JSON (``.json``) formats.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in (".joblib",):
            import joblib
            self._model = joblib.load(path)
        elif suffix in (".pkl", ".pickle"):
            import pickle
            with open(path, "rb") as fh:
                self._model = pickle.load(fh)  # noqa: S301
        elif suffix == ".json":
            import xgboost as xgb
            self._model = xgb.XGBClassifier()
            self._model.load_model(str(path))
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

        logger.info("Loaded ML fraud model from %s", path)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, enriched: EnrichedTransaction) -> np.ndarray:
        """Convert an ``EnrichedTransaction`` into the numeric feature
        vector expected by the model.

        Categorical variables (merchant_category_code, channel) are one-hot
        encoded to avoid leaking false ordinality into tree-based models.
        Unknown categories get all-zero encoding rather than being mapped
        to a shared bucket.

        Returns a 1-D ``np.ndarray``.
        """
        txn = enriched.transaction

        # Parse hour / day-of-week from ISO timestamp.
        try:
            dt = datetime.fromisoformat(txn.timestamp)
            hour_of_day = dt.hour
            day_of_week = dt.weekday()
        except (ValueError, TypeError):
            hour_of_day = 12
            day_of_week = 3

        avg_amount = enriched.avg_amount_30d if enriched.avg_amount_30d is not None else txn.amount
        amount_deviation_ratio = (
            (txn.amount / avg_amount) if avg_amount > 0 else 1.0
        )

        # One-hot encode MCC. Unknown MCCs get all-zero (no false grouping).
        mcc_onehot = [
            1.0 if txn.merchant_category_code == mcc else 0.0
            for mcc in _KNOWN_MCCS
        ]

        # One-hot encode channel. Unknown channels get all-zero.
        channel_onehot = [
            1.0 if txn.channel.value == ch else 0.0
            for ch in _KNOWN_CHANNELS
        ]

        numeric_features = [
            txn.amount,
            float(txn.is_recurring),
            float(hour_of_day),
            float(day_of_week),
            enriched.days_since_last_txn if enriched.days_since_last_txn is not None else -1.0,
            avg_amount if avg_amount is not None else -1.0,
            float(enriched.txn_count_30d) if enriched.txn_count_30d is not None else -1.0,
            amount_deviation_ratio,
            float(enriched.is_new_merchant),
            float(enriched.is_high_risk_country),
        ]

        features = np.array(
            numeric_features + mcc_onehot + channel_onehot,
            dtype=np.float64,
        )
        return features

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, enriched: EnrichedTransaction) -> MLScoringResult:
        """Score a single enriched transaction.

        Raises ``RuntimeError`` if no model has been loaded.
        """
        if self._model is None:
            raise RuntimeError(
                "No model loaded. Call load_model() before predict()."
            )

        features = self.extract_features(enriched)
        proba = self._model.predict_proba(features.reshape(1, -1))[0]
        # Convention: index 1 is the fraud class.
        score = float(proba[1]) if len(proba) > 1 else float(proba[0])

        importances = self._get_feature_importances()
        top_factors = self._top_risk_factors(features, importances, top_n=3)

        return MLScoringResult(
            score=score,
            feature_importances=importances,
            top_risk_factors=top_factors,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_feature_importances(self) -> dict[str, float]:
        """Extract feature importances from the underlying model."""
        if self._model is None:
            return {}
        try:
            raw = self._model.feature_importances_
            total = float(np.sum(raw)) or 1.0
            return {
                name: round(float(raw[i]) / total, 4)
                for i, name in enumerate(self._feature_columns)
                if i < len(raw)
            }
        except AttributeError:
            return {}

    def _top_risk_factors(
        self,
        features: np.ndarray,
        importances: dict[str, float],
        top_n: int = 3,
    ) -> list[str]:
        """Identify the most influential risk factors for this prediction."""
        if not importances:
            return []
        scored = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        return [name for name, _ in scored[:top_n]]


# ---------------------------------------------------------------------------
# Simulated scorer (demo / testing)
# ---------------------------------------------------------------------------


class SimulatedMLScorer(MLFraudScorer):
    """Generates realistic fraud-probability scores without a trained model.

    Uses hand-crafted heuristic rules that approximate the behaviour of a
    gradient-boosted-trees model, producing scores in [0, 1].  Useful for
    integration testing and demonstrations.
    """

    def __init__(self, feature_columns: list[str] | None = None) -> None:
        super().__init__(feature_columns)
        # Mark as "loaded" so predict() does not raise.
        self._model = self  # sentinel -- overridden predict bypasses model call

    # Override load_model to be a no-op (no real model to load).
    def load_model(self, path: str | Path) -> None:  # noqa: ARG002
        logger.info("SimulatedMLScorer: load_model is a no-op (using heuristics)")

    def predict(self, enriched: EnrichedTransaction) -> MLScoringResult:
        """Heuristic-based scoring that mimics XGBoost output."""
        features = self.extract_features(enriched)
        txn = enriched.transaction

        risk = 0.0
        factors: list[str] = []

        # --- Amount-based risk ---
        if txn.amount > 10_000:
            risk += 0.25
            factors.append("very_high_amount")
        elif txn.amount > 5_000:
            risk += 0.15
            factors.append("high_amount")
        elif txn.amount > 1_000:
            risk += 0.05

        # --- Channel risk ---
        if txn.channel.value == "online":
            risk += 0.10
            factors.append("online_channel")
        elif txn.channel.value == "atm":
            risk += 0.07

        # --- High-risk country ---
        if enriched.is_high_risk_country:
            risk += 0.20
            factors.append("high_risk_country")

        # --- New merchant ---
        if enriched.is_new_merchant:
            risk += 0.12
            factors.append("new_merchant")

        # --- Amount deviation ---
        avg = enriched.avg_amount_30d or txn.amount
        deviation_ratio = (txn.amount / avg) if avg > 0 else 1.0
        if deviation_ratio > 3.0:
            risk += 0.15
            factors.append("amount_deviation_high")
        elif deviation_ratio > 1.5:
            risk += 0.05

        # --- Low recent activity ---
        if enriched.txn_count_30d is not None and enriched.txn_count_30d < 3:
            risk += 0.08
            factors.append("low_recent_activity")

        # --- Time-of-day risk (late night) ---
        try:
            hour = datetime.fromisoformat(txn.timestamp).hour
        except (ValueError, TypeError):
            hour = 12
        if hour < 5 or hour >= 23:
            risk += 0.06
            factors.append("unusual_hour")

        # --- Recurring payment reduces risk ---
        if txn.is_recurring:
            risk -= 0.10

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, risk))

        # Build synthetic feature importances
        importances = {
            "amount": 0.18,
            "merchant_category_code": 0.08,
            "channel": 0.10,
            "is_recurring": 0.05,
            "hour_of_day": 0.06,
            "day_of_week": 0.03,
            "days_since_last_txn": 0.07,
            "avg_amount_30d": 0.09,
            "txn_count_30d": 0.08,
            "amount_deviation_ratio": 0.12,
            "is_new_merchant": 0.07,
            "is_high_risk_country": 0.07,
        }

        return MLScoringResult(
            score=round(score, 4),
            feature_importances=importances,
            top_risk_factors=factors[:3],
        )
