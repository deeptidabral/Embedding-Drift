"""
Covariate shift detection via a domain classifier.

Trains a binary classifier to distinguish reference from production
input features.  If the classifier achieves accuracy significantly
above 50%, the input distribution has shifted.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class CovariateShiftResult(BaseModel):
    """Outcome of a covariate shift test."""

    is_drift: bool
    classifier_auc: float
    auc_threshold: float
    feature_importances: dict[str, float] = Field(default_factory=dict)
    details: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CovariateShiftDetector:
    """Detect covariate shift using a domain-classifier approach.

    A gradient-boosted classifier is trained to discriminate between
    reference (label 0) and production (label 1) samples.  Cross-
    validated AUC above *auc_threshold* signals distributional shift.

    Parameters
    ----------
    auc_threshold:
        AUC above which drift is declared.
    cv_folds:
        Number of cross-validation folds.
    feature_names:
        Optional names for the input features (used in the importance
        report).
    n_estimators:
        Number of boosting rounds.
    max_depth:
        Maximum tree depth.
    """

    def __init__(
        self,
        auc_threshold: float = 0.60,
        cv_folds: int = 5,
        feature_names: list[str] | None = None,
        n_estimators: int = 100,
        max_depth: int = 3,
    ) -> None:
        self._auc_threshold = auc_threshold
        self._cv_folds = cv_folds
        self._feature_names = feature_names
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._reference: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, reference_features: np.ndarray) -> CovariateShiftDetector:
        """Store the reference feature matrix.

        Parameters
        ----------
        reference_features:
            ``(N, F)`` array of input features from the reference period.
        """
        self._reference = np.asarray(reference_features, dtype=np.float32)
        if self._feature_names is None:
            self._feature_names = [
                f"feature_{i}" for i in range(self._reference.shape[1])
            ]
        return self

    def detect(
        self, production_features: np.ndarray
    ) -> CovariateShiftResult:
        """Test for covariate shift between reference and production.

        Returns a ``CovariateShiftResult`` with the cross-validated AUC
        and per-feature importances.
        """
        if self._reference is None:
            raise RuntimeError("Call fit() before detect().")

        prod = np.asarray(production_features, dtype=np.float32)

        x = np.concatenate([self._reference, prod], axis=0)
        y = np.concatenate([
            np.zeros(self._reference.shape[0]),
            np.ones(prod.shape[0]),
        ])

        clf = GradientBoostingClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=42,
            subsample=0.8,
        )

        scores = cross_val_score(
            clf, x, y, cv=self._cv_folds, scoring="roc_auc"
        )
        mean_auc = float(np.mean(scores))
        std_auc = float(np.std(scores))

        # Fit on full data to extract feature importances
        clf.fit(x, y)
        importances = clf.feature_importances_
        feature_imp: dict[str, float] = {}
        if self._feature_names is not None:
            for name, imp in zip(self._feature_names, importances):
                feature_imp[name] = float(imp)

        is_drift = mean_auc > self._auc_threshold

        return CovariateShiftResult(
            is_drift=is_drift,
            classifier_auc=mean_auc,
            auc_threshold=self._auc_threshold,
            feature_importances=feature_imp,
            details={
                "auc_std": std_auc,
                "n_reference": float(self._reference.shape[0]),
                "n_production": float(prod.shape[0]),
            },
        )
