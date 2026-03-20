"""
Concept shift detection for fraud models.

Monitors whether the relationship between model predictions and true
labels has changed -- i.e., whether the model's learned decision
boundary still aligns with reality.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import chi2_contingency, ks_2samp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class ConceptShiftResult(BaseModel):
    """Outcome of a concept-shift detection test."""

    is_drift: bool
    confidence_ks_statistic: float
    confidence_ks_p_value: float
    calibration_shift: float
    label_distribution_chi2: float
    label_distribution_p_value: float
    details: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class ConceptShiftDetector:
    """Detect changes in the prediction-to-label relationship.

    Concept shift manifests as:
      1. A change in the distribution of prediction confidences
         conditional on the true label.
      2. A change in the calibration curve (predicted probability vs
         observed frequency).

    Parameters
    ----------
    n_bins:
        Number of bins for the calibration analysis.
    alpha:
        Significance level for the KS and chi-squared tests.
    """

    def __init__(
        self,
        n_bins: int = 10,
        alpha: float = 0.05,
    ) -> None:
        self._n_bins = n_bins
        self._alpha = alpha
        self._ref_predictions: np.ndarray | None = None
        self._ref_labels: np.ndarray | None = None
        self._ref_calibration: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        reference_predictions: np.ndarray,
        reference_labels: np.ndarray,
    ) -> ConceptShiftDetector:
        """Learn reference statistics from labelled predictions.

        Parameters
        ----------
        reference_predictions:
            Model-predicted fraud probabilities on the reference set.
        reference_labels:
            Ground-truth binary labels (0/1) for the reference set.
        """
        self._ref_predictions = np.asarray(reference_predictions, dtype=np.float64).ravel()
        self._ref_labels = np.asarray(reference_labels, dtype=np.float64).ravel()
        self._ref_calibration = self._compute_calibration(
            self._ref_predictions, self._ref_labels
        )
        return self

    def detect(
        self,
        production_predictions: np.ndarray,
        production_labels: np.ndarray,
    ) -> ConceptShiftResult:
        """Test for concept shift between reference and production.

        Returns a ``ConceptShiftResult`` summarising the findings.
        """
        if self._ref_predictions is None or self._ref_labels is None:
            raise RuntimeError("Call fit() before detect().")

        prod_preds = np.asarray(production_predictions, dtype=np.float64).ravel()
        prod_labels = np.asarray(production_labels, dtype=np.float64).ravel()

        # 1. KS test on prediction confidences conditioned on positive label
        ref_pos = self._ref_predictions[self._ref_labels == 1]
        prod_pos = prod_preds[prod_labels == 1]

        if len(ref_pos) < 2 or len(prod_pos) < 2:
            ks_stat, ks_p = 0.0, 1.0
        else:
            ks_stat, ks_p = ks_2samp(ref_pos, prod_pos)

        # 2. Calibration shift
        prod_calibration = self._compute_calibration(prod_preds, prod_labels)
        calibration_shift = float(
            np.mean(np.abs(self._ref_calibration - prod_calibration))
        )

        # 3. Label distribution chi-squared test
        ref_counts = np.array([
            np.sum(self._ref_labels == 0),
            np.sum(self._ref_labels == 1),
        ])
        prod_counts = np.array([
            np.sum(prod_labels == 0),
            np.sum(prod_labels == 1),
        ])
        contingency = np.array([ref_counts, prod_counts])

        if contingency.min() == 0:
            chi2_stat, chi2_p = 0.0, 1.0
        else:
            chi2_stat, chi2_p, _, _ = chi2_contingency(contingency)

        is_drift = ks_p < self._alpha or chi2_p < self._alpha

        return ConceptShiftResult(
            is_drift=is_drift,
            confidence_ks_statistic=float(ks_stat),
            confidence_ks_p_value=float(ks_p),
            calibration_shift=calibration_shift,
            label_distribution_chi2=float(chi2_stat),
            label_distribution_p_value=float(chi2_p),
            details={
                "ref_fraud_rate": float(self._ref_labels.mean()),
                "prod_fraud_rate": float(prod_labels.mean()),
                "n_ref": float(len(self._ref_labels)),
                "n_prod": float(len(prod_labels)),
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_calibration(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Compute binned calibration curve (observed frequency per bin)."""
        bin_edges = np.linspace(0.0, 1.0, self._n_bins + 1)
        calibration = np.zeros(self._n_bins, dtype=np.float64)

        for i in range(self._n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if mask.sum() > 0:
                calibration[i] = labels[mask].mean()
            else:
                calibration[i] = 0.0

        return calibration
