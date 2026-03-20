"""
Target shift detection across fraud-rate segments.

Monitors whether the distribution of the fraud label itself has
changed, both globally and stratified by merchant category,
geography, and amount band.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import chi2_contingency, fisher_exact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class SegmentShiftResult(BaseModel):
    """Shift result for a single segment."""

    segment_name: str
    segment_value: str
    reference_fraud_rate: float
    production_fraud_rate: float
    statistic: float
    p_value: float
    is_significant: bool


class TargetShiftResult(BaseModel):
    """Aggregate target-shift detection result."""

    is_drift: bool
    global_chi2: float
    global_p_value: float
    reference_fraud_rate: float
    production_fraud_rate: float
    segment_results: list[SegmentShiftResult] = Field(default_factory=list)
    n_significant_segments: int = 0
    details: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class TargetShiftDetector:
    """Detect changes in the fraud-rate distribution across segments.

    Parameters
    ----------
    alpha:
        Significance level (Bonferroni-corrected per segment).
    min_segment_size:
        Minimum number of samples in a segment for it to be tested.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_segment_size: int = 30,
    ) -> None:
        self._alpha = alpha
        self._min_segment_size = min_segment_size
        self._ref_labels: np.ndarray | None = None
        self._ref_segments: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        reference_labels: np.ndarray,
        reference_segments: dict[str, np.ndarray],
    ) -> TargetShiftDetector:
        """Store reference label and segment data.

        Parameters
        ----------
        reference_labels:
            Binary labels (0/1) for the reference set.
        reference_segments:
            Dict mapping segment name (e.g., ``"merchant_category"``) to
            an array of segment values aligned with *reference_labels*.
        """
        self._ref_labels = np.asarray(reference_labels, dtype=np.int64).ravel()
        self._ref_segments = {
            k: np.asarray(v).ravel() for k, v in reference_segments.items()
        }
        return self

    def detect(
        self,
        production_labels: np.ndarray,
        production_segments: dict[str, np.ndarray],
    ) -> TargetShiftResult:
        """Test for target shift between reference and production.

        Both a global chi-squared test and per-segment tests are
        performed.
        """
        if self._ref_labels is None or self._ref_segments is None:
            raise RuntimeError("Call fit() before detect().")

        prod_labels = np.asarray(production_labels, dtype=np.int64).ravel()
        prod_segments = {
            k: np.asarray(v).ravel() for k, v in production_segments.items()
        }

        # Global test
        ref_fraud = int(self._ref_labels.sum())
        ref_legit = int(len(self._ref_labels) - ref_fraud)
        prod_fraud = int(prod_labels.sum())
        prod_legit = int(len(prod_labels) - prod_fraud)

        global_table = np.array([[ref_legit, ref_fraud], [prod_legit, prod_fraud]])
        if global_table.min() == 0:
            global_chi2, global_p = 0.0, 1.0
        else:
            global_chi2, global_p, _, _ = chi2_contingency(global_table)

        ref_fraud_rate = float(self._ref_labels.mean())
        prod_fraud_rate = float(prod_labels.mean())

        # Per-segment tests
        segment_results: list[SegmentShiftResult] = []
        all_segment_values = self._collect_unique_segments(
            self._ref_segments, prod_segments
        )

        total_tests = sum(len(vals) for vals in all_segment_values.values())
        corrected_alpha = self._alpha / max(total_tests, 1)

        for seg_name, unique_vals in all_segment_values.items():
            ref_seg = self._ref_segments.get(seg_name, np.array([]))
            prod_seg = prod_segments.get(seg_name, np.array([]))

            for val in unique_vals:
                ref_mask = ref_seg == val
                prod_mask = prod_seg == val

                ref_count = int(ref_mask.sum())
                prod_count = int(prod_mask.sum())

                if (
                    ref_count < self._min_segment_size
                    or prod_count < self._min_segment_size
                ):
                    continue

                ref_seg_fraud = int(self._ref_labels[ref_mask].sum())
                ref_seg_legit = ref_count - ref_seg_fraud
                prod_seg_fraud = int(prod_labels[prod_mask].sum())
                prod_seg_legit = prod_count - prod_seg_fraud

                table = np.array([
                    [ref_seg_legit, ref_seg_fraud],
                    [prod_seg_legit, prod_seg_fraud],
                ])

                if table.min() == 0 or table.sum() < 20:
                    # Use Fisher's exact test for small samples
                    _, p_val = fisher_exact(table)
                    stat = 0.0
                else:
                    stat, p_val, _, _ = chi2_contingency(table)

                seg_ref_rate = ref_seg_fraud / ref_count if ref_count > 0 else 0.0
                seg_prod_rate = prod_seg_fraud / prod_count if prod_count > 0 else 0.0

                segment_results.append(
                    SegmentShiftResult(
                        segment_name=seg_name,
                        segment_value=str(val),
                        reference_fraud_rate=seg_ref_rate,
                        production_fraud_rate=seg_prod_rate,
                        statistic=float(stat),
                        p_value=float(p_val),
                        is_significant=p_val < corrected_alpha,
                    )
                )

        n_sig = sum(1 for r in segment_results if r.is_significant)
        is_drift = global_p < self._alpha or n_sig > 0

        return TargetShiftResult(
            is_drift=is_drift,
            global_chi2=float(global_chi2),
            global_p_value=float(global_p),
            reference_fraud_rate=ref_fraud_rate,
            production_fraud_rate=prod_fraud_rate,
            segment_results=segment_results,
            n_significant_segments=n_sig,
            details={
                "n_ref": float(len(self._ref_labels)),
                "n_prod": float(len(prod_labels)),
                "total_segments_tested": float(len(segment_results)),
                "corrected_alpha": corrected_alpha,
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_unique_segments(
        ref_segments: dict[str, np.ndarray],
        prod_segments: dict[str, np.ndarray],
    ) -> dict[str, list[Any]]:
        """Gather the union of unique segment values across both sets."""
        all_names = set(ref_segments.keys()) | set(prod_segments.keys())
        result: dict[str, list[Any]] = {}
        for name in sorted(all_names):
            vals: set[Any] = set()
            if name in ref_segments:
                vals.update(ref_segments[name].tolist())
            if name in prod_segments:
                vals.update(prod_segments[name].tolist())
            result[name] = sorted(vals, key=str)
        return result
