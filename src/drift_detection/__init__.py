"""
Drift detection algorithms and ensemble detectors.

Includes pure statistical metric functions (cosine distance, MMD,
KS test, Wasserstein distance, PSI) and an ensemble-based detector
that combines multiple metrics with configurable severity thresholds.
"""

from src.drift_detection.metrics import (
    cosine_distance_drift,
    kolmogorov_smirnov_per_component,
    maximum_mean_discrepancy,
    population_stability_index,
    wasserstein_distance_drift,
)
from src.drift_detection.detectors import DriftReport, EmbeddingDriftDetector

__all__ = [
    "cosine_distance_drift",
    "kolmogorov_smirnov_per_component",
    "maximum_mean_discrepancy",
    "population_stability_index",
    "wasserstein_distance_drift",
    "DriftReport",
    "EmbeddingDriftDetector",
]
