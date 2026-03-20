"""
Drift detection using Maximum Mean Discrepancy (MMD).

MMD is the sole drift metric because dense embedding dimensions are
highly entangled.  Univariate tests (KS per dimension) suffer from
massive multiple-testing problems and miss multivariate rotations.
Cosine distance only captures mean shift.  PCA explained-variance
comparisons miss mean shift entirely.  Only MMD correctly assesses the
full high-dimensional distribution.
"""

from src.drift_detection.metrics import maximum_mean_discrepancy
from src.drift_detection.detectors import DriftReport, EmbeddingDriftDetector

__all__ = [
    "maximum_mean_discrepancy",
    "DriftReport",
    "EmbeddingDriftDetector",
]
