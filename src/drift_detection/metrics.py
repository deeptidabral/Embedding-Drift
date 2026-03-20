"""
Pure functions for computing embedding drift metrics.

Every public function accepts two numpy arrays -- a reference distribution
and a production distribution -- and returns a ``DriftResult`` that
carries the metric value, an optional p-value, and a boolean indicating
statistical significance at the configured alpha level.

Design rationale
----------------
Dense embedding dimensions are highly entangled.  Univariate tests (e.g.
KS per dimension) suffer from massive multiple-testing problems and miss
multivariate rotations.  Cosine distance only captures mean shift.  PCA
explained-variance comparisons miss mean shift entirely.  Only MMD --
a kernel-based two-sample test that operates on the full joint
distribution -- correctly assesses high-dimensional distributional
divergence.  Therefore MMD is the sole drift metric used here.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class DriftResult(BaseModel):
    """Outcome of a single drift metric computation."""

    metric_name: str
    value: float
    p_value: float | None = None
    is_significant: bool = False
    alpha: float = DEFAULT_ALPHA
    details: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy (MMD)
# ---------------------------------------------------------------------------


def maximum_mean_discrepancy(
    reference: np.ndarray,
    production: np.ndarray,
    kernel: Literal["rbf", "linear"] = "rbf",
    alpha: float = DEFAULT_ALPHA,
) -> DriftResult:
    """Unbiased estimate of the squared MMD with an RBF or linear kernel.

    The bandwidth for the RBF kernel is set to the median pairwise
    distance (the *median heuristic*).  A permutation p-value is
    computed with 200 shuffles.
    """
    _validate_shapes(reference, production)

    def _kernel_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if kernel == "linear":
            return x @ y.T
        # RBF
        dists = cdist(x, y, metric="sqeuclidean")
        sigma2 = np.median(dists) + 1e-8
        return np.exp(-dists / (2.0 * sigma2))

    k_rr = _kernel_matrix(reference, reference)
    k_pp = _kernel_matrix(production, production)
    k_rp = _kernel_matrix(reference, production)

    n = reference.shape[0]
    m = production.shape[0]

    # Unbiased estimate
    np.fill_diagonal(k_rr, 0.0)
    np.fill_diagonal(k_pp, 0.0)

    mmd2 = (
        k_rr.sum() / (n * (n - 1))
        + k_pp.sum() / (m * (m - 1))
        - 2.0 * k_rp.sum() / (n * m)
    )

    # Permutation test
    combined = np.concatenate([reference, production], axis=0)
    n_permutations = 200
    count_ge = 0
    rng = np.random.default_rng(seed=42)

    for _ in range(n_permutations):
        perm = rng.permutation(combined.shape[0])
        perm_ref = combined[perm[:n]]
        perm_prod = combined[perm[n:]]

        kp_rr = _kernel_matrix(perm_ref, perm_ref)
        kp_pp = _kernel_matrix(perm_prod, perm_prod)
        kp_rp = _kernel_matrix(perm_ref, perm_prod)

        np.fill_diagonal(kp_rr, 0.0)
        np.fill_diagonal(kp_pp, 0.0)

        perm_mmd2 = (
            kp_rr.sum() / (n * (n - 1))
            + kp_pp.sum() / (perm_prod.shape[0] * (perm_prod.shape[0] - 1))
            - 2.0 * kp_rp.sum() / (n * perm_prod.shape[0])
        )
        if perm_mmd2 >= mmd2:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)

    return DriftResult(
        metric_name="mmd",
        value=float(mmd2),
        p_value=p_value,
        is_significant=p_value < alpha,
        alpha=alpha,
        details={"kernel": 0.0 if kernel == "rbf" else 1.0},
    )


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate_shapes(reference: np.ndarray, production: np.ndarray) -> None:
    """Ensure both arrays are 2-D with matching dimensionality."""
    if reference.ndim != 2 or production.ndim != 2:
        raise ValueError(
            "Both reference and production must be 2-D arrays; got shapes "
            f"{reference.shape} and {production.shape}."
        )
    if reference.shape[1] != production.shape[1]:
        raise ValueError(
            "Embedding dimensions do not match: "
            f"reference has {reference.shape[1]}, "
            f"production has {production.shape[1]}."
        )
    if reference.shape[0] < 2 or production.shape[0] < 2:
        raise ValueError(
            "Each distribution must contain at least 2 samples; got "
            f"{reference.shape[0]} and {production.shape[0]}."
        )
