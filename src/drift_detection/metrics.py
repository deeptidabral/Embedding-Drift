"""
Pure functions for computing embedding drift metrics.

Every public function accepts two numpy arrays -- a reference distribution
and a production distribution -- and returns a ``DriftResult`` that
carries the metric value, an optional p-value, and a boolean indicating
statistical significance at the configured alpha level.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, norm
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA

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
# 1. Cosine distance drift
# ---------------------------------------------------------------------------


def cosine_distance_drift(
    reference: np.ndarray,
    production: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> DriftResult:
    """Mean pairwise cosine distance between reference and production centroids.

    A permutation-based p-value is estimated by shuffling the combined
    pool and re-computing the statistic 200 times.
    """
    _validate_shapes(reference, production)

    ref_centroid = reference.mean(axis=0, keepdims=True)
    prod_centroid = production.mean(axis=0, keepdims=True)

    observed = float(cdist(ref_centroid, prod_centroid, metric="cosine")[0, 0])

    # Permutation test
    combined = np.concatenate([reference, production], axis=0)
    n_ref = reference.shape[0]
    n_permutations = 200
    count_ge = 0

    rng = np.random.default_rng(seed=42)
    for _ in range(n_permutations):
        perm = rng.permutation(combined.shape[0])
        perm_ref = combined[perm[:n_ref]].mean(axis=0, keepdims=True)
        perm_prod = combined[perm[n_ref:]].mean(axis=0, keepdims=True)
        perm_dist = float(cdist(perm_ref, perm_prod, metric="cosine")[0, 0])
        if perm_dist >= observed:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)

    return DriftResult(
        metric_name="cosine_distance",
        value=observed,
        p_value=p_value,
        is_significant=p_value < alpha,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# 2. Maximum Mean Discrepancy (MMD)
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
# 3. Kolmogorov-Smirnov test per PCA component
# ---------------------------------------------------------------------------


def kolmogorov_smirnov_per_component(
    reference: np.ndarray,
    production: np.ndarray,
    n_components: int = 10,
    alpha: float = DEFAULT_ALPHA,
) -> DriftResult:
    """KS test on the leading PCA components of the embedding space.

    PCA is fit on the *reference* distribution; both sets are projected.
    The returned value is the maximum KS statistic across components,
    and drift is declared significant if any component rejects the
    null hypothesis after Bonferroni correction.
    """
    _validate_shapes(reference, production)

    n_components = min(n_components, reference.shape[1], reference.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    ref_proj = pca.fit_transform(reference)
    prod_proj = pca.transform(production)

    ks_stats: list[float] = []
    p_values: list[float] = []
    corrected_alpha = alpha / n_components  # Bonferroni correction

    for comp in range(n_components):
        stat, pval = ks_2samp(ref_proj[:, comp], prod_proj[:, comp])
        ks_stats.append(float(stat))
        p_values.append(float(pval))

    max_stat = max(ks_stats)
    min_p = min(p_values)
    is_significant = min_p < corrected_alpha

    component_details = {
        f"ks_stat_pc{i}": s for i, s in enumerate(ks_stats)
    }
    component_details.update(
        {f"p_value_pc{i}": p for i, p in enumerate(p_values)}
    )

    return DriftResult(
        metric_name="ks_per_component",
        value=max_stat,
        p_value=min_p,
        is_significant=is_significant,
        alpha=alpha,
        details=component_details,
    )


# ---------------------------------------------------------------------------
# 4. Wasserstein (Earth-Mover) distance drift
# ---------------------------------------------------------------------------


def wasserstein_distance_drift(
    reference: np.ndarray,
    production: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> DriftResult:
    """Sliced Wasserstein distance between two high-dimensional distributions.

    Projects both distributions onto 50 random unit vectors, computes the
    1-D Wasserstein distance for each projection, and averages.  A
    permutation p-value is provided via 200 shuffles.
    """
    _validate_shapes(reference, production)

    n_projections = 50
    rng = np.random.default_rng(seed=42)
    directions = rng.standard_normal((n_projections, reference.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    def _sliced_wd(x: np.ndarray, y: np.ndarray) -> float:
        total = 0.0
        for d in directions:
            px = np.sort(x @ d)
            py = np.sort(y @ d)
            # Interpolate to equal length for 1-D comparison
            min_len = min(len(px), len(py))
            px_interp = np.interp(
                np.linspace(0, 1, min_len), np.linspace(0, 1, len(px)), px
            )
            py_interp = np.interp(
                np.linspace(0, 1, min_len), np.linspace(0, 1, len(py)), py
            )
            total += float(np.mean(np.abs(px_interp - py_interp)))
        return total / n_projections

    observed = _sliced_wd(reference, production)

    # Permutation test
    combined = np.concatenate([reference, production], axis=0)
    n_ref = reference.shape[0]
    n_permutations = 200
    count_ge = 0

    for _ in range(n_permutations):
        perm = rng.permutation(combined.shape[0])
        perm_val = _sliced_wd(combined[perm[:n_ref]], combined[perm[n_ref:]])
        if perm_val >= observed:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)

    return DriftResult(
        metric_name="wasserstein",
        value=observed,
        p_value=p_value,
        is_significant=p_value < alpha,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# 5. Population Stability Index (PSI)
# ---------------------------------------------------------------------------


def population_stability_index(
    reference: np.ndarray,
    production: np.ndarray,
    n_bins: int = 20,
    alpha: float = DEFAULT_ALPHA,
) -> DriftResult:
    """PSI computed on the first principal component.

    PSI thresholds (conventional):
      * < 0.10  -- no significant shift
      * 0.10-0.25 -- moderate shift
      * > 0.25  -- significant shift

    The ``is_significant`` flag is set when PSI exceeds 0.25.
    """
    _validate_shapes(reference, production)

    pca = PCA(n_components=1, random_state=42)
    ref_1d = pca.fit_transform(reference).ravel()
    prod_1d = pca.transform(production).ravel()

    # Build bins from reference
    breakpoints = np.linspace(
        min(ref_1d.min(), prod_1d.min()) - 1e-6,
        max(ref_1d.max(), prod_1d.max()) + 1e-6,
        n_bins + 1,
    )

    ref_counts, _ = np.histogram(ref_1d, bins=breakpoints)
    prod_counts, _ = np.histogram(prod_1d, bins=breakpoints)

    # Convert to proportions with smoothing
    eps = 1e-8
    ref_prop = (ref_counts + eps) / (ref_counts.sum() + eps * n_bins)
    prod_prop = (prod_counts + eps) / (prod_counts.sum() + eps * n_bins)

    psi = float(np.sum((prod_prop - ref_prop) * np.log(prod_prop / ref_prop)))

    # Conventional thresholds mapped to a synthetic p-value for uniformity.
    if psi < 0.10:
        synthetic_p = 0.50
    elif psi < 0.25:
        synthetic_p = 0.05
    else:
        synthetic_p = 0.001

    return DriftResult(
        metric_name="psi",
        value=psi,
        p_value=synthetic_p,
        is_significant=psi > 0.25,
        alpha=alpha,
        details={"n_bins": float(n_bins)},
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
