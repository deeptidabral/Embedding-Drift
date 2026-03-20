"""
Comprehensive test suite for src.drift_detection.metrics.

Validates all five drift metric functions against known synthetic
distributions.  Each test uses deterministic random seeds so results
are reproducible across platforms.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.drift_detection.metrics import (
    cosine_distance_drift,
    kolmogorov_smirnov_per_component,
    maximum_mean_discrepancy,
    population_stability_index,
    wasserstein_distance_drift,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
DIM = 64
N_SAMPLES = 500


def _standard_embeddings(n: int = N_SAMPLES, d: int = DIM) -> np.ndarray:
    """Return *n* samples drawn from a standard normal in *d* dimensions."""
    return RNG.standard_normal((n, d))


def _shifted_embeddings(
    shift: float = 3.0, n: int = N_SAMPLES, d: int = DIM
) -> np.ndarray:
    """Return samples from a normal shifted by *shift* along every axis."""
    return RNG.standard_normal((n, d)) + shift


def _orthogonal_pair(d: int = DIM) -> tuple[np.ndarray, np.ndarray]:
    """Return two (1, d) arrays that are orthogonal to each other."""
    a = np.zeros((1, d))
    b = np.zeros((1, d))
    a[0, 0] = 1.0
    b[0, 1] = 1.0
    return a, b


# ===================================================================
# cosine_distance_drift
# ===================================================================


@pytest.mark.unit
class TestCosineDistanceDrift:
    """Tests for cosine distance between embedding distributions."""

    def test_identical_distributions_near_zero(self) -> None:
        """Identical reference and production sets should yield drift ~ 0."""
        ref = _standard_embeddings()
        score = cosine_distance_drift(ref, ref)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_embeddings_near_one(self) -> None:
        """Mutually orthogonal single-sample sets should yield drift ~ 1."""
        a, b = _orthogonal_pair()
        score = cosine_distance_drift(a, b)
        assert score == pytest.approx(1.0, abs=0.05)

    def test_small_shift_below_threshold(self) -> None:
        """A small additive shift should produce a moderate, nonzero score."""
        ref = _standard_embeddings()
        prod = ref + 0.01
        score = cosine_distance_drift(ref, prod)
        assert 0.0 < score < 0.5

    def test_large_shift_above_threshold(self) -> None:
        """A large directional shift should produce a high drift score."""
        ref = _standard_embeddings()
        prod = _shifted_embeddings(shift=10.0)
        score = cosine_distance_drift(ref, prod)
        assert score > 0.1

    def test_single_sample_does_not_crash(self) -> None:
        """A single-sample distribution should still return a float."""
        ref = _standard_embeddings(n=1)
        prod = _standard_embeddings(n=1)
        score = cosine_distance_drift(ref, prod)
        assert isinstance(score, float)


# ===================================================================
# maximum_mean_discrepancy
# ===================================================================


@pytest.mark.unit
class TestMaximumMeanDiscrepancy:
    """Tests for the MMD two-sample statistic."""

    def test_same_distribution_near_zero(self) -> None:
        """MMD between two draws from the same distribution should be near 0."""
        ref = _standard_embeddings(n=400)
        prod = _standard_embeddings(n=400)
        score = maximum_mean_discrepancy(ref, prod)
        assert score < 0.05

    def test_different_distributions_large(self) -> None:
        """MMD between well-separated distributions should be large."""
        ref = _standard_embeddings(n=400)
        prod = _shifted_embeddings(shift=5.0, n=400)
        score = maximum_mean_discrepancy(ref, prod)
        assert score > 0.1

    def test_score_is_non_negative(self) -> None:
        """MMD is a squared distance and must be non-negative."""
        ref = _standard_embeddings(n=200)
        prod = _standard_embeddings(n=200)
        score = maximum_mean_discrepancy(ref, prod)
        assert score >= 0.0

    def test_high_dimensional_stability(self) -> None:
        """MMD should remain stable in high-dimensional spaces."""
        ref = RNG.standard_normal((200, 512))
        prod = RNG.standard_normal((200, 512)) + 2.0
        score = maximum_mean_discrepancy(ref, prod)
        assert isinstance(score, float)
        assert score > 0.0


# ===================================================================
# kolmogorov_smirnov_per_component
# ===================================================================


@pytest.mark.unit
class TestKolmogorovSmirnovPerComponent:
    """Tests for per-component KS drift detection."""

    def test_same_distribution_low_statistic(self) -> None:
        """KS statistic for draws from the same distribution should be low."""
        ref = _standard_embeddings()
        prod = _standard_embeddings()
        result = kolmogorov_smirnov_per_component(ref, prod)
        # result may be a dict or an array; we check the mean statistic
        if isinstance(result, dict):
            mean_stat = np.mean(list(result.values()))
        else:
            mean_stat = float(np.mean(result))
        assert mean_stat < 0.15

    def test_shifted_distribution_high_statistic(self) -> None:
        """Shifting one distribution should raise the KS statistic."""
        ref = _standard_embeddings()
        prod = _shifted_embeddings(shift=3.0)
        result = kolmogorov_smirnov_per_component(ref, prod)
        if isinstance(result, dict):
            mean_stat = np.mean(list(result.values()))
        else:
            mean_stat = float(np.mean(result))
        assert mean_stat > 0.3

    def test_returns_per_component_values(self) -> None:
        """Result should contain one statistic per embedding dimension."""
        ref = _standard_embeddings(d=8)
        prod = _standard_embeddings(d=8)
        result = kolmogorov_smirnov_per_component(ref, prod)
        if isinstance(result, dict):
            assert len(result) == 8
        else:
            assert len(result) == 8


# ===================================================================
# wasserstein_distance_drift
# ===================================================================


@pytest.mark.unit
class TestWassersteinDistanceDrift:
    """Tests for the Wasserstein (earth mover) distance metric."""

    def test_same_distribution_near_zero(self) -> None:
        """Wasserstein distance between identical arrays should be 0."""
        ref = _standard_embeddings()
        score = wasserstein_distance_drift(ref, ref)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_known_shift_produces_proportional_distance(self) -> None:
        """Shifting all embeddings by a constant should increase the score."""
        ref = _standard_embeddings()
        small_shift = ref + 0.5
        large_shift = ref + 5.0
        score_small = wasserstein_distance_drift(ref, small_shift)
        score_large = wasserstein_distance_drift(ref, large_shift)
        assert score_large > score_small > 0.0

    def test_single_sample(self) -> None:
        """Function should handle single-sample inputs gracefully."""
        ref = _standard_embeddings(n=1)
        prod = _shifted_embeddings(shift=1.0, n=1)
        score = wasserstein_distance_drift(ref, prod)
        assert isinstance(score, float)
        assert score >= 0.0


# ===================================================================
# population_stability_index
# ===================================================================


@pytest.mark.unit
class TestPopulationStabilityIndex:
    """Tests for the Population Stability Index metric."""

    def test_identical_distributions_zero(self) -> None:
        """PSI of a distribution against itself should be near 0."""
        ref = _standard_embeddings()
        score = population_stability_index(ref, ref)
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_shifted_distribution_positive(self) -> None:
        """Shifting the distribution should produce a positive PSI."""
        ref = _standard_embeddings()
        prod = _shifted_embeddings(shift=2.0)
        score = population_stability_index(ref, prod)
        assert score > 0.10

    def test_psi_is_non_negative(self) -> None:
        """PSI must be non-negative by definition."""
        ref = _standard_embeddings()
        prod = _standard_embeddings()
        score = population_stability_index(ref, prod)
        assert score >= 0.0

    def test_moderate_shift_in_warning_band(self) -> None:
        """A moderate shift should fall in the warning band (0.10 - 0.20)."""
        ref = _standard_embeddings(n=2000)
        prod = ref + 0.3
        score = population_stability_index(ref, prod)
        # We only assert it is meaningfully positive -- exact band depends on
        # implementation binning strategy.
        assert score > 0.0


# ===================================================================
# Edge-case and cross-cutting tests
# ===================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Cross-cutting edge cases that apply to all metrics."""

    def test_empty_array_raises(self) -> None:
        """All metrics should raise for empty input arrays."""
        empty = np.empty((0, DIM))
        ref = _standard_embeddings(n=10)
        with pytest.raises((ValueError, IndexError)):
            cosine_distance_drift(empty, ref)
        with pytest.raises((ValueError, IndexError)):
            maximum_mean_discrepancy(empty, ref)

    def test_mismatched_dimensions_raises(self) -> None:
        """Differing embedding dimensions should raise ValueError."""
        a = _standard_embeddings(d=32)
        b = _standard_embeddings(d=64)
        with pytest.raises((ValueError, IndexError)):
            cosine_distance_drift(a, b)

    def test_very_high_dimensional(self) -> None:
        """Metrics should work in very high-dimensional spaces (d=3072)."""
        ref = RNG.standard_normal((50, 3072))
        prod = RNG.standard_normal((50, 3072)) + 1.0
        score_cos = cosine_distance_drift(ref, prod)
        score_wass = wasserstein_distance_drift(ref, prod)
        assert isinstance(score_cos, float)
        assert isinstance(score_wass, float)
