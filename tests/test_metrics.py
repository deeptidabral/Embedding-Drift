"""
Comprehensive test suite for src.drift_detection.metrics.

Validates the MMD drift metric function against known synthetic
distributions.  Each test uses deterministic random seeds so results
are reproducible across platforms.

Design rationale: Dense embedding dimensions are highly entangled.
Only MMD correctly assesses the full high-dimensional distribution.
See src/drift_detection/metrics.py for the full rationale.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.drift_detection.metrics import (
    maximum_mean_discrepancy,
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
        result = maximum_mean_discrepancy(ref, prod)
        assert result.value < 0.05

    def test_different_distributions_large(self) -> None:
        """MMD between well-separated distributions should be large."""
        ref = _standard_embeddings(n=400)
        prod = _shifted_embeddings(shift=5.0, n=400)
        result = maximum_mean_discrepancy(ref, prod)
        assert result.value > 0.1

    def test_score_is_non_negative(self) -> None:
        """MMD is a squared distance and must be non-negative."""
        ref = _standard_embeddings(n=200)
        prod = _standard_embeddings(n=200)
        result = maximum_mean_discrepancy(ref, prod)
        assert result.value >= 0.0

    def test_high_dimensional_stability(self) -> None:
        """MMD should remain stable in high-dimensional spaces."""
        ref = RNG.standard_normal((200, 512))
        prod = RNG.standard_normal((200, 512)) + 2.0
        result = maximum_mean_discrepancy(ref, prod)
        assert isinstance(result.value, float)
        assert result.value > 0.0

    def test_p_value_returned(self) -> None:
        """MMD result should include a permutation p-value."""
        ref = _standard_embeddings(n=100)
        prod = _standard_embeddings(n=100)
        result = maximum_mean_discrepancy(ref, prod)
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0

    def test_significant_drift_detected(self) -> None:
        """MMD should detect significant drift for well-separated distributions."""
        ref = _standard_embeddings(n=200)
        prod = _shifted_embeddings(shift=5.0, n=200)
        result = maximum_mean_discrepancy(ref, prod)
        assert result.is_significant is True

    def test_no_drift_not_significant(self) -> None:
        """MMD should not flag identical distributions as significant."""
        ref = _standard_embeddings(n=200)
        result = maximum_mean_discrepancy(ref, ref)
        assert result.is_significant is False

    def test_metric_name(self) -> None:
        """Result should carry the correct metric name."""
        ref = _standard_embeddings(n=50)
        prod = _standard_embeddings(n=50)
        result = maximum_mean_discrepancy(ref, prod)
        assert result.metric_name == "mmd"

    def test_linear_kernel(self) -> None:
        """MMD with linear kernel should also work."""
        ref = _standard_embeddings(n=100)
        prod = _shifted_embeddings(shift=3.0, n=100)
        result = maximum_mean_discrepancy(ref, prod, kernel="linear")
        assert result.value > 0.0
        assert result.details.get("kernel") == 1.0


# ===================================================================
# Edge-case and cross-cutting tests
# ===================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Cross-cutting edge cases for the MMD metric."""

    def test_empty_array_raises(self) -> None:
        """MMD should raise for empty input arrays."""
        empty = np.empty((0, DIM))
        ref = _standard_embeddings(n=10)
        with pytest.raises((ValueError, IndexError)):
            maximum_mean_discrepancy(empty, ref)

    def test_mismatched_dimensions_raises(self) -> None:
        """Differing embedding dimensions should raise ValueError."""
        a = _standard_embeddings(d=32)
        b = _standard_embeddings(d=64)
        with pytest.raises((ValueError, IndexError)):
            maximum_mean_discrepancy(a, b)

    def test_single_sample_raises(self) -> None:
        """Single-sample distributions should raise ValueError (need >= 2)."""
        ref = _standard_embeddings(n=1)
        prod = _standard_embeddings(n=10)
        with pytest.raises(ValueError):
            maximum_mean_discrepancy(ref, prod)

    def test_very_high_dimensional(self) -> None:
        """MMD should work in very high-dimensional spaces (d=3072)."""
        ref = RNG.standard_normal((50, 3072))
        prod = RNG.standard_normal((50, 3072)) + 1.0
        result = maximum_mean_discrepancy(ref, prod)
        assert isinstance(result.value, float)
        assert result.value > 0.0
