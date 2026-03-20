"""
Test suite for src.drift_detection.detectors.

Validates the EmbeddingDriftDetector ensemble logic, severity classification,
and DriftReport model structure.  All external configuration is mocked so
that tests are fully self-contained.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.drift_detection.detectors import DriftReport, EmbeddingDriftDetector

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(99)
DIM = 64
N_SAMPLES = 300

# Default threshold config that mirrors configs/drift_thresholds.yaml
_THRESHOLDS: dict[str, dict[str, Any]] = {
    "cosine_distance": {
        "nominal_upper": 0.05,
        "warning_upper": 0.15,
        "critical_upper": 0.30,
    },
    "maximum_mean_discrepancy": {
        "nominal_upper": 0.02,
        "warning_upper": 0.08,
        "critical_upper": 0.20,
    },
    "kolmogorov_smirnov": {
        "nominal_upper": 0.05,
        "warning_upper": 0.12,
        "critical_upper": 0.25,
    },
    "wasserstein_distance": {
        "nominal_upper": 0.03,
        "warning_upper": 0.10,
        "critical_upper": 0.22,
    },
    "population_stability_index": {
        "nominal_upper": 0.10,
        "warning_upper": 0.20,
        "critical_upper": 0.35,
    },
}


def _make_detector(
    min_metrics_agreeing: int = 2,
) -> EmbeddingDriftDetector:
    """Create a detector with mocked config loading."""
    with patch.object(
        EmbeddingDriftDetector,
        "_load_thresholds",
        return_value=_THRESHOLDS,
    ):
        detector = EmbeddingDriftDetector(
            thresholds=_THRESHOLDS,
            min_metrics_agreeing=min_metrics_agreeing,
        )
    return detector


def _standard_embeddings(n: int = N_SAMPLES, d: int = DIM) -> np.ndarray:
    return RNG.standard_normal((n, d))


# ===================================================================
# Initialization
# ===================================================================


@pytest.mark.unit
class TestDetectorInit:
    """Verify that EmbeddingDriftDetector initializes correctly."""

    def test_default_construction(self) -> None:
        """Detector should store thresholds and ensemble parameter."""
        detector = _make_detector()
        assert detector.min_metrics_agreeing == 2

    def test_custom_min_metrics(self) -> None:
        """min_metrics_agreeing should accept arbitrary positive integers."""
        detector = _make_detector(min_metrics_agreeing=4)
        assert detector.min_metrics_agreeing == 4

    def test_thresholds_loaded(self) -> None:
        """All five metrics should be present in the loaded thresholds."""
        detector = _make_detector()
        expected_keys = {
            "cosine_distance",
            "maximum_mean_discrepancy",
            "kolmogorov_smirnov",
            "wasserstein_distance",
            "population_stability_index",
        }
        assert expected_keys.issubset(set(detector.thresholds.keys()))


# ===================================================================
# evaluate() -- severity classification
# ===================================================================


@pytest.mark.unit
class TestEvaluateSeverity:
    """Verify that evaluate() returns the correct severity level."""

    def test_no_drift_returns_nominal(self) -> None:
        """When reference and production are the same, severity is nominal."""
        detector = _make_detector()
        ref = _standard_embeddings()
        report = detector.evaluate(ref, ref)
        assert report.severity == "nominal"

    def test_moderate_drift_returns_warning(self) -> None:
        """A moderate distributional shift should trigger a warning."""
        detector = _make_detector()
        ref = _standard_embeddings()
        # Shift enough to exceed warning thresholds on most metrics
        prod = ref + 0.8
        report = detector.evaluate(ref, prod)
        assert report.severity in ("warning", "critical")

    def test_severe_drift_returns_critical(self) -> None:
        """A large distributional shift should trigger critical severity."""
        detector = _make_detector()
        ref = _standard_embeddings()
        prod = _standard_embeddings() + 10.0
        report = detector.evaluate(ref, prod)
        assert report.severity == "critical"


# ===================================================================
# Ensemble logic
# ===================================================================


@pytest.mark.unit
class TestEnsembleLogic:
    """Verify ensemble voting respects min_metrics_agreeing."""

    def test_high_agreement_threshold_reduces_sensitivity(self) -> None:
        """Requiring all 5 metrics to agree makes detection less sensitive."""
        strict_detector = _make_detector(min_metrics_agreeing=5)
        ref = _standard_embeddings()
        # A marginal shift may only push some metrics past warning
        prod = ref + 0.5
        report = strict_detector.evaluate(ref, prod)
        # With a strict agreement threshold, mild drift is more likely nominal
        assert report.severity in ("nominal", "warning")

    def test_low_agreement_threshold_increases_sensitivity(self) -> None:
        """Requiring only 1 metric to agree makes detection more sensitive."""
        sensitive_detector = _make_detector(min_metrics_agreeing=1)
        ref = _standard_embeddings()
        prod = ref + 0.5
        report = sensitive_detector.evaluate(ref, prod)
        # At least one metric likely fires warning for a 0.5 shift
        assert report.severity in ("warning", "critical")


# ===================================================================
# DriftReport model validation
# ===================================================================


@pytest.mark.unit
class TestDriftReport:
    """Validate the DriftReport pydantic model."""

    def test_construction_with_required_fields(self) -> None:
        """DriftReport should accept all required fields."""
        report = DriftReport(
            severity="warning",
            metric_scores={
                "cosine_distance": 0.12,
                "maximum_mean_discrepancy": 0.06,
            },
            timestamp=datetime.now(tz=timezone.utc),
            window_size=1000,
        )
        assert report.severity == "warning"
        assert "cosine_distance" in report.metric_scores

    def test_severity_must_be_valid(self) -> None:
        """DriftReport should reject an unknown severity string."""
        with pytest.raises((ValueError, KeyError)):
            DriftReport(
                severity="catastrophic",  # invalid
                metric_scores={"cosine_distance": 0.5},
                timestamp=datetime.now(tz=timezone.utc),
                window_size=1000,
            )

    def test_metric_scores_dict(self) -> None:
        """metric_scores should be a dict mapping metric names to floats."""
        report = DriftReport(
            severity="nominal",
            metric_scores={
                "cosine_distance": 0.01,
                "wasserstein_distance": 0.02,
            },
            timestamp=datetime.now(tz=timezone.utc),
            window_size=500,
        )
        assert isinstance(report.metric_scores, dict)
        for v in report.metric_scores.values():
            assert isinstance(v, float)

    def test_report_from_evaluate(self) -> None:
        """evaluate() should return a well-formed DriftReport."""
        detector = _make_detector()
        ref = _standard_embeddings()
        report = detector.evaluate(ref, ref)
        assert isinstance(report, DriftReport)
        assert hasattr(report, "severity")
        assert hasattr(report, "metric_scores")
        assert hasattr(report, "timestamp")
