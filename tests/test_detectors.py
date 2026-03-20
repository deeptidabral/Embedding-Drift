"""
Test suite for src.drift_detection.detectors.

Validates the EmbeddingDriftDetector MMD-based detection, severity
classification, and DriftReport model structure.  All external
configuration is mocked so that tests are fully self-contained.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.drift_detection.detectors import (
    DriftReport,
    DriftSeverity,
    EmbeddingDriftDetector,
    MetricThresholds,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(99)
DIM = 64
N_SAMPLES = 300


def _make_detector() -> EmbeddingDriftDetector:
    """Create a detector with mocked config loading (uses built-in defaults)."""
    with patch.object(
        EmbeddingDriftDetector,
        "_load_thresholds",
        return_value={
            "mmd": MetricThresholds(
                low=0.005, moderate=0.02, high=0.05, critical=0.10
            ),
        },
    ):
        detector = EmbeddingDriftDetector()
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
        """Detector should initialize without errors."""
        detector = _make_detector()
        assert detector is not None

    def test_thresholds_contain_mmd(self) -> None:
        """Only MMD thresholds should be present."""
        detector = _make_detector()
        assert "mmd" in detector._thresholds
        assert len(detector._thresholds) == 1


# ===================================================================
# evaluate() -- severity classification
# ===================================================================


@pytest.mark.unit
class TestEvaluateSeverity:
    """Verify that evaluate() returns the correct severity level."""

    def test_no_drift_returns_none_or_low(self) -> None:
        """When reference and production are the same, severity should be NONE."""
        detector = _make_detector()
        ref = _standard_embeddings()
        report = detector.evaluate(ref, ref)
        assert report.overall_severity in (DriftSeverity.NONE, DriftSeverity.LOW)

    def test_moderate_drift(self) -> None:
        """A moderate distributional shift should trigger at least LOW severity."""
        detector = _make_detector()
        ref = _standard_embeddings()
        prod = ref + 0.8
        report = detector.evaluate(ref, prod)
        assert report.overall_severity != DriftSeverity.NONE

    def test_severe_drift_returns_critical(self) -> None:
        """A large distributional shift should trigger HIGH or CRITICAL severity."""
        detector = _make_detector()
        ref = _standard_embeddings()
        prod = _standard_embeddings() + 10.0
        report = detector.evaluate(ref, prod)
        assert report.overall_severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)


# ===================================================================
# MMD-only detection (no ensemble)
# ===================================================================


@pytest.mark.unit
class TestMMDOnly:
    """Verify detector uses only MMD, not an ensemble."""

    def test_single_metric_result(self) -> None:
        """evaluate() should return exactly one metric result (MMD)."""
        detector = _make_detector()
        ref = _standard_embeddings()
        report = detector.evaluate(ref, ref)
        assert len(report.metric_results) == 1
        assert report.metric_results[0].metric_name == "mmd"

    def test_per_metric_severity_has_only_mmd(self) -> None:
        """per_metric_severity should contain only the 'mmd' key."""
        detector = _make_detector()
        ref = _standard_embeddings()
        report = detector.evaluate(ref, ref)
        assert list(report.per_metric_severity.keys()) == ["mmd"]


# ===================================================================
# DriftReport model validation
# ===================================================================


@pytest.mark.unit
class TestDriftReport:
    """Validate the DriftReport pydantic model."""

    def test_report_from_evaluate(self) -> None:
        """evaluate() should return a well-formed DriftReport."""
        detector = _make_detector()
        ref = _standard_embeddings()
        report = detector.evaluate(ref, ref)
        assert isinstance(report, DriftReport)
        assert hasattr(report, "overall_severity")
        assert hasattr(report, "metric_results")
        assert hasattr(report, "timestamp")

    def test_report_has_recommended_actions(self) -> None:
        """DriftReport should include recommended actions."""
        detector = _make_detector()
        ref = _standard_embeddings()
        report = detector.evaluate(ref, ref)
        assert isinstance(report.recommended_actions, list)
        assert len(report.recommended_actions) > 0

    def test_report_has_sample_counts(self) -> None:
        """DriftReport should record the number of reference and production samples."""
        detector = _make_detector()
        ref = _standard_embeddings(n=100)
        prod = _standard_embeddings(n=50)
        report = detector.evaluate(ref, prod)
        assert report.n_reference == 100
        assert report.n_production == 50

    def test_report_window_metadata(self) -> None:
        """DriftReport should carry window start/end when provided."""
        detector = _make_detector()
        ref = _standard_embeddings()
        prod = _standard_embeddings()
        report = detector.evaluate(
            ref, prod,
            window_start="2025-01-01T00:00:00Z",
            window_end="2025-01-02T00:00:00Z",
        )
        assert report.window_start == "2025-01-01T00:00:00Z"
        assert report.window_end == "2025-01-02T00:00:00Z"


# ===================================================================
# Windowed evaluation
# ===================================================================


@pytest.mark.unit
class TestEvaluateWindowed:
    """Verify windowed evaluation across multiple time windows."""

    def test_returns_list_of_reports(self) -> None:
        """evaluate_windowed should return one report per window."""
        detector = _make_detector()
        ref = _standard_embeddings()
        windows = [
            ("2025-01-01", "2025-01-02", _standard_embeddings(n=50)),
            ("2025-01-02", "2025-01-03", _standard_embeddings(n=50)),
        ]
        reports = detector.evaluate_windowed(ref, windows)
        assert len(reports) == 2
        for r in reports:
            assert isinstance(r, DriftReport)

    def test_skips_small_windows(self) -> None:
        """Windows with fewer than 2 samples should be skipped."""
        detector = _make_detector()
        ref = _standard_embeddings()
        windows = [
            ("2025-01-01", "2025-01-02", _standard_embeddings(n=1)),  # too small
            ("2025-01-02", "2025-01-03", _standard_embeddings(n=50)),
        ]
        reports = detector.evaluate_windowed(ref, windows)
        assert len(reports) == 1
