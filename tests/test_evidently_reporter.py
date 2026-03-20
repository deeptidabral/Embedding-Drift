"""
Tests for the Evidently AI drift reporter.

Validates embedding drift report generation, feature drift analysis,
dual-layer combined reporting, and the bridge to the project's
DriftReport model.  Uses synthetic data to avoid external dependencies.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.drift_detection.detectors import DriftSeverity
from src.monitoring.evidently_reporter import (
    EvidentlyDriftReporter,
    EvidentlyDriftSummary,
    EvidentlyFeatureDriftSummary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    """Temporary directory for HTML reports."""
    return tmp_path / "evidently_reports"


@pytest.fixture
def reporter(reports_dir: Path) -> EvidentlyDriftReporter:
    """Reporter instance pointing at a temp directory."""
    return EvidentlyDriftReporter(
        reports_dir=reports_dir,
        embedding_dim=32,
        confidence_threshold=0.05,
    )


@pytest.fixture
def reference_embeddings() -> np.ndarray:
    """Stable reference embedding distribution."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((500, 32)).astype(np.float32)


@pytest.fixture
def production_embeddings_no_drift(reference_embeddings: np.ndarray) -> np.ndarray:
    """Production embeddings drawn from the same distribution."""
    rng = np.random.default_rng(99)
    return rng.standard_normal((300, 32)).astype(np.float32)


@pytest.fixture
def production_embeddings_with_drift() -> np.ndarray:
    """Production embeddings with significant mean shift."""
    rng = np.random.default_rng(7)
    shifted = rng.standard_normal((300, 32)).astype(np.float32) + 3.0
    return shifted


@pytest.fixture
def reference_features() -> pd.DataFrame:
    """Baseline feature DataFrame for ML model inputs."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "amount": rng.exponential(200, size=n),
        "hour_of_day": rng.integers(0, 24, size=n),
        "txn_count_30d": rng.integers(1, 50, size=n),
        "avg_amount_30d": rng.exponential(150, size=n),
        "is_new_merchant": rng.integers(0, 2, size=n),
        "is_high_risk_country": rng.integers(0, 2, size=n),
    })


@pytest.fixture
def production_features_no_drift(reference_features: pd.DataFrame) -> pd.DataFrame:
    """Production features from the same distribution."""
    rng = np.random.default_rng(99)
    n = 300
    return pd.DataFrame({
        "amount": rng.exponential(200, size=n),
        "hour_of_day": rng.integers(0, 24, size=n),
        "txn_count_30d": rng.integers(1, 50, size=n),
        "avg_amount_30d": rng.exponential(150, size=n),
        "is_new_merchant": rng.integers(0, 2, size=n),
        "is_high_risk_country": rng.integers(0, 2, size=n),
    })


@pytest.fixture
def production_features_with_drift() -> pd.DataFrame:
    """Production features with shifted distributions."""
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame({
        "amount": rng.exponential(2000, size=n),
        "hour_of_day": rng.integers(0, 6, size=n),
        "txn_count_30d": rng.integers(1, 5, size=n),
        "avg_amount_30d": rng.exponential(1500, size=n),
        "is_new_merchant": np.ones(n, dtype=int),
        "is_high_risk_country": np.ones(n, dtype=int),
    })


# ---------------------------------------------------------------------------
# Summary model tests
# ---------------------------------------------------------------------------


class TestEvidentlyDriftSummary:
    """Tests for the EvidentlyDriftSummary pydantic model."""

    @pytest.mark.unit
    def test_default_construction(self) -> None:
        """Default summary should indicate no drift."""
        summary = EvidentlyDriftSummary()
        assert summary.dataset_drift_detected is False
        assert summary.embedding_drift_detected is False
        assert summary.embedding_drift_score == 0.0
        assert summary.html_report_path is None

    @pytest.mark.unit
    def test_populated_construction(self) -> None:
        """Summary with drift values should store them correctly."""
        summary = EvidentlyDriftSummary(
            dataset_drift_detected=True,
            share_of_drifted_columns=0.6,
            number_of_drifted_columns=12,
            number_of_columns=20,
            embedding_drift_detected=True,
            embedding_drift_score=0.25,
            per_column_drift={"emb_0": 0.03, "emb_1": 0.45},
            html_report_path="/tmp/report.html",
        )
        assert summary.dataset_drift_detected is True
        assert summary.share_of_drifted_columns == 0.6
        assert summary.embedding_drift_score == 0.25
        assert len(summary.per_column_drift) == 2


class TestEvidentlyFeatureDriftSummary:
    """Tests for the EvidentlyFeatureDriftSummary pydantic model."""

    @pytest.mark.unit
    def test_default_construction(self) -> None:
        summary = EvidentlyFeatureDriftSummary()
        assert summary.overall_drift_detected is False
        assert summary.drifted_features == []

    @pytest.mark.unit
    def test_with_drifted_features(self) -> None:
        summary = EvidentlyFeatureDriftSummary(
            drifted_features=["amount", "hour_of_day"],
            feature_drift_scores={"amount": 0.001, "hour_of_day": 0.02},
            overall_drift_detected=True,
            share_drifted=0.33,
        )
        assert len(summary.drifted_features) == 2
        assert summary.share_drifted == pytest.approx(0.33)


# ---------------------------------------------------------------------------
# Embedding drift report tests
# ---------------------------------------------------------------------------


class TestEmbeddingDriftReport:
    """Tests for embedding drift report generation."""

    @pytest.mark.unit
    def test_report_returns_summary(
        self,
        reporter: EvidentlyDriftReporter,
        reference_embeddings: np.ndarray,
        production_embeddings_no_drift: np.ndarray,
    ) -> None:
        """Report should return an EvidentlyDriftSummary."""
        summary = reporter.generate_embedding_drift_report(
            reference_embeddings,
            production_embeddings_no_drift,
            save_html=False,
        )
        assert isinstance(summary, EvidentlyDriftSummary)
        assert summary.report_timestamp is not None

    @pytest.mark.unit
    def test_no_drift_low_score(
        self,
        reporter: EvidentlyDriftReporter,
        reference_embeddings: np.ndarray,
        production_embeddings_no_drift: np.ndarray,
    ) -> None:
        """Same-distribution embeddings should produce low drift score."""
        summary = reporter.generate_embedding_drift_report(
            reference_embeddings,
            production_embeddings_no_drift,
            save_html=False,
        )
        assert summary.embedding_drift_score < 0.5

    @pytest.mark.unit
    def test_drift_detected_for_shifted(
        self,
        reporter: EvidentlyDriftReporter,
        reference_embeddings: np.ndarray,
        production_embeddings_with_drift: np.ndarray,
    ) -> None:
        """Significantly shifted embeddings should trigger drift detection."""
        summary = reporter.generate_embedding_drift_report(
            reference_embeddings,
            production_embeddings_with_drift,
            save_html=False,
        )
        assert summary.embedding_drift_detected is True

    @pytest.mark.unit
    def test_html_report_saved(
        self,
        reporter: EvidentlyDriftReporter,
        reference_embeddings: np.ndarray,
        production_embeddings_no_drift: np.ndarray,
    ) -> None:
        """When save_html is True, an HTML file should be written."""
        summary = reporter.generate_embedding_drift_report(
            reference_embeddings,
            production_embeddings_no_drift,
            save_html=True,
            report_name="test_embedding_report",
        )
        assert summary.html_report_path is not None
        assert Path(summary.html_report_path).exists()
        assert Path(summary.html_report_path).suffix == ".html"


# ---------------------------------------------------------------------------
# Feature drift report tests
# ---------------------------------------------------------------------------


class TestFeatureDriftReport:
    """Tests for ML model feature drift report generation."""

    @pytest.mark.unit
    def test_no_drift_stable_features(
        self,
        reporter: EvidentlyDriftReporter,
        reference_features: pd.DataFrame,
        production_features_no_drift: pd.DataFrame,
    ) -> None:
        """Same-distribution features should show minimal drift."""
        summary = reporter.generate_feature_drift_report(
            reference_features,
            production_features_no_drift,
            save_html=False,
        )
        assert isinstance(summary, EvidentlyFeatureDriftSummary)
        assert summary.share_drifted < 0.5

    @pytest.mark.unit
    def test_drift_detected_shifted_features(
        self,
        reporter: EvidentlyDriftReporter,
        reference_features: pd.DataFrame,
        production_features_with_drift: pd.DataFrame,
    ) -> None:
        """Shifted feature distributions should trigger drift detection."""
        summary = reporter.generate_feature_drift_report(
            reference_features,
            production_features_with_drift,
            save_html=False,
        )
        assert summary.overall_drift_detected is True
        assert len(summary.drifted_features) > 0

    @pytest.mark.unit
    def test_feature_scores_populated(
        self,
        reporter: EvidentlyDriftReporter,
        reference_features: pd.DataFrame,
        production_features_with_drift: pd.DataFrame,
    ) -> None:
        """Per-feature drift scores should be populated."""
        summary = reporter.generate_feature_drift_report(
            reference_features,
            production_features_with_drift,
            save_html=False,
        )
        assert len(summary.feature_drift_scores) > 0
        for score in summary.feature_drift_scores.values():
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Bridge to DriftReport tests
# ---------------------------------------------------------------------------


class TestToDriftReport:
    """Tests for converting Evidently summary to project DriftReport."""

    @pytest.mark.unit
    def test_no_drift_maps_to_none_severity(
        self, reporter: EvidentlyDriftReporter
    ) -> None:
        summary = EvidentlyDriftSummary(embedding_drift_score=0.01)
        report = reporter.to_drift_report(summary)
        assert report.overall_severity == DriftSeverity.NONE

    @pytest.mark.unit
    def test_high_score_maps_to_critical(
        self, reporter: EvidentlyDriftReporter
    ) -> None:
        summary = EvidentlyDriftSummary(
            embedding_drift_score=0.50,
            embedding_drift_detected=True,
        )
        report = reporter.to_drift_report(summary)
        assert report.overall_severity == DriftSeverity.CRITICAL
        assert any("CRITICAL" in a for a in report.recommended_actions)

    @pytest.mark.unit
    def test_moderate_score_maps_correctly(
        self, reporter: EvidentlyDriftReporter
    ) -> None:
        summary = EvidentlyDriftSummary(embedding_drift_score=0.12)
        report = reporter.to_drift_report(summary)
        assert report.overall_severity == DriftSeverity.MODERATE

    @pytest.mark.unit
    def test_report_contains_evidently_metrics(
        self, reporter: EvidentlyDriftReporter
    ) -> None:
        summary = EvidentlyDriftSummary(
            embedding_drift_score=0.08,
            share_of_drifted_columns=0.3,
        )
        report = reporter.to_drift_report(summary)
        metric_names = [m.metric_name for m in report.metric_results]
        assert "evidently_embedding_drift" in metric_names
        assert "evidently_share_drifted_components" in metric_names


# ---------------------------------------------------------------------------
# Dual-layer report tests
# ---------------------------------------------------------------------------


class TestDualLayerReport:
    """Tests for the combined dual-layer drift report."""

    @pytest.mark.unit
    def test_compound_drift_detected(
        self,
        reporter: EvidentlyDriftReporter,
        reference_embeddings: np.ndarray,
        production_embeddings_with_drift: np.ndarray,
        reference_features: pd.DataFrame,
        production_features_with_drift: pd.DataFrame,
    ) -> None:
        """When both layers drift, compound_drift_detected should be True."""
        result = reporter.generate_dual_layer_report(
            reference_embeddings,
            production_embeddings_with_drift,
            reference_features,
            production_features_with_drift,
            save_html=False,
        )
        assert result["compound_drift_detected"] is True
        assert "CRITICAL" in result["risk_assessment"]

    @pytest.mark.unit
    def test_no_compound_when_stable(
        self,
        reporter: EvidentlyDriftReporter,
        reference_embeddings: np.ndarray,
        production_embeddings_no_drift: np.ndarray,
        reference_features: pd.DataFrame,
        production_features_no_drift: pd.DataFrame,
    ) -> None:
        """When both layers are stable, compound drift should not fire."""
        result = reporter.generate_dual_layer_report(
            reference_embeddings,
            production_embeddings_no_drift,
            reference_features,
            production_features_no_drift,
            save_html=False,
        )
        assert result["compound_drift_detected"] is False

    @pytest.mark.unit
    def test_result_contains_both_summaries(
        self,
        reporter: EvidentlyDriftReporter,
        reference_embeddings: np.ndarray,
        production_embeddings_no_drift: np.ndarray,
        reference_features: pd.DataFrame,
        production_features_no_drift: pd.DataFrame,
    ) -> None:
        """Result dict should contain both layer summaries."""
        result = reporter.generate_dual_layer_report(
            reference_embeddings,
            production_embeddings_no_drift,
            reference_features,
            production_features_no_drift,
            save_html=False,
        )
        assert isinstance(result["embedding_summary"], EvidentlyDriftSummary)
        assert isinstance(
            result["feature_summary"], EvidentlyFeatureDriftSummary
        )
        assert "risk_assessment" in result
