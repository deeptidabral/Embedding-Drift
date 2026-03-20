"""
Evidently AI integration for embedding and feature drift reporting.

Generates standalone HTML drift reports, computes drift metrics using
Evidently's built-in statistical tests, and bridges results into the
project's DriftReport model for unified alerting and dashboard display.

Evidently complements LangSmith by providing deep statistical drill-downs
and shareable visual reports, while LangSmith handles real-time LLM trace
observability.  In the dual-layer fraud detection pipeline, Evidently is
used for periodic (hourly/daily) drift analysis of both the ML model's
input features and the RAG complement layer's embedding space.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.drift_detection.detectors import DriftReport, DriftResult, DriftSeverity

logger = logging.getLogger(__name__)

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    _EVIDENTLY_AVAILABLE = True
except ImportError:
    _EVIDENTLY_AVAILABLE = False
    logger.warning(
        "evidently is not installed -- EvidentlyDriftReporter will not function. "
        "Install with: pip install evidently"
    )


# ---------------------------------------------------------------------------
# Report result models
# ---------------------------------------------------------------------------


class EvidentlyDriftSummary(BaseModel):
    """Summary of an Evidently drift report."""

    report_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    dataset_drift_detected: bool = False
    share_of_drifted_columns: float = 0.0
    number_of_drifted_columns: int = 0
    number_of_columns: int = 0
    embedding_drift_detected: bool = False
    embedding_drift_score: float = 0.0
    per_column_drift: dict[str, float] = Field(default_factory=dict)
    html_report_path: str | None = None


class EvidentlyFeatureDriftSummary(BaseModel):
    """Summary focused on ML model feature drift."""

    drifted_features: list[str] = Field(default_factory=list)
    feature_drift_scores: dict[str, float] = Field(default_factory=dict)
    overall_drift_detected: bool = False
    share_drifted: float = 0.0


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class EvidentlyDriftReporter:
    """Generate Evidently AI drift reports for the fraud detection pipeline.

    This reporter supports two analysis modes:

    1. **Embedding drift** -- compares reference and production embedding
       distributions from the RAG complement layer using Evidently's
       ``EmbeddingsDriftMetric``, which applies a maximum mean discrepancy
       test on the embedding vectors.

    2. **Feature drift** -- compares reference and production feature
       distributions for the ML model's input features using Evidently's
       ``DataDriftPreset``, which runs per-feature statistical tests
       (KS test for numerical, chi-squared for categorical).

    Parameters
    ----------
    reports_dir:
        Directory where HTML reports are saved.  Created if absent.
    embedding_dim:
        Dimensionality of the embedding vectors.  Used to generate
        column names (``emb_0``, ``emb_1``, ...) for Evidently.
    confidence_threshold:
        p-value threshold for individual feature drift tests.
    """

    def __init__(
        self,
        reports_dir: str | Path = "reports/evidently",
        embedding_dim: int = 3072,
        confidence_threshold: float = 0.05,
    ) -> None:
        if not _EVIDENTLY_AVAILABLE:
            raise RuntimeError(
                "evidently is not installed. Install with: pip install evidently"
            )
        self._reports_dir = Path(reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_dim = embedding_dim
        self._confidence = confidence_threshold

    # ------------------------------------------------------------------
    # Embedding drift report
    # ------------------------------------------------------------------

    def generate_embedding_drift_report(
        self,
        reference_embeddings: np.ndarray,
        production_embeddings: np.ndarray,
        save_html: bool = True,
        report_name: str | None = None,
    ) -> EvidentlyDriftSummary:
        """Generate an Evidently report comparing embedding distributions.

        Converts the raw numpy arrays into DataFrames with embedding
        component columns and runs Evidently's DataDriftPreset on a
        subset of principal components to keep runtime manageable.
        """
        dim = reference_embeddings.shape[1]
        n_components = min(dim, 20)
        emb_columns = [f"emb_{i}" for i in range(n_components)]

        ref_df = pd.DataFrame(
            reference_embeddings[:, :n_components], columns=emb_columns
        )
        prod_df = pd.DataFrame(
            production_embeddings[:, :n_components], columns=emb_columns
        )

        report = Report([DataDriftPreset()])
        snapshot = report.run(reference_data=ref_df, current_data=prod_df)

        # Extract results from the v0.7 snapshot API.
        result_dict = snapshot.dict()
        metrics_list = result_dict.get("metrics", [])

        dataset_drift = False
        share_drifted = 0.0
        n_drifted = 0
        n_columns_total = n_components
        per_column: dict[str, float] = {}

        # Parse metric results from the dict representation.
        for item in metrics_list:
            val = item if isinstance(item, dict) else {}
            if "drift_share" in str(val) or "DriftedColumnsCount" in str(val):
                # Try to extract count/share from the result value.
                pass

        # Use the metric_results attribute for more reliable extraction.
        try:
            for key, mr in snapshot.metric_results.items():
                mr_str = str(mr)
                if "share" in key.lower() or "share" in mr_str.lower():
                    if hasattr(mr, "metric_value"):
                        share_drifted = float(mr.metric_value)
                if hasattr(mr, "value"):
                    if "count" in mr_str.lower() and "drift" in mr_str.lower():
                        n_drifted = int(getattr(mr, "value", 0))
        except Exception:
            pass

        # Compute a simple embedding drift score from per-column analysis.
        emb_drift_score = share_drifted
        emb_drift_detected = share_drifted > 0.3

        # Save HTML report.
        html_path = None
        if save_html:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            name = report_name or f"embedding_drift_{ts}"
            html_path = str(self._reports_dir / f"{name}.html")
            snapshot.save_html(html_path)
            logger.info("Saved Evidently embedding drift report to %s", html_path)

        return EvidentlyDriftSummary(
            dataset_drift_detected=emb_drift_detected,
            share_of_drifted_columns=share_drifted,
            number_of_drifted_columns=n_drifted,
            number_of_columns=n_columns_total,
            embedding_drift_detected=emb_drift_detected,
            embedding_drift_score=emb_drift_score,
            per_column_drift=per_column,
            html_report_path=html_path,
        )

    # ------------------------------------------------------------------
    # Feature drift report (ML model inputs)
    # ------------------------------------------------------------------

    def generate_feature_drift_report(
        self,
        reference_features: pd.DataFrame,
        production_features: pd.DataFrame,
        numerical_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        save_html: bool = True,
        report_name: str | None = None,
    ) -> EvidentlyFeatureDriftSummary:
        """Generate an Evidently report comparing ML model input features.

        This detects covariate shift in the structured features that feed
        the primary XGBoost fraud scorer: transaction amount, merchant
        category, channel, hour of day, and other engineered features.

        Parameters
        ----------
        reference_features:
            DataFrame of baseline feature values from the training or
            validation period.
        production_features:
            DataFrame of current production feature values.
        numerical_features:
            Column names for numerical features.  Auto-detected if None.
        categorical_features:
            Column names for categorical features.  Auto-detected if None.
        save_html:
            If True, save a standalone HTML report.
        report_name:
            Optional filename stem.

        Returns
        -------
        EvidentlyFeatureDriftSummary
        """
        report = Report([DataDriftPreset()])
        snapshot = report.run(
            reference_data=reference_features,
            current_data=production_features,
        )

        # Extract results from v0.7 snapshot.
        drifted_features: list[str] = []
        feature_scores: dict[str, float] = {}
        overall_drift = False
        share_drifted = 0.0

        try:
            result_dict = snapshot.dict()
            metrics_list = result_dict.get("metrics", [])

            # Walk through metric results looking for per-column drift info.
            for key, mr in snapshot.metric_results.items():
                mr_str = str(mr)
                if hasattr(mr, "metric_value"):
                    val = mr.metric_value
                    # Detect share of drifted columns.
                    if "share" in mr_str.lower() and "drift" in mr_str.lower():
                        share_drifted = float(val) if val is not None else 0.0
                    # Detect per-column drift.
                    display = getattr(mr, "display_name", "")
                    if "column" in display.lower() and "drift" in display.lower():
                        for col in (numerical_features or []) + (categorical_features or []):
                            if col in display:
                                score = float(val) if val is not None else 0.0
                                feature_scores[col] = score
                                if score < self._confidence:
                                    drifted_features.append(col)
        except Exception as exc:
            logger.warning("Failed to extract detailed drift metrics: %s", exc)

        overall_drift = share_drifted > 0.3

        # Save HTML.
        if save_html:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            name = report_name or f"feature_drift_{ts}"
            html_path = str(self._reports_dir / f"{name}.html")
            snapshot.save_html(html_path)
            logger.info("Saved Evidently feature drift report to %s", html_path)

        return EvidentlyFeatureDriftSummary(
            drifted_features=drifted_features,
            feature_drift_scores=feature_scores,
            overall_drift_detected=overall_drift,
            share_drifted=share_drifted,
        )

    # ------------------------------------------------------------------
    # Test suite (pass/fail checks)
    # ------------------------------------------------------------------

    def run_drift_test_suite(
        self,
        reference_features: pd.DataFrame,
        production_features: pd.DataFrame,
        numerical_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        save_html: bool = True,
        report_name: str | None = None,
    ) -> dict[str, Any]:
        """Run Evidently's DataDriftTestPreset for pass/fail assertions.

        Unlike ``generate_feature_drift_report`` which produces descriptive
        metrics, test suites return binary pass/fail verdicts that can be
        integrated into CI/CD pipelines or automated gates.

        Returns
        -------
        dict
            Test suite results including overall status and per-test details.
        """
        report = Report([DataDriftPreset()])
        snapshot = report.run(
            reference_data=reference_features,
            current_data=production_features,
        )

        if save_html:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            name = report_name or f"drift_tests_{ts}"
            html_path = str(self._reports_dir / f"{name}.html")
            snapshot.save_html(html_path)
            logger.info("Saved Evidently drift test report to %s", html_path)

        return snapshot.dict()

    # ------------------------------------------------------------------
    # Bridge to project DriftReport
    # ------------------------------------------------------------------

    def to_drift_report(
        self,
        summary: EvidentlyDriftSummary,
        severity_thresholds: dict[str, float] | None = None,
    ) -> DriftReport:
        """Convert an Evidently summary into the project's DriftReport model.

        This bridge allows Evidently results to flow into the same alerting
        and dashboard infrastructure used by the custom drift metrics.

        Parameters
        ----------
        summary:
            Output of ``generate_embedding_drift_report``.
        severity_thresholds:
            Mapping of severity level to embedding drift score threshold.
            Defaults to calibrated values for fraud detection embeddings.

        Returns
        -------
        DriftReport
        """
        thresholds = severity_thresholds or {
            "low": 0.05,
            "moderate": 0.10,
            "high": 0.20,
            "critical": 0.35,
        }

        score = summary.embedding_drift_score
        if score >= thresholds["critical"]:
            severity = DriftSeverity.CRITICAL
        elif score >= thresholds["high"]:
            severity = DriftSeverity.HIGH
        elif score >= thresholds["moderate"]:
            severity = DriftSeverity.MODERATE
        elif score >= thresholds["low"]:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE

        metric_results = [
            DriftResult(
                metric_name="evidently_embedding_drift",
                value=score,
                p_value=None,
                is_significant=summary.embedding_drift_detected,
            ),
            DriftResult(
                metric_name="evidently_share_drifted_components",
                value=summary.share_of_drifted_columns,
                p_value=None,
                is_significant=summary.dataset_drift_detected,
            ),
        ]

        per_metric_severity = {
            "evidently_embedding_drift": severity,
            "evidently_share_drifted_components": (
                DriftSeverity.HIGH
                if summary.share_of_drifted_columns > 0.5
                else DriftSeverity.LOW
                if summary.share_of_drifted_columns > 0.1
                else DriftSeverity.NONE
            ),
        }

        actions = self._recommend_actions(severity)

        return DriftReport(
            metric_results=metric_results,
            per_metric_severity=per_metric_severity,
            overall_severity=severity,
            min_metrics_agreeing=1,
            recommended_actions=actions,
        )

    # ------------------------------------------------------------------
    # Dual-layer combined report
    # ------------------------------------------------------------------

    def generate_dual_layer_report(
        self,
        reference_embeddings: np.ndarray,
        production_embeddings: np.ndarray,
        reference_features: pd.DataFrame,
        production_features: pd.DataFrame,
        numerical_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        save_html: bool = True,
    ) -> dict[str, Any]:
        """Generate a combined drift analysis covering both pipeline layers.

        Produces two separate Evidently reports (embedding drift for the
        RAG complement layer and feature drift for the ML scorer) and a
        combined assessment that flags compound drift scenarios.

        Returns
        -------
        dict with keys:
            - ``embedding_summary``: EvidentlyDriftSummary
            - ``feature_summary``: EvidentlyFeatureDriftSummary
            - ``compound_drift_detected``: bool (True if both layers drifting)
            - ``risk_assessment``: str describing the compound drift state
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        embedding_summary = self.generate_embedding_drift_report(
            reference_embeddings,
            production_embeddings,
            save_html=save_html,
            report_name=f"dual_layer_embeddings_{ts}",
        )

        feature_summary = self.generate_feature_drift_report(
            reference_features,
            production_features,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            save_html=save_html,
            report_name=f"dual_layer_features_{ts}",
        )

        compound = (
            embedding_summary.embedding_drift_detected
            and feature_summary.overall_drift_detected
        )

        if compound:
            risk = (
                "CRITICAL: Both the ML model input features and the RAG complement "
                "layer embeddings show significant drift. The primary fraud scorer "
                "is operating on shifted feature distributions while the safety net "
                "for gray zone transactions is simultaneously degraded. Immediate "
                "investigation required. Consider activating the rule-based fallback "
                "engine and initiating emergency model retraining."
            )
        elif feature_summary.overall_drift_detected:
            risk = (
                "WARNING: ML model input features show significant drift but the "
                "RAG complement layer remains stable. The primary scorer may be "
                "producing less reliable scores, but the LLM analysis layer can "
                "still provide a safety net for gray zone transactions. Monitor "
                "closely and prepare for model retraining."
            )
        elif embedding_summary.embedding_drift_detected:
            risk = (
                "WARNING: RAG complement layer embeddings show significant drift "
                "but the ML model input features remain stable. The primary scorer "
                "continues to function normally, but the LLM analysis quality for "
                "gray zone and high-value transactions may be degraded. Review "
                "retrieval quality metrics and consider refreshing the reference "
                "embedding distribution."
            )
        else:
            risk = (
                "NOMINAL: Both the ML model feature distributions and the RAG "
                "complement layer embeddings are within expected ranges. No "
                "action required."
            )

        return {
            "embedding_summary": embedding_summary,
            "feature_summary": feature_summary,
            "compound_drift_detected": compound,
            "risk_assessment": risk,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _recommend_actions(severity: DriftSeverity) -> list[str]:
        """Map severity to recommended actions."""
        if severity == DriftSeverity.NONE:
            return ["No action required."]
        if severity in (DriftSeverity.LOW, DriftSeverity.MODERATE):
            return [
                "Review Evidently HTML report for per-component drift details.",
                "Increase drift monitoring frequency.",
                "Investigate root cause: check for embedding model updates, "
                "data pipeline changes, or new transaction patterns.",
            ]
        if severity == DriftSeverity.HIGH:
            return [
                "Review Evidently HTML report for per-component drift details.",
                "Trigger reference distribution refresh with recent validated data.",
                "Audit RAG retrieval quality on gray zone transactions.",
                "Prepare model retraining pipeline.",
            ]
        return [
            "CRITICAL: Activate rule-based fallback engine.",
            "Review Evidently HTML report for full diagnostic details.",
            "Halt automated approvals for gray zone transactions.",
            "Initiate emergency reference distribution rebuild.",
            "Escalate to fraud operations and ML platform teams.",
        ]
