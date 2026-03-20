"""
LangSmith integration for reporting drift metrics and managing
drift-related evaluation datasets.

Drift results are attached as custom feedback to LangSmith runs so
that they can be correlated with RAG retrieval quality and LLM
assessment accuracy.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from langsmith import Client as LangSmithClient
from pydantic import BaseModel, Field

from src.drift_detection.detectors import DriftReport, DriftSeverity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------


class LangSmithConfig(BaseModel):
    """Connection parameters for the LangSmith reporter."""

    api_key: str | None = None
    project_name: str = "fraud-detection-drift"
    dataset_name: str = "drift-evaluation"
    feedback_prefix: str = "drift"


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class LangSmithDriftReporter:
    """Report embedding drift metrics to LangSmith.

    Parameters
    ----------
    config:
        LangSmith connection and naming configuration.
    """

    def __init__(self, config: LangSmithConfig | None = None) -> None:
        self._config = config or LangSmithConfig()
        self._client = LangSmithClient(api_key=self._config.api_key)
        self._project = self._config.project_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report_drift(
        self,
        drift_report: DriftReport,
        run_id: str | UUID,
    ) -> None:
        """Attach drift metrics as custom feedback to a LangSmith run.

        Each metric in the ``DriftReport`` is recorded as an individual
        feedback entry so that it can be filtered and charted inside the
        LangSmith UI independently.

        Parameters
        ----------
        drift_report:
            The drift evaluation report produced by
            ``EmbeddingDriftDetector.evaluate``.
        run_id:
            The LangSmith run ID to annotate (typically the parent RAG
            chain run).
        """
        run_id_str = str(run_id)

        for result in drift_report.metric_results:
            feedback_key = f"{self._config.feedback_prefix}.{result.metric_name}"
            self._client.create_feedback(
                run_id=run_id_str,
                key=feedback_key,
                score=result.value,
                value=result.model_dump_json(),
                comment=(
                    f"Drift metric {result.metric_name}: "
                    f"value={result.value:.6f}, "
                    f"p_value={result.p_value}, "
                    f"significant={result.is_significant}"
                ),
            )

        # Overall severity as a separate feedback entry
        severity_score = {
            DriftSeverity.NONE: 0.0,
            DriftSeverity.LOW: 0.25,
            DriftSeverity.MODERATE: 0.50,
            DriftSeverity.HIGH: 0.75,
            DriftSeverity.CRITICAL: 1.0,
        }
        self._client.create_feedback(
            run_id=run_id_str,
            key=f"{self._config.feedback_prefix}.overall_severity",
            score=severity_score.get(drift_report.overall_severity, 0.0),
            value=drift_report.overall_severity.value,
            comment=(
                f"Overall drift severity: {drift_report.overall_severity.value} "
                f"({drift_report.n_reference} ref, {drift_report.n_production} prod)"
            ),
        )
        logger.info(
            "Reported drift feedback for run %s -- severity=%s",
            run_id_str,
            drift_report.overall_severity.value,
        )

    def create_drift_dataset(
        self,
        examples: list[dict[str, Any]],
        dataset_name: str | None = None,
    ) -> str:
        """Create or update a LangSmith dataset for drift evaluation.

        Each element of *examples* should contain:
          - ``inputs``: dict with ``reference_embeddings`` and
            ``production_embeddings`` (serialised as lists).
          - ``outputs``: dict with expected ``severity`` and metric values.

        Returns the dataset ID.
        """
        name = dataset_name or self._config.dataset_name
        dataset = self._client.create_dataset(
            dataset_name=name,
            description=(
                "Evaluation dataset for embedding drift detection in the "
                "fraud detection pipeline."
            ),
        )

        for example in examples:
            self._client.create_example(
                inputs=example.get("inputs", {}),
                outputs=example.get("outputs", {}),
                dataset_id=dataset.id,
            )

        logger.info(
            "Created drift dataset '%s' with %d examples (id=%s)",
            name,
            len(examples),
            dataset.id,
        )
        return str(dataset.id)

    def log_evaluation(
        self,
        drift_report: DriftReport,
        run_id: str | UUID,
        retrieval_scores: list[float] | None = None,
    ) -> None:
        """Log a combined evaluation linking drift state to retrieval quality.

        When *retrieval_scores* (e.g. NDCG or MRR values per query) are
        provided, they are logged alongside the drift severity so that
        degradation in retrieval quality can be correlated with
        distributional shift.

        Parameters
        ----------
        drift_report:
            Drift evaluation report.
        run_id:
            LangSmith run to annotate.
        retrieval_scores:
            Optional list of per-query retrieval quality scores observed
            during the same production window.
        """
        run_id_str = str(run_id)
        self.report_drift(drift_report, run_id_str)

        if retrieval_scores is not None and len(retrieval_scores) > 0:
            import numpy as np

            mean_score = float(np.mean(retrieval_scores))
            self._client.create_feedback(
                run_id=run_id_str,
                key=f"{self._config.feedback_prefix}.retrieval_quality",
                score=mean_score,
                comment=(
                    f"Mean retrieval score={mean_score:.4f} over "
                    f"{len(retrieval_scores)} queries during drift window "
                    f"{drift_report.window_start} -- {drift_report.window_end}"
                ),
            )
            logger.info(
                "Logged retrieval quality (mean=%.4f) alongside drift for run %s",
                mean_score,
                run_id_str,
            )

    def get_drift_history(
        self,
        project_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve recent drift feedback entries for trend analysis.

        Returns a list of dicts with keys ``run_id``, ``key``, ``score``,
        and ``created_at``.
        """
        proj = project_name or self._project
        runs = list(self._client.list_runs(project_name=proj, limit=limit))

        history: list[dict[str, Any]] = []
        for run in runs:
            feedbacks = list(
                self._client.list_feedback(run_ids=[str(run.id)])
            )
            for fb in feedbacks:
                if fb.key.startswith(self._config.feedback_prefix):
                    history.append(
                        {
                            "run_id": str(run.id),
                            "key": fb.key,
                            "score": fb.score,
                            "created_at": fb.created_at.isoformat()
                            if fb.created_at
                            else None,
                        }
                    )
        return history
