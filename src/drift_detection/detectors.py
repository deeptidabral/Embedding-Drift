"""
MMD-based embedding drift detector.

Uses Maximum Mean Discrepancy as the sole drift metric and configurable
thresholds (loaded from YAML) to produce a drift severity assessment
with recommended remediation actions.

Design rationale
----------------
Dense embedding dimensions are highly entangled.  Ensemble approaches
that combine univariate tests (KS per dimension), cosine distance, or
PCA-based metrics suffer from multiple-testing problems, miss
multivariate rotations, or capture only partial distributional
information.  MMD is a kernel-based two-sample test that operates on
the full joint distribution and is the only metric that correctly
assesses high-dimensional distributional divergence.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from pydantic import BaseModel, Field

from src.drift_detection.metrics import (
    DriftResult,
    maximum_mean_discrepancy,
)

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "drift_thresholds.yaml"
)


# ---------------------------------------------------------------------------
# Enums and models
# ---------------------------------------------------------------------------


class DriftSeverity(str, Enum):
    """Qualitative severity level for a drift event."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class MetricThresholds(BaseModel):
    """Per-metric threshold configuration."""

    low: float = 0.0
    moderate: float = 0.0
    high: float = 0.0
    critical: float = 0.0


class DriftReport(BaseModel):
    """Complete report produced by the drift detector."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metric_results: list[DriftResult] = Field(default_factory=list)
    per_metric_severity: dict[str, DriftSeverity] = Field(default_factory=dict)
    overall_severity: DriftSeverity = DriftSeverity.NONE
    n_reference: int = 0
    n_production: int = 0
    recommended_actions: list[str] = Field(default_factory=list)
    window_start: str | None = None
    window_end: str | None = None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class EmbeddingDriftDetector:
    """MMD-based drift detector.

    Parameters
    ----------
    thresholds_path:
        Path to the YAML file containing MMD severity thresholds.
        If the file is absent, built-in defaults are used.
    """

    _BUILT_IN_THRESHOLDS: dict[str, MetricThresholds] = {
        "mmd": MetricThresholds(
            low=0.005, moderate=0.02, high=0.05, critical=0.10
        ),
    }

    def __init__(
        self,
        thresholds_path: str | Path = DEFAULT_THRESHOLDS_PATH,
        **kwargs: Any,
    ) -> None:
        self._thresholds = self._load_thresholds(Path(thresholds_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        window_start: str | None = None,
        window_end: str | None = None,
    ) -> DriftReport:
        """Run MMD drift detection and produce a ``DriftReport``.

        Parameters
        ----------
        reference:
            ``(N, D)`` array of reference embeddings.
        production:
            ``(M, D)`` array of production embeddings.
        window_start / window_end:
            Optional ISO-8601 strings denoting the production window
            (included in the report for traceability).
        """
        result = maximum_mean_discrepancy(reference, production, kernel="rbf")

        severity = self._classify_severity(result.metric_name, result.value)
        actions = self._recommend_actions(severity)

        report = DriftReport(
            metric_results=[result],
            per_metric_severity={result.metric_name: severity},
            overall_severity=severity,
            n_reference=reference.shape[0],
            n_production=production.shape[0],
            recommended_actions=actions,
            window_start=window_start,
            window_end=window_end,
        )
        logger.info("Drift evaluation complete -- severity=%s", severity.value)
        return report

    def evaluate_windowed(
        self,
        reference: np.ndarray,
        production_windows: Sequence[tuple[str, str, np.ndarray]],
    ) -> list[DriftReport]:
        """Evaluate drift across a sequence of time windows.

        Each element of *production_windows* is a tuple of
        ``(window_start, window_end, embeddings_array)``.
        """
        reports: list[DriftReport] = []
        for start, end, prod in production_windows:
            if prod.shape[0] < 2:
                logger.warning(
                    "Skipping window %s--%s: fewer than 2 samples", start, end
                )
                continue
            reports.append(
                self.evaluate(reference, prod, window_start=start, window_end=end)
            )
        return reports

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # Mapping from YAML metric keys to the metric_name values produced
    # by the functions in ``metrics.py``.
    _YAML_KEY_TO_METRIC: dict[str, str] = {
        "maximum_mean_discrepancy": "mmd",
    }

    def _load_thresholds(
        self, path: Path
    ) -> dict[str, MetricThresholds]:
        """Load thresholds from YAML, falling back to built-in defaults.

        Supports two YAML layouts:

        1. **Simple format** -- a top-level ``thresholds`` mapping with
           ``low / moderate / high / critical`` keys per metric.
        2. **Extended format** -- the ``drift_detection.metrics`` structure
           used in the repository's ``configs/drift_thresholds.yaml``, which
           stores ``nominal_upper / warning_upper / critical_upper`` per
           metric.  The ``nominal_upper`` is mapped to *low*, ``warning_upper``
           to both *moderate* and *high* (split at the midpoint), and
           ``critical_upper`` to *critical*.
        """
        if path.exists():
            try:
                with open(path, "r") as fh:
                    raw: dict[str, Any] = yaml.safe_load(fh) or {}

                # Try simple format first.
                if "thresholds" in raw:
                    thresholds: dict[str, MetricThresholds] = {}
                    for metric, vals in raw["thresholds"].items():
                        metric_name = self._YAML_KEY_TO_METRIC.get(metric, metric)
                        if metric_name == "mmd":
                            thresholds[metric_name] = MetricThresholds(**vals)
                    if thresholds:
                        logger.info("Loaded drift thresholds (simple) from %s", path)
                        return thresholds

                # Try extended format.
                dd = raw.get("drift_detection", {})
                metrics_cfg: dict[str, Any] = dd.get("metrics", {})
                if metrics_cfg:
                    thresholds = {}
                    for yaml_key, vals in metrics_cfg.items():
                        metric_name = self._YAML_KEY_TO_METRIC.get(yaml_key, yaml_key)
                        if metric_name != "mmd":
                            continue
                        nom = float(vals.get("nominal_upper", 0.0))
                        warn = float(vals.get("warning_upper", 0.0))
                        crit = float(vals.get("critical_upper", 0.0))
                        midpoint = (warn + crit) / 2.0
                        thresholds[metric_name] = MetricThresholds(
                            low=nom,
                            moderate=warn,
                            high=midpoint,
                            critical=crit,
                        )
                    if thresholds:
                        logger.info("Loaded drift thresholds (extended) from %s", path)
                        return thresholds

            except Exception:
                logger.warning(
                    "Failed to parse %s -- falling back to built-in thresholds",
                    path,
                    exc_info=True,
                )
        return dict(self._BUILT_IN_THRESHOLDS)

    def _classify_severity(
        self, metric_name: str, value: float
    ) -> DriftSeverity:
        thresholds = self._thresholds.get(
            metric_name,
            MetricThresholds(low=0.005, moderate=0.02, high=0.05, critical=0.10),
        )
        if value >= thresholds.critical:
            return DriftSeverity.CRITICAL
        if value >= thresholds.high:
            return DriftSeverity.HIGH
        if value >= thresholds.moderate:
            return DriftSeverity.MODERATE
        if value >= thresholds.low:
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    @staticmethod
    def _recommend_actions(
        severity: DriftSeverity,
    ) -> list[str]:
        actions: list[str] = []
        if severity == DriftSeverity.NONE:
            return ["No action required."]
        if severity in (DriftSeverity.LOW, DriftSeverity.MODERATE):
            actions.append(
                "Increase monitoring frequency and review recent model predictions."
            )
            actions.append("Investigate root cause of distributional shift.")
        if severity == DriftSeverity.HIGH:
            actions.append(
                "Trigger model retraining pipeline with updated reference data."
            )
            actions.append("Activate enhanced manual review for flagged transactions.")
        if severity == DriftSeverity.CRITICAL:
            actions.append(
                "CRITICAL: Engage fallback rule-based fraud engine immediately."
            )
            actions.append(
                "Halt automated approvals and escalate to fraud operations team."
            )
            actions.append("Begin emergency model retraining with fresh data.")
        return actions
