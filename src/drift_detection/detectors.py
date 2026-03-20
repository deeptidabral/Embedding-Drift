"""
Ensemble-based embedding drift detector.

Combines multiple statistical metrics and uses configurable thresholds
(loaded from YAML) to produce an overall drift severity assessment with
recommended remediation actions.
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
    cosine_distance_drift,
    kolmogorov_smirnov_per_component,
    maximum_mean_discrepancy,
    population_stability_index,
    wasserstein_distance_drift,
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
    """Complete report produced by the ensemble detector."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metric_results: list[DriftResult] = Field(default_factory=list)
    per_metric_severity: dict[str, DriftSeverity] = Field(default_factory=dict)
    overall_severity: DriftSeverity = DriftSeverity.NONE
    min_metrics_agreeing: int = 2
    n_reference: int = 0
    n_production: int = 0
    recommended_actions: list[str] = Field(default_factory=list)
    window_start: str | None = None
    window_end: str | None = None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class EmbeddingDriftDetector:
    """Ensemble drift detector that aggregates multiple metric signals.

    Parameters
    ----------
    thresholds_path:
        Path to the YAML file containing per-metric severity thresholds.
        If the file is absent, built-in defaults are used.
    min_metrics_agreeing:
        Number of metrics that must agree on a severity level for the
        ensemble to declare that level.
    """

    _BUILT_IN_THRESHOLDS: dict[str, MetricThresholds] = {
        "cosine_distance": MetricThresholds(
            low=0.01, moderate=0.03, high=0.06, critical=0.10
        ),
        "mmd": MetricThresholds(
            low=0.005, moderate=0.02, high=0.05, critical=0.10
        ),
        "ks_per_component": MetricThresholds(
            low=0.10, moderate=0.20, high=0.35, critical=0.50
        ),
        "wasserstein": MetricThresholds(
            low=0.05, moderate=0.15, high=0.30, critical=0.50
        ),
        "psi": MetricThresholds(
            low=0.05, moderate=0.10, high=0.25, critical=0.50
        ),
    }

    def __init__(
        self,
        thresholds_path: str | Path = DEFAULT_THRESHOLDS_PATH,
        min_metrics_agreeing: int = 2,
    ) -> None:
        self._min_agreeing = min_metrics_agreeing
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
        """Run all drift metrics and produce an ensemble ``DriftReport``.

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
        results: list[DriftResult] = [
            cosine_distance_drift(reference, production),
            maximum_mean_discrepancy(reference, production, kernel="rbf"),
            kolmogorov_smirnov_per_component(reference, production),
            wasserstein_distance_drift(reference, production),
            population_stability_index(reference, production),
        ]

        per_metric_severity: dict[str, DriftSeverity] = {}
        for result in results:
            per_metric_severity[result.metric_name] = self._classify_severity(
                result.metric_name, result.value
            )

        overall = self._ensemble_severity(per_metric_severity)
        actions = self._recommend_actions(overall, per_metric_severity)

        report = DriftReport(
            metric_results=results,
            per_metric_severity=per_metric_severity,
            overall_severity=overall,
            min_metrics_agreeing=self._min_agreeing,
            n_reference=reference.shape[0],
            n_production=production.shape[0],
            recommended_actions=actions,
            window_start=window_start,
            window_end=window_end,
        )
        logger.info("Drift evaluation complete -- severity=%s", overall.value)
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
        "cosine_distance": "cosine_distance",
        "maximum_mean_discrepancy": "mmd",
        "kolmogorov_smirnov": "ks_per_component",
        "wasserstein_distance": "wasserstein",
        "population_stability_index": "psi",
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
                        thresholds[metric] = MetricThresholds(**vals)
                    logger.info("Loaded drift thresholds (simple) from %s", path)
                    return thresholds

                # Try extended format.
                dd = raw.get("drift_detection", {})
                metrics_cfg: dict[str, Any] = dd.get("metrics", {})
                if metrics_cfg:
                    thresholds = {}
                    for yaml_key, vals in metrics_cfg.items():
                        metric_name = self._YAML_KEY_TO_METRIC.get(yaml_key, yaml_key)
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
            MetricThresholds(low=0.05, moderate=0.15, high=0.30, critical=0.50),
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

    def _ensemble_severity(
        self, per_metric: dict[str, DriftSeverity]
    ) -> DriftSeverity:
        """Determine overall severity using a voting strategy.

        The highest severity level reached by at least
        ``min_metrics_agreeing`` metrics is selected.
        """
        ordered = [
            DriftSeverity.CRITICAL,
            DriftSeverity.HIGH,
            DriftSeverity.MODERATE,
            DriftSeverity.LOW,
        ]
        severity_counts: dict[DriftSeverity, int] = {}
        for sev in per_metric.values():
            for level in ordered:
                if self._severity_gte(sev, level):
                    severity_counts[level] = severity_counts.get(level, 0) + 1

        for level in ordered:
            if severity_counts.get(level, 0) >= self._min_agreeing:
                return level
        return DriftSeverity.NONE

    @staticmethod
    def _severity_gte(a: DriftSeverity, b: DriftSeverity) -> bool:
        order = {
            DriftSeverity.NONE: 0,
            DriftSeverity.LOW: 1,
            DriftSeverity.MODERATE: 2,
            DriftSeverity.HIGH: 3,
            DriftSeverity.CRITICAL: 4,
        }
        return order[a] >= order[b]

    @staticmethod
    def _recommend_actions(
        overall: DriftSeverity,
        per_metric: dict[str, DriftSeverity],
    ) -> list[str]:
        actions: list[str] = []
        if overall == DriftSeverity.NONE:
            return ["No action required."]
        if overall in (DriftSeverity.LOW, DriftSeverity.MODERATE):
            actions.append(
                "Increase monitoring frequency and review recent model predictions."
            )
            actions.append("Investigate root cause of distributional shift.")
        if overall == DriftSeverity.HIGH:
            actions.append(
                "Trigger model retraining pipeline with updated reference data."
            )
            actions.append("Activate enhanced manual review for flagged transactions.")
        if overall == DriftSeverity.CRITICAL:
            actions.append(
                "CRITICAL: Engage fallback rule-based fraud engine immediately."
            )
            actions.append(
                "Halt automated approvals and escalate to fraud operations team."
            )
            actions.append("Begin emergency model retraining with fresh data.")
        return actions
