"""
Dashboard data generation for drift visualisation.

Produces structured panel definitions (time-series, heatmaps,
correlation matrices) that can be consumed by any frontend renderer
such as Grafana, Streamlit, or a custom React dashboard.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, Field

from src.drift_detection.detectors import DriftReport, DriftSeverity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Panel models
# ---------------------------------------------------------------------------


class TimeSeriesPoint(BaseModel):
    """A single data point on a time-series chart."""

    timestamp: str
    value: float
    label: str = ""


class TimeSeriesPanel(BaseModel):
    """Panel definition for a metric time-series view."""

    title: str
    metric_name: str
    series: list[TimeSeriesPoint] = Field(default_factory=list)
    threshold_low: float | None = None
    threshold_high: float | None = None
    threshold_critical: float | None = None


class HeatmapCell(BaseModel):
    """One cell of the metric-vs-time heatmap."""

    row: str
    column: str
    value: float
    severity: str = "none"


class HeatmapPanel(BaseModel):
    """Panel definition for a metric heatmap."""

    title: str
    rows: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    cells: list[HeatmapCell] = Field(default_factory=list)


class CorrelationEntry(BaseModel):
    """One entry of the metric correlation matrix."""

    metric_a: str
    metric_b: str
    correlation: float


class CorrelationPanel(BaseModel):
    """Panel definition for the inter-metric correlation view."""

    title: str
    metrics: list[str] = Field(default_factory=list)
    entries: list[CorrelationEntry] = Field(default_factory=list)


class DashboardPayload(BaseModel):
    """Complete dashboard data payload."""

    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    time_series_panels: list[TimeSeriesPanel] = Field(default_factory=list)
    heatmap_panels: list[HeatmapPanel] = Field(default_factory=list)
    correlation_panels: list[CorrelationPanel] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dashboard builder
# ---------------------------------------------------------------------------


class DriftDashboard:
    """Aggregate drift reports into renderable dashboard panels.

    Parameters
    ----------
    reports:
        Ordered sequence of ``DriftReport`` objects (one per evaluation
        window).
    """

    def __init__(self, reports: Sequence[DriftReport] | None = None) -> None:
        self._reports: list[DriftReport] = list(reports) if reports else []

    def add_report(self, report: DriftReport) -> None:
        """Append a new drift report to the internal buffer."""
        self._reports.append(report)

    # ------------------------------------------------------------------
    # Panel builders
    # ------------------------------------------------------------------

    def build_time_series_panel(
        self,
        metric_name: str,
        title: str | None = None,
    ) -> TimeSeriesPanel:
        """Build a time-series panel for a single drift metric.

        Extracts the metric value from each stored report and pairs it
        with the report timestamp.
        """
        panel = TimeSeriesPanel(
            title=title or f"Drift -- {metric_name}",
            metric_name=metric_name,
        )

        for report in self._reports:
            for result in report.metric_results:
                if result.metric_name == metric_name:
                    panel.series.append(
                        TimeSeriesPoint(
                            timestamp=report.window_end or report.timestamp,
                            value=result.value,
                            label=report.per_metric_severity.get(
                                metric_name, DriftSeverity.NONE
                            ).value,
                        )
                    )
                    break

        return panel

    def build_heatmap_panel(
        self,
        title: str = "Drift Metric Heatmap",
    ) -> HeatmapPanel:
        """Build a heatmap of metric values across time windows.

        Rows are metric names; columns are window timestamps.
        """
        metric_names: list[str] = []
        if self._reports:
            metric_names = [
                r.metric_name for r in self._reports[0].metric_results
            ]

        columns: list[str] = []
        cells: list[HeatmapCell] = []

        for report in self._reports:
            col_label = report.window_end or report.timestamp
            columns.append(col_label)
            for result in report.metric_results:
                sev = report.per_metric_severity.get(
                    result.metric_name, DriftSeverity.NONE
                )
                cells.append(
                    HeatmapCell(
                        row=result.metric_name,
                        column=col_label,
                        value=result.value,
                        severity=sev.value,
                    )
                )

        return HeatmapPanel(
            title=title,
            rows=metric_names,
            columns=columns,
            cells=cells,
        )

    def build_correlation_panel(
        self,
        title: str = "Inter-Metric Correlation",
    ) -> CorrelationPanel:
        """Compute pairwise Pearson correlations among drift metrics
        across all stored reports.
        """
        if not self._reports:
            return CorrelationPanel(title=title)

        metric_names = [
            r.metric_name for r in self._reports[0].metric_results
        ]
        n_metrics = len(metric_names)

        # Build a (n_reports x n_metrics) matrix
        matrix = np.zeros((len(self._reports), n_metrics), dtype=np.float64)
        for i, report in enumerate(self._reports):
            for j, result in enumerate(report.metric_results):
                matrix[i, j] = result.value

        # Pearson correlation -- handle degenerate case
        if matrix.shape[0] < 2:
            corr = np.eye(n_metrics)
        else:
            corr = np.corrcoef(matrix, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)

        entries: list[CorrelationEntry] = []
        for i in range(n_metrics):
            for j in range(i, n_metrics):
                entries.append(
                    CorrelationEntry(
                        metric_a=metric_names[i],
                        metric_b=metric_names[j],
                        correlation=float(corr[i, j]),
                    )
                )

        return CorrelationPanel(
            title=title,
            metrics=metric_names,
            entries=entries,
        )

    # ------------------------------------------------------------------
    # Full dashboard
    # ------------------------------------------------------------------

    def build(self) -> DashboardPayload:
        """Build the complete dashboard payload containing all panels."""
        metric_names: list[str] = []
        if self._reports:
            metric_names = [
                r.metric_name for r in self._reports[0].metric_results
            ]

        ts_panels = [
            self.build_time_series_panel(name) for name in metric_names
        ]

        return DashboardPayload(
            time_series_panels=ts_panels,
            heatmap_panels=[self.build_heatmap_panel()],
            correlation_panels=[self.build_correlation_panel()],
        )
