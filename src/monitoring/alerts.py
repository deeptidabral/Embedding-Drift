"""
Alert routing for drift events.

Dispatches notifications to Slack and PagerDuty based on the severity
of the detected drift, with built-in deduplication to prevent alert
fatigue during sustained drift episodes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, Field

from src.drift_detection.detectors import DriftReport, DriftSeverity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class AlertConfig(BaseModel):
    """Alert routing configuration."""

    slack_webhook_url: str | None = None
    pagerduty_routing_key: str | None = None
    dedup_window_seconds: int = 3600
    min_severity_for_slack: DriftSeverity = DriftSeverity.LOW
    min_severity_for_pagerduty: DriftSeverity = DriftSeverity.HIGH


# ---------------------------------------------------------------------------
# Alert manager
# ---------------------------------------------------------------------------


class AlertManager:
    """Route drift alerts to Slack and PagerDuty with deduplication.

    Parameters
    ----------
    config:
        Alert routing configuration including webhook URLs and severity
        thresholds.
    """

    def __init__(self, config: AlertConfig) -> None:
        self._config = config
        self._http = httpx.Client(timeout=15.0)
        # Dedup state: maps dedup_key -> last_sent_epoch
        self._sent: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(self, drift_report: DriftReport) -> dict[str, bool]:
        """Dispatch alerts based on severity and deduplication rules.

        Returns a dict indicating which channels were notified::

            {"slack": True, "pagerduty": False}
        """
        results: dict[str, bool] = {"slack": False, "pagerduty": False}

        if self._should_deduplicate(drift_report):
            logger.info(
                "Alert deduplicated -- severity=%s already notified within window",
                drift_report.overall_severity.value,
            )
            return results

        severity_order = {
            DriftSeverity.NONE: 0,
            DriftSeverity.LOW: 1,
            DriftSeverity.MODERATE: 2,
            DriftSeverity.HIGH: 3,
            DriftSeverity.CRITICAL: 4,
        }

        report_level = severity_order.get(drift_report.overall_severity, 0)

        # Slack
        slack_min = severity_order.get(self._config.min_severity_for_slack, 1)
        if (
            self._config.slack_webhook_url
            and report_level >= slack_min
        ):
            results["slack"] = self._send_slack(drift_report)

        # PagerDuty
        pd_min = severity_order.get(self._config.min_severity_for_pagerduty, 3)
        if (
            self._config.pagerduty_routing_key
            and report_level >= pd_min
        ):
            results["pagerduty"] = self._send_pagerduty(drift_report)

        # Record dedup key
        self._record_dedup(drift_report)
        return results

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _should_deduplicate(self, drift_report: DriftReport) -> bool:
        """Return *True* if an equivalent alert was sent within the window."""
        key = self._dedup_key(drift_report)
        last_sent = self._sent.get(key)
        if last_sent is None:
            return False
        elapsed = time.monotonic() - last_sent
        return elapsed < self._config.dedup_window_seconds

    def _record_dedup(self, drift_report: DriftReport) -> None:
        self._sent[self._dedup_key(drift_report)] = time.monotonic()

    @staticmethod
    def _dedup_key(drift_report: DriftReport) -> str:
        """Produce a stable hash covering severity and the set of significant metrics."""
        significant = sorted(
            r.metric_name
            for r in drift_report.metric_results
            if r.is_significant
        )
        raw = f"{drift_report.overall_severity.value}|{'|'.join(significant)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Slack
    # ------------------------------------------------------------------

    def _send_slack(self, drift_report: DriftReport) -> bool:
        payload = self._format_slack_message(drift_report)
        try:
            resp = self._http.post(
                self._config.slack_webhook_url,  # type: ignore[arg-type]
                json=payload,
            )
            resp.raise_for_status()
            logger.info("Slack alert sent -- severity=%s", drift_report.overall_severity.value)
            return True
        except httpx.HTTPError:
            logger.exception("Failed to send Slack alert")
            return False

    @staticmethod
    def _format_slack_message(drift_report: DriftReport) -> dict[str, Any]:
        """Build a Slack Block Kit message from a drift report."""
        severity = drift_report.overall_severity.value.upper()
        header = f"Embedding Drift Alert -- {severity}"

        metric_lines: list[str] = []
        for result in drift_report.metric_results:
            flag = "[DRIFT]" if result.is_significant else "[ok]"
            metric_lines.append(
                f"  {flag} {result.metric_name}: {result.value:.6f} "
                f"(p={result.p_value})"
            )

        body = "\n".join(metric_lines)
        actions = "\n".join(f"  - {a}" for a in drift_report.recommended_actions)

        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": header},
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*Window:* {drift_report.window_start} -- "
                            f"{drift_report.window_end}\n"
                            f"*Samples:* {drift_report.n_reference} ref / "
                            f"{drift_report.n_production} prod\n\n"
                            f"```\n{body}\n```\n\n"
                            f"*Recommended actions:*\n{actions}"
                        ),
                    },
                },
            ],
        }

    # ------------------------------------------------------------------
    # PagerDuty
    # ------------------------------------------------------------------

    def _send_pagerduty(self, drift_report: DriftReport) -> bool:
        event = self._format_pagerduty_event(drift_report)
        try:
            resp = self._http.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=event,
            )
            resp.raise_for_status()
            logger.info(
                "PagerDuty event sent -- severity=%s",
                drift_report.overall_severity.value,
            )
            return True
        except httpx.HTTPError:
            logger.exception("Failed to send PagerDuty event")
            return False

    def _format_pagerduty_event(
        self, drift_report: DriftReport
    ) -> dict[str, Any]:
        """Build a PagerDuty Events API v2 payload."""
        severity_map = {
            DriftSeverity.HIGH: "warning",
            DriftSeverity.CRITICAL: "critical",
        }
        pd_severity = severity_map.get(
            drift_report.overall_severity, "warning"
        )

        return {
            "routing_key": self._config.pagerduty_routing_key,
            "event_action": "trigger",
            "dedup_key": self._dedup_key(drift_report),
            "payload": {
                "summary": (
                    f"Embedding drift detected -- "
                    f"severity={drift_report.overall_severity.value}"
                ),
                "source": "embedding-drift-monitor",
                "severity": pd_severity,
                "timestamp": drift_report.timestamp,
                "custom_details": {
                    "n_reference": drift_report.n_reference,
                    "n_production": drift_report.n_production,
                    "metrics": {
                        r.metric_name: r.value
                        for r in drift_report.metric_results
                    },
                    "recommended_actions": drift_report.recommended_actions,
                },
            },
        }
