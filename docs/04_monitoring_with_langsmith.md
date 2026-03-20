# Monitoring Embedding Drift with LangSmith

## Overview

- LangSmith serves as the centralized observability layer for the async RAG+LLM investigation pipeline in the dual-layer fraud detection system. It monitors the investigation path, not the real-time authorization path.
- The primary ML model scores every transaction and drives real-time authorization decisions (sub-10ms, sub-50ms end-to-end). Flagged transactions (gray zone, high-value) are queued for async investigation by the RAG+LLM pipeline, which generates investigation reports, analyst recommendations, and audit documentation to support human analysts in manual review queues.
- If embeddings drift from their reference distribution, retrieval quality silently degrades and the LLM generates investigation reports based on irrelevant context. This degrades the quality of manual review for the transactions the ML model flagged as uncertain.
- LangSmith makes this investigation quality degradation visible before it leads to missed fraud in manual review or weak chargeback defense.

---

## LangSmith Architecture in the Fraud Detection Pipeline

### Where LangSmith Sits

```
          REAL-TIME AUTHORIZATION PATH (sub-50ms SLA)
          ============================================

+------------------+     +-------------------+     +-------------------+
| Transaction      |     | Primary ML Model  |     | Decision Router   |
| Ingestion        | --> | (XGBoost, <10ms)  | --> | (authorize + flag)|
+------------------+     +-------------------+     +-------------------+
                                                          |
                         +--------------------------------+----------------+
                         |                                                 |
                         v                                                 v
                  +--------------+                              +-------------------+
                  | Auto-Approve |                              | Authorize + Flag  |
                  | or Decline   |                              | for Manual Review |
                  | (ML score    |                              | (gray zone,       |
                  |  alone)      |                              |  high-value)      |
                  +--------------+                              +-------------------+
                                                                           |
- - - - - - - - - - - - - - - - - ASYNC BOUNDARY - - - - - - - - - - - - -+- -
                                                                           |
          ASYNC INVESTIGATION PATH (no latency SLA)                        |
          ==========================================                       v
                                                               +-------------------+
                                                               | Async             |
                                                               | Investigation     |
                                                               | Queue             |
                                                               +-------------------+
                                                                        |
                    +-----------------------------------+-------------------+
                    |                  |                 |                   |
                    v                  v                 v                   v
             +------------+    +-------------+   +--------------+   +-------------+
             | Text       |    | Embedding   |   | Vector       |   | LLM         |
             | Rendering  |    | Generation  |   | Retrieval    |   | Investigation|
             +------------+    +-------------+   +--------------+   | Report      |
                                     |                              +-------------+
                                     |  (sampled embeddings)
                                     v
                          +---------------------+
                          |    LangSmith        |
                          |                     |
                          |  - Run Tracing      |
                          |  - Drift Evaluators |
                          |  - Dashboards       |
                          |  - Alerts           |
                          +---------------------+
```

- Every transaction first passes through the primary ML model, which produces a fraud probability score and drives the real-time authorization decision.
- Confident decisions (scores below 0.3 or above 0.7) are authorized on the ML score alone.
- Gray zone and high-value transactions are authorized (with conservative thresholds) and flagged for manual review. Flagged transactions are queued for async RAG+LLM investigation.
- LangSmith traces the async investigation pipeline, not the real-time authorization path. Each investigation run is logged as a traced run, with each stage (text rendering, embedding, retrieval, LLM assessment) as a child span. The ML model score and routing decision are recorded as metadata for cross-layer correlation.
- LangSmith monitors investigation quality, not authorization latency. The drift monitor operates as a parallel observer on the async path. It samples embeddings, computes drift statistics, and reports them as custom feedback scores.

### Integration Points

- **Run tracing**: every async investigation run is logged as a LangSmith run (raw transaction text, embedding vector, retrieved documents, LLM-generated investigation report). This traces the investigation pipeline, not the authorization path.
- **Custom evaluators**: asynchronous evaluators consume batches of recent investigation runs, compute drift statistics against the reference distribution, and post results as feedback scores.
- **Dashboard and alerting**: dashboard panels visualize investigation quality and drift over time. Threshold-based alerts trigger Slack and PagerDuty notifications when investigation quality degrades.

---

## Instrumenting the Pipeline

### Basic Tracing Setup

```python
import os
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "fraud-detection-embedding-drift"

ls_client = Client()


@traceable(name="fraud-investigation-async", run_type="chain")
def investigate_flagged_transaction(transaction: dict, ml_score: float) -> dict:
    """Async RAG+LLM investigation for transactions flagged during authorization.

    This runs asynchronously after the authorization decision has been made.
    It generates investigation reports to assist human analysts in manual review.
    """

    # Stage 1: Render the transaction into text
    transaction_text = render_transaction_template(transaction)

    # Stage 2: Generate embedding
    embedding = generate_embedding(transaction_text)

    # Attach embedding and ML model context to the investigation run for drift analysis
    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.metadata["embedding_vector"] = embedding.tolist()
        run_tree.metadata["merchant_category_code"] = transaction["merchant_category_code"]
        run_tree.metadata["transaction_amount_band"] = classify_amount_band(transaction["amount"])
        run_tree.metadata["ml_model_score"] = ml_score
        run_tree.metadata["routing_reason"] = classify_routing_reason(transaction, ml_score)

    # Stage 3: Retrieve similar fraud patterns from vector store
    retrieved_docs = retrieve_fraud_patterns(embedding, top_k=5)

    # Stage 4: LLM risk assessment
    risk_assessment = llm_assess_risk(transaction_text, retrieved_docs)

    return risk_assessment


def classify_routing_reason(transaction: dict, ml_score: float) -> str:
    """Determine why this transaction was flagged for async investigation."""
    if 0.3 <= ml_score <= 0.7:
        return "gray_zone"
    elif transaction["amount"] > 10000:
        return "high_value"
    elif transaction.get("explainability_required"):
        return "explainability"
    else:
        return "novel_pattern"


@traceable(name="generate-embedding", run_type="embedding")
def generate_embedding(text: str):
    """Generate embedding using sentence-transformers (all-MiniLM-L6-v2)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(text).tolist()


@traceable(name="retrieve-fraud-patterns", run_type="retriever")
def retrieve_fraud_patterns(embedding, top_k: int = 5):
    """Query the vector store for historically similar fraud patterns."""
    from src.embeddings.store import EmbeddingStore
    store = EmbeddingStore()
    results = store.query_similar(query_embedding=embedding, top_k=top_k)
    return results
```

- Each async investigation run produces a full trace with child spans for embedding generation and retrieval.
- The embedding vector, ML score, and flagging reason are stored in run metadata for batch drift evaluation and cross-layer correlation.
- These traces monitor the async investigation pipeline, not the real-time authorization path. Authorization latency is tracked separately in the ML model's own monitoring infrastructure.

---

## Custom Evaluators for Drift Metrics

- LangSmith custom evaluators run against completed runs asynchronously.
- The evaluator pulls recent embedding vectors from run metadata, computes statistical distance against the reference distribution, and posts results as feedback scores.

### Implementing the Drift Evaluator

```python
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp
from langsmith import Client
from langsmith.evaluation import EvaluationResult, RunEvaluator

ls_client = Client()


class EmbeddingDriftEvaluator(RunEvaluator):
    """Evaluates embedding drift by comparing production vectors to a reference set."""

    def __init__(self, reference_embeddings: np.ndarray, metric: str = "cosine"):
        self.reference_centroid = np.mean(reference_embeddings, axis=0)
        self.reference_embeddings = reference_embeddings
        self.metric = metric

    def evaluate_run(self, run, example=None) -> EvaluationResult:
        embedding = run.metadata.get("embedding_vector")
        if embedding is None:
            return EvaluationResult(key="embedding_drift", score=None, comment="No embedding found")

        embedding = np.array(embedding)

        if self.metric == "cosine":
            score = cosine(embedding, self.reference_centroid)
        elif self.metric == "ks":
            ref_proj = self.reference_embeddings[:, 0]
            score = ks_2samp(ref_proj, [embedding[0]]).statistic
        else:
            score = float(np.linalg.norm(embedding - self.reference_centroid))

        return EvaluationResult(
            key=f"embedding_drift_{self.metric}",
            score=float(score),
            comment=f"Drift score ({self.metric}): {score:.6f}",
        )


# Load reference embeddings from the versioned store
reference_embeddings = load_reference_embeddings(collection="ref_embeddings_v12")

# Register the evaluator against the project
drift_evaluator = EmbeddingDriftEvaluator(
    reference_embeddings=reference_embeddings,
    metric="cosine",
)
```

### Batch Evaluation Over Recent Runs

- In production, the drift evaluator runs on a schedule, not synchronously per transaction.
- A background job fetches recent runs, evaluates in batch, and posts aggregated drift statistics.

```python
from datetime import datetime, timedelta


def evaluate_recent_drift(lookback_minutes: int = 5):
    """Pull recent runs and compute aggregate drift metrics."""
    cutoff = datetime.utcnow() - timedelta(minutes=lookback_minutes)

    runs = list(ls_client.list_runs(
        project_name="fraud-detection-embedding-drift",
        run_type="chain",
        start_time=cutoff,
    ))

    embeddings = []
    for run in runs:
        vec = run.metadata.get("embedding_vector")
        if vec:
            embeddings.append(np.array(vec))

    if len(embeddings) < 50:
        return  # Not enough samples for reliable statistics

    production_batch = np.array(embeddings)
    reference = load_reference_embeddings(collection="ref_embeddings_v12")

    prod_centroid = np.mean(production_batch, axis=0)
    ref_centroid = np.mean(reference, axis=0)
    cosine_drift = float(cosine(prod_centroid, ref_centroid))

    latest_run = runs[0]
    ls_client.create_feedback(
        run_id=latest_run.id,
        key="aggregate_cosine_drift",
        score=cosine_drift,
        comment=f"Aggregate drift over {len(embeddings)} transactions: {cosine_drift:.6f}",
    )

    return cosine_drift
```

---

## Dashboard Design

Four sections. The first three cover async investigation pipeline embedding drift and investigation quality. The fourth provides cross-layer visibility between the ML authorization model and the async investigation pipeline. LangSmith monitors investigation quality, not authorization latency.

### Dashboard Panel Summary

```
+--------+---------------------------------------+-----------------------------------------+
| Panel  | Name                                  | What It Detects                         |
+--------+---------------------------------------+-----------------------------------------+
|  1     | Cosine Drift Over Time                | Centroid shift between production and    |
|        |                                       | reference distributions (5-min window). |
+--------+---------------------------------------+-----------------------------------------+
|  2     | MMD Two-Sample Test                   | Distributional shifts that centroid      |
|        |                                       | cosine misses (10-min window).          |
+--------+---------------------------------------+-----------------------------------------+
|  3     | Drift Severity Gauge                  | Ensemble severity (nominal/warning/     |
|        |                                       | critical) requiring 2+ metrics agree.   |
+--------+---------------------------------------+-----------------------------------------+
|  4     | Retrieval Quality vs Drift Scatter    | Correlation between drift and retrieval |
|        |                                       | relevance degradation.                  |
+--------+---------------------------------------+-----------------------------------------+
|  5     | Drift by Merchant Category Heatmap    | Drift localized to specific merchant    |
|        |                                       | verticals (e.g., targeted fraud).       |
+--------+---------------------------------------+-----------------------------------------+
|  6     | Precision Trend                       | Tandem movement of detection precision  |
|        |                                       | and drift score.                        |
+--------+---------------------------------------+-----------------------------------------+
|  7     | Transaction Volume by Severity Band   | Count of transactions processed under   |
|        |                                       | each drift severity level.              |
+--------+---------------------------------------+-----------------------------------------+
|  8     | Alert History                         | Audit trail of drift alerts with        |
|        |                                       | severity, values, and resolution.       |
+--------+---------------------------------------+-----------------------------------------+
|  9     | ML Score Distribution (Routed Txns)   | Whether the gray zone is widening       |
|        |                                       | (ML feature drift) or stable.           |
+--------+---------------------------------------+-----------------------------------------+
| 10     | Gray Zone Routing Rate Over Time      | Rising rate = ML less decisive.         |
|        |                                       | Stable rate + poor accuracy = embedding |
|        |                                       | drift.                                  |
+--------+---------------------------------------+-----------------------------------------+
| 11     | ML/LLM Agreement Rate                 | Declining agreement isolates which      |
|        |                                       | layer is degrading.                     |
+--------+---------------------------------------+-----------------------------------------+
| 12     | Embedding vs ML Feature Drift         | Correlated movement = common cause.     |
|        |                                       | Divergent = isolated layer issue.       |
+--------+---------------------------------------+-----------------------------------------+
```

### Section 1: Drift Health Overview

- **Panel 1. Cosine Drift Over Time**: time-series of aggregate cosine distance between production and reference centroids (5-min intervals, 24-hour rolling window). Threshold lines at 0.05 (nominal), 0.15 (warning), 0.30 (critical).
- **Panel 2. MMD Two-Sample Test**: time-series of MMD score (10-min intervals, 2,000-transaction sliding window). Thresholds at 0.02, 0.08, 0.20. Catches distributional shifts that centroid-based cosine misses.
- **Panel 3. Drift Severity Gauge**: single-value indicator of current ensemble severity (nominal/warning/critical) based on at least two metrics agreeing.

### Section 2: Correlation and Segmentation

- **Panel 4. Retrieval Quality vs Drift Scatter**: cosine drift (x) against retrieval relevance (y) per evaluation window. A strong negative correlation confirms drift is causing retrieval degradation.
- **Panel 5. Drift by Merchant Category Heatmap**: rows = merchant categories, columns = time intervals, color intensity = cosine drift. Reveals whether drift is localized to specific verticals.
- **Panel 6. Precision Trend**: rolling fraud detection precision overlaid with drift score. Tandem movement establishes the causal link.

### Section 3: Operational Metrics

- **Panel 7. Transaction Volume by Severity Band**: stacked area chart of transactions per minute, colored by active drift severity. Quantifies how many transactions may be affected by elevated drift.
- **Panel 8. Alert History**: table of recent drift alerts (severity, triggering values, timestamp, resolution status) for audit trail and compliance.

### Section 4: Cross-Layer Correlation

- **Panel 9. ML Score Distribution for Routed Transactions**: histogram of ML scores for complement-layer transactions. A widening gray zone suggests ML feature drift. A stable zone with poor complement outcomes points to embedding drift.
- **Panel 10. Gray Zone Routing Rate Over Time**: percentage of transactions routed to complement layer by reason. Rising rate means the ML model is becoming less decisive. Stable rate with declining accuracy means embedding drift.
- **Panel 11. ML/LLM Agreement Rate**: tracks how often the async LLM investigation assessment aligns with the ML score direction. Declining agreement with a stable ML distribution suggests embedding drift in the investigation pipeline. Declining agreement with a shifting ML distribution suggests ML feature drift in the authorization model.
- **Panel 12. Embedding Drift vs ML Feature Drift Correlation**: dual-axis time-series of embedding drift and ML feature drift (e.g., mean PSI across top features). Correlated movement indicates a common upstream cause. Divergent movement isolates which layer is degrading.

---

## Correlating Drift with Retrieval Quality and Fraud Detection Accuracy

- The value of drift monitoring is its ability to predict downstream degradation in the async investigation pipeline.
- For each evaluation window, record three values.
  - Aggregate drift score (cosine distance, MMD, or ensemble severity).
  - Mean retrieval relevance (average cosine similarity between query embedding and top-k retrieved documents in the investigation pipeline).
  - Investigation quality (from chargeback outcomes, manual review analyst feedback, and post-hoc accuracy assessment, typically 30-90 day lag).
- Two correlation modes exist.
  - **Near-real-time**: correlates drift with retrieval relevance in the investigation pipeline (available immediately after each async investigation run).
  - **Retrospective** (weekly/monthly): joins drift scores with actual fraud labels and manual review outcomes to validate that drift alerts corresponded to genuine investigation quality degradation.

```python
def compute_drift_retrieval_correlation(lookback_hours: int = 24):
    """Compute Pearson correlation between drift and retrieval quality."""
    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

    runs = list(ls_client.list_runs(
        project_name="fraud-detection-embedding-drift",
        run_type="chain",
        start_time=cutoff,
    ))

    drift_scores = []
    relevance_scores = []

    for run in runs:
        feedbacks = list(ls_client.list_feedback(run_ids=[run.id]))
        drift_fb = [f for f in feedbacks if f.key == "aggregate_cosine_drift"]
        rel_fb = [f for f in feedbacks if f.key == "retrieval_relevance"]

        if drift_fb and rel_fb:
            drift_scores.append(drift_fb[0].score)
            relevance_scores.append(rel_fb[0].score)

    if len(drift_scores) < 10:
        return None

    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(drift_scores, relevance_scores)

    return {
        "pearson_r": correlation,
        "p_value": p_value,
        "n_samples": len(drift_scores),
        "interpretation": "strong_negative" if correlation < -0.7 else "moderate" if correlation < -0.4 else "weak",
    }
```

---

## Alerting Configuration and Escalation

### Alert Escalation Path

```
+-------------------+      Cosine > 0.15      +-------------------+
|                   |      or MMD > 0.08       |                   |
|     NOMINAL       | -----------------------> |     WARNING       |
|                   |                          |                   |
| - Metrics logged  |                          | - Slack notify    |
| - Dashboards      |                          |   #fraud-model-   |
|   updated         |                          |   alerts          |
| - No human        |                          | - Monitoring      |
|   notification    |                          |   frequency 2x    |
|                   |                          | - Per-dimension   |
|                   |                          |   audit triggered |
|                   |                          | - Affected txns   |
|                   |                          |   flagged for     |
|                   |                          |   manual review   |
+-------------------+                          +-------------------+
                                                       |
                                                       | Cosine > 0.30
                                                       | or MMD > 0.20
                                                       v
                                               +-------------------+
                                               |                   |
                                               |     CRITICAL      |
                                               |                   |
                                               | - PagerDuty page  |
                                               | - Complement      |
                                               |   layer bypassed  |
                                               | - Emergency       |
                                               |   reindex/retrain |
                                               | - Embedding       |
                                               |   snapshot for    |
                                               |   forensics       |
                                               | - ML model        |
                                               |   continues if    |
                                               |   own metrics OK  |
                                               +-------------------+
```

### Alert Routing

- **Nominal**: metrics logged, dashboards updated, no human notification.
- **Warning**: Slack notification to #fraud-model-alerts with on-call ML engineer mentioned. Monitoring frequency automatically doubled. Embedding quality audit triggered (per-dimension drift stats). Recent transactions under warning-level drift flagged for manual review.
- **Critical**: PagerDuty incident page to on-call engineer. Complement layer bypassed so gray zone and high-value transactions go to manual review or ML-only with conservative thresholds. Emergency reindexing/retraining pipeline triggered. Production embedding snapshot captured for forensics. ML model continues normally if its own drift metrics are within bounds.

### Deduplication and Alert Fatigue Prevention

- At payment processor scale (billions of transactions per year), drift metrics fluctuate rapidly.
- Alerts deduplicated within a 30-minute window. Maximum 3 alerts per window across severity levels.
- Severity escalation resets the deduplication window. A critical alert fires immediately even if a warning dedup is active.

```python
import hashlib
from datetime import datetime, timedelta

_alert_cache = {}


def should_send_alert(severity: str, metric_name: str) -> bool:
    """Check deduplication cache before sending an alert."""
    cache_key = hashlib.sha256(f"{severity}:{metric_name}".encode()).hexdigest()
    now = datetime.utcnow()
    window = timedelta(minutes=30)

    if cache_key in _alert_cache:
        last_sent, count = _alert_cache[cache_key]
        if now - last_sent < window and count >= 3:
            return False

    if cache_key in _alert_cache:
        last_sent, count = _alert_cache[cache_key]
        if now - last_sent < window:
            _alert_cache[cache_key] = (last_sent, count + 1)
        else:
            _alert_cache[cache_key] = (now, 1)
    else:
        _alert_cache[cache_key] = (now, 1)

    return True
```

---

## Best Practices for Production Monitoring at Scale

### Sampling Strategy

- At payment processor scale, computing drift over every embedding is neither necessary nor feasible.
- Sample 1 in 100 transactions, stratified by merchant category and transaction amount band.
- Stratification ensures low-volume but high-risk categories (digital goods, wire transfers) are represented even when high-volume categories (groceries, fuel) dominate.

### Windowed Computation

- Always compute drift over sliding windows, not individual embeddings. A single outlier embedding is not meaningful. Drift is a distributional phenomenon.
- Use windows of 1,000, 5,000, and 20,000 transactions to capture drift at different temporal scales.

### Reference Distribution Management

- Refresh the reference distribution periodically (rolling 90-day window, minimum 50,000 samples, stratified by merchant category, amount band, and geography).
- Version every reference snapshot so drift can be recomputed against historical references.

### Latency Considerations

- Drift computation runs entirely asynchronously. Zero latency added to transaction processing.
- Embeddings are logged to run metadata during processing (negligible overhead). The drift evaluator pulls them in batch on a 5-minute schedule.
- Storage overhead: approximately 3 KB per transaction for a 384-dimensional embedding (64-bit floats).

### Compliance and Auditability

- Every drift alert, severity escalation, and fallback activation must be logged with full traceability. This is a regulatory requirement for payment processors.
- Run metadata must include drift severity at time of processing, reference distribution version, and specific metric values that triggered actions.
- LangSmith run traces provide this audit trail natively.

---

## Complementing LangSmith with Evidently AI

- LangSmith excels at observability for the async LLM investigation pipeline (tracing, custom feedback scores, live dashboards) but is not a native drift detection engine. It visualizes drift scores computed externally. It monitors investigation quality, not real-time authorization performance.
- Evidently AI fills this gap. It is free, open-source (Apache 2.0), and provides built-in statistical drift detection with publication-quality HTML reports.

### LangSmith vs Evidently AI Capabilities

```
+----------------------------+-----------------------------+-----------------------------+
| Capability                 | LangSmith                   | Evidently AI                |
+----------------------------+-----------------------------+-----------------------------+
| Monitoring mode            | Continuous, real-time        | Periodic (hourly/daily)     |
+----------------------------+-----------------------------+-----------------------------+
| Per-transaction tracing    | Yes (every LLM call)         | No                          |
+----------------------------+-----------------------------+-----------------------------+
| Drift score computation    | External (custom evaluators) | Built-in (MMD, KS, chi-sq) |
+----------------------------+-----------------------------+-----------------------------+
| Per-minute drift tracking  | Yes                          | No                          |
+----------------------------+-----------------------------+-----------------------------+
| Retrieval quality          | Yes (per-transaction)        | No                          |
| correlation                |                              |                             |
+----------------------------+-----------------------------+-----------------------------+
| Trace-level debugging      | Yes                          | No                          |
+----------------------------+-----------------------------+-----------------------------+
| HTML reports for           | No                           | Yes (publication-quality)   |
| stakeholders               |                              |                             |
+----------------------------+-----------------------------+-----------------------------+
| Per-feature drill-down     | No                           | Yes (KS, chi-sq per        |
| (ML model inputs)          |                              | feature)                    |
+----------------------------+-----------------------------+-----------------------------+
| Dual-layer compound        | No (single layer)            | Yes (embedding + feature    |
| assessment                 |                              | drift together)             |
+----------------------------+-----------------------------+-----------------------------+
| CI/CD drift test suites    | No                           | Yes (pass/fail gates)       |
+----------------------------+-----------------------------+-----------------------------+
| Regulatory audit artifacts | Run traces                   | Standalone HTML reports     |
+----------------------------+-----------------------------+-----------------------------+
| License                    | Commercial SaaS              | Apache 2.0 (open source)   |
+----------------------------+-----------------------------+-----------------------------+
```

### Division of Responsibilities

- **LangSmith**: continuous monitoring of the async investigation pipeline. Every investigation run traced. Rolling drift scores attached to runs. Alerts on threshold breach. Monitors investigation quality, not authorization latency.
- **Evidently**: periodic deep-dive analysis (hourly/daily). Comprehensive drift reports with per-feature statistical tests, embedding distribution comparisons, and visual diagnostics. Standalone HTML reports for stakeholders, incident tickets, and regulatory archives.

### Evidently Integration Points

The ``EvidentlyDriftReporter`` class (``src/monitoring/evidently_reporter.py``) supports three modes.

- **Embedding drift reports**: compares reference and production embedding distributions using ``EmbeddingsDriftMetric`` (model-based MMD test). Covers the RAG complement layer.
- **Feature drift reports**: compares reference and production feature distributions for ML model inputs using ``DataDriftPreset`` (KS tests for numerical, chi-squared for categorical). Covers the primary XGBoost layer.
- **Dual-layer combined reports**: runs both analyses. If both layers drift simultaneously, flags a critical compound drift state. This is the most dangerous scenario because the primary scorer is unreliable and the safety net is degraded.

### Generating Reports

```python
from src.monitoring.evidently_reporter import EvidentlyDriftReporter

reporter = EvidentlyDriftReporter(reports_dir="reports/evidently")

# Embedding drift for RAG complement layer
embedding_summary = reporter.generate_embedding_drift_report(
    reference_embeddings=ref_embeddings,
    production_embeddings=prod_embeddings,
    save_html=True,
)

# Feature drift for ML model
feature_summary = reporter.generate_feature_drift_report(
    reference_features=ref_features_df,
    production_features=prod_features_df,
    save_html=True,
)

# Dual-layer compound assessment
result = reporter.generate_dual_layer_report(
    reference_embeddings=ref_embeddings,
    production_embeddings=prod_embeddings,
    reference_features=ref_features_df,
    production_features=prod_features_df,
)
print(result["risk_assessment"])
```

### Bridging Evidently Results into the Alerting Pipeline

```python
drift_report = reporter.to_drift_report(embedding_summary)
# drift_report.overall_severity can now drive AlertManager
```

- Evidently-derived drift scores flow into the same alerting and dashboard infrastructure used by custom statistical metrics and LangSmith feedback scores.

### When to Use Each Tool

- **LangSmith**: operational monitoring of every async LLM investigation run, continuous per-minute drift tracking on the investigation pipeline, per-investigation retrieval quality correlation, trace-level debugging of investigation reports. Does not monitor authorization latency.
- **Evidently**: hourly/daily deep-dive reports, stakeholder-facing HTML with visual diagnostics, per-feature drill-down for ML model inputs, dual-layer compound assessment, CI/CD pass/fail test suites, regulatory audit artifacts.
- In production, both run simultaneously. LangSmith provides the live pulse on investigation quality. Evidently provides the periodic health examination across both layers.
