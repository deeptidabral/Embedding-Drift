# Drift Metrics and Thresholds

## Overview

Five embedding-specific drift metrics monitor the RAG complement layer, each capturing a different aspect of how the production distribution diverges from its reference. All metrics are calibrated for the default 384-dimensional embedding space (all-MiniLM-L6-v2) within payment processor fraud detection pipelines. Thresholds can be recalibrated for higher-dimensional spaces if using OpenAI text-embedding-3-large (3,072 dimensions) as an optional backend.

### Metric Summary

```
+---------------------+-------------------------------+---------------------------+--------------------+
| Metric              | What It Measures              | Best For                  | Computational Cost |
+---------------------+-------------------------------+---------------------------+--------------------+
| Cosine Distance     | Angular separation between    | Detecting rotational      | Low                |
|                     | centroids                     | shifts from new merchant  | O(d)               |
|                     |                               | types or categories       |                    |
+---------------------+-------------------------------+---------------------------+--------------------+
| MMD                 | Full distributional shape     | Variance changes, new     | High               |
|                     | difference via kernel test    | clusters, tail shifts     | O(n^2) or O(nD)   |
+---------------------+-------------------------------+---------------------------+--------------------+
| KS Test             | Max CDF difference per        | Identifying which         | Medium             |
|                     | principal component           | dimensions are drifting   | O(n log n) per PC  |
+---------------------+-------------------------------+---------------------------+--------------------+
| Wasserstein         | Minimum cost to transform     | Gradual continuous drift  | Medium             |
|                     | one distribution into another | over weeks or months      | O(n log n)         |
+---------------------+-------------------------------+---------------------------+--------------------+
| PSI                 | Symmetric distributional      | Tail changes where fraud  | Low                |
|                     | shift over binned projections | clusters, regulatory      | O(bins)            |
|                     |                               | familiarity               |                    |
+---------------------+-------------------------------+---------------------------+--------------------+
```

### Threshold Summary

```
+---------------------+------------------+------------------+------------------+
| Metric              | Nominal          | Warning          | Critical         |
+---------------------+------------------+------------------+------------------+
| Cosine Distance     | 0.00 - 0.05      | 0.05 - 0.15     | > 0.15           |
| MMD                 | 0.00 - 0.02      | 0.02 - 0.08     | > 0.08           |
| KS Test             | 0.00 - 0.05      | 0.05 - 0.12     | > 0.12           |
| Wasserstein         | 0.00 - 0.03      | 0.03 - 0.10     | > 0.10           |
| PSI                 | 0.00 - 0.10      | 0.10 - 0.20     | > 0.20           |
+---------------------+------------------+------------------+------------------+
```

---

## Metric 1: Cosine Distance

### What It Measures

- Angular separation between the production centroid and the reference centroid, ignoring magnitude.
- Formula: cosine_distance(r, p) = 1 - (r . p) / (||r|| * ||p||).
- Range: 0 (identical direction) to 2 (opposite), though drift values rarely exceed 0.5 in practice.

```
  Cosine Distance: Angle Between Centroids

  Reference centroid (r)
        \
         \  angle = drift
          \
           +----------> Production centroid (p)

  Small angle (< 0.05) = nominal.
  Large angle (> 0.15) = critical rotation.
```

### Why It Matters

- Detects rotational shifts in embedding space. For example, when new merchant types (cryptocurrency-related) appear in the transaction stream, the production centroid rotates away from the reference.
- Blind spot: insensitive to uniform magnitude changes, which can indicate shifts in embedding model confidence. Do not use in isolation.

### Thresholds

- **Nominal (0.00-0.05)**: normal daily variation (weekday/weekend, seasonal spending) stays below 0.03. Up to 0.05 is expected operating range.
- **Warning (0.05-0.15)**: meaningful directional drift. Commonly seen during holiday seasons, major events, or new merchant category volume growth. Retrieval quality may degrade at the upper end.
- **Critical (above 0.15)**: substantial rotation from reference. RAG retriever is matching against irrelevant historical patterns. Immediate intervention required.

---

## Metric 2: Maximum Mean Discrepancy (MMD)

### What It Measures

- Kernel-based two-sample test comparing the full shape of two distributions, not just centroids.
- Uses RBF kernel with bandwidth set via the median heuristic.
- Formula: MMD^2(P, Q) = (1/m^2) sum k(x_i, x_j) - (2/mn) sum k(x_i, y_j) + (1/n^2) sum k(y_i, y_j).

```
  MMD: Distribution Shape Difference

  Reference               Production
  +--------+              +--------+
  | *  * * |              |  *     |
  | * ** * |              | * * *  |
  |  *  *  |              | * * ** |
  | * *  * |              |   * *  |
  +--------+              +--------+
  (tight cluster)         (spread out, shifted)

  MMD measures the full shape mismatch,
  catching variance changes, new clusters,
  and tail shifts invisible to centroid comparison.
```

### Why It Matters

- Catches distributional shifts that centroid comparison misses.
- Variance/spread changes (e.g., fraudsters diversifying transaction patterns).
- Emergence of new clusters (genuinely novel transaction types with no historical reference).
- Tail shifts (where many fraud patterns reside).

### Thresholds

- **Nominal (0.00-0.02)**: distributions statistically indistinguishable. Minor kernel fluctuations from sampling noise expected below 0.01.
- **Warning (0.02-0.08)**: specific regions of embedding space are shifting, likely tied to particular transaction types or merchant categories. May be benign market evolution or new fraud vectors.
- **Critical (above 0.08)**: production distribution is fundamentally different from reference. Vector store has no relevant historical patterns for these transactions.

---

## Metric 3: Kolmogorov-Smirnov (KS) Test

### What It Measures

- Maximum absolute difference between empirical CDFs, applied per principal component (top 10) after projecting both distributions.
- Formula: KS_statistic = max_x |F_ref(x) - F_prod(x)|.
- Aggregate score = maximum KS statistic across all components.
- Null hypothesis rejected at p < 0.01.

```
  KS Test: Max CDF Gap Per Principal Component

  CDF
  1.0 |          ____-------  <- Reference
      |       __/ .--/----    <- Production
      |     _/ .-/
      |    / .-    |<-- KS statistic = max gap
      |   /./
      |  //
  0.0 +--/--------------------> value along PC axis

  Each PC tested independently.
  The largest gap across all PCs becomes the aggregate score.
```

### Why It Matters

- Identifies which directions in embedding space are drifting, not just that drift has occurred or its magnitude.
- These axes often map to interpretable dimensions (e.g., first PC = transaction amount, third PC = geographic spread).
- Enables targeted investigation rather than blanket response.

### Thresholds

- **Nominal (0.00-0.05)**: PC distributions well-aligned. Fluctuations below 0.03 are normal sampling variability.
- **Warning (0.05-0.12)**: one or more PCs show significant shift. Specific semantic dimensions are changing. Investigate affected components.
- **Critical (above 0.12)**: multiple PCs show large differences. Embedding space restructuring across several dimensions simultaneously.

---

## Metric 4: Wasserstein Distance (Earth Mover's Distance)

### What It Measures

- Minimum cost of transforming one distribution into another (probability mass moved x distance traveled).
- Applied to first principal component projections.
- Formula: W_1(F_ref, F_prod) = integral |F_ref(x) - F_prod(x)| dx.

```
  Wasserstein: Cost to Move One Distribution Into Another

  Reference:   [===####====]
  Production:          [====####=======]
                       <------>
                       movement cost

  Unlike KS (max pointwise gap), Wasserstein
  integrates the total area between CDFs,
  making it sensitive to slow, continuous drift.
```

### Why It Matters

- Unlike KS (which measures only the max pointwise CDF difference), Wasserstein integrates across the entire distribution.
- Particularly sensitive to gradual, continuous drift that accumulates over weeks or months. For example, transaction amount inflation or geographic mix shifts as a payment network expands into new markets.
- These slow drifts are easy to miss with other metrics but can materially degrade retrieval.

### Thresholds

- **Nominal (0.00-0.03)**: negligible transformation cost. Daily operational variations stay below 0.02.
- **Warning (0.03-0.10)**: embedding space geometry is meaningfully different from reference. Check retrieval relevance for concurrent degradation.
- **Critical (above 0.10)**: production embeddings occupy a meaningfully different region. Vector store retrieval is operating on a distribution it was not calibrated for.

---

## Metric 5: Population Stability Index (PSI)

### What It Measures

- Symmetric measure of distributional shift from a baseline, computed over binned PC projections (20 bins).
- Formula: PSI = sum (prod_i - ref_i) * ln(prod_i / ref_i).
- Laplace smoothing (0.0001) applied to prevent division by zero.

```
  PSI: Binned Distribution Comparison

  Reference bins:   |##|##|####|######|####|##|##|  |
  Production bins:  |# |# |##  |####  |####|##|###|##|

  Each bin compared: (prod_i - ref_i) * ln(prod_i / ref_i).
  Tail bins (where fraud clusters) contribute disproportionately
  when their proportions change.
```

### Why It Matters

- The metric most familiar to risk management teams at payment processors due to its widespread use in credit scoring model monitoring. Stakeholder communication is easier.
- Particularly sensitive to tail changes, which is where fraudulent transactions tend to cluster in the embedding distribution.
- 20 bins balances resolution against noise sensitivity.

### Thresholds

- **Nominal (0.00-0.10)**: industry standard threshold for "no significant shift" in credit risk monitoring. Same applies to embedding distributions.
- **Warning (0.10-0.20)**: moderate shift. In traditional model monitoring, PSI 0.10-0.25 triggers review. Investigate whether the shift is benign market evolution or emerging fraud patterns.
- **Critical (above 0.20)**: significant change. Regulatory guidance in financial services generally treats PSI above 0.25 as requiring model revalidation.

---

## ML Model Feature Drift vs Embedding Drift

The dual-layer architecture requires monitoring two distinct drift metric sets that apply to different layers and measure different phenomena.

### ML Model Feature Drift (Primary Scoring Layer)

- **Per-Feature PSI**: computed per input feature (transaction amount, velocity counters, merchant risk scores, geographic features). PSI > 0.10 on any top-importance feature triggers investigation. PSI > 0.25 triggers revalidation.
- **Prediction Distribution Shift**: monitors ML output score distribution. Gray zone widening (0.3-0.7) indicates degrading confidence calibration, directly affecting complement layer volume.
- **Model Performance Metrics**: precision, recall, AUC-ROC, FPR tracked against labeled outcomes (30-90 day lag from chargebacks/manual review).
- **Feature Correlation Stability**: monitors correlation matrix of top features. Structural changes indicate the relationships the model learned may no longer hold.

### Comparison Table

```
+-----------------+----------------------------------+----------------------------------+
| Aspect          | ML Feature Drift                 | Embedding Drift                  |
+-----------------+----------------------------------+----------------------------------+
| Layer           | Primary ML model                 | RAG complement layer             |
+-----------------+----------------------------------+----------------------------------+
| Input type      | Structured features              | Dense embedding vectors          |
|                 |                                  | (384-dim default)                |
+-----------------+----------------------------------+----------------------------------+
| Primary cause   | Population shift, feature        | Embedding model changes,         |
|                 | engineering changes              | serialization changes,           |
|                 |                                  | population shift                 |
+-----------------+----------------------------------+----------------------------------+
| Impact          | Score calibration, routing       | Retrieval quality, LLM context   |
|                 | accuracy                         | relevance                        |
+-----------------+----------------------------------+----------------------------------+
| Detection       | Real-time feature monitoring     | Batch statistical tests on       |
|                 |                                  | embedding windows                |
+-----------------+----------------------------------+----------------------------------+
| Remediation     | Model retraining, threshold      | Reference distribution refresh,  |
|                 | recalibration                    | vector DB reindexing             |
+-----------------+----------------------------------+----------------------------------+
```

### When Both Drift Simultaneously

- Combined effect is multiplicative, not additive. The ML model routes more to the gray zone (because its features drifted) while the complement layer handles them poorly (because its embeddings drifted).
- This is the most dangerous scenario and requires the most urgent response.
- Detected by monitoring cross-layer correlation (see LangSmith dashboard, Panel 12).

---

## Threshold Calibration Methodology

### Empirical Approach

Thresholds are calibrated through a systematic process, not set arbitrarily.

1. Collect historical data spanning at least four quarters to capture seasonal patterns and known drift events.
2. Compute each drift metric on a rolling basis using production window sizes and evaluation frequencies.
3. Identify known events: model deployments, holiday seasons, market disruptions, fraud campaigns, system outages.
4. Correlate drift metric values with retrieval quality degradation and fraud detection accuracy drops.
5. Set nominal upper bound at the 95th percentile of scores during normal operating periods.
6. Set warning threshold where retrieval relevance begins to degrade measurably (approximately 5% drop in mean retrieval similarity).
7. Set critical threshold where fraud detection precision drops below the operating floor defined by risk management.

### Periodic Recalibration

- Recalibrate quarterly, coinciding with reference distribution refresh.
- As the reference shifts to incorporate recent data, the nominal range of metric values changes. A threshold appropriate for Q1 may be too tight or loose for Q2.

---

## Action Matrix

Actions are cumulative: warning includes all nominal actions. Critical includes all warning actions. Covers both embedding drift and ML feature drift, including the compound scenario.

### Action Matrix Summary

```
+-------------------+------------------------------------+-----------------------------------+
| Severity          | Automated Actions                  | Manual Actions                    |
+-------------------+------------------------------------+-----------------------------------+
| Nominal           | Log metrics, update dashboards,    | None required. Weekly review      |
|                   | store in time-series DB.           | during governance meeting.        |
+-------------------+------------------------------------+-----------------------------------+
| Warning:          | Slack alert, double monitoring     | On-call acknowledges within       |
| Embedding Drift   | frequency, trigger embedding       | 30 min. Investigate root cause.   |
|                   | quality audit, flag transactions   | Review category heatmap and       |
|                   | for manual review, compute         | retrieval quality. Check ML       |
|                   | per-merchant breakdown.            | agreement rate. Document.         |
+-------------------+------------------------------------+-----------------------------------+
| Warning:          | Slack alert with ML context,       | Review ML feature drift           |
| ML Feature Drift  | monitor complement layer volume,   | dashboard. Assess routing         |
|                   | flag ML revalidation for next      | impact on complement layer.       |
|                   | governance cycle.                  | Evaluate gray zone absorption.    |
+-------------------+------------------------------------+-----------------------------------+
| Critical:         | PagerDuty page, bypass complement  | On-call acknowledges within       |
| Embedding Drift   | layer, trigger emergency           | 15 min. Initiate incident         |
|                   | reindexing, snapshot embeddings    | response. Assess adversarial      |
|                   | for forensics, notify risk mgmt.   | activity. Coordinate fraud ops.   |
+-------------------+------------------------------------+-----------------------------------+
| Critical:         | All critical embedding actions     | Dual on-call response (one per    |
| Compound Drift    | plus page secondary on-call,       | layer). Investigate common        |
|                   | activate rule-based fallback for   | upstream cause. Stabilize ML      |
|                   | ALL transactions, halt automated   | model first. Assess total         |
|                   | approvals above threshold,         | cross-layer exposure. Prepare     |
|                   | trigger emergency retraining and   | joint risk committee report.      |
|                   | reindexing, VP-level escalation.   |                                   |
+-------------------+------------------------------------+-----------------------------------+
```

### Nominal (All Metrics Within Bounds)

Embedding drift condition: cosine < 0.05, MMD < 0.02, KS < 0.05, Wasserstein < 0.03, PSI < 0.10.

ML feature drift condition: per-feature PSI < 0.10, score distribution shift within bounds, gray zone routing rate within historical norms.

- **Automated**: log both-layer metrics, update LangSmith dashboards (including cross-layer correlation), store in time-series DB.
- **Manual**: none required. Weekly review during model governance meeting.

### Warning: Embedding Drift (2+ metrics exceed warning thresholds)

Condition (any two of): cosine >= 0.05, MMD >= 0.02, KS >= 0.05, Wasserstein >= 0.03, PSI >= 0.10.

- **Automated**: Slack to #fraud-model-alerts, double monitoring frequency, trigger embedding quality audit, flag affected transactions for manual review, compute per-merchant-category breakdown, check ML drift correlation.
- **Manual**: on-call acknowledges within 30 min. Investigate root cause (market evolution vs embedding model change vs adversarial). Review category heatmap and retrieval quality correlation. Check ML agreement rate. Document findings.

### Warning: ML Feature Drift Only (Embedding Metrics Nominal)

Condition: per-feature PSI >= 0.10 on 2+ top features, or score distribution shift exceeds warning, or gray zone routing rate exceeds 95th percentile.

- **Automated**: Slack to #fraud-model-alerts with ML drift context, monitor complement layer volume for routing rate increases, flag ML revalidation for next governance cycle.
- **Manual**: review ML feature drift dashboard. Assess routing impact on complement layer. Evaluate whether complement layer can absorb increased gray zone volume. Document findings.

### Critical: Embedding Drift (2+ metrics exceed critical thresholds)

Condition (any two of): cosine >= 0.15, MMD >= 0.08, KS >= 0.12, Wasserstein >= 0.10, PSI >= 0.20.

- **Automated**: PagerDuty page, bypass complement layer (route to manual review or ML-only with conservative thresholds), trigger emergency reindexing, snapshot production embeddings for forensics, notify risk management.
- **Manual**: on-call acknowledges within 15 min. Initiate incident response. Assess if adversarial (active fraud campaign). Coordinate with fraud ops on manual review queue surge. Assess financial exposure. Prepare risk committee report.

### Critical: Compound Drift (Both Layers Simultaneously)

Condition: embedding drift at warning or critical AND ML feature drift at warning or critical.

- **Automated**: all critical embedding actions plus page secondary on-call for ML model, activate rule-based fallback for ALL transactions, halt automated approvals above configurable amount threshold, trigger emergency retraining for ML model and reindexing for complement layer, VP-level risk management escalation.
- **Manual**: dual on-call response (one per layer). Investigate common upstream cause (data pipeline issue, population shift, regulatory change). Stabilize primary ML model first (higher volume). Assess total cross-layer financial exposure. Prepare joint risk committee report.

---

## Ensemble Decision Logic

No single metric provides a complete picture. Each has blind spots. The ensemble approach requires agreement among multiple metrics before declaring severity.

### Voting Mechanism

- At least 2 of 5 metrics must independently reach a severity level before it is declared.
- Reduces false alarms from a single metric responding to a benign distributional quirk.

### Sustained Period Requirement

- A severity level must persist for 3 consecutive evaluation periods before actions trigger.
- Transient single-window spikes (burst of unusual but legitimate transactions) do not fire alerts.

### Multi-Scale Windows

- **Short (1,000 transactions)**: detects rapid acute events (sudden new merchant category, coordinated fraud attack). Logged as early warnings only, no automated actions.
- **Medium (5,000 transactions)**: captures drift developing over hours (daily transaction mix shifts).
- **Long (20,000 transactions)**: identifies slow gradual drift over days or weeks (seasonal shifts, new payment channel adoption).
- Severity confirmed only at medium or long window scale.

---

## Practical Considerations for High-Throughput Processing

### Computational Budget

At large processor scale (approximately 300M transactions/day), full pairwise kernel matrices for MMD on every window are not feasible. Key optimizations:

- **Stratified sampling**: evaluate drift on a 1% stratified sample.
- **Random Fourier features**: approximate RBF kernel in MMD, reducing cost from O(n^2) to O(n * D) where D = 500-1,000.
- **Incremental PCA**: maintain running PCA via incremental SVD, avoiding full recomputation each window.
- **Pre-aggregated PSI bins**: update bin counts incrementally as transactions arrive rather than rebinning the entire window.

### Numerical Stability

- PSI: Laplace smoothing constant of 0.0001 added to all bin proportions to handle zero-count bins.
- Cosine distance: minimum norm threshold enforced for near-zero-norm vectors (from degenerate embeddings or malformed text). Sub-threshold embeddings logged as anomalies and excluded.

### Metric Storage and Retention

- Raw per-window metrics: retained 90 days.
- Hourly aggregates: retained 1 year.
- Daily aggregates: retained indefinitely.
- Supports real-time alerting and long-term trend analysis for quarterly model governance reviews.
