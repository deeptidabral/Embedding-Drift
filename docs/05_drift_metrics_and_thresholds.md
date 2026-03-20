# Drift Metrics and Thresholds

## Overview

Maximum Mean Discrepancy (MMD) is the sole embedding drift metric for the RAG complement layer. MMD is a kernel-based two-sample test that compares the full shape of the production embedding distribution against a calibrated reference. It is computed over the default 384-dimensional embedding space (all-MiniLM-L6-v2 for simplified demos; production systems use entity embeddings, autoencoders, or GNNs). Thresholds can be recalibrated for alternative embedding dimensionalities.

### Why MMD -- and Why Not the Alternatives

Dense embedding vectors (hundreds or thousands of dimensions, non-independent, continuous) invalidate assumptions underlying most classical drift statistics. The following methods were evaluated and rejected.

#### Kolmogorov-Smirnov (KS) Test

- **Fundamental flaw**: the KS test is a univariate statistic. Applying it per dimension (or per principal component) decomposes a multivariate problem into independent marginal tests.
- Multivariate rotations and shape changes that preserve each marginal distribution are invisible to KS.
- Testing hundreds of dimensions simultaneously creates a massive multiple testing problem. Even with Bonferroni or FDR correction, either statistical power collapses or false discovery rates become uncontrollable.
- Aggregating per-dimension KS statistics (e.g., taking the max) has no formal distributional theory, making p-value calibration unreliable.

#### Cosine Distance (Centroid Comparison)

- **Fundamental flaw**: cosine distance between centroids reduces each distribution to a single point (the mean), discarding all information about variance, shape, multimodality, and tail behavior.
- A distribution can split into two clusters, develop heavy tails, or undergo a complete variance collapse -- all while the centroid remains unchanged.
- In fraud detection, emerging attack clusters and tail shifts (where novel fraud resides) are precisely the signals that centroid comparison misses.

#### PCA Explained Variance

- **Fundamental flaw**: explained variance ratios capture only the relative spread along principal axes. They are blind to mean shifts entirely.
- A production distribution can translate arbitrarily far from the reference without changing PCA eigenvalues.
- Additionally, PCA is a linear projection; nonlinear distributional changes in embedding space (common with deep models) are poorly represented.

#### Population Stability Index (PSI)

- **Fundamental flaw**: PSI requires binning continuous distributions. In high dimensions, bin counts explode exponentially (curse of dimensionality), and most bins end up empty.
- The standard workaround (project onto principal components, then bin) loses multivariate structure, reducing PSI to a marginal test with the same problems as KS.
- Laplace smoothing on empty bins introduces arbitrary bias. Bin boundary choices heavily influence the result in high dimensions.

#### Wasserstein Distance (Earth Mover's Distance)

- **Fundamental flaw**: computing the exact Wasserstein distance is O(n^3) via linear programming, which is intractable for the sample sizes and dimensionalities in production fraud monitoring.
- Sliced Wasserstein (random 1D projections) is computationally feasible but reduces to a collection of univariate comparisons, losing sensitivity to the full multivariate structure -- the same core limitation as KS.
- The Wasserstein distance also lacks a straightforward hypothesis testing framework, making threshold calibration ad hoc.

#### Why MMD Works

- MMD operates on pairwise kernel evaluations, inherently capturing the full multivariate distributional shape without binning, marginalizing, or projecting.
- With an RBF kernel, MMD is sensitive to differences in all moments of the distribution (mean, variance, skewness, tails, multimodality).
- It has a well-defined permutation test for formal hypothesis testing with calibrated p-values.
- Random Fourier feature approximations reduce computational cost from O(n^2) to O(nD) where D is 500-1,000, making it feasible at production scale.

---

## Maximum Mean Discrepancy (MMD) -- Detailed Specification

### What It Measures

- Kernel-based two-sample test comparing the full shape of two distributions.
- Formula: MMD^2(P, Q) = E[k(x, x')] - 2*E[k(x, y)] + E[k(y, y')], where x, x' ~ P and y, y' ~ Q.
- In practice, estimated from finite samples: MMD^2_u = (1/m(m-1)) sum_{i!=j} k(x_i, x_j) - (2/mn) sum_{i,j} k(x_i, y_j) + (1/n(n-1)) sum_{i!=j} k(y_i, y_j).

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
  and tail shifts that centroid comparison,
  marginal tests, and binning methods miss.
```

### Kernel Choice: RBF with Median Heuristic

- **Kernel**: Radial Basis Function (Gaussian), k(x, y) = exp(-||x - y||^2 / (2 * sigma^2)).
- **Bandwidth selection**: the median heuristic sets sigma to the median of pairwise distances in the combined reference + production sample. This is a data-adaptive choice that avoids manual tuning and has strong empirical performance across a wide range of distributional differences.
- **Why RBF**: the RBF kernel induces a rich RKHS (reproducing kernel Hilbert space) that makes MMD a consistent test -- it converges to zero if and only if the two distributions are identical. Polynomial or linear kernels lack this universal property.
- **Multi-bandwidth option**: for additional robustness, compute MMD with multiple bandwidth values (e.g., median * {0.5, 1.0, 2.0}) and aggregate (sum or max). This widens sensitivity across different spatial scales of distributional change.

### Permutation Test for P-Value Calibration

- Under the null hypothesis (P = Q), the MMD statistic has a known but complex distribution that depends on the kernel and the unknown distribution.
- The permutation test provides an exact, distribution-free p-value:
  1. Compute the observed MMD^2 on the reference and production samples.
  2. Pool all samples and repeatedly (e.g., 1,000 permutations) randomly split into two groups of the original sizes.
  3. Compute MMD^2 for each random split.
  4. The p-value is the fraction of permutation MMD^2 values that exceed the observed value.
- Reject the null at p < 0.01 (configurable). This controls the false positive rate without distributional assumptions.

### Threshold Calibration

Thresholds map the continuous MMD statistic to operational severity levels. They are calibrated empirically, not set arbitrarily.

1. Collect historical embedding data spanning at least four quarters to capture seasonal patterns and known drift events.
2. Compute MMD on a rolling basis using production window sizes and evaluation frequencies.
3. Identify known events: model deployments, holiday seasons, market disruptions, fraud campaigns, system outages.
4. Correlate MMD values with retrieval quality degradation and fraud detection accuracy drops.
5. Set nominal upper bound at the 95th percentile of MMD during normal operating periods.
6. Set warning threshold where retrieval relevance begins to degrade measurably (approximately 5% drop in mean retrieval similarity).
7. Set critical threshold where fraud detection precision drops below the operating floor defined by risk management.

### MMD Thresholds

```
+---------------------+------------------+------------------+------------------+
| Metric              | Nominal          | Warning          | Critical         |
+---------------------+------------------+------------------+------------------+
| MMD                 | 0.00 - 0.02      | 0.02 - 0.08     | > 0.08           |
| MMD p-value         | > 0.01           | 0.001 - 0.01    | < 0.001          |
+---------------------+------------------+------------------+------------------+
```

- **Nominal (MMD < 0.02, p > 0.01)**: distributions statistically indistinguishable. Minor kernel fluctuations from sampling noise expected below 0.01. No action required.
- **Warning (MMD 0.02-0.08, p 0.001-0.01)**: specific regions of embedding space are shifting. May be benign market evolution or emerging fraud vectors. Investigate.
- **Critical (MMD > 0.08, p < 0.001)**: production distribution is fundamentally different from reference. Vector store has no relevant historical patterns for these transactions. Immediate intervention required.

---

## ML Model Feature Drift vs Embedding Drift

The dual-layer architecture requires monitoring two distinct drift surfaces that apply to different layers and measure different phenomena.

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
| Detection       | Real-time feature monitoring     | MMD-based batch statistical      |
|                 |                                  | test on embedding windows        |
+-----------------+----------------------------------+----------------------------------+
| Remediation     | Model retraining, threshold      | Reference distribution refresh,  |
|                 | recalibration                    | vector DB reindexing             |
+-----------------+----------------------------------+----------------------------------+
```

### When Both Drift Simultaneously

- Combined effect is multiplicative, not additive. The ML model routes more to the gray zone (because its features drifted) while the complement layer handles them poorly (because its embeddings drifted).
- This is the most dangerous scenario and requires the most urgent response.
- Detected by monitoring cross-layer correlation (see monitoring dashboard, MMD panel correlated with ML score distribution shift).

---

## Action Matrix

Actions are cumulative: warning includes all nominal actions. Critical includes all warning actions. Covers both embedding drift and ML feature drift, including the compound scenario.

### Action Matrix Summary

```
+-------------------+------------------------------------+-----------------------------------+
| Severity          | Automated Actions                  | Manual Actions                    |
+-------------------+------------------------------------+-----------------------------------+
| Nominal           | Log MMD metric, update dashboards, | None required. Weekly review      |
|                   | store in time-series DB.           | during governance meeting.        |
+-------------------+------------------------------------+-----------------------------------+
| Warning:          | Alert on-call team, double monitoring     | On-call acknowledges within       |
| Embedding Drift   | frequency, trigger embedding       | 30 min. Investigate root cause.   |
| (MMD warning)     | quality audit, flag transactions   | Review category heatmap and       |
|                   | for manual review, compute         | retrieval quality. Check ML       |
|                   | per-merchant breakdown.            | agreement rate. Document.         |
+-------------------+------------------------------------+-----------------------------------+
| Warning:          | Alert on-call team with ML context,       | Review ML feature drift           |
| ML Feature Drift  | monitor complement layer volume,   | dashboard. Assess routing         |
|                   | flag ML revalidation for next      | impact on complement layer.       |
|                   | governance cycle.                  | Evaluate gray zone absorption.    |
+-------------------+------------------------------------+-----------------------------------+
| Critical:         | Page on-call engineer, bypass complement  | On-call acknowledges within       |
| Embedding Drift   | layer, trigger emergency           | 15 min. Initiate incident         |
| (MMD critical)    | reindexing, snapshot embeddings    | response. Assess adversarial      |
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

### Nominal (MMD Within Bounds)

Embedding drift condition: MMD < 0.02 (p > 0.01).

ML feature drift condition: per-feature PSI < 0.10, score distribution shift within bounds, gray zone routing rate within historical norms.

- **Automated**: log MMD metric and ML layer metrics, update dashboards (including cross-layer correlation), store in time-series DB.
- **Manual**: none required. Weekly review during model governance meeting.

### Warning: Embedding Drift (MMD Exceeds Warning Threshold)

Condition: MMD >= 0.02 (p <= 0.01).

- **Automated**: alert on-call team, double monitoring frequency, trigger embedding quality audit, flag affected transactions for manual review, compute per-merchant-category breakdown, check ML drift correlation.
- **Manual**: on-call acknowledges within 30 min. Investigate root cause (market evolution vs embedding model change vs adversarial). Review category heatmap and retrieval quality correlation. Check ML agreement rate. Document findings.

### Warning: ML Feature Drift Only (Embedding MMD Nominal)

Condition: per-feature PSI >= 0.10 on 2+ top features, or score distribution shift exceeds warning, or gray zone routing rate exceeds 95th percentile.

- **Automated**: alert on-call team with ML drift context, monitor complement layer volume for routing rate increases, flag ML revalidation for next governance cycle.
- **Manual**: review ML feature drift dashboard. Assess routing impact on complement layer. Evaluate whether complement layer can absorb increased gray zone volume. Document findings.

### Critical: Embedding Drift (MMD Exceeds Critical Threshold)

Condition: MMD >= 0.08 (p < 0.001).

- **Automated**: Page on-call engineer, bypass complement layer (route to manual review or ML-only with conservative thresholds), trigger emergency reindexing, snapshot production embeddings for forensics, notify risk management.
- **Manual**: on-call acknowledges within 15 min. Initiate incident response. Assess if adversarial (active fraud campaign). Coordinate with fraud ops on manual review queue surge. Assess financial exposure. Prepare risk committee report.

### Critical: Compound Drift (Both Layers Simultaneously)

Condition: embedding MMD at warning or critical AND ML feature drift at warning or critical.

- **Automated**: all critical embedding actions plus page secondary on-call for ML model, activate rule-based fallback for ALL transactions, halt automated approvals above configurable amount threshold, trigger emergency retraining for ML model and reindexing for complement layer, VP-level risk management escalation.
- **Manual**: dual on-call response (one per layer). Investigate common upstream cause (data pipeline issue, population shift, regulatory change). Stabilize primary ML model first (higher volume). Assess total cross-layer financial exposure. Prepare joint risk committee report.

---

## Multi-Scale Window Strategy

MMD is evaluated across multiple transaction window sizes to capture drift at different time scales.

### Window Configuration

- **Short (1,000 transactions)**: detects rapid acute events (sudden new merchant category, coordinated fraud attack). Logged as early warnings only, no automated actions.
- **Medium (5,000 transactions)**: captures drift developing over hours (daily transaction mix shifts).
- **Long (20,000 transactions)**: identifies slow gradual drift over days or weeks (seasonal shifts, new payment channel adoption).
- Severity confirmed only at medium or long window scale.

### Sustained Period Requirement

- A severity level must persist for 3 consecutive evaluation periods before actions trigger.
- Transient single-window spikes (burst of unusual but legitimate transactions) do not fire alerts.

---

## Practical Considerations for High-Throughput Processing

### Computational Budget

At large processor scale (approximately 300M transactions/day), full pairwise kernel matrices for MMD on every window are not feasible. Key optimizations:

- **Stratified sampling**: evaluate drift on a 1% stratified sample.
- **Random Fourier features**: approximate RBF kernel in MMD, reducing cost from O(n^2) to O(n * D) where D = 500-1,000. This is the primary scalability mechanism.
- **Block MMD**: partition samples into blocks and compute MMD on each block, then aggregate. Provides unbiased estimates with reduced variance.

### Numerical Stability

- Kernel bandwidth: enforce a minimum bandwidth floor to prevent numerical overflow in exp(-||x-y||^2 / (2*sigma^2)) when sigma is very small.
- Permutation test: use at least 1,000 permutations for stable p-value estimates. For critical decisions, use 10,000.

### Metric Storage and Retention

- Raw per-window MMD values and p-values: retained 90 days.
- Hourly aggregates: retained 1 year.
- Daily aggregates: retained indefinitely.
- Supports real-time alerting and long-term trend analysis for quarterly model governance reviews.

### Periodic Recalibration

- Recalibrate thresholds quarterly, coinciding with reference distribution refresh.
- As the reference shifts to incorporate recent data, the nominal range of MMD values changes. A threshold appropriate for Q1 may be too tight or loose for Q2.
