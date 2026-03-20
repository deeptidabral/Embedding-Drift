# Embedding Drift Theory

## Overview

- Embedding drift is silent degradation of the embedding space powering the RAG complement layer in dual-layer fraud detection systems.
- Unlike crashes or errors, drift produces subtly wrong results that accumulate until the complement layer operates well below intended performance.
- Distinct from ML model feature drift (which affects the primary scorer's structured inputs). Both must be tracked as they can compound.
- **Note on embedding approach**: the text embedding method described in this project (e.g., all-MiniLM-L6-v2 over transaction text) is a simplified demonstration. Production fraud detection systems typically represent transactions via entity embeddings (learned representations of merchants, cardholders, devices), autoencoders over structured feature sets, or graph neural networks (GNNs) that capture relational structure across transaction networks. The drift theory and MMD-based detection framework apply regardless of the embedding method.

---

## Formal Definition

- Let Q_t = f_t(P_t) be the embedding distribution at time t, where f_t is the embedding function and P_t is the input transaction distribution.
- Drift has occurred when D(Q_t, Q_ref) > tau, where D is a divergence measure and tau is the acceptable performance degradation threshold.
- Two sources.
  - Data drift: input distribution changes while the embedding function is unchanged.
  - Model drift: embedding function changes (model update, API change, infrastructure modification) while inputs are unchanged.
- Both often co-occur. Monitoring must detect drift regardless of source.

---

## Mathematical Framing

### Distribution Shift

- First-order (mean shift): ||mu_t - mu_ref||_2. Captures global directional shifts such as portfolio shifting toward different demographics or geographies.
- Second-order (covariance change): ||Sigma_t - Sigma_ref||_F. Captures changing relationships between embedding dimensions, altering decision boundary geometry.
- Higher-order (skewness, kurtosis, tail behavior): requires nonparametric tests like MMD.

### Divergence Measures

- MMD (Maximum Mean Discrepancy): kernel-based, sensitive to all distribution moments, no density estimation needed. This is the recommended detection method for embedding drift. With an RBF kernel and median heuristic bandwidth, MMD captures mean shifts, variance changes, tail behavior, and multimodality in a single statistic with a well-defined permutation test for p-value calibration.
- Other measures (Wasserstein, KL divergence, KS test, PSI, cosine distance) are either computationally intractable in high dimensions, reduce to univariate tests that miss multivariate structure, or depend on binning that breaks down in dense embedding spaces. See the drift metrics and thresholds document for a detailed analysis of why these alternatives are inappropriate.

---

## Types of Embedding Drift

### Drift Pattern Signatures

```
Sudden Drift                    Gradual Drift
(step function)                 (slow curve)

Value                           Value
  ^                               ^
  |         +--------             |              .......
  |         |                     |          ....
  |         |                     |      ...
  |---------+                     |  ...
  |                               |..
  +--------+---------> Time       +--------------------> Time


Incremental Drift               Recurring Drift
(staircase)                     (wave / seasonal)

Value                           Value
  ^                               ^
  |              +----            |    .        .        .
  |         +----|                |   . .      . .      .
  |    +----|                     |  .   .    .   .    .
  |----+                          | .     .  .     .  .
  |                               |.       ..       ..
  +--------------------> Time     +--------------------> Time
```
### Drift Type Mapping

| Drift Type | Description | Root Cause | Business Impact | Detection Difficulty |
|---|---|---|---|---|
| Sudden | Abrupt, large shift in embedding distribution occurring over minutes or hours. | Embedding model update, pipeline bug, GPU migration. | Immediate retrieval precision collapse. False negatives spike within hours. | Low. Change-point algorithms catch it quickly. |
| Gradual | Slow, continuous movement of the distribution over weeks or months. | Evolving consumer behavior, new merchant categories, slow model decay. | Retrieval precision erodes over weeks to months. Complement layer slowly loses value. | High. Accumulates below alert thresholds. |
| Incremental | Series of small, discrete steps that compound over time. | Phased product rollouts, sequential acquirer onboarding, iterative rule changes. | Stepwise degradation. Each step looks acceptable but cumulative effect is severe. | Medium. Requires long-window cumulative statistics. |
| Recurring | Periodic pattern that shifts away from reference and returns to baseline. | Holiday spending, weekday/weekend mix, payroll cycles. | False alerts during seasonal shifts desensitize operators. Real drift hidden behind expected variation. | Medium. Requires seasonal baseline separation. |

---

## Summary

- Embedding drift is a distributional shift that degrades the complement layer's performance on the system's most uncertain decisions.
- Four patterns (sudden, gradual, incremental, recurring), each with different root causes in the fraud domain.
- Degradation operates through retrieval quality, decision boundary alignment, and risk score calibration.
- Can compound with ML model feature drift. When both layers drift, the system loses accuracy at both tiers.
