# Why Embedding Drift Should Be Monitored

## Overview

- The ML model drives real-time authorization decisions (sub-10ms, sub-50ms end-to-end). The RAG+LLM layer operates asynchronously after authorization, supporting human analysts in manual review of flagged transactions, chargeback defense, and regulatory audit documentation.
- Embedding drift degrades the async investigation layer, which is the system's support mechanism for the cases the ML model has flagged as uncertain (gray zone, high-value, and novel-pattern transactions).

```
+---------------------+       +------------------------+       +------------------+
| Unmonitored         | ----> | Retrieval              | ----> | Poor LLM         |
| Embedding Drift     |       | Degradation            |       | Investigation    |
+---------------------+       +------------------------+       | Context          |
                                                                +------------------+
                                                                       |
                                                                       v
                              +------------------------+       +------------------+
                              | Financial              | <---- | Degraded Manual  |
                              | Loss                   |       | Review Quality   |
                              +------------------------+       +------------------+
```

---

## The Cost of Undetected Drift

### Missed Fraud in Manual Review (Degraded Investigation Quality)

- Drifted embeddings cause fraudulent transactions to no longer map near known fraud patterns in the async investigation pipeline.
- Retrieval fails to surface relevant cases. The LLM generates investigation reports with irrelevant context. Human analysts reviewing flagged transactions receive poor-quality supporting evidence, increasing the likelihood of incorrect manual review decisions.
- Each missed fraud case means chargeback liability, cardholder trust violated, and a reinforced fraud pattern.
- The financial impact comes from degraded investigation quality leading to missed fraud in manual review, not from real-time false negatives in the authorization path (which is driven by the ML model alone).

### False Escalations in Manual Review

- Drift causes legitimate transactions to map into fraud-associated regions. The LLM generates investigation reports that overstate risk.
- Analysts waste time investigating legitimate transactions flagged with misleading LLM-generated evidence, reducing manual review throughput and increasing operational costs.
- False declines driven by ML model drift in the authorization path are a separate concern. Aite-Novarica estimates US e-commerce false declines exceed $440B/year vs. $12B in actual fraud.

### Asymmetry and Concentration

- Drift can degrade investigation quality for both fraud and legitimate flagged transactions simultaneously. Different transaction types drift in different directions.
- Financial impact concentrates on gray zone transactions (ML score 0.3-0.7) that are flagged for manual review, which carry higher per-transaction exposure than the portfolio average.

### Compounding Drift

- ML model feature drift and embedding drift can compound. Degraded ML confidence pushes more transactions into the gray zone (more flagged for manual review) while the async investigation layer is simultaneously producing lower-quality reports for analysts.
- This is the most dangerous operational state for the dual-layer system: more transactions needing human review, with less useful LLM-generated investigation support.

---

## Drift and RAG Retrieval Quality

### Nearest Neighbor Relevance

- Retrieval assumes geometric proximity equals semantic similarity. Drift breaks this assumption.
- Example: a $2,300 electronics transaction that previously retrieved stolen-card fraud cases now also retrieves irrelevant subscription renewals that drifted into the same neighborhood.

### Recall of Rare Fraud Patterns

- Rare fraud patterns form small, tight clusters. Even modest drift can push queries out of top-k retrieval range.
- The rarest, most dangerous tactics degrade disproportionately.

### Stale Index Compounding

- Historical embeddings were computed with older model versions. New queries use the current version.
- Oldest indexed embeddings (most established fraud patterns with best documentation) become the most misaligned.
- The most valuable historical patterns become the least retrievable.

---

## Drift and LLM Investigation Quality

### Context Poisoning

- Drift-degraded retrieval feeds irrelevant context to the LLM in the async investigation pipeline. The LLM has no mechanism to detect the mismatch and generates investigation reports based on poor-quality retrieved evidence.

### Confidence Miscalibration in Investigation Reports

- True high-risk transactions receive moderate-confidence investigation assessments due to diluted context.
- Genuinely ambiguous transactions receive high-confidence assessments due to misleadingly consistent retrieved examples.
- Since analysts rely on LLM-generated investigation reports to prioritize and decide on flagged cases, miscalibrated reports lead to incorrect manual review outcomes.

### Explanation Quality

- Degraded retrieval produces vague investigation reports ("shares some characteristics with flagged transactions") instead of specific ones ("matches stolen-card pattern at electronics retailers in the tri-state area").
- This reduces manual review effectiveness, weakens chargeback defense, and degrades regulatory audit documentation quality.

---

## Regulatory and Compliance Implications

- PCI DSS v4.0 requires continuous monitoring of security controls (Req 6.3.2, 11.6.1). Both the ML authorization model and the LLM investigation pipeline are security controls, and unmonitored drift is an undetected change to their effectiveness.
- SOX: for publicly traded processors, unmonitored drift causing material fraud loss increases could trigger a material weakness finding requiring public disclosure.
- SR 11-7: Federal Reserve guidance requires ongoing performance monitoring for all models used in material decisions, including AI/ML systems.
- EU AI Act: fraud detection AI is high-risk. Article 72 requires post-market monitoring for performance changes. Embedding drift monitoring satisfies this requirement.

---

## Case Study: Unmonitored Drift at a Payment Processor

### Week-by-Week Incident Timeline

```
Week 0          Week 1-2            Week 3-6            Week 7
  |                |                   |                   |
  v                v                   v                   v
+-----------+  +----------------+  +------------------+  +------------------+
| Embedding |  | Silent         |  | Safety Net       |  | First Visible    |
| Provider  |  | Degradation    |  | Failure          |  | Symptom          |
| Deploys   |  |                |  |                  |  |                  |
| Change    |  | Cosine sim     |  | Detection rate   |  | Monthly report   |
| (Friday   |  | 0.96-0.99      |  | drops to 92.8%   |  | shows 18% fraud  |
|  evening) |  | 12% of txns    |  | $14.7M fraud     |  | increase         |
|           |  | affected        |  | approved         |  |                  |
+-----------+  +----------------+  +------------------+  +------------------+

Week 8-10           Week 11
  |                   |
  v                   v
+-----------------+  +------------------+
| Root Cause      |  | Remediation      |
| Hunt            |  |                  |
|                 |  | Full reindex     |
| 2 weeks testing |  | 500M embeddings  |
| hypotheses      |  | 9 days downtime  |
| before finding  |  | Total cost: $85M |
| embedding API   |  |                  |
| change          |  |                  |
+-----------------+  +------------------+
```
---

## Summary

- Drift monitoring is a necessary control for both the real-time ML authorization layer and the async RAG+LLM investigation layer.
- In the authorization path, ML feature drift directly degrades real-time decisions. In the investigation path, embedding drift degrades the quality of LLM-generated reports that human analysts rely on for manual review, chargeback defense, and audit documentation.
- Undetected drift costs tens to hundreds of millions per quarter, concentrated in the highest-risk segments.
- Regulatory exposure spans PCI DSS, SOX, SR 11-7, and the EU AI Act.
- Monitoring cost ($500K-$2M/year) is less than 1% of potential annual drift impact.
