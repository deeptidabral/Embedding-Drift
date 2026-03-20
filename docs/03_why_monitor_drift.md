# Why Embedding Drift Should Be Monitored

## Overview

- Embedding drift degrades the system precisely where the primary ML model has already acknowledged uncertainty. This includes gray zone, high-value, and novel-pattern transactions.

```
+---------------------+       +------------------------+       +------------------+
| Unmonitored         | ----> | Retrieval              | ----> | Poor LLM         |
| Embedding Drift     |       | Degradation            |       | Context          |
+---------------------+       +------------------------+       +------------------+
                                                                       |
                                                                       v
                              +------------------------+       +------------------+
                              | Financial              | <---- | Wrong Fraud      |
                              | Loss                   |       | Decisions        |
                              +------------------------+       +------------------+
```

---

## The Cost of Undetected Drift

### False Negatives (Fraud Gets Through)

- Drifted embeddings cause fraudulent transactions to no longer map near known fraud patterns.
- Retrieval fails to surface relevant cases. The LLM underestimates risk. The transaction is approved.
- Each false negative means chargeback liability, cardholder trust violated, and a reinforced fraud pattern.

### False Positives (Legitimate Transactions Blocked)

- Drift causes legitimate transactions to map into fraud-associated regions. The LLM overestimates risk.
- False declines are less visible but far more costly in aggregate. Aite-Novarica estimates US e-commerce false declines exceed $440B/year vs. $12B in actual fraud.

### Asymmetry and Concentration

- Drift can increase both false negatives and false positives simultaneously. Different transaction types drift in different directions.
- Financial impact concentrates on gray zone transactions (ML score 0.3-0.7), which carry higher per-transaction exposure than the portfolio average.

### Compounding Drift

- ML model feature drift and embedding drift can compound. Degraded ML confidence pushes more transactions into the gray zone while the complement layer is simultaneously less capable.
- This is the most dangerous operational state for the dual-layer system.

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

## Drift and LLM Decision Quality

### Context Poisoning

- Drift-degraded retrieval feeds irrelevant context to the LLM, which has no mechanism to detect the mismatch.

### Confidence Miscalibration

- True high-risk transactions receive moderate-confidence scores due to diluted context.
- Genuinely ambiguous transactions receive high-confidence scores due to misleadingly consistent retrieved examples.
- Since confidence drives routing (auto-decline, manual review, auto-approve), miscalibration makes routing unreliable.

### Explanation Quality

- Degraded retrieval produces vague explanations ("shares some characteristics with flagged transactions") instead of specific ones ("matches stolen-card pattern at electronics retailers in the tri-state area").
- This reduces manual review effectiveness and system auditability.

---

## Regulatory and Compliance Implications

- PCI DSS v4.0 requires continuous monitoring of security controls (Req 6.3.2, 11.6.1). An LLM fraud system is a security control, and unmonitored drift is an undetected change to its effectiveness.
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

- Drift monitoring is a necessary control for the complement layer in any production dual-layer fraud detection system.
- Undetected drift costs tens to hundreds of millions per quarter, concentrated in the highest-risk segments.
- Regulatory exposure spans PCI DSS, SOX, SR 11-7, and the EU AI Act.
- Monitoring cost ($500K-$2M/year) is less than 1% of potential annual drift impact.
