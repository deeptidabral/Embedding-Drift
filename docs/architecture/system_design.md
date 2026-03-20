# System Architecture: Dual-Layer Fraud Detection Pipeline with Drift Monitoring

## Overview

- Dual-layer fraud detection pipeline for large-scale payment processors.
- Layer 1: XGBoost model scores all transactions at high speed.
- Layer 2: RAG+LLM pipeline selectively invoked for gray zone, high-value, explainability, and novel pattern cases.
- Unified monitoring plane tracks ML feature drift and RAG embedding drift across both layers.

Design priorities:

- Sub-10ms ML scoring on the primary path. Sub-2s end-to-end for RAG+LLM path.
- 99.99% uptime on the ML scoring critical path.
- Drift-induced accuracy loss detected before it causes material financial impact.

---

## High-Level Architecture Diagram

```
                         TRANSACTION PROCESSING PIPELINE
                         ===============================

+-------------------+       +----------------------+       +-------------------+
|                   |       |                      |       |                   |
|  Transaction      |------>|  Feature             |------>|  ML Scorer        |
|  Ingestion        |       |  Extraction          |       |  (XGBoost)        |
|  Gateway          |       |  Service             |       |                   |
|                   |       |                      |       |  Output:          |
+-------------------+       +----------------------+       |  fraud_score      |
                                                           |  (0.0 - 1.0)     |
                                                           +--------+----------+
                                                                    |
                                                                    v
                                                           +--------+----------+
                                                           |                   |
                                                           |  Decision Router  |
                                                           |                   |
                                                           |  Routes based on: |
                                                           |  - ML score       |
                                                           |  - Txn amount     |
                                                           |  - Explainability |
                                                           |    requirements   |
                                                           +---+----------+----+
                                                               |          |
                              +--------------------------------+          |
                              |                                           |
                              v                                           v
              +---------------+-----------+           +-------------------+---------+
              |                           |           |                             |
              |  Direct Decision          |           |  RAG+LLM Complement Layer  |
              |  (70-80% of transactions) |           |  (20-30% of transactions)  |
              |                           |           |                             |
              |  score < 0.3 -> approve   |           |  Embedding Service          |
              |  score > 0.7 -> decline   |           |  (all-MiniLM-L6-v2 default; |
              |                           |           |   text-embed-3-large opt.)  |
              |                           |           |         |                   |
              +---------------+-----------+           |         v                   |
                              |                       |  Vector Store (ChromaDB)    |
                              |                       |  (top-k fraud patterns)     |
                              |                       |         |                   |
                              |                       |         v                   |
                              |                       |  RAG Retriever + Reranker   |
                              |                       |         |                   |
                              |                       |         v                   |
                              |                       |  LLM Assessor (GPT-4o)      |
                              |                       |                             |
                              |                       |  Output:                    |
                              |                       |  - fraud_probability        |
                              |                       |  - reasoning                |
                              |                       |  - matching_patterns        |
                              |                       |  - recommended_action       |
                              |                       |  - audit_trail              |
                              |                       +-------------------+---------+
                              |                                           |
                              v                                           v
                       +------+-------------------------------------------+------+
                       |                                                         |
                       |                    Final Decision                       |
                       |          approve | flag_for_review | decline | block    |
                       |                                                         |
                       +---------------------------------------------------------+


                         MONITORING PLANE (Asynchronous)
                         ===============================

+---------------------------+                    +---------------------------+
|                           |                    |                           |
|  ML Feature Drift         |                    |  Embedding Drift          |
|  Monitor                  |                    |  Monitor                  |
|                           |                    |                           |
|  - Covariate shift        |                    |  - Cosine distance        |
|  - Concept drift          |                    |  - MMD                    |
|  - Target shift           |                    |  - KS test                |
|  - PSI per feature        |                    |  - Wasserstein            |
|  - Feature importance     |                    |  - PSI                    |
|    stability              |                    |  - CUSUM (online)         |
+------------+--------------+                    +------------+--------------+
             |                                                |
             +---------------------+  +-----------------------+
                                   |  |
                                   v  v
                          +--------+--+---------+       +-------------------+
                          |                     |       |                   |
                          |  Drift Correlation  |------>|  LangSmith        |
                          |  Engine             |       |  (Tracing +       |
                          |                     |       |   Evaluation +    |
                          +--------+------------+       |   Dashboard)      |
                                   |                    +-------------------+
                                   v
                          +--------+------------+       +-------------------+
                          |                     |       |                   |
                          |  Alert Router       |------>|  Slack +          |
                          |                     |       |  PagerDuty        |
                          +--------+------------+       +-------------------+
                                   |
                                   v
                          +--------+------------+
                          |                     |
                          |  Fallback           |
                          |  Controller         |
                          |                     |
                          |  ML failure:        |
                          |   -> rule engine    |
                          |  LLM failure:       |
                          |   -> ML-only mode   |
                          |  Critical drift:    |
                          |   -> safe defaults  |
                          +---------------------+
```

---

## Component Details

```
+----------------------------+---------------------------+-----------+-----------------------------+
| Component                  | Technology                | Latency   | Failure Mode                |
+----------------------------+---------------------------+-----------+-----------------------------+
| Transaction Ingestion      | API Gateway + Pydantic    | <1ms      | Queue timeout -> rule       |
| Gateway                    | + Load Balancer           |           | engine fallback.            |
+----------------------------+---------------------------+-----------+-----------------------------+
| Feature Extraction         | In-memory lookup tables   | <1ms      | No features -> equivalent   |
| Service                    | + cached aggregations     |           | to ML scorer failure.       |
+----------------------------+---------------------------+-----------+-----------------------------+
| ML Scorer                  | XGBoost (500-2000 trees,  | <5ms      | All txns -> rule-based      |
|                            | max depth 6-8)            |           | fallback engine.            |
+----------------------------+---------------------------+-----------+-----------------------------+
| Decision Router            | Stateless routing logic   | <1ms      | Defaults to ML-only         |
|                            |                           |           | decision path.              |
+----------------------------+---------------------------+-----------+-----------------------------+
| Embedding Service          | sentence-transformers     | ~50ms     | ML-score-only decision.     |
|                            | all-MiniLM-L6-v2 (default,| (local)   | Falls back to ML-only.      |
|                            | 384-dim, local) or        | ~100ms    | 30s timeout, 3 retries      |
|                            | text-embedding-3-large    | (API)     | (OpenAI API path).          |
|                            | (optional, 3072-dim)      |           |                             |
+----------------------------+---------------------------+-----------+-----------------------------+
| Vector Store               | ChromaDB (HNSW, m=32,     | ~10ms     | In-memory cache of top 500  |
|                            | ef_search=100, cosine)    |           | patterns. Then ML-only.     |
+----------------------------+---------------------------+-----------+-----------------------------+
| RAG Retriever + Reranker   | Top-5 + ms-marco-MiniLM  | ~50ms     | Skip reranking. Use raw     |
|                            | cross-encoder reranker    |           | retrieval results.          |
+----------------------------+---------------------------+-----------+-----------------------------+
| LLM Assessor               | GPT-4o (temp 0.0)        | ~1500ms   | ML-score-only with          |
|                            |                           |           | LLM-unavailable flag.       |
+----------------------------+---------------------------+-----------+-----------------------------+
| Embedding Sample Buffer    | Ring buffer (20K entries,  | <0.1ms   | Drift monitor suspends.     |
|                            | lock-free concurrent)     |           | Transactions unaffected.    |
+----------------------------+---------------------------+-----------+-----------------------------+
| ML Feature Drift Monitor   | PSI, KS, chi-squared,    | <2s per   | Standby takeover via        |
|                            | SHAP stability            | window    | leader election (~30s).     |
+----------------------------+---------------------------+-----------+-----------------------------+
| Embedding Drift Monitor    | Cosine, MMD, KS,         | <1s per   | Standby takeover via        |
|                            | Wasserstein, PSI, CUSUM   | window    | leader election (~30s).     |
+----------------------------+---------------------------+-----------+-----------------------------+
| Drift Correlation Engine   | Cross-layer temporal      | <1s       | Monitors run independently. |
|                            | alignment analysis        |           | No cascade.                 |
+----------------------------+---------------------------+-----------+-----------------------------+
| LangSmith                  | Tracing + evaluation +    | Async     | No impact on transactions.  |
|                            | dashboard (60s refresh)   |           | Monitoring visibility lost. |
+----------------------------+---------------------------+-----------+-----------------------------+
| Alert Router               | Slack + PagerDuty         | Async     | No impact on transactions.  |
|                            | (dedup: 3/30min/source)   |           | Alerts queued for retry.    |
+----------------------------+---------------------------+-----------+-----------------------------+
| Fallback Controller        | Mode switching logic      | <1ms      | Defaults to most            |
|                            |                           |           | conservative mode.          |
+----------------------------+---------------------------+-----------+-----------------------------+
```

### Transaction Ingestion Gateway

- Receives raw transaction data from the payment network (transaction ID, amount, currency, merchant info, location, timestamp, payment channel, device fingerprint, historical patterns).
- Validates schema via Pydantic, applies rate limiting, routes to feature extraction.
- Handles backpressure: queues transactions when downstream is overloaded, falls back to rule engine on queue timeout.
- At large processor scale: ~3,500 TPS (300M/day), horizontally scaled at ~500 TPS per instance.

### Feature Extraction Service

- Transforms raw transaction data into the structured feature vector for ML scoring.
- Computes derived features: velocity (txn count/amount in last 1/6/24 hours), geographic risk, merchant category risk, time encoding, device fingerprint match, behavioral deviation.
- Output: fixed-dimension vector (150-300 features), computed in under 1ms.

### ML Scorer (XGBoost)

- Processes every transaction without exception. Produces fraud probability (0.0-1.0) in sub-5ms.
- Model: 500-2,000 trees, max depth 6-8, retrained weekly on rolling 90-day labeled data.
- Feature importance tracked per retraining cycle for drift correlation.
- On failure, system falls back to rule-based engine.

### Decision Router

- Determines processing path after ML scoring.
- Score < 0.3: approve on ML score alone.
- Score > 0.7: decline on ML score alone.
- Score 0.3-0.7 (gray zone): route to RAG+LLM.
- Amount > $10K: route to RAG+LLM regardless of score.
- Explainability required: route to RAG+LLM for reasoning and pattern documentation.
- During critical drift, routing thresholds can be adjusted to increase manual review.

### Embedding Service

- Defaults to sentence-transformers all-MiniLM-L6-v2, which produces 384-dimensional vectors locally with no API key required.
- OpenAI text-embedding-3-large (3,072 dimensions) is available as an optional production backend for higher-fidelity embeddings.
- Local model operates in micro-batches; OpenAI path uses micro-batches of up to 128 txns per API call.
- Config (OpenAI path): 30s timeout, 3 retries with exponential backoff. On failure, reverts to ML-score-only decision.
- Writes each embedding to the sample buffer for async drift monitoring.

### Vector Store (ChromaDB)

- Two collections: fraud knowledge base (curated patterns, updated monthly) and reference embeddings (50K+ snapshots, refreshed quarterly on 90-day rolling window).
- HNSW indexing (m=32, ef_construction=200, ef_search=100), cosine distance similarity.

### RAG Retriever and Reranker

- Queries fraud knowledge base with transaction embedding. Returns top-5 patterns above 0.75 cosine similarity threshold.
- Cross-encoder reranker (ms-marco-MiniLM-L-12-v2) rescores on full text.

### LLM Assessor

- Receives transaction text, ML fraud score, and reranked context.
- Uses GPT-4o at temperature 0.0 with fraud analyst system prompt.
- Output: fraud_probability, risk_level, reasoning, matching_patterns, recommended_action, audit_trail.
- Strict 2,000ms timeout. On timeout, reverts to ML-score-only with LLM-unavailable flag.

### Embedding Sample Buffer

- Ring buffer storing most recent 20,000 production embeddings with metadata.
- Lock-free concurrent access: write-only for embedding service, read-only for drift monitor.
- Supports three window sizes: 1,000 (short), 5,000 (medium), 20,000 (long).

### Drift Monitors

- ML Feature Drift Monitor runs every 15 min for PSI/KS and hourly for concept drift.
- Embedding Drift Monitor runs every 5 min (cosine/KS), every 10 min (MMD/Wasserstein), every 15 min (PSI). Also runs CUSUM for continuous online detection.
- Both are singleton services with leader election for standby takeover.

### Drift Correlation Engine

- Analyzes cross-layer drift signals and tracks temporal alignment between drift events.
- Escalates alert severity for correlated cross-layer drift.

### Alert Router

- Slack: warning-level and above to #fraud-model-alerts. @oncall-ml-engineer on critical.
- PagerDuty: critical-level only with severity-based routing.
- Deduplication: max 3 alerts per 30-min window per severity level per drift source.

### Fallback Controller

- ML scorer failure: all transactions to rule-based fallback. Critical alert dispatched.
- RAG+LLM failure: ML scorer continues for all transactions. Gray zone handled with adjusted thresholds or manual review routing.
- Recovery: dual-layer processing restored when drift returns to nominal or failed service recovers.

---

## Data Flow

```
                    NUMBERED DATA FLOW SEQUENCE
                    ===========================

[1] Payment Network
     |
     v
[2] Ingestion Gateway (validate schema, rate limit)
     |
     v
[3] Feature Extraction Service (compute 150-300 features, <1ms)
     |
     v
[4] ML Scorer / XGBoost (fraud_score 0.0-1.0, <5ms)
     |
     v
[5] Decision Router
     |                              |
     | score<0.3 or score>0.7       | score 0.3-0.7, >$10K,
     | (70-80% of txns)             | or explainability needed
     |                              | (20-30% of txns)
     v                              v
[6a] Direct Decision           [6b] Embedding Service (384-dim default / 3072-dim optional)
     |                              |
     |                         [7]  +--> Sample Buffer (for drift monitoring)
     |                              |
     |                         [8]  Vector Store / ChromaDB (top-5 patterns, ~10ms)
     |                              |
     |                         [9]  Cross-Encoder Reranker (~50ms)
     |                              |
     |                         [10] LLM Assessor / GPT-4o (~1500ms)
     |                              |
     v                              v
[11] Final Decision --> Payment Network
     (approve | flag_for_review | decline | block)

                    ASYNC MONITORING (parallel)
                    ===========================

[12] LangSmith Trace Logging (every LLM-assessed transaction)
[13] ML Feature Drift Monitor (every 15 min PSI/KS, hourly concept drift)
[14] Embedding Drift Monitor (every 5-15 min, plus continuous CUSUM)
[15] Drift Correlation Engine (cross-layer signal analysis)
[16] Alert Router --> Slack / PagerDuty (if thresholds exceeded)
[17] Fallback Controller (adjust processing mode if needed)
```

---

## Throughput Targets

```
+----------------------------+-------------------+---------------------------------------------+
| Component                  | Transactions/sec  | Bottleneck                                  |
+----------------------------+-------------------+---------------------------------------------+
| Ingestion Gateway          | 5,000 TPS         | Network I/O. Horizontally scalable.         |
+----------------------------+-------------------+---------------------------------------------+
| Feature Extraction         | 5,000 TPS         | In-memory. Scales with instances.           |
+----------------------------+-------------------+---------------------------------------------+
| ML Scorer (XGBoost)        | 5,000 TPS         | CPU inference (<5ms). Scales with instances. |
+----------------------------+-------------------+---------------------------------------------+
| Decision Router            | 5,000 TPS         | Stateless logic. No bottleneck.             |
+----------------------------+-------------------+---------------------------------------------+
| Embedding Service          | 1,000 TPS         | External API call. Only 20-30% of traffic.  |
+----------------------------+-------------------+---------------------------------------------+
| RAG Retrieval (ChromaDB)   | 3,000 QPS         | HNSW index lookup. In-memory.               |
+----------------------------+-------------------+---------------------------------------------+
| LLM Assessor (GPT-4o)     | 150 TPS            | LLM API rate limit. System bottleneck for   |
|                            |                   | complement path. Only 20-30% of traffic.    |
+----------------------------+-------------------+---------------------------------------------+
| ML Feature Drift Monitor   | N/A (batch)       | Feature window processed in <2s.            |
+----------------------------+-------------------+---------------------------------------------+
| Embedding Drift Monitor    | N/A (batch)       | 2,000-embedding window processed in <1s.    |
+----------------------------+-------------------+---------------------------------------------+
```

The LLM assessor is the most expensive component per-transaction, but since it only processes a routed subset, it does not constrain overall throughput.

---

## Scalability Considerations

- The ML scorer handles all transactions at high speed. Only 20-30% require the more expensive RAG+LLM processing.
- Ingestion gateway, feature extraction, ML scorer, and decision router are stateless and horizontally scalable.
- Embedding service and LLM assessor only serve the routed subset, reducing required capacity.
- ChromaDB supports sharding. The fraud knowledge base (tens of thousands of patterns) does not require it. Reference embedding collection (50K+) benefits from in-memory indexing.
- Both drift monitors are singleton services. Leader election ensures standby takeover if primary fails.

---

## Failure Modes and Fallback Strategies

### ML Scorer Unavailable (Critical Path)

- All transactions immediately routed to rule-based fallback engine.
- Critical alert to on-call team and fraud ops leadership.
- RAG+LLM layer cannot substitute as primary scorer (insufficient throughput).

### Feature Extraction Service Failure

- Equivalent to ML scorer failure. Rule-based fallback activated.

### Embedding API Unavailable

- RAG+LLM layer disabled. ML scorer continues for all transactions.
- Gray zone transactions processed ML-score-only with adjusted thresholds or flagged for manual review.
- Degraded but not critical.

### Vector Store Unavailable

- RAG retriever switches to in-memory cache of top 500 most frequently matched patterns.
- If cache also unavailable, RAG+LLM layer disabled. ML-only mode.

### LLM API Unavailable

- RAG+LLM layer disabled. ML scorer continues for all transactions.
- Automatic transition, no human intervention required.

### Drift Monitor Failure

- Standby instance takes over via leader election (typically under 30s).
- If both primary and standby down, transactions continue without drift monitoring. Infrastructure team alerted.

### Cascading Failure Prevention

- Critical transaction path (ingestion, feature extraction, ML scoring, routing) is isolated from RAG+LLM layer and monitoring plane.
- Failure in RAG+LLM, drift monitors, LangSmith, or alert router does not affect primary scoring.
- Circuit breakers at each external dependency boundary (embedding API, LLM API, ChromaDB, LangSmith).
- ML scorer has no external API dependencies at inference time (model loaded locally).

---

## Security and Compliance

- Embedding vectors: non-reversible lossy transformation. Original transaction text cannot be reconstructed.
- ML feature vectors: derived statistical features only. PII stripped before feature computation.
- LangSmith traces: include transaction text but stored in PCI DSS Level 1 compliant environment with access control and audit logging.
- Sample buffer: stores only embedding vectors and non-sensitive metadata (merchant category, amount band, geography).
- Reference snapshots: versioned and access-controlled. Retained for regulatory audit.
- Network: all inter-service communication uses mutual TLS.
- Secrets: API keys (OpenAI if used, LangSmith) in secrets manager, rotated on 90-day cycle. The default local embedding model (sentence-transformers) requires no API key.
- Model integrity: ML model binary signed and verified at load time.
