# Embeddings Fundamentals for Fraud Detection

## Overview

- A dual-layer fraud detection architecture pairs a primary ML model (XGBoost or similar) for real-time scoring with a RAG+LLM complement layer for gray zone cases, high-value transactions, explainability, and novel pattern detection.
- Embeddings are the computational foundation of the complement layer. They determine retrieval quality for the hardest, highest-stakes decisions.
- This document covers what embeddings are, how transaction data is encoded, key geometric properties, and why embedding quality matters disproportionately for cases the ML model cannot resolve alone.

---

## What Are Embeddings

- A learned function that maps inputs into a fixed-length numeric vector, typically 384 to 3072 dimensions.
- Semantically similar inputs land close together in vector space. Dissimilar inputs land far apart.
- In payments, the input is a transaction record (merchant, amount, location, cardholder profile) and the output is a vector where transactions with similar risk profiles are geometrically proximate.
- Common production dimensions: 384 (all-MiniLM-L6-v2), 768 (BERT-based), 1536 (text-embedding-ada-002), 3072 (text-embedding-3-large).
- This project defaults to sentence-transformers all-MiniLM-L6-v2 (384 dimensions) for local execution, with no API key required. OpenAI text-embedding-3-large (3072 dimensions) is available as a production alternative.
- Higher dimensions capture finer distinctions but cost more in storage, compute, and training data.

### Embedding Transformation Pipeline

```
+-------------------------+       +----------------------+       +-----------------+       +-----------------------------+
| Raw Transaction Data    |       | Text Serialization   |       | Embedding Model |       | Dense Vector                |
|                         | ----> |                      | ----> |                 | ----> |                             |
| amt: $247.83            |       | "Transaction:        |       | all-MiniLM-L6-v2|       | [0.12, -0.45, 0.78, 0.03,  |
| merchant: BEST BUY     |       |  $247.83 USD at      |       | (default) or    |       |  -0.91, 0.34, ..., 0.07]   |
| mcc: 5732               |       |  BEST BUY #0423      |       | text-embedding- |       |                             |
| city: New York, NY      |       |  (MCC 5732:          |       | 3-large         |       | 384 dimensions (default),   |
| terminal: POS chip-read |       |  Electronics)..."    |       | (optional)      |       | L2-normalized               |
+-------------------------+       +----------------------+       +-----------------+       +-----------------------------+
```

### Similarity in Embedding Space

| Concept | Embedding Representation |
|---|---|
| Two grocery purchases at the same store, same cardholder | Vectors nearly identical. Cosine similarity above 0.95. |
| Grocery purchase vs. electronics purchase, same cardholder | Vectors moderately close. Cosine similarity 0.60 to 0.80. |
| Legitimate retail purchase vs. card-testing micro-charge | Vectors far apart. Cosine similarity below 0.40. |
| Two fraud-ring transactions at different merchants | Vectors cluster together due to shared behavioral signature. |

---

## Why Embeddings Matter for Fraud Detection

- Rule-based systems are brittle and cannot capture high-dimensional fraud relationships.
- Primary ML models handle most transactions well but struggle with gray zone cases, novel patterns, and contextual reasoning.
- Embeddings close that gap, powering the RAG complement layer for the 5 to 15 percent of transactions the ML model cannot resolve confidently.

### RAG Retrieval Pipeline

```
+------------------+     +-------------------+     +-----------------------+     +-------------------+
| Incoming         |     | Embed Transaction |     | Vector Database       |     | LLM Risk          |
| Transaction      | --> | (all-MiniLM-L6-v2 | --> | Query (ANN search,    | --> | Assessment        |
| (gray zone,      |     |  default, or      |     |  k=20 neighbors,      |     | (weigh retrieved  |
|  high-value, or  |     |  text-embedding-  |     |  MCC + time filters)  |     |  cases, produce   |
|  novel pattern)  |     |  3-large)         |     |                       |     |  score + reason)  |
|                  |     | Output: 384-dim   |     | Output: 20 similar    |     |                   |
|                  |     |                   |     | labeled transactions  |     | Output: risk      |
|                  |     |                   |     | with fraud labels     |     | score, decision,  |
|                  |     |                   |     | and investigation     |     | explanation       |
|                  |     |                   |     | notes                 |     |                   |
+------------------+     +-------------------+     +-----------------------+     +-------------------+
```

Patterns embeddings capture that structured features miss:

- Card-testing sequences such as small gas station charges followed by a large electronics purchase.
- Organized fraud rings with correlated merchant, geographic, and temporal patterns across multiple cards.
- Cardholder behavioral trajectory anomalies indicating account compromise.

---

## How Transaction Data Gets Embedded

### Stage 1: Feature Serialization

Raw authorization messages (100+ fields) are serialized into natural language descriptions using a template.

**Template:**

```
Transaction: ${amt} ${currency} at ${merchant} (MCC ${mcc}: ${category})
Location: ${lat}, ${long} (${city}, ${state})
Terminal: ${terminal_type}, ${entry_mode}
Time: ${timestamp} (${day_of_week} ${time_of_day}, local)
Cardholder profile: ${account_age}, avg monthly spend ${avg_monthly}
Recent activity: ${recent_txn_count} transactions in past 24h, ${recent_pattern}
```

**Example 1: Legitimate grocery purchase.**

| Field | Value |
|---|---|
| amt | $67.50 |
| merchant | KROGER #1204 |
| mcc / category | 5411 / Grocery |
| city, state | Columbus, OH |
| terminal | POS chip-read, card-present |
| timestamp | 2025-03-12T09:15:22Z |
| account_age | 6-year account |
| avg_monthly | $2,800 |
| recent_24h | 0 transactions |

```
Transaction: $67.50 USD at KROGER #1204 (MCC 5411: Grocery)
Location: 39.9612, -82.9988 (Columbus, OH)
Terminal: POS chip-read, card-present
Time: 2025-03-12T09:15:22Z (Wednesday morning, local)
Cardholder profile: 6-year account, avg monthly spend $2,800
Recent activity: 0 transactions in past 24h
```

**Example 2: Suspicious electronics purchase.**

| Field | Value |
|---|---|
| amt | $1,847.23 |
| merchant | TIFFANY & CO #0891 |
| mcc / category | 5944 / Jewelry |
| city, state | New York, NY |
| terminal | POS chip-read, card-present |
| timestamp | 2025-03-15T14:23:07Z |
| account_age | 8-year account |
| avg_monthly | $4,100 |
| recent_24h | 1 transaction (grocery, $67.50) |

```
Transaction: $1,847.23 USD at TIFFANY & CO #0891 (MCC 5944: Jewelry)
Location: 40.7589, -73.9851 (New York, NY)
Terminal: POS chip-read, card-present
Time: 2025-03-15T14:23:07Z (Saturday afternoon, local)
Cardholder profile: 8-year account, avg monthly spend $4,100
Recent activity: 1 transaction in past 24h (grocery, $67.50)
```

**Example 3: Card-testing micro-charge.**

| Field | Value |
|---|---|
| amt | $1.00 |
| merchant | SHELL OIL 57442 |
| mcc / category | 5541 / Gas Station |
| city, state | Miami, FL |
| terminal | Card-not-present, manual key entry |
| timestamp | 2025-03-16T03:41:55Z |
| account_age | 2-year account |
| avg_monthly | $1,200 |
| recent_24h | 4 transactions (3 gas, 1 online retail) |

```
Transaction: $1.00 USD at SHELL OIL 57442 (MCC 5541: Gas Station)
Location: 25.7617, -80.1918 (Miami, FL)
Terminal: Card-not-present, manual key entry
Time: 2025-03-16T03:41:55Z (Sunday early morning, local)
Cardholder profile: 2-year account, avg monthly spend $1,200
Recent activity: 4 transactions in past 24h (3 gas, 1 online retail)
```

### Stage 2: Model Inference

- BPE tokenization (approximately 80 to 150 tokens per transaction), transformer forward pass, pooling into a fixed-length vector, and L2 normalization to the unit hypersphere.

### Stage 3: Post-Processing

- Dimensionality reduction (PCA or learned projection, e.g. 384 to 128 or 3072 to 256) for storage efficiency.
- Quantization (32-bit to 8-bit) for faster similarity computation.
- Optional concatenation with engineered numeric features (amount, velocity counters).

---

## Embedding Space Geometry

### Cosine Similarity

- L2-normalized embeddings make cosine similarity equal to the dot product, ranging from -1 to +1.
- Fraud detection thresholds.
  - Above 0.90: near-identical risk profile. If one is fraud, the other almost certainly is.
  - 0.70 to 0.90: significant shared characteristics. Useful for fraud pattern variants.
  - 0.40 to 0.70: partial overlap. May indicate same fraud family, different execution.
  - Below 0.40: largely unrelated.

### Cluster Structure

- Legitimate transactions form tight per-cardholder clusters reflecting habitual patterns.
- Fraud clusters are often tighter (repeatable playbooks) and sit at boundaries between legitimate clusters.
- The gap between legitimate and fraud clusters provides discriminative power. Drift that erodes this gap degrades detection.

### 2D Projection of Embedding Space

```
    Cosine Dim 2
    ^
    |
    |        L L L                      G G G
    |      L L L L L                  G G G G G
    |     L L L L L L                G G G G G G
    |      L L L L L     * * *        G G G G G
    |        L L L      * F F *         G G G
    |                  * F F F *
    |                   * F F *    D D D
    |     T T T          * *     D D D D D
    |   T T T T T               D D D D D D
    |    T T T T T               D D D D D
    |     T T T T                  D D D
    |      T T
    |
    +---------------------------------------------> Cosine Dim 1

    L = Legitimate grocery cluster     G = Legitimate gas station cluster
    T = Legitimate travel cluster       D = Legitimate dining cluster
    F = Fraud cluster (sits between legitimate clusters)
    * = Fraud boundary region
```

### What Embedding Dimensions Capture

| Dimension Group | Captures | Example |
|---|---|---|
| Amount patterns (dims 0 to 50) | Transaction size relative to cardholder norm. | $1 micro-charge vs. $4,100 average spend flags anomaly. |
| Merchant category similarity (dims 51 to 150) | Semantic grouping of merchant types. | Electronics and jewelry cluster near each other, far from grocery. |
| Geographic proximity (dims 151 to 300) | Spatial relationships between transaction locations. | Miami transaction for an Ohio-based cardholder encodes distance. |
| Temporal patterns (dims 301 to 450) | Time-of-day, day-of-week, and recency signals. | 3 AM transaction on a Sunday far from typical weekday pattern. |
| Behavioral trajectory (dims 451 to 512) | Sequence-level features across recent transactions. | Four rapid gas station charges encode card-testing velocity. |

Note: dimension ranges are illustrative. In practice, transformer-based embeddings distribute information across all dimensions.

### Manifold Structure

- Embeddings concentrate on lower-dimensional manifolds (intrinsic dimensionality 50 to 200) within the full space.
- Drift can change manifold shape, position, or topology, not just simple distributional shifts.
- Kernel-based detection methods outperform dimension-independent methods for this reason.

---

## The Dual-Layer Architecture

### Architecture Flowchart

```
+-------------------+     +--------------------+     +---------------------+
| Incoming          |     | Layer 1:           |     | Decision Router     |
| Transaction       | --> | Primary ML Model   | --> | (score thresholds)  |
| (all volume)      |     | (XGBoost on        |     |                     |
|                   |     |  structured        |     | score < 0.3: APPROVE|
|                   |     |  features, ~5ms)   |     | score > 0.7: DECLINE|
+-------------------+     +--------------------+     +-----+-------+-------+
                                                            |       |
                                     +----------------------+       |
                                     | score 0.3 to 0.7,           |
                                     | high-value, or               |
                                     | explainability-required      |
                                     v                              v
                          +---------------------+         +------------------+
                          | Layer 2:            |         | Decisioned       |
                          | RAG+LLM Complement  |         | (85-95% of       |
                          |                     |         |  transactions)   |
                          | 1. Embed transaction|         +------------------+
                          | 2. Retrieve k=20    |
                          |    similar cases    |
                          | 3. LLM synthesizes  |
                          |    risk assessment  |
                          | (~100-300ms)        |
                          +----------+----------+
                                     |
                                     v
                          +---------------------+
                          | Final Decision      |
                          | (approve/decline/   |
                          |  escalate + reason) |
                          +---------------------+
```

### Layer 1: Primary ML Model

- Gradient boosted trees on structured features: transaction attributes, velocity features, historical aggregates, network features, and derived risk indicators.
- Outputs fraud probability 0 to 1. Below 0.3 approved, above 0.7 declined or escalated.
- Handles 85 to 95 percent of transactions decisively.
- Feature drift monitored separately (PSI, performance tracking).

### Layer 2: RAG+LLM Complement

- Processes gray zone (0.3 to 0.7), high-value (above $10K), explainability-required, and novel-pattern transactions.
- Uses embeddings to retrieve similar historical fraud patterns, then an LLM synthesizes context into a risk assessment, explanation, and recommendation.
- Embedding drift is consequential because it degrades the system exactly where the ML model has acknowledged uncertainty.

---

## Embeddings in RAG-Based Fraud Detection

### Retrieval Phase

- Each routed transaction is embedded and queried against a vector database of labeled historical transactions (typically k=10 to 50 nearest neighbors).
- Entirely dependent on embedding quality. Drifted embeddings produce less relevant retrievals for exactly the cases that need it most.

```python
results = index.query(
    vector=transaction_embedding,
    top_k=20,
    filter={"mcc": {"$eq": transaction.mcc}, "days_ago": {"$lte": 90}},
    include_metadata=True
)
```

### Generation Phase

- Retrieved transactions (with fraud labels and investigation notes) are assembled into an LLM prompt for risk assessment.
- Wrong embedding neighborhood means irrelevant context and unreliable assessments.
- Since only uncertain transactions reach this layer, degraded embeddings remove the system's safety net.

---

## Concrete Example

Input:

```
Amount: $1,847.23 USD at TIFFANY & CO #0891 (MCC 5944, Jewelry)
Terminal: POS chip-read, card-present, Manhattan NY
Cardholder: 8-year tenure, $4,100/mo avg spend
Past 24h: 1 transaction (grocery, $67.50)
```

- Embedded via all-MiniLM-L6-v2 (default, 384 dimensions) or text-embedding-3-large (optional, 3072 dimensions) into a dense vector.
- Vector database query returns: similar legitimate luxury purchases from comparable profiles, known fraud at luxury retailers with stolen cards, and same-cardholder history at similar merchants.
- LLM weighs risk factors: high tenure, chip auth, and geographic consistency reduce risk. High amount at a luxury retailer (common fraud target) increases it.

---

## Embedding Quality and Detection Accuracy

- Embedding quality determines complement layer effectiveness, not the primary ML model.
- Because the complement layer handles the most uncertain transactions, quality has disproportionate impact on high-risk outcomes.

### Retrieval Precision

- High quality: precision at k=20 is 0.75 to 0.85 (15 to 17 of 20 results relevant).
- Degraded: precision drops to 0.40 to 0.55. Nearly half the context is noise.

### Decision Boundary Sharpness

- High quality: clear fraud/legitimate cluster separation, silhouette score 0.6 to 0.8.
- Drifted: clusters overlap. Silhouette below 0.3 means severe degradation.

### Downstream LLM Performance

- LLM accuracy degrades roughly linearly with retrieval precision.
- A 10-point retrieval precision drop (0.80 to 0.70) corresponds to a 3 to 5 point AUC-ROC decline.
- At scale (10B complement layer transactions per year), a 3-point AUC-ROC drop means hundreds of thousands of additional misclassified high-risk transactions.

---

## Dataset: Sparkov Credit Card Transactions

This project uses the Sparkov Credit Card Fraud Detection dataset as its primary data source for embedding generation, drift analysis, and pipeline evaluation.

- **Source:** Kaggle, kartik2112/fraud-detection.
- **Size:** Approximately 1.8 million simulated transactions (1.3M train, 0.5M test).
- **Customers:** Approximately 1,000 unique cardholders with realistic spending profiles.
- **Merchants:** Approximately 800 merchants across diverse categories.
- **Time span:** January 2019 through December 2020.
- **Fraud rate:** Approximately 0.58 percent (realistic class imbalance for credit card fraud).

### Sample Data

| trans_date_trans_time | merchant | category | amt | city | state | is_fraud |
|---|---|---|---|---|---|---|
| 2019-01-01 00:00:18 | fraud_Rippin | misc_net | $4.97 | Moravian Falls | NC | 0 |
| 2019-03-15 14:23:07 | fraud_Heller | grocery_pos | $107.23 | Columbus | OH | 0 |
| 2019-06-22 03:41:55 | fraud_Kutch | shopping_net | $1,298.64 | Miami | FL | 1 |

### Key Fields for Transaction Text Construction

| Field | Purpose | Example |
|---|---|---|
| `merchant` | Primary merchant identifier. | fraud_Rippin |
| `category` | Merchant category for semantic grouping. | grocery_pos |
| `amt` | Transaction amount in USD. | $107.23 |
| `city`, `state` | Geographic context. | Columbus, OH |
| `lat`, `long` | Precise location for proximity calculations. | 39.96, -82.99 |
| `trans_date_trans_time` | Timestamp for temporal feature extraction. | 2019-03-15 14:23:07 |
| `is_fraud` | Ground truth label for evaluation. | 0 |

- **Transaction text template:** Fields are serialized into natural language descriptions (for example, "Transaction: $107.23 USD at fraud_Heller (category: grocery_pos), Location: Columbus, OH") before being passed to the embedding model.
- The seasonal and geographic patterns in Sparkov data produce natural embedding drift when comparing temporal windows, making it well suited for drift detection experiments.

---

## Summary

- Embeddings power the RAG complement layer, the safety net for gray zone, high-value, and explainability-required transactions.
- Embedding quality directly determines retrieval quality, which determines whether the complement layer adds signal or noise.
- Understanding how embeddings encode transaction data and what geometric properties they depend on is prerequisite to understanding how drift degrades them.
