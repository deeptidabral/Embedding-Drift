# Embedding Drift Monitoring for ML+LLM-powered Fraud Detection

Real-time fraud detection at global payment processors increasingly relies on two layers: an XGBoost model for fast primary scoring and a RAG+LLM pipeline for edge cases, explainability, and audit compliance. I have used the **Sparkov Credit Card Fraud Detection** dataset (1.8M simulated transactions, ~1,000 customers, ~800 merchants) from Kaggle as the primary data source for this project to conduct experimentation and showcase embedding drift analysis. In this repo, I explore approaches to detect and mitigate embedding drift in the RAG layer and feature/data drift in the ML model, both of which can slowly degrade model performance when left unchecked.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Core Topics Covered](#2-core-topics-covered)
3. [Architecture Overview](#3-architecture-overview)
4. [Repository Structure](#4-repository-structure)
5. [How to Navigate This Repo](#5-how-to-navigate-this-repo)
6. [Notebooks](#6-notebooks)
7. [Quick Start](#7-quick-start)
8. [Technology Stack](#8-technology-stack)
9. [References](#9-references)
10. [License](#10-license)

---

## 1: Problem Statement

- Payment processors, card networks and issuers handle millions of daily transactions worldwide.
- Every transaction scoring and processing is a low-latency workflow where the core ML model (XGBoost/LightGBM) scores every transaction in sub-10ms, providing clear authorization decisions (approvals and declines). Payment processors operate under sub-50ms to sub-100ms latency SLAs for authorization, so the ML model alone drives the real-time decision.
- A significant fraction of transactions fall into a gray zone (score 0.3-0.7) where the model is uncertain, or are large (above >= $10K), or require explainability for stakeholders such as auditors and investigators. These transactions are flagged for manual review by human analysts.
- A RAG+LLM layer operates **asynchronously, after the authorization decision**, to support post-transaction investigation. It retrieves historical fraud patterns from a vector database, performs pattern matching, and synthesizes contextual risk assessments with full documented reasoning. The LLM never blocks a transaction decision. It assists human analysts in reviewing flagged cases, generates explainability reports for audit compliance, supports chargeback defense, and produces regulatory audit documentation.

**Why drift monitoring matters:**

- Silent performance degradation: Reporting dashboards could stay green, latency be normal, error rates are flat, but decisions get worse
- Feature drift in the ML model (covariate shift, concept drift, target shift) erodes primary scoring accuracy, directly impacting real-time authorization quality
- Embedding drift in the RAG layer causes retrieval quality to degrade, feeding the LLM lower-quality context for post-transaction investigation. This leads to degraded investigation quality, missed fraud in manual review queues, weaker chargeback defense, and unreliable audit documentation
- Undetected false negatives can cost tens of millions per fraud ring; false positives cost the industry an estimated $440B+ annually in declined legitimate transactions
- Under PCI DSS and SOX, undetected drift in either layer is a control failure

This repository provides approaches for detection theory, statistical tooling, monitoring infrastructure, and mitigation strategies for both layers.

---

## 2: Core Topics Covered

### Topic 1: Embeddings Fundamentals
- How raw transaction data (merchant codes, amounts, geolocation, temporal features) is transformed into dense vectors for the RAG layer
- How those embeddings enable retrieval of similar historical fraud patterns for LLM-based analysis
- **Important note on embedding approach:** This project uses text embedding models (sentence-transformers) as a simplified demonstration of drift monitoring concepts. Text embeddings are not the right tool for encoding structured tabular transaction data (amounts, MCCs, coordinates) in production -- they are poor at capturing numerical magnitudes and geospatial relationships. Production fraud detection systems use Deep Entity Embeddings, Autoencoders, or Graph Neural Networks (GNNs) to embed tabular transaction entities. The drift monitoring framework demonstrated here applies equally to any embedding type.
- See: `docs/01_embeddings_fundamentals.md`

### Topic 2: Embedding Drift Theory
- Formal definition of drift as distributional shift in embedding space
- Four drift types in fraud detection: sudden (model updates/pipeline breaks), gradual (evolving consumer behavior), incremental (slow merchant ecosystem changes), recurring (seasonal spending)
- See: `docs/02_embedding_drift_theory.md`

### Topic 3: Why Monitor Drift
- Business case: quantified financial impact at processor scale -- direct fraud losses from false negatives, revenue losses from false positive declines
- Regulatory implications under PCI DSS and SOX
- See: `docs/03_why_monitor_drift.md`

### Topic 4: Monitoring with LangSmith
- LangSmith as the observability backend for the async RAG+LLM investigation pipeline (not the real-time authorization path)
- How post-transaction LLM investigation runs are traced, drift metrics posted as feedback scores, and dashboard displays drift trends alongside investigation quality
- See: `docs/04_monitoring_with_langsmith.md`

### Topic 5: Drift Metrics and Thresholds
- Statistical methods: MMD, KS tests, cosine similarity monitoring, PCA-based drift detection, PSI
- Threshold selection guidance for each method in the dual-layer architecture
- See: `docs/05_drift_metrics_and_thresholds.md`

### Topic 6: Drift Interactions
- How embedding drift and ML feature drift interact -- amplification and masking effects
- Strategies for correlated monitoring across both layers
- See: `docs/06_drift_interactions.md`

---

## 3: Architecture Overview

### Dual-Layer Fraud Detection Pipeline

- Every transaction is scored by the ML model first (XGBoost, sub-10ms). The ML model alone drives the real-time authorization decision within the payment processor's sub-50ms latency SLA.
- A Decision Router determines the authorization outcome and whether post-transaction investigation is needed:
  - **Clear decisions (score < 0.3 or > 0.7):** Approved or declined on ML score alone
  - **Gray zone (score 0.3-0.7):** Authorized based on ML score (with conservative thresholds), then flagged for manual review
  - **High-value (> $10K):** Authorized based on ML score, then flagged for post-transaction investigation regardless of score
- The RAG+LLM layer operates **asynchronously after authorization**. It processes flagged transactions to assist human analysts in manual review queues, generate explainability reports, support chargeback defense, and produce regulatory audit documentation. LLM API calls take hundreds of milliseconds to seconds, making them incompatible with real-time authorization latency requirements.

### Decision Flow

```
                    REAL-TIME AUTHORIZATION PATH (sub-50ms SLA)
                    ============================================

Transaction -> ML Model (XGBoost, sub-10ms) -> fraud_score -> Decision Router
                                                                   |
                          +----------------------------------------+------------------+
                          |                                                           |
                   Clear decision                                         Gray zone / High value
                   (score < 0.3 or > 0.7)                                            |
                          |                                                           v
                          v                                              Authorization decision
                   Authorization decision                                (approve/decline with
                   (approve/decline)                                      conservative thresholds)
                                                                                      |
                                                                                      v
                                                                         Flag for manual review
                                                                                      |
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +- -
                                                                                      |
                    ASYNC INVESTIGATION PATH (no latency SLA)                         |
                    ==========================================                        |
                                                                                      v
                                                                         Async queue -> RAG+LLM
                                                                         (retrieve + assess)
                                                                                      |
                                                                                      v
                                                                         Investigation report
                                                                         (reasoning, audit trail,
                                                                          analyst recommendations)
```

### Drift Monitoring Plane

- **ML Model Feature/Data Drift:** Monitors input feature distributions (transaction amount, merchant category, geographic patterns, temporal features) for covariate shift, concept drift, and target shift against training data
- **RAG Embedding Drift:** Compares incoming embedding distributions against a validated baseline using MMD, KS tests, cosine similarity monitoring, and PCA drift detection
- **Graduated alert sequence:** Low drift triggers logging; moderate drift alerts the ML platform team; high drift triggers automatic fallback and escalation to fraud ops leadership
- All drift events are logged for compliance reporting

---

## 4: Repository Structure

```
embedding-drift/
|
|-- README.md
|-- LICENSE
|-- pyproject.toml
|-- requirements.txt
|-- .gitignore
|
|-- configs/
|   |-- drift_thresholds.yaml
|   |-- monitoring.yaml
|   |-- pipeline.yaml
|
|-- data/
|   |-- README.md                  # Download instructions for Sparkov dataset
|   |-- fraudTrain.csv             # Training set (~1.3M transactions, not committed)
|   |-- fraudTest.csv              # Test set (~0.5M transactions, not committed)
|
|-- docs/
|   |-- 01_embeddings_fundamentals.md
|   |-- 02_embedding_drift_theory.md
|   |-- 03_why_monitor_drift.md
|   |-- 04_monitoring_with_langsmith.md
|   |-- 05_drift_metrics_and_thresholds.md
|   |-- 06_drift_interactions.md
|   |-- architecture/
|       |-- system_design.md
|
|-- notebooks/
|   |-- 01_data_exploration.ipynb          # Sparkov dataset EDA and statistics
|   |-- 02_embedding_generation.ipynb      # Transaction text construction and embedding
|   |-- 03_drift_detection_analysis.ipynb  # Drift metric computation and comparison
|   |-- 04_visualization_gallery.ipynb     # All visualization types in one notebook
|   |-- 04_dual_layer_monitoring.ipynb     # Dual-layer pipeline and Evidently reports
|
|-- src/
|   |-- __init__.py
|   |-- data/
|   |   |-- __init__.py
|   |   |-- loader.py              # SparkovDataLoader for CSV ingestion and preprocessing
|   |
|   |-- embeddings/
|   |   |-- __init__.py
|   |   |-- generator.py
|   |   |-- store.py
|   |
|   |-- drift_detection/
|   |   |-- __init__.py
|   |   |-- detectors.py
|   |   |-- metrics.py
|   |
|   |-- data_drift/
|   |   |-- __init__.py
|   |   |-- concept_shift.py
|   |   |-- covariate_shift.py
|   |   |-- target_shift.py
|   |
|   |-- fraud_detection/
|   |   |-- __init__.py
|   |   |-- ml_scorer.py
|   |   |-- pipeline.py
|   |   |-- rag_retriever.py
|   |   |-- transaction_processor.py
|   |
|   |-- visualization/
|   |   |-- __init__.py
|   |   |-- plots.py               # Drift metrics, embedding space, and score distribution plots
|   |   |-- schematics.py          # Architecture and threshold diagrams
|   |
|   |-- monitoring/
|       |-- __init__.py
|       |-- alerts.py
|       |-- dashboard.py
|       |-- evidently_reporter.py
|       |-- langsmith_reporter.py
|
|-- examples/
|   |-- drift_monitoring_demo.py
|   |-- evidently_drift_report_demo.py
|   |-- fraud_detection_demo.py
|
|-- tests/
|   |-- __init__.py
|   |-- test_detectors.py
|   |-- test_metrics.py
|   |-- test_pipeline.py
```

---

## 5: How to Navigate This Repo

If you are new to this project, follow this path:

### Start here

- Read this **README** for the full picture: what the project does, why it matters, how everything fits together

### Understand the concepts (docs/)

- `docs/01_embeddings_fundamentals.md` -- What are embeddings and why fraud detection needs them
- `docs/02_embedding_drift_theory.md` -- What goes wrong when embeddings drift
- `docs/03_why_monitor_drift.md` -- The business cost of ignoring drift
- `docs/architecture/system_design.md` -- How the dual-layer pipeline is designed
- `docs/04_monitoring_with_langsmith.md` -- How to monitor in production (LangSmith + Evidently)
- `docs/05_drift_metrics_and_thresholds.md` -- Which metrics to use and what thresholds trigger action
- `docs/06_drift_interactions.md` -- How embedding drift compounds with other drift types
### Get hands-on (notebooks/)

Download the Sparkov dataset first (see Quick Start), then run these in order:

1. `notebooks/01_data_exploration.ipynb` -- Explore the transaction data, see fraud patterns
2. `notebooks/02_embedding_analysis.ipynb` -- Visualize embeddings with t-SNE and PCA
3. `notebooks/03_drift_detection.ipynb` -- Detect drift with 5 metrics, see threshold bands
4. `notebooks/04_dual_layer_monitoring.ipynb` -- ML + LLM pipeline in action, Evidently reports

### Explore the source code (src/)

- `src/data/loader.py` -- How raw transactions become embeddable text
- `src/fraud_detection/ml_scorer.py` -- The primary ML model (XGBoost interface)
- `src/fraud_detection/pipeline.py` -- Dual-layer orchestration (ML first, then conditionally RAG+LLM)
- `src/drift_detection/metrics.py` -- The 5 drift metrics (cosine, MMD, KS, Wasserstein, PSI)
- `src/drift_detection/detectors.py` -- Ensemble detector with severity classification
- `src/monitoring/evidently_reporter.py` -- Evidently AI integration for HTML reports
- `src/visualization/plots.py` -- 14 reusable chart functions for drift and embedding analysis

### Run the examples (examples/)

- `examples/fraud_detection_demo.py` -- End-to-end fraud scoring with dual-layer routing
- `examples/drift_monitoring_demo.py` -- Watch drift escalate through severity levels
- `examples/evidently_drift_report_demo.py` -- Generate shareable HTML drift reports
### Review quality and configuration (tests/, configs/)

- `tests/` -- How every component is tested
- `configs/drift_thresholds.yaml` -- Threshold values and automated response actions
- `configs/pipeline.yaml` -- Full pipeline configuration for the dual-layer system

---

## 6: Notebooks

The `notebooks/` directory contains Jupyter notebooks that provide interactive walkthroughs of the project's key components, all using the Sparkov dataset when available.

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Exploratory data analysis of the Sparkov dataset: transaction volume over time, fraud rate by category, amount distributions, and geographic patterns across 1.8M transactions. |
| `02_embedding_generation.ipynb` | Demonstrates how raw Sparkov transaction fields (merchant, category, amount, location, timestamp) are serialized into natural language descriptions and embedded into dense vectors for the RAG layer. |
| `03_drift_detection_analysis.ipynb` | Computes and compares all five drift metrics (MMD, KS, cosine distance, Wasserstein, PSI) on Sparkov embeddings across temporal windows, illustrating how seasonal spending patterns manifest as measurable drift. |
| `04_dual_layer_monitoring.ipynb` | Dual-layer pipeline demo: ML model scoring, decision routing, Evidently drift reports, and compound drift assessment across both layers. |

---

## 7: Quick Start

### Prerequisites

- Python 3.10+
- No API keys required. Runs locally using sentence-transformers.
- ChromaDB (installed automatically via requirements.txt)

### Installation

```bash
git clone https://github.com/your-org/embedding-drift.git
cd embedding-drift
pip install -e ".[dev]"
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

**Note:** The project runs fully locally with no API keys required. The embedding model is sentence-transformers (all-MiniLM-L6-v2), which runs on your machine at no cost.

**Note on embedding approach:** This project uses text embedding models (sentence-transformers) as a simplified demonstration of drift monitoring concepts. In production, structured tabular transaction data (amounts, MCCs, coordinates) would be embedded using Deep Entity Embeddings, Autoencoders, or Graph Neural Networks (GNNs), which properly capture numerical magnitudes and geospatial relationships. The drift monitoring framework and statistical methods demonstrated here transfer directly to any embedding type.

### Download the Sparkov Dataset

The project uses the Sparkov Credit Card Fraud Detection dataset (1.8M simulated transactions). Download it via the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d kartik2112/fraud-detection -p data/ --unzip
```

This places `fraudTrain.csv` (~1.3M transactions) and `fraudTest.csv` (~0.5M transactions) into the `data/` directory. See `data/README.md` for manual download instructions and expected file structure.

### Configuration

```bash
cp configs/monitoring.yaml configs/monitoring.local.yaml
```

Edit `configs/monitoring.local.yaml`:

```yaml
ml_scorer:
  model_type: xgboost
  model_path: models/fraud_xgb_v3.json
  feature_columns: configs/feature_schema.yaml
  gray_zone:
    lower_bound: 0.3
    upper_bound: 0.7
  high_value_threshold: 10000

embedding_model:
  provider: sentence-transformers
  model_name: all-MiniLM-L6-v2
  dimension: 384

vector_database:
  provider: pinecone
  index_name: fraud-embeddings-prod
  environment: us-east-1

monitoring:
  ml_feature_drift:
    reference_window_days: 30
    detection_window_minutes: 60
  embedding_drift:
    reference_window_days: 30
    detection_window_minutes: 60
  alert_cooldown_minutes: 15

thresholds:
  embedding:
    mmd_warning: 0.05
    mmd_critical: 0.15
    cosine_mean_shift_warning: 0.02
    cosine_mean_shift_critical: 0.08
    ks_statistic_warning: 0.10
    ks_statistic_critical: 0.25
  ml_features:
    psi_warning: 0.10
    psi_critical: 0.25
    ks_statistic_warning: 0.10
    ks_statistic_critical: 0.25
```

### Running the Monitoring Pipeline

```bash
python -m src.monitoring.dashboard --config configs/monitoring.local.yaml
```

One-time drift assessment:

```bash
python -m src.drift_detection.detectors \
  --reference-start 2025-01-01 \
  --reference-end 2025-01-31 \
  --test-start 2025-02-01 \
  --test-end 2025-02-28 \
  --output reports/drift_report_feb2025.json
```

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### Running the Examples

```bash
python examples/fraud_detection_demo.py
python examples/drift_monitoring_demo.py
```

---

## 8: Technology Stack

| Component              | Technology                                    | Purpose                                           |
|------------------------|-----------------------------------------------|---------------------------------------------------|
| ML Scoring Model       | XGBoost, scikit-learn                         | Primary fraud probability scoring for all transactions |
| Embedding Models       | sentence-transformers (all-MiniLM-L6-v2)              | Transaction text to vector conversion for RAG layer |
| Vector Database        | ChromaDB                                      | Storage and retrieval of transaction embeddings    |
| LLM Inference          | Ollama (phi3:mini)                            | Async post-transaction investigation, analyst co-pilot for manual review, chargeback defense, and audit documentation |
| Statistical Computing  | NumPy, SciPy, scikit-learn                    | Drift detection computations for both layers       |
| Drift Monitoring       | Evidently AI, alibi-detect                    | Statistical drift reports and advanced detection   |
| LLM Observability      | LangSmith                                     | Async investigation pipeline tracing, evaluation, and dashboards |
| Alerting               | PagerDuty, Slack webhooks                     | Drift alert delivery                               |
| Data Processing        | Pandas                                        | Transaction data manipulation and feature engineering |
| Visualization          | Matplotlib, Seaborn                           | Embedding space, drift metrics, and pipeline visualization |
| Configuration          | Pydantic, PyYAML                              | Data validation, model schemas, and config management |
| Testing                | pytest                                        | Unit and integration testing                       |

---

## 9: References

### Academic Papers

1. Gretton et al. (2012). "A Kernel Two-Sample Test." JMLR 13, 723-773. Foundational paper on MMD -- the primary statistical test for embedding drift detection in high-dimensional spaces.

2. Rabanser et al. (2019). "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift." NeurIPS. Empirical comparison of drift detection methods including dimensionality reduction approaches.

3. Lu et al. (2019). "Learning under Concept Drift: A Review." IEEE TKDE 31(12), 2346-2363. Survey of concept drift types and detection methods informing the drift taxonomy used here.

4. Reimers and Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP. Key embedding architecture for encoding transaction descriptions.

5. Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS. Foundational RAG paper underpinning the retrieval-augmented fraud detection architecture.

6. Dal Pozzolo et al. (2014). "Learned Lessons in Credit Card Fraud Detection from a Practitioner Perspective." Expert Systems with Applications 41(10), 4915-4928. Practical lessons on distribution shift in deployed fraud detection models.

7. Chen and Guestrin (2016). "XGBoost: A Scalable Tree Boosting System." KDD, 785-794. The gradient boosted tree framework used as the primary ML scorer.

### Industry Standards and Frameworks

8. PCI Security Standards Council (2022). "PCI DSS v4.0." Requirements for fraud detection system monitoring and validation.

9. NIST (2023). "AI Risk Management Framework (AI RMF 1.0)." Guidance on production AI monitoring, including drift detection as a trustworthy AI component.

### Datasets

10. Kartik Shenoy (2022). "Sparkov Credit Card Fraud Detection." Kaggle. https://www.kaggle.com/datasets/kartik2112/fraud-detection -- 1.8M simulated credit card transactions (1,000 customers, 800 merchants, Jan 2019 to Dec 2020) generated using the Sparkov Data Generation tool by Brandon Harris. Used as the primary dataset for embedding generation, drift analysis, and pipeline evaluation.

### Tools and Libraries

11. Evidently AI. "Evidently: An open-source ML monitoring tool." https://github.com/evidentlyai/evidently

12. Alibi Detect. "Alibi Detect: Algorithms for outlier, adversarial, and drift detection." https://github.com/SeldonIO/alibi-detect

---

## 10: License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request. All contributions must include appropriate tests and documentation updates.
