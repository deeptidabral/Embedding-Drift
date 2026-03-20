"""
Evidently AI Drift Reporting Demo

Demonstrates how to use Evidently AI to generate standalone HTML drift
reports for the dual-layer fraud detection pipeline.  This example covers
three scenarios:

1. Embedding drift analysis for the RAG complement layer
2. Feature drift analysis for the ML model input features
3. Dual-layer combined report detecting compound drift

All data is synthetic.  In production, reference data would come from a
validated baseline period (for example, the first 90 days after model
deployment), and production data would be sampled from the live
transaction stream.

For more realistic results, download the Sparkov Credit Card Fraud
Detection dataset (https://www.kaggle.com/datasets/kartik2112/fraud-detection)
into the data/ directory. The Sparkov dataset provides 1.8M simulated
transactions with realistic feature distributions that better exercise
Evidently's statistical drift tests.

Usage:
    python examples/evidently_drift_report_demo.py

Reports are saved to reports/evidently/ as standalone HTML files that
can be opened in any browser and shared with stakeholders.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.monitoring.evidently_reporter import EvidentlyDriftReporter

# ---------------------------------------------------------------------------
# Local embedding generator (falls back to synthetic if unavailable)
# ---------------------------------------------------------------------------

_USE_REAL_EMBEDDINGS = False
_embedding_generator = None

try:
    from src.embeddings.generator import LocalEmbeddingGenerator
    _embedding_generator = LocalEmbeddingGenerator()
    _USE_REAL_EMBEDDINGS = True
except (ImportError, Exception) as _exc:
    import warnings
    warnings.warn(
        f"sentence-transformers not available ({_exc}). "
        "Falling back to synthetic embedding vectors. Install with: "
        "pip install sentence-transformers",
        stacklevel=1,
    )

_SPARKOV_TRAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "fraudTrain.csv"
_EMBEDDING_DIM = 384 if _USE_REAL_EMBEDDINGS else 64


def _transaction_to_text(row: dict) -> str:
    """Convert a Sparkov row into a text description for embedding."""
    return (
        f"{row.get('merchant', 'Unknown')} "
        f"{row.get('category', 'misc')} "
        f"${row.get('amt', 0):.2f} "
        f"{row.get('city', '')}, {row.get('state', '')} "
        f"card {str(row.get('cc_num', ''))[-4:]}"
    )


def _try_load_sparkov_embeddings(
    n_ref: int = 1000, n_prod: int = 500, seed: int = 42
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load Sparkov data and produce real embeddings for reference and production.

    Returns (ref_embeddings, prod_embeddings) or (None, None) when
    the dataset or sentence-transformers is not available.
    """
    if not _USE_REAL_EMBEDDINGS or _embedding_generator is None:
        return None, None
    if not _SPARKOV_TRAIN_PATH.exists():
        return None, None

    print("  Loading Sparkov data for real embeddings...")
    df = pd.read_csv(_SPARKOV_TRAIN_PATH, nrows=15_000)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values("trans_date_trans_time")

    split_idx = int(len(df) * 0.7)
    ref_df = df.iloc[:split_idx].sample(n=min(n_ref, split_idx), random_state=seed)
    prod_df = df.iloc[split_idx:].sample(n=min(n_prod, len(df) - split_idx), random_state=seed)

    ref_texts = [_transaction_to_text(row) for _, row in ref_df.iterrows()]
    prod_texts = [_transaction_to_text(row) for _, row in prod_df.iterrows()]

    ref_emb = _embedding_generator.encode(ref_texts)
    prod_emb = _embedding_generator.encode(prod_texts)

    print(f"  Real reference embeddings: {ref_emb.shape}")
    print(f"  Real production embeddings: {prod_emb.shape}")
    return ref_emb, prod_emb


# ---------------------------------------------------------------------------
# Data generators (real embeddings when possible, synthetic fallback)
# ---------------------------------------------------------------------------


def generate_reference_embeddings(
    n_samples: int = 1000, dim: int = _EMBEDDING_DIM, seed: int = 42
) -> np.ndarray:
    """Generate stable reference embeddings.

    Uses real sentence-transformer embeddings from Sparkov data when
    available.  Falls back to synthetic two-cluster vectors otherwise.
    """
    ref, _ = _try_load_sparkov_embeddings(n_ref=n_samples, seed=seed)
    if ref is not None:
        return ref

    rng = np.random.default_rng(seed)
    n_legit = int(n_samples * 0.95)
    n_fraud = n_samples - n_legit

    legitimate = rng.standard_normal((n_legit, dim)) * 0.5
    fraud_center = np.full(dim, 2.0)
    fraudulent = rng.standard_normal((n_fraud, dim)) * 0.3 + fraud_center

    embeddings = np.vstack([legitimate, fraudulent])
    rng.shuffle(embeddings)
    return embeddings.astype(np.float32)


def generate_production_embeddings(
    n_samples: int = 500,
    dim: int = _EMBEDDING_DIM,
    drift_magnitude: float = 0.0,
    seed: int = 99,
) -> np.ndarray:
    """Generate production embeddings with configurable drift.

    Uses real embeddings (with optional shift applied) when available.
    Falls back to synthetic otherwise.
    """
    _, prod = _try_load_sparkov_embeddings(n_prod=n_samples, seed=seed)
    if prod is not None:
        return prod + drift_magnitude

    rng = np.random.default_rng(seed)
    n_legit = int(n_samples * 0.95)
    n_fraud = n_samples - n_legit

    shift = np.full(dim, drift_magnitude)
    legitimate = rng.standard_normal((n_legit, dim)) * 0.5 + shift
    fraud_center = np.full(dim, 2.0) + shift * 1.5
    fraudulent = rng.standard_normal((n_fraud, dim)) * 0.3 + fraud_center

    embeddings = np.vstack([legitimate, fraudulent])
    rng.shuffle(embeddings)
    return embeddings.astype(np.float32)


def generate_reference_features(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate baseline ML model input features.

    Represents the feature distributions during the training/validation
    period of the XGBoost fraud scorer.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "amount": rng.exponential(200, size=n_samples),
        "hour_of_day": rng.integers(0, 24, size=n_samples),
        "day_of_week": rng.integers(0, 7, size=n_samples),
        "txn_count_30d": rng.integers(1, 60, size=n_samples),
        "avg_amount_30d": rng.exponential(180, size=n_samples),
        "days_since_last_txn": rng.exponential(3, size=n_samples),
        "amount_deviation_ratio": rng.standard_normal(n_samples) * 0.5 + 1.0,
        "is_new_merchant": rng.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
        "is_high_risk_country": rng.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        "is_recurring": rng.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
    })


def generate_production_features(
    n_samples: int = 500,
    drift: bool = False,
    seed: int = 99,
) -> pd.DataFrame:
    """Generate production features, optionally with distribution shift.

    When drift=True, simulates a scenario where transaction patterns
    have changed: higher amounts, more new merchants, more high-risk
    countries (consistent with an emerging fraud campaign).
    """
    rng = np.random.default_rng(seed)

    if drift:
        return pd.DataFrame({
            "amount": rng.exponential(800, size=n_samples),
            "hour_of_day": rng.integers(0, 6, size=n_samples),
            "day_of_week": rng.integers(5, 7, size=n_samples),
            "txn_count_30d": rng.integers(1, 10, size=n_samples),
            "avg_amount_30d": rng.exponential(700, size=n_samples),
            "days_since_last_txn": rng.exponential(15, size=n_samples),
            "amount_deviation_ratio": rng.standard_normal(n_samples) * 1.5 + 3.0,
            "is_new_merchant": rng.choice([0, 1], size=n_samples, p=[0.4, 0.6]),
            "is_high_risk_country": rng.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
            "is_recurring": rng.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        })

    return pd.DataFrame({
        "amount": rng.exponential(200, size=n_samples),
        "hour_of_day": rng.integers(0, 24, size=n_samples),
        "day_of_week": rng.integers(0, 7, size=n_samples),
        "txn_count_30d": rng.integers(1, 60, size=n_samples),
        "avg_amount_30d": rng.exponential(180, size=n_samples),
        "days_since_last_txn": rng.exponential(3, size=n_samples),
        "amount_deviation_ratio": rng.standard_normal(n_samples) * 0.5 + 1.0,
        "is_new_merchant": rng.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
        "is_high_risk_country": rng.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        "is_recurring": rng.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
    })


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------


def demo_embedding_drift_report(reporter: EvidentlyDriftReporter) -> None:
    """Scenario 1: RAG complement layer embedding drift analysis."""
    print("=" * 72)
    print("SCENARIO 1: Embedding Drift in the RAG Complement Layer")
    print("=" * 72)
    print()
    print("Context: The RAG complement layer retrieves historical fraud")
    print("patterns for gray zone transactions (ML score 0.3-0.7) and")
    print("high-value transactions (above $10,000). Embedding drift in")
    print("this layer degrades retrieval quality for exactly the")
    print("transactions where the ML model is least confident.")
    print()

    ref = generate_reference_embeddings(n_samples=1000, dim=_EMBEDDING_DIM)
    prod_stable = generate_production_embeddings(dim=_EMBEDDING_DIM, drift_magnitude=0.0)
    prod_drifted = generate_production_embeddings(dim=_EMBEDDING_DIM, drift_magnitude=1.5)

    print("--- Stable production embeddings (no drift) ---")
    summary_stable = reporter.generate_embedding_drift_report(
        ref, prod_stable, save_html=True, report_name="embedding_no_drift"
    )
    print(f"  Drift detected: {summary_stable.embedding_drift_detected}")
    print(f"  Drift score:    {summary_stable.embedding_drift_score:.4f}")
    print(f"  Components drifted: {summary_stable.number_of_drifted_columns}"
          f" / {summary_stable.number_of_columns}")
    if summary_stable.html_report_path:
        print(f"  HTML report: {summary_stable.html_report_path}")
    print()

    print("--- Drifted production embeddings (magnitude=1.5) ---")
    summary_drift = reporter.generate_embedding_drift_report(
        ref, prod_drifted, save_html=True, report_name="embedding_with_drift"
    )
    print(f"  Drift detected: {summary_drift.embedding_drift_detected}")
    print(f"  Drift score:    {summary_drift.embedding_drift_score:.4f}")
    print(f"  Components drifted: {summary_drift.number_of_drifted_columns}"
          f" / {summary_drift.number_of_columns}")
    if summary_drift.html_report_path:
        print(f"  HTML report: {summary_drift.html_report_path}")
    print()

    # Bridge to project DriftReport
    report = reporter.to_drift_report(summary_drift)
    print(f"  Mapped severity: {report.overall_severity.value}")
    print(f"  Actions: {report.recommended_actions[0]}")
    print()


def demo_feature_drift_report(reporter: EvidentlyDriftReporter) -> None:
    """Scenario 2: ML model feature drift analysis."""
    print("=" * 72)
    print("SCENARIO 2: ML Model Feature Drift (XGBoost Input Features)")
    print("=" * 72)
    print()
    print("Context: The primary XGBoost fraud scorer operates on structured")
    print("transaction features. When these feature distributions shift,")
    print("the model's fraud probability scores become less calibrated.")
    print("This is covariate shift affecting the primary scoring layer.")
    print()

    ref_features = generate_reference_features(n_samples=1000)
    prod_stable = generate_production_features(n_samples=500, drift=False)
    prod_drifted = generate_production_features(n_samples=500, drift=True)

    numerical = [
        "amount", "hour_of_day", "day_of_week", "txn_count_30d",
        "avg_amount_30d", "days_since_last_txn", "amount_deviation_ratio",
    ]
    categorical = ["is_new_merchant", "is_high_risk_country", "is_recurring"]

    print("--- Stable production features (no drift) ---")
    summary_stable = reporter.generate_feature_drift_report(
        ref_features, prod_stable,
        numerical_features=numerical,
        categorical_features=categorical,
        save_html=True,
        report_name="features_no_drift",
    )
    print(f"  Overall drift: {summary_stable.overall_drift_detected}")
    print(f"  Share drifted: {summary_stable.share_drifted:.2%}")
    print(f"  Drifted features: {summary_stable.drifted_features or 'none'}")
    print()

    print("--- Drifted production features (fraud campaign scenario) ---")
    summary_drift = reporter.generate_feature_drift_report(
        ref_features, prod_drifted,
        numerical_features=numerical,
        categorical_features=categorical,
        save_html=True,
        report_name="features_with_drift",
    )
    print(f"  Overall drift: {summary_drift.overall_drift_detected}")
    print(f"  Share drifted: {summary_drift.share_drifted:.2%}")
    print(f"  Drifted features: {summary_drift.drifted_features}")
    print()

    if summary_drift.feature_drift_scores:
        print("  Per-feature drift scores:")
        for feat, score in sorted(
            summary_drift.feature_drift_scores.items(),
            key=lambda x: x[1],
        ):
            marker = " [DRIFTED]" if feat in summary_drift.drifted_features else ""
            print(f"    {feat:30s} {score:.4f}{marker}")
    print()


def demo_dual_layer_report(reporter: EvidentlyDriftReporter) -> None:
    """Scenario 3: Dual-layer compound drift analysis."""
    print("=" * 72)
    print("SCENARIO 3: Dual-Layer Compound Drift Analysis")
    print("=" * 72)
    print()
    print("Context: The most dangerous scenario for a payment processor is")
    print("when both layers drift simultaneously. The ML model produces")
    print("unreliable scores AND the RAG complement layer provides degraded")
    print("analysis for the gray zone. This is compound drift, and it")
    print("requires immediate intervention.")
    print()

    ref_emb = generate_reference_embeddings(n_samples=1000, dim=_EMBEDDING_DIM)
    ref_feat = generate_reference_features(n_samples=1000)

    scenarios = [
        ("Both stable", 0.0, False),
        ("Embedding drift only (RAG degraded)", 1.5, False),
        ("Feature drift only (ML degraded)", 0.0, True),
        ("Compound drift (CRITICAL)", 1.5, True),
    ]

    for name, emb_drift, feat_drift in scenarios:
        print(f"--- {name} ---")
        prod_emb = generate_production_embeddings(
            dim=_EMBEDDING_DIM, drift_magnitude=emb_drift
        )
        prod_feat = generate_production_features(
            n_samples=500, drift=feat_drift
        )

        result = reporter.generate_dual_layer_report(
            ref_emb, prod_emb, ref_feat, prod_feat,
            save_html=True,
        )

        emb_summary = result["embedding_summary"]
        feat_summary = result["feature_summary"]

        print(f"  Embedding drift detected: {emb_summary.embedding_drift_detected}")
        print(f"  Feature drift detected:   {feat_summary.overall_drift_detected}")
        print(f"  Compound drift:           {result['compound_drift_detected']}")
        print(f"  Assessment: {result['risk_assessment'][:120]}...")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _plot_drift_summary(reports_dir: Path) -> None:
    """Generate a grouped bar chart summarizing drift scores across all scenarios."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Collect drift scores from each scenario by re-running the generators
        # (lightweight since the data is small and synthetic)
        ref_emb = generate_reference_embeddings(n_samples=1000, dim=_EMBEDDING_DIM)
        ref_feat = generate_reference_features(n_samples=1000)

        scenarios = {
            "Both Stable": (0.0, False),
            "Embedding\nDrift Only": (1.5, False),
            "Feature\nDrift Only": (0.0, True),
            "Compound\nDrift": (1.5, True),
        }

        emb_scores = []
        feat_shares = []
        labels = list(scenarios.keys())

        for name, (emb_mag, feat_drift) in scenarios.items():
            prod_emb = generate_production_embeddings(dim=_EMBEDDING_DIM, drift_magnitude=emb_mag)
            prod_feat = generate_production_features(n_samples=500, drift=feat_drift)

            # Compute simple drift proxies (mean L2 distance for embeddings,
            # fraction of features with significant KS stat for features)
            emb_diff = np.linalg.norm(ref_emb.mean(axis=0) - prod_emb.mean(axis=0))
            emb_scores.append(float(emb_diff))

            # Feature drift: fraction of columns where KS > 0.1
            from scipy.stats import ks_2samp
            n_drifted = 0
            for col in ref_feat.columns:
                stat, _ = ks_2samp(ref_feat[col].values, prod_feat[col].values)
                if stat > 0.1:
                    n_drifted += 1
            feat_shares.append(n_drifted / len(ref_feat.columns))

        x = np.arange(len(labels))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(10, 6))

        bars1 = ax1.bar(x - width / 2, emb_scores, width, label="Embedding Drift (L2 centroid shift)",
                        color="#2196F3", edgecolor="white")
        ax1.set_ylabel("Embedding Centroid L2 Distance", color="#2196F3")
        ax1.tick_params(axis="y", labelcolor="#2196F3")

        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width / 2, feat_shares, width, label="Feature Drift (share drifted)",
                        color="#FF9800", edgecolor="white")
        ax2.set_ylabel("Share of Features Drifted", color="#FF9800")
        ax2.tick_params(axis="y", labelcolor="#FF9800")
        ax2.set_ylim(0, 1.0)

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_title("Drift Scores Across All Evidently Scenarios")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        out_path = reports_dir / "drift_summary_bar_chart.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Drift summary bar chart saved to {out_path}")
    except ImportError:
        print("matplotlib not installed -- skipping summary chart.")


def main() -> None:
    """Run all three Evidently drift reporting scenarios."""
    print()
    print("Evidently AI Drift Reporting for Dual-Layer Fraud Detection")
    print("Payment Processor Context")
    print()
    print("This demo generates standalone HTML reports saved to")
    print("reports/evidently/ that can be opened in any browser.")
    print()

    if _USE_REAL_EMBEDDINGS:
        print("Using real sentence-transformer embeddings (all-MiniLM-L6-v2, 384 dims).")
        if _SPARKOV_TRAIN_PATH.exists():
            print(f"Sparkov data found at {_SPARKOV_TRAIN_PATH}.")
        else:
            print("Sparkov data not found; real embeddings will use random text fallback.")
    else:
        print("Falling back to synthetic embedding vectors (install sentence-transformers for real embeddings).")
    print()

    reports_dir = Path("reports/evidently")
    reporter = EvidentlyDriftReporter(
        reports_dir=reports_dir,
        embedding_dim=_EMBEDDING_DIM,
    )

    demo_embedding_drift_report(reporter)
    demo_feature_drift_report(reporter)
    demo_dual_layer_report(reporter)

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print("Evidently AI complements LangSmith in the monitoring stack:")
    print()
    print("  LangSmith  -- Real-time LLM trace observability. Tracks every")
    print("                 RAG retrieval and LLM assessment. Custom drift")
    print("                 feedback scores on individual runs.")
    print()
    print("  Evidently  -- Periodic deep-dive drift analysis. Generates")
    print("                 shareable HTML reports with statistical details.")
    print("                 Covers both embedding drift (RAG layer) and")
    print("                 feature drift (ML model). Detects compound drift.")
    print()
    print("In a production pipeline at a payment processor, LangSmith runs")
    print("continuously while Evidently reports are generated hourly or daily")
    print("for trend analysis and stakeholder communication.")
    print()

    # Generate grouped bar chart summarizing drift across all scenarios
    _plot_drift_summary(reports_dir)
    print()

    html_files = list(reports_dir.glob("*.html"))
    if html_files:
        print(f"Generated {len(html_files)} HTML reports in {reports_dir}/:")
        for f in sorted(html_files):
            print(f"  - {f.name}")
    print()


if __name__ == "__main__":
    main()
