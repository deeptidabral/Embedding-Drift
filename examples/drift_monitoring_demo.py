"""
Drift Monitoring Demo -- End-to-end example of embedding drift detection
for a dual-layer fraud detection system (ML model + RAG+LLM).

This script walks through the complete monitoring workflow:

    1. Create a reference embedding distribution (simulating a trained model).
    2. Compute all five drift metrics against a production window.
    3. Run the ensemble drift detector with configurable agreement logic.
    4. Simulate gradual drift over multiple time windows.
    5. Show how alerts fire at different severity levels.
    6. Demonstrate monitoring of both ML feature drift and RAG embedding drift.
    7. Show how drift in the RAG layer affects gray zone analysis quality.

In the dual-layer architecture:
  - ML feature drift degrades the primary scorer, affecting ALL transactions.
  - RAG embedding drift degrades pattern retrieval, affecting gray zone and
    high-value transactions that are escalated to the LLM layer.
  - Monitoring both drift types provides comprehensive pipeline health.

All data is synthetic.  No external services are required.

Run with:
    python examples/drift_monitoring_demo.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Local embedding generator (falls back to random vectors if unavailable)
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
        "Falling back to random embedding vectors. Install with: "
        "pip install sentence-transformers",
        stacklevel=1,
    )

_SPARKOV_TRAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "fraudTrain.csv"


def _transaction_to_text(row: dict) -> str:
    """Convert a transaction row into a natural-language string for embedding."""
    return (
        f"{row.get('merchant', 'Unknown')} "
        f"{row.get('category', 'misc')} "
        f"${row.get('amt', 0):.2f} "
        f"{row.get('city', '')}, {row.get('state', '')} "
        f"card {str(row.get('cc_num', ''))[-4:]}"
    )


# ---------------------------------------------------------------------------
# Configuration (mirrors configs/drift_thresholds.yaml)
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, dict[str, float]] = {
    "cosine_distance": {
        "nominal_upper": 0.05,
        "warning_upper": 0.15,
        "critical_upper": 0.30,
    },
    "maximum_mean_discrepancy": {
        "nominal_upper": 0.02,
        "warning_upper": 0.08,
        "critical_upper": 0.20,
    },
    "kolmogorov_smirnov": {
        "nominal_upper": 0.05,
        "warning_upper": 0.12,
        "critical_upper": 0.25,
    },
    "wasserstein_distance": {
        "nominal_upper": 0.03,
        "warning_upper": 0.10,
        "critical_upper": 0.22,
    },
    "population_stability_index": {
        "nominal_upper": 0.10,
        "warning_upper": 0.20,
        "critical_upper": 0.35,
    },
}

MIN_METRICS_AGREEING = 2

# ---------------------------------------------------------------------------
# Data generation -- real embeddings from Sparkov when available
# ---------------------------------------------------------------------------

DIM = 384 if _USE_REAL_EMBEDDINGS else 64
N_REF = 2000
N_PROD = 500

_sparkov_ref_cache: np.ndarray | None = None
_sparkov_prod_cache: np.ndarray | None = None


def _load_sparkov_embeddings() -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load Sparkov data and generate real embeddings split by time.

    Returns (reference_embeddings, production_embeddings) or (None, None)
    if the dataset is unavailable.
    """
    global _sparkov_ref_cache, _sparkov_prod_cache
    if _sparkov_ref_cache is not None:
        return _sparkov_ref_cache, _sparkov_prod_cache

    if not _USE_REAL_EMBEDDINGS or _embedding_generator is None:
        return None, None

    try:
        import pandas as pd
    except ImportError:
        return None, None

    if not _SPARKOV_TRAIN_PATH.exists():
        return None, None

    print("  Loading Sparkov data and generating real embeddings...")
    df = pd.read_csv(_SPARKOV_TRAIN_PATH, nrows=20_000)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values("trans_date_trans_time")

    # Split by time: first 70% as reference, last 30% as production
    split_idx = int(len(df) * 0.7)
    ref_df = df.iloc[:split_idx]
    prod_df = df.iloc[split_idx:]

    # Sample to target sizes
    rng = np.random.default_rng(42)
    ref_sample = ref_df.sample(n=min(N_REF, len(ref_df)), random_state=42)
    prod_sample = prod_df.sample(n=min(N_PROD, len(prod_df)), random_state=42)

    ref_texts = [_transaction_to_text(row) for _, row in ref_sample.iterrows()]
    prod_texts = [_transaction_to_text(row) for _, row in prod_sample.iterrows()]

    _sparkov_ref_cache = _embedding_generator.encode(ref_texts)
    _sparkov_prod_cache = _embedding_generator.encode(prod_texts)

    print(f"  Reference embeddings: {_sparkov_ref_cache.shape}")
    print(f"  Production embeddings: {_sparkov_prod_cache.shape}")
    return _sparkov_ref_cache, _sparkov_prod_cache


def create_reference_distribution(
    n: int = N_REF, d: int = DIM, seed: int = 0
) -> np.ndarray:
    """Generate a reference embedding distribution.

    When Sparkov data and sentence-transformers are available, returns real
    embeddings from reference-period transactions.  Otherwise falls back
    to synthetic random vectors.
    """
    ref, _ = _load_sparkov_embeddings()
    if ref is not None:
        return ref

    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


def create_production_window(
    shift: float = 0.0, n: int = N_PROD, d: int = DIM, seed: int = 1
) -> np.ndarray:
    """Generate a production embedding window with an optional mean shift.

    When real embeddings are available and shift is 0.0, returns the actual
    production-period embeddings.  For non-zero shifts, applies the shift
    to the real embeddings (or falls back to synthetic).
    """
    _, prod = _load_sparkov_embeddings()
    if prod is not None:
        return prod + shift

    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)) + shift


# ---------------------------------------------------------------------------
# ML feature drift simulation
# ---------------------------------------------------------------------------

N_ML_FEATURES = 12  # matches FEATURE_COLUMNS in ml_scorer.py


def create_ml_reference_features(
    n: int = N_REF, d: int = N_ML_FEATURES, seed: int = 10
) -> np.ndarray:
    """Generate reference ML feature distributions.

    Each column represents one of the 12 features used by the XGBoost
    model (amount, channel encoding, country risk, etc.).
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


def create_ml_production_features(
    shift: float = 0.0,
    n: int = N_PROD,
    d: int = N_ML_FEATURES,
    seed: int = 11,
) -> np.ndarray:
    """Generate production ML feature window with optional drift."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)) + shift


# ---------------------------------------------------------------------------
# Metric implementations (simplified, self-contained versions)
# ---------------------------------------------------------------------------

def cosine_distance_drift(ref: np.ndarray, prod: np.ndarray) -> float:
    """Mean cosine distance between reference and production centroids."""
    ref_centroid = ref.mean(axis=0)
    prod_centroid = prod.mean(axis=0)
    cos_sim = np.dot(ref_centroid, prod_centroid) / (
        np.linalg.norm(ref_centroid) * np.linalg.norm(prod_centroid) + 1e-10
    )
    return float(1.0 - cos_sim)


def maximum_mean_discrepancy(ref: np.ndarray, prod: np.ndarray) -> float:
    """Simplified MMD with RBF kernel using the median heuristic."""
    from scipy.spatial.distance import cdist

    # Median heuristic for bandwidth
    combined = np.vstack([ref[:200], prod[:200]])
    dists = cdist(combined, combined, "sqeuclidean")
    bandwidth = float(np.median(dists[dists > 0]))
    if bandwidth == 0:
        bandwidth = 1.0

    def rbf(x: np.ndarray, y: np.ndarray) -> float:
        d = cdist(x, y, "sqeuclidean")
        return float(np.mean(np.exp(-d / (2 * bandwidth))))

    k_rr = rbf(ref, ref)
    k_pp = rbf(prod, prod)
    k_rp = rbf(ref, prod)
    return float(max(k_rr + k_pp - 2 * k_rp, 0.0))


def kolmogorov_smirnov_mean(ref: np.ndarray, prod: np.ndarray) -> float:
    """Mean KS statistic across all embedding components."""
    from scipy.stats import ks_2samp

    stats = []
    for j in range(ref.shape[1]):
        stat, _ = ks_2samp(ref[:, j], prod[:, j])
        stats.append(stat)
    return float(np.mean(stats))


def wasserstein_distance_drift(ref: np.ndarray, prod: np.ndarray) -> float:
    """Mean 1D Wasserstein distance across embedding components."""
    from scipy.stats import wasserstein_distance as wd

    dists = []
    for j in range(ref.shape[1]):
        dists.append(wd(ref[:, j], prod[:, j]))
    return float(np.mean(dists))


def population_stability_index(
    ref: np.ndarray, prod: np.ndarray, n_bins: int = 20
) -> float:
    """PSI averaged over embedding components."""
    psi_total = 0.0
    for j in range(ref.shape[1]):
        # Determine bin edges from reference
        edges = np.histogram_bin_edges(ref[:, j], bins=n_bins)
        ref_counts = np.histogram(ref[:, j], bins=edges)[0].astype(float)
        prod_counts = np.histogram(prod[:, j], bins=edges)[0].astype(float)
        # Avoid division by zero
        ref_frac = (ref_counts + 1e-6) / ref_counts.sum()
        prod_frac = (prod_counts + 1e-6) / prod_counts.sum()
        psi_total += float(np.sum((prod_frac - ref_frac) * np.log(prod_frac / ref_frac)))
    return psi_total / ref.shape[1]


# ---------------------------------------------------------------------------
# Ensemble detector
# ---------------------------------------------------------------------------

def classify_severity(value: float, thresholds: dict[str, float]) -> str:
    """Map a metric value to a severity level using the threshold config."""
    if value <= thresholds["nominal_upper"]:
        return "nominal"
    elif value <= thresholds["warning_upper"]:
        return "warning"
    elif value <= thresholds["critical_upper"]:
        return "critical"
    else:
        return "critical"


def run_ensemble_detector(
    ref: np.ndarray,
    prod: np.ndarray,
    min_agreeing: int = MIN_METRICS_AGREEING,
) -> dict[str, Any]:
    """Run all five metrics and apply ensemble voting logic.

    Returns a dict with per-metric scores, per-metric severities,
    and the overall ensemble severity.
    """
    scores = {
        "cosine_distance": cosine_distance_drift(ref, prod),
        "maximum_mean_discrepancy": maximum_mean_discrepancy(ref, prod),
        "kolmogorov_smirnov": kolmogorov_smirnov_mean(ref, prod),
        "wasserstein_distance": wasserstein_distance_drift(ref, prod),
        "population_stability_index": population_stability_index(ref, prod),
    }

    severities = {
        name: classify_severity(val, THRESHOLDS[name])
        for name, val in scores.items()
    }

    # Count how many metrics vote for each severity level
    severity_counts: dict[str, int] = {"nominal": 0, "warning": 0, "critical": 0}
    for sev in severities.values():
        severity_counts[sev] += 1

    # Determine overall severity: highest level with enough agreement
    if severity_counts["critical"] >= min_agreeing:
        overall = "critical"
    elif severity_counts["warning"] >= min_agreeing:
        overall = "warning"
    else:
        overall = "nominal"

    return {
        "scores": scores,
        "severities": severities,
        "severity_counts": severity_counts,
        "overall_severity": overall,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Alert simulation
# ---------------------------------------------------------------------------

_ACTIONS: dict[str, list[str]] = {
    "nominal": ["log_metrics", "update_dashboard"],
    "warning": [
        "log_metrics",
        "update_dashboard",
        "notify_oncall_slack",
        "increase_monitoring_frequency",
    ],
    "critical": [
        "log_metrics",
        "update_dashboard",
        "page_oncall_engineer",
        "activate_fallback_model",
        "halt_automated_approvals",
    ],
}


def simulate_alert(severity: str) -> None:
    """Print the actions that would be triggered at the given severity."""
    actions = _ACTIONS.get(severity, [])
    print(f"    Actions triggered:")
    for action in actions:
        print(f"      -> {action}")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_report(report: dict[str, Any], window_label: str, layer: str = "") -> None:
    """Pretty-print a drift detection report."""
    layer_tag = f" [{layer}]" if layer else ""
    print(f"\n{'=' * 64}")
    print(f"  DRIFT REPORT{layer_tag} -- {window_label}")
    print(f"{'=' * 64}")
    print(f"  Overall Severity: {report['overall_severity'].upper()}")
    print(f"  Timestamp       : {report['timestamp']}")
    print()
    print(f"  {'Metric':<35s} {'Score':>8s}  {'Severity':>8s}")
    print(f"  {'-' * 53}")
    for name in report["scores"]:
        score = report["scores"][name]
        sev = report["severities"][name]
        print(f"  {name:<35s} {score:>8.4f}  {sev:>8s}")
    print()
    simulate_alert(report["overall_severity"])
    print(f"{'=' * 64}\n")


def print_dual_layer_impact(
    ml_severity: str,
    rag_severity: str,
    shift: float,
) -> None:
    """Explain the dual-layer impact of current drift levels."""
    print(f"  --- Dual-Layer Impact Analysis (shift={shift:.1f}) ---")
    print(f"  ML Feature Drift  : {ml_severity.upper()}")
    print(f"  RAG Embedding Drift: {rag_severity.upper()}")
    print()

    if ml_severity == "critical":
        print("  [!] ML model accuracy is degraded for ALL transactions.")
        print("      -> Rule-based fallback should be activated.")
    elif ml_severity == "warning":
        print("  [*] ML model showing early signs of feature drift.")
        print("      -> Monitor closely; retrain if trend continues.")
    else:
        print("  [ok] ML model features are stable.")

    if rag_severity == "critical":
        print("  [!] RAG embedding drift is critical.")
        print("      -> Gray zone and high-value LLM analyses are unreliable.")
        print("      -> Pattern retrieval is returning stale/irrelevant results.")
    elif rag_severity == "warning":
        print("  [*] RAG embeddings showing moderate drift.")
        print("      -> Gray zone analysis quality may be reduced.")
        print("      -> Consider refreshing the pattern knowledge base.")
    else:
        print("  [ok] RAG embeddings are stable; LLM analysis is reliable.")

    # Interaction effects
    if ml_severity in ("warning", "critical") and rag_severity in ("warning", "critical"):
        print()
        print("  [!!] COMPOUND DRIFT: Both layers are drifting simultaneously.")
        print("       -> More transactions fall into the gray zone due to ML drift,")
        print("          AND the LLM layer that handles them is also degraded.")
        print("       -> This is the highest-risk scenario for the pipeline.")
    print()


# ---------------------------------------------------------------------------
# Main -- simulate drift over time (dual-layer)
# ---------------------------------------------------------------------------

def main() -> None:
    """Simulate gradual drift across both the ML feature space and
    the RAG embedding space, showing their interaction."""
    print()
    print("Dual-Layer Drift Monitoring Demo")
    print("Monitoring ML feature drift AND RAG embedding drift")
    print("-" * 64)

    if _USE_REAL_EMBEDDINGS:
        print("\nUsing real sentence-transformer embeddings (all-MiniLM-L6-v2, 384 dims).")
        if _SPARKOV_TRAIN_PATH.exists():
            print(f"Sparkov data found at {_SPARKOV_TRAIN_PATH}.")
        else:
            print("Sparkov data not found; using random text for real embeddings.")
    else:
        print("\nFalling back to synthetic random vectors (install sentence-transformers for real embeddings).")

    # Create reference distributions for both layers
    rag_ref = create_reference_distribution()
    ml_ref = create_ml_reference_features()
    print(f"\nRAG reference distribution : {rag_ref.shape[0]} samples, {rag_ref.shape[1]} dims")
    print(f"ML feature reference       : {ml_ref.shape[0]} samples, {ml_ref.shape[1]} features")

    # Collect reports for visualization
    rag_reports: list[dict[str, Any]] = []
    ml_reports: list[dict[str, Any]] = []

    # ---------------------------------------------------------------------------
    # Part 1: RAG embedding drift (same as original demo)
    # ---------------------------------------------------------------------------
    print()
    print("=" * 64)
    print("  PART 1: RAG Embedding Drift Over Time")
    print("=" * 64)
    print("  (Affects gray zone and high-value transaction analysis)")

    shift_schedule = [0.0, 0.2, 0.5, 1.0, 1.8, 2.5]

    for step, shift in enumerate(shift_schedule):
        window_label = f"Window {step + 1} / {len(shift_schedule)}  (shift={shift:.1f})"
        prod = create_production_window(shift=shift, seed=step + 100)
        report = run_ensemble_detector(rag_ref, prod)
        report["_shift"] = shift
        rag_reports.append(report)
        print_report(report, window_label, layer="RAG Embeddings")

    # ---------------------------------------------------------------------------
    # Part 2: ML feature drift
    # ---------------------------------------------------------------------------
    print()
    print("=" * 64)
    print("  PART 2: ML Feature Drift Over Time")
    print("=" * 64)
    print("  (Affects ALL transaction scoring)")

    ml_shift_schedule = [0.0, 0.3, 0.8, 1.5]

    for step, shift in enumerate(ml_shift_schedule):
        window_label = f"Window {step + 1} / {len(ml_shift_schedule)}  (shift={shift:.1f})"
        ml_prod = create_ml_production_features(shift=shift, seed=step + 200)
        report = run_ensemble_detector(ml_ref, ml_prod)
        report["_shift"] = shift
        ml_reports.append(report)
        print_report(report, window_label, layer="ML Features")

    # ---------------------------------------------------------------------------
    # Part 3: Dual-layer interaction
    # ---------------------------------------------------------------------------
    print()
    print("=" * 64)
    print("  PART 3: Dual-Layer Drift Interaction")
    print("=" * 64)
    print("  Showing how drift in one layer compounds with the other.\n")

    interaction_scenarios = [
        # (ml_shift, rag_shift, description)
        (0.0, 0.0, "Baseline: both layers stable"),
        (0.0, 1.5, "RAG drift only: LLM analysis degrades"),
        (1.0, 0.0, "ML drift only: primary scorer degrades"),
        (1.0, 1.5, "Compound drift: both layers degraded"),
    ]

    for ml_shift, rag_shift, description in interaction_scenarios:
        print(f"  Scenario: {description}")
        print(f"  ML shift={ml_shift:.1f}, RAG shift={rag_shift:.1f}")

        ml_prod = create_ml_production_features(shift=ml_shift, seed=300)
        rag_prod = create_production_window(shift=rag_shift, seed=301)

        ml_report = run_ensemble_detector(ml_ref, ml_prod)
        rag_report = run_ensemble_detector(rag_ref, rag_prod)

        print_dual_layer_impact(
            ml_severity=ml_report["overall_severity"],
            rag_severity=rag_report["overall_severity"],
            shift=max(ml_shift, rag_shift),
        )
        print(f"  {'-' * 56}")

    # Summary
    print()
    print("-" * 64)
    print("Demo complete.")
    print(
        "In production, both ML feature drift and RAG embedding drift are\n"
        "monitored continuously. Compound drift (both layers shifting) is the\n"
        "highest-risk scenario because more transactions enter the gray zone\n"
        "while the LLM analysis that handles them is also unreliable.\n"
        "Reports are streamed to LangSmith and alert channels (Slack, PagerDuty)."
    )
    print()

    # ---------------------------------------------------------------------------
    # Visualization: drift metrics over time for both layers
    # ---------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metric_names = list(THRESHOLDS.keys())
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

        # --- Part 1 chart: RAG embedding drift metrics over shift schedule ---
        ax = axes[0]
        for metric in metric_names:
            values = [r["scores"][metric] for r in rag_reports]
            shifts = [r["_shift"] for r in rag_reports]
            ax.plot(shifts, values, marker="o", label=metric.replace("_", " ").title(), linewidth=1.5)
        # Add threshold bands
        ax.axhspan(0, 0.05, alpha=0.08, color="green", label="Nominal zone")
        ax.axhspan(0.05, 0.15, alpha=0.08, color="orange")
        ax.axhspan(0.15, ax.get_ylim()[1] if ax.get_ylim()[1] > 0.3 else 0.5, alpha=0.08, color="red")
        ax.set_xlabel("Mean Shift Magnitude")
        ax.set_ylabel("Metric Value")
        ax.set_title("RAG Embedding Drift Metrics Over Increasing Shift")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # --- Part 2 chart: ML feature drift metrics over shift schedule ---
        ax = axes[1]
        for metric in metric_names:
            values = [r["scores"][metric] for r in ml_reports]
            shifts = [r["_shift"] for r in ml_reports]
            ax.plot(shifts, values, marker="s", label=metric.replace("_", " ").title(), linewidth=1.5)
        ax.axhspan(0, 0.05, alpha=0.08, color="green", label="Nominal zone")
        ax.axhspan(0.05, 0.15, alpha=0.08, color="orange")
        ax.axhspan(0.15, ax.get_ylim()[1] if ax.get_ylim()[1] > 0.3 else 0.5, alpha=0.08, color="red")
        ax.set_xlabel("Mean Shift Magnitude")
        ax.set_ylabel("Metric Value")
        ax.set_title("ML Feature Drift Metrics Over Increasing Shift")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = Path(__file__).resolve().parent.parent / "reports" / "drift_metrics_over_time.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Drift metrics chart saved to {out_path}")
    except ImportError:
        print("matplotlib not installed -- skipping visualization.")
    print()


if __name__ == "__main__":
    main()
