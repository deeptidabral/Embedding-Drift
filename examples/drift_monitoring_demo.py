"""
Drift Monitoring Demo -- End-to-end example of MMD-based embedding drift
detection for a fraud detection system.

This script walks through the complete monitoring workflow:

    1. Create a reference embedding distribution (simulating a trained model).
    2. Compute MMD against a production window.
    3. Run the drift detector with severity classification.
    4. Simulate gradual drift over multiple time windows.
    5. Show how alerts fire at different severity levels.
    6. Demonstrate monitoring of both ML feature drift and RAG embedding drift.
    7. Show how drift in the RAG layer affects investigation quality.

Architecture:
  - ML model runs in the real-time authorization path (synchronous).
  - LLM/RAG investigation runs ASYNCHRONOUSLY for flagged transactions.
  - Drift is monitored using MMD only (the sole metric -- see design note).

Design note on drift detection:
  Dense embedding dimensions are highly entangled.  Univariate tests
  (KS per dimension) suffer from massive multiple-testing problems and
  miss multivariate rotations.  Cosine distance only captures mean
  shift.  PCA explained-variance comparisons miss mean shift entirely.
  Only MMD correctly assesses the full high-dimensional distribution.

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
# Configuration -- MMD thresholds only
# ---------------------------------------------------------------------------

MMD_THRESHOLDS: dict[str, float] = {
    "nominal_upper": 0.02,
    "warning_upper": 0.08,
    "critical_upper": 0.20,
}

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
    """Generate reference ML feature distributions."""
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
# MMD metric (simplified, self-contained version for this demo)
# ---------------------------------------------------------------------------

def compute_mmd(ref: np.ndarray, prod: np.ndarray) -> float:
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


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

def classify_severity(mmd_value: float) -> str:
    """Map an MMD value to a severity level."""
    if mmd_value <= MMD_THRESHOLDS["nominal_upper"]:
        return "nominal"
    elif mmd_value <= MMD_THRESHOLDS["warning_upper"]:
        return "warning"
    else:
        return "critical"


def run_drift_detector(
    ref: np.ndarray,
    prod: np.ndarray,
) -> dict[str, Any]:
    """Run MMD and classify severity.

    Returns a dict with the MMD score, severity, and timestamp.
    """
    mmd_value = compute_mmd(ref, prod)
    severity = classify_severity(mmd_value)

    return {
        "mmd": mmd_value,
        "severity": severity,
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
        "notify_oncall",
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
    print(f"  Overall Severity: {report['severity'].upper()}")
    print(f"  Timestamp       : {report['timestamp']}")
    print()
    print(f"  {'Metric':<35s} {'Score':>8s}  {'Severity':>8s}")
    print(f"  {'-' * 53}")
    print(f"  {'maximum_mean_discrepancy':<35s} {report['mmd']:>8.4f}  {report['severity']:>8s}")
    print()
    simulate_alert(report["severity"])
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
        print("      -> Async LLM investigations are unreliable.")
        print("      -> Pattern retrieval is returning stale/irrelevant results.")
    elif rag_severity == "warning":
        print("  [*] RAG embeddings showing moderate drift.")
        print("      -> Async investigation quality may be reduced.")
        print("      -> Consider refreshing the pattern knowledge base.")
    else:
        print("  [ok] RAG embeddings are stable; async LLM investigation is reliable.")

    # Interaction effects
    if ml_severity in ("warning", "critical") and rag_severity in ("warning", "critical"):
        print()
        print("  [!!] COMPOUND DRIFT: Both layers are drifting simultaneously.")
        print("       -> More transactions fall into the gray zone due to ML drift,")
        print("          AND the async LLM layer that investigates them is also degraded.")
        print("       -> This is the highest-risk scenario for the pipeline.")
    print()


# ---------------------------------------------------------------------------
# Main -- simulate drift over time (dual-layer)
# ---------------------------------------------------------------------------

def main() -> None:
    """Simulate gradual drift across both the ML feature space and
    the RAG embedding space, showing their interaction."""
    print()
    print("MMD-Based Drift Monitoring Demo")
    print("Monitoring ML feature drift AND RAG embedding drift using MMD")
    print("-" * 64)
    print()
    print("Architecture:")
    print("  ML model  -> synchronous authorization (approve / decline / flag)")
    print("  LLM/RAG   -> async post-transaction investigation (flagged txns only)")
    print("  Drift     -> MMD-based monitoring (sole metric)")
    print()

    if _USE_REAL_EMBEDDINGS:
        print("Using real sentence-transformer embeddings (all-MiniLM-L6-v2, 384 dims).")
        if _SPARKOV_TRAIN_PATH.exists():
            print(f"Sparkov data found at {_SPARKOV_TRAIN_PATH}.")
        else:
            print("Sparkov data not found; using random text for real embeddings.")
    else:
        print("Falling back to synthetic random vectors (install sentence-transformers for real embeddings).")

    # Create reference distributions for both layers
    rag_ref = create_reference_distribution()
    ml_ref = create_ml_reference_features()
    print(f"\nRAG reference distribution : {rag_ref.shape[0]} samples, {rag_ref.shape[1]} dims")
    print(f"ML feature reference       : {ml_ref.shape[0]} samples, {ml_ref.shape[1]} features")

    # Collect reports for visualization
    rag_reports: list[dict[str, Any]] = []
    ml_reports: list[dict[str, Any]] = []

    # ---------------------------------------------------------------------------
    # Part 1: RAG embedding drift
    # ---------------------------------------------------------------------------
    print()
    print("=" * 64)
    print("  PART 1: RAG Embedding Drift Over Time (MMD)")
    print("=" * 64)
    print("  (Affects async LLM investigation for flagged transactions)")

    shift_schedule = [0.0, 0.2, 0.5, 1.0, 1.8, 2.5]

    for step, shift in enumerate(shift_schedule):
        window_label = f"Window {step + 1} / {len(shift_schedule)}  (shift={shift:.1f})"
        prod = create_production_window(shift=shift, seed=step + 100)
        report = run_drift_detector(rag_ref, prod)
        report["_shift"] = shift
        rag_reports.append(report)
        print_report(report, window_label, layer="RAG Embeddings")

    # ---------------------------------------------------------------------------
    # Part 2: ML feature drift
    # ---------------------------------------------------------------------------
    print()
    print("=" * 64)
    print("  PART 2: ML Feature Drift Over Time (MMD)")
    print("=" * 64)
    print("  (Affects ALL transaction scoring in the authorization path)")

    ml_shift_schedule = [0.0, 0.3, 0.8, 1.5]

    for step, shift in enumerate(ml_shift_schedule):
        window_label = f"Window {step + 1} / {len(ml_shift_schedule)}  (shift={shift:.1f})"
        ml_prod = create_ml_production_features(shift=shift, seed=step + 200)
        report = run_drift_detector(ml_ref, ml_prod)
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
        (0.0, 1.5, "RAG drift only: async LLM investigation degrades"),
        (1.0, 0.0, "ML drift only: primary scorer degrades"),
        (1.0, 1.5, "Compound drift: both layers degraded"),
    ]

    for ml_shift, rag_shift, description in interaction_scenarios:
        print(f"  Scenario: {description}")
        print(f"  ML shift={ml_shift:.1f}, RAG shift={rag_shift:.1f}")

        ml_prod = create_ml_production_features(shift=ml_shift, seed=300)
        rag_prod = create_production_window(shift=rag_shift, seed=301)

        ml_report = run_drift_detector(ml_ref, ml_prod)
        rag_report = run_drift_detector(rag_ref, rag_prod)

        print_dual_layer_impact(
            ml_severity=ml_report["severity"],
            rag_severity=rag_report["severity"],
            shift=max(ml_shift, rag_shift),
        )
        print(f"  {'-' * 56}")

    # Summary
    print()
    print("-" * 64)
    print("Demo complete.")
    print(
        "In production, both ML feature drift and RAG embedding drift are\n"
        "monitored continuously using MMD.  Compound drift (both layers\n"
        "shifting) is the highest-risk scenario because more transactions\n"
        "enter the gray zone while the async LLM investigation that handles\n"
        "them is also unreliable.\n"
        "Reports are streamed to LangSmith for dashboard visibility."
    )
    print()

    # ---------------------------------------------------------------------------
    # Visualization: MMD over time for both layers
    # ---------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        # --- Part 1 chart: RAG embedding MMD over shift schedule ---
        ax = axes[0]
        mmd_values = [r["mmd"] for r in rag_reports]
        shifts = [r["_shift"] for r in rag_reports]
        ax.plot(shifts, mmd_values, marker="o", color="#1B4F72", linewidth=1.5, label="MMD")
        # Add threshold bands
        ax.axhline(MMD_THRESHOLDS["nominal_upper"], color="green", linestyle="--", alpha=0.7, label="Nominal upper")
        ax.axhline(MMD_THRESHOLDS["warning_upper"], color="orange", linestyle="--", alpha=0.7, label="Warning upper")
        ax.axhline(MMD_THRESHOLDS["critical_upper"], color="red", linestyle="--", alpha=0.7, label="Critical upper")
        ax.set_xlabel("Mean Shift Magnitude")
        ax.set_ylabel("MMD Value")
        ax.set_title("RAG Embedding Drift (MMD) Over Increasing Shift")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Part 2 chart: ML feature MMD over shift schedule ---
        ax = axes[1]
        mmd_values = [r["mmd"] for r in ml_reports]
        shifts = [r["_shift"] for r in ml_reports]
        ax.plot(shifts, mmd_values, marker="s", color="#2E86C1", linewidth=1.5, label="MMD")
        ax.axhline(MMD_THRESHOLDS["nominal_upper"], color="green", linestyle="--", alpha=0.7, label="Nominal upper")
        ax.axhline(MMD_THRESHOLDS["warning_upper"], color="orange", linestyle="--", alpha=0.7, label="Warning upper")
        ax.axhline(MMD_THRESHOLDS["critical_upper"], color="red", linestyle="--", alpha=0.7, label="Critical upper")
        ax.set_xlabel("Mean Shift Magnitude")
        ax.set_ylabel("MMD Value")
        ax.set_title("ML Feature Drift (MMD) Over Increasing Shift")
        ax.legend(fontsize=8)
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
