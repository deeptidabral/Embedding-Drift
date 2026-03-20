"""Reusable plotting functions for embedding drift analysis and fraud detection.

All functions use a consistent professional style (seaborn ``whitegrid`` theme,
a curated colour palette) and return the ``matplotlib.figure.Figure`` object so
that callers can further customise or display the plot.  Every function accepts
an optional ``save_path`` argument; when provided the figure is written to disk
before being returned.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "accent": "#E67E22",
    "fraud": "#E74C3C",
    "legit": "#2ECC71",
    "nominal": "#27AE60",
    "warning": "#F39C12",
    "critical": "#C0392B",
    "light_grey": "#BDC3C7",
    "dark_grey": "#2C3E50",
    "palette": [
        "#1B4F72",
        "#2E86C1",
        "#E67E22",
        "#27AE60",
        "#8E44AD",
        "#C0392B",
        "#16A085",
        "#D4AC0D",
        "#7F8C8D",
        "#2C3E50",
    ],
}

SEVERITY_COLORS = {
    "nominal": COLORS["nominal"],
    "warning": COLORS["warning"],
    "critical": COLORS["critical"],
}


def set_style() -> None:
    """Apply the project-wide matplotlib / seaborn style."""
    sns.set_theme(style="whitegrid", palette=COLORS["palette"])
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_and_return(fig: Figure, save_path: Optional[Union[str, Path]]) -> Figure:
    """Optionally save *fig* to *save_path*, then return it."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        logger.info("Figure saved to %s", path)
    return fig


# ---------------------------------------------------------------------------
# Transaction-level plots
# ---------------------------------------------------------------------------


def plot_transaction_volume_over_time(
    df: pd.DataFrame,
    freq: str = "D",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Daily (or other frequency) transaction volume with a fraud overlay.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe with ``timestamp`` and ``is_fraud`` columns.
    freq : str
        Pandas offset alias for resampling (default ``"D"`` for daily).
    save_path : str or Path, optional
        If provided, save the figure to this path.

    Returns
    -------
    Figure
    """
    set_style()
    ts = df.set_index("timestamp")
    total = ts.resample(freq).size()
    fraud = ts[ts["is_fraud"] == 1].resample(freq).size()

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.fill_between(total.index, total.values, alpha=0.3, color=COLORS["primary"], label="Total")
    ax1.plot(total.index, total.values, color=COLORS["primary"], linewidth=0.8)
    ax1.set_ylabel("Total transactions")
    ax1.set_xlabel("Date")
    ax1.set_title("Transaction Volume Over Time")

    ax2 = ax1.twinx()
    ax2.bar(fraud.index, fraud.values, width=1.0, alpha=0.7, color=COLORS["fraud"], label="Fraud")
    ax2.set_ylabel("Fraud transactions")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_amount_distribution(
    df: pd.DataFrame,
    by_fraud: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Histogram / KDE of transaction amounts, optionally split by fraud label.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``amt`` and (if *by_fraud*) ``is_fraud``.
    by_fraud : bool
        Whether to separate fraud and legitimate distributions.
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    if by_fraud:
        legit = df.loc[df["is_fraud"] == 0, "amt"]
        fraud = df.loc[df["is_fraud"] == 1, "amt"]
        ax.hist(legit, bins=100, density=True, alpha=0.5, color=COLORS["legit"], label="Legitimate")
        ax.hist(fraud, bins=100, density=True, alpha=0.6, color=COLORS["fraud"], label="Fraud")
        ax.legend()
    else:
        ax.hist(df["amt"], bins=100, density=True, alpha=0.6, color=COLORS["primary"])

    ax.set_xlabel("Transaction Amount (USD)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction Amount Distribution")
    ax.set_xlim(left=0)
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_category_fraud_rate(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Horizontal bar chart of fraud rate per transaction category.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``category`` and ``is_fraud``.
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    set_style()
    rates = df.groupby("category")["is_fraud"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(rates.index, rates.values, color=COLORS["secondary"], edgecolor="white")

    # Highlight categories above mean fraud rate.
    mean_rate = df["is_fraud"].mean()
    for bar, val in zip(bars, rates.values):
        if val > mean_rate:
            bar.set_color(COLORS["fraud"])

    ax.axvline(mean_rate, color=COLORS["dark_grey"], linestyle="--", linewidth=0.8, label=f"Mean ({mean_rate:.3f})")
    ax.set_xlabel("Fraud Rate")
    ax.set_title("Fraud Rate by Category")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# Embedding space plots
# ---------------------------------------------------------------------------


def plot_embedding_space_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """2-D scatter of embeddings coloured by label (fraud / legitimate).

    Parameters
    ----------
    embeddings : np.ndarray
        (N, D) embedding matrix.
    labels : np.ndarray
        Binary labels (0 = legit, 1 = fraud) of length N.
    method : str
        Dimensionality reduction method -- ``"tsne"`` or ``"umap"``.
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    set_style()
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == "umap":
        import umap  # type: ignore[import-untyped]
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'tsne' or 'umap'.")

    coords = reducer.fit_transform(embeddings)
    labels = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask_legit = labels == 0
    mask_fraud = labels == 1
    ax.scatter(
        coords[mask_legit, 0], coords[mask_legit, 1],
        c=COLORS["legit"], s=4, alpha=0.3, label="Legitimate",
    )
    ax.scatter(
        coords[mask_fraud, 0], coords[mask_fraud, 1],
        c=COLORS["fraud"], s=12, alpha=0.7, label="Fraud",
    )
    ax.set_title(f"Embedding Space ({method.upper()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(markerscale=3)
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_embedding_space_3d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "pca",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """3-D scatter of embeddings (PCA by default).

    Parameters
    ----------
    embeddings : np.ndarray
        (N, D) embedding matrix.
    labels : np.ndarray
        Binary labels of length N.
    method : str
        Currently only ``"pca"`` is supported for the 3-D view.
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    set_style()
    if method != "pca":
        raise ValueError("Only 'pca' is supported for 3-D plots.")

    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(embeddings)
    labels = np.asarray(labels)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    mask_legit = labels == 0
    mask_fraud = labels == 1
    ax.scatter(
        coords[mask_legit, 0], coords[mask_legit, 1], coords[mask_legit, 2],
        c=COLORS["legit"], s=4, alpha=0.3, label="Legitimate",
    )
    ax.scatter(
        coords[mask_fraud, 0], coords[mask_fraud, 1], coords[mask_fraud, 2],
        c=COLORS["fraud"], s=12, alpha=0.7, label="Fraud",
    )
    ax.set_title("Embedding Space (PCA 3-D)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    ax.legend(markerscale=3)
    fig.tight_layout()
    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# Drift monitoring plots
# ---------------------------------------------------------------------------


def plot_drift_metrics_over_time(
    drift_values: dict[str, Sequence[float]],
    metric_names: Sequence[str],
    timestamps: Sequence,
    thresholds: Optional[dict[str, tuple[float, float]]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Multi-line time series of drift metrics with threshold bands.

    Parameters
    ----------
    drift_values : dict[str, Sequence[float]]
        Mapping of metric name to its time series of values.
    metric_names : Sequence[str]
        Which keys to plot from *drift_values*.
    timestamps : Sequence
        X-axis values (dates or indices).
    thresholds : dict, optional
        ``{metric: (warning, critical)}`` to draw horizontal bands.
        Nominal zone = green (below warning), warning zone = yellow
        (between warning and critical), critical zone = red (above critical).
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    set_style()
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for ax, name in zip(axes, metric_names):
        values = drift_values[name]
        ax.plot(timestamps, values, color=COLORS["primary"], linewidth=1.5, label=name)

        if thresholds and name in thresholds:
            warn, crit = thresholds[name]
            y_max = max(max(values) * 1.2, crit * 1.3)
            ax.axhspan(0, warn, alpha=0.10, color=COLORS["nominal"])
            ax.axhspan(warn, crit, alpha=0.12, color=COLORS["warning"])
            ax.axhspan(crit, y_max, alpha=0.10, color=COLORS["critical"])
            ax.axhline(warn, color=COLORS["warning"], linestyle="--", linewidth=0.7, label="Warning")
            ax.axhline(crit, color=COLORS["critical"], linestyle="--", linewidth=0.7, label="Critical")

        ax.set_ylabel(name)
        ax.legend(loc="upper left", framealpha=0.9)

    axes[-1].set_xlabel("Time")
    axes[0].set_title("Drift Metrics Over Time")
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_drift_heatmap(
    drift_matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Heatmap of drift scores (e.g. category x time window).

    Parameters
    ----------
    drift_matrix : np.ndarray
        2-D array of drift values, shape ``(len(row_labels), len(col_labels))``.
    row_labels : Sequence[str]
        Labels for rows (e.g. categories).
    col_labels : Sequence[str]
        Labels for columns (e.g. time windows).
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.9), max(5, len(row_labels) * 0.5)))
    sns.heatmap(
        drift_matrix,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Drift by Category and Time Window")
    ax.set_xlabel("Time Window")
    ax.set_ylabel("Category")
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_severity_timeline(
    severities: Sequence[str],
    timestamps: Sequence,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Colour-coded bar chart showing severity level over time.

    Parameters
    ----------
    severities : Sequence[str]
        One of ``"nominal"``, ``"warning"``, ``"critical"`` per time step.
    timestamps : Sequence
        X-axis values.
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    set_style()
    bar_colors = [SEVERITY_COLORS.get(s, COLORS["light_grey"]) for s in severities]

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.bar(range(len(severities)), [1] * len(severities), color=bar_colors, width=1.0, edgecolor="white")
    ax.set_yticks([])
    ax.set_xticks(range(len(timestamps)))
    ax.set_xticklabels(timestamps, rotation=45, ha="right", fontsize=8)
    ax.set_title("Drift Severity Timeline")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SEVERITY_COLORS["nominal"], label="Nominal"),
        Patch(facecolor=SEVERITY_COLORS["warning"], label="Warning"),
        Patch(facecolor=SEVERITY_COLORS["critical"], label="Critical"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", ncol=3)
    fig.tight_layout()
    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# ML and routing plots
# ---------------------------------------------------------------------------


def plot_ml_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Overlapping histograms of ML fraud scores for true fraud vs true legit.

    Parameters
    ----------
    scores : np.ndarray
        Predicted fraud probability scores.
    labels : np.ndarray
        Ground-truth binary labels.
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    set_style()
    scores, labels = np.asarray(scores), np.asarray(labels)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores[labels == 0], bins=80, density=True, alpha=0.5, color=COLORS["legit"], label="Legitimate")
    ax.hist(scores[labels == 1], bins=80, density=True, alpha=0.6, color=COLORS["fraud"], label="Fraud")
    ax.set_xlabel("Fraud Score")
    ax.set_ylabel("Density")
    ax.set_title("ML Fraud Score Distribution")
    ax.legend()
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_routing_breakdown(
    ml_only_count: int,
    ml_plus_llm_count: int,
    fallback_count: int,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Pie chart showing the breakdown of analysis tiers.

    Parameters
    ----------
    ml_only_count : int
        Transactions handled by ML model alone.
    ml_plus_llm_count : int
        Transactions escalated to ML + LLM analysis.
    fallback_count : int
        Transactions routed to the fallback / manual review tier.
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    set_style()
    sizes = [ml_only_count, ml_plus_llm_count, fallback_count]
    labels = ["ML Only", "ML + LLM", "Fallback / Manual"]
    colors = [COLORS["legit"], COLORS["secondary"], COLORS["accent"]]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    ax.set_title("Decision Routing Breakdown")
    fig.tight_layout()
    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# Advanced / dual-layer plots
# ---------------------------------------------------------------------------


def plot_dual_layer_drift_correlation(
    embedding_drift: Sequence[float],
    feature_drift: Sequence[float],
    timestamps: Sequence,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Dual-axis time series showing embedding drift and feature drift together.

    Parameters
    ----------
    embedding_drift : Sequence[float]
        Embedding-based drift metric over time.
    feature_drift : Sequence[float]
        Feature-based (tabular) drift metric over time.
    timestamps : Sequence
        Shared x-axis.
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    set_style()
    fig, ax1 = plt.subplots(figsize=(14, 5))

    color_emb = COLORS["primary"]
    color_feat = COLORS["accent"]

    ax1.plot(timestamps, embedding_drift, color=color_emb, linewidth=1.5, label="Embedding Drift")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Embedding Drift", color=color_emb)
    ax1.tick_params(axis="y", labelcolor=color_emb)

    ax2 = ax1.twinx()
    ax2.plot(timestamps, feature_drift, color=color_feat, linewidth=1.5, linestyle="--", label="Feature Drift")
    ax2.set_ylabel("Feature Drift", color=color_feat)
    ax2.tick_params(axis="y", labelcolor=color_feat)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Dual-Layer Drift Correlation")
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_pca_explained_variance(
    embeddings: np.ndarray,
    n_components: int = 20,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Scree plot of PCA explained variance.

    Parameters
    ----------
    embeddings : np.ndarray
        (N, D) embedding matrix.
    n_components : int
        Number of principal components to display.
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    set_style()
    n_components = min(n_components, min(embeddings.shape))
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(embeddings)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(1, n_components + 1)
    ax.bar(x, explained, alpha=0.6, color=COLORS["secondary"], label="Individual")
    ax.plot(x, cumulative, color=COLORS["fraud"], marker="o", markersize=4, label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance (Scree Plot)")
    ax.set_xticks(list(x))
    ax.legend()
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_cosine_distance_distribution(
    reference: np.ndarray,
    production: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Overlapping histograms of pairwise cosine distances within two sets.

    A random subsample of pairwise distances is used when the sets are large
    to keep computation tractable.

    Parameters
    ----------
    reference : np.ndarray
        (N, D) reference embeddings.
    production : np.ndarray
        (M, D) production embeddings.
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    from sklearn.metrics.pairwise import cosine_distances

    set_style()

    def _sample_distances(arr: np.ndarray, max_pairs: int = 50_000) -> np.ndarray:
        n = arr.shape[0]
        if n * (n - 1) // 2 <= max_pairs:
            dists = cosine_distances(arr)
            return dists[np.triu_indices(n, k=1)]
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=min(n, int(np.sqrt(max_pairs * 2))), replace=False)
        sub = arr[idx]
        dists = cosine_distances(sub)
        return dists[np.triu_indices(len(idx), k=1)]

    ref_dists = _sample_distances(reference)
    prod_dists = _sample_distances(production)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ref_dists, bins=80, density=True, alpha=0.5, color=COLORS["primary"], label="Reference")
    ax.hist(prod_dists, bins=80, density=True, alpha=0.5, color=COLORS["accent"], label="Production")
    ax.set_xlabel("Cosine Distance")
    ax.set_ylabel("Density")
    ax.set_title("Pairwise Cosine Distance Distribution")
    ax.legend()
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_cusum_chart(
    cusum_values: Sequence[float],
    alarm_threshold: float,
    timestamps: Optional[Sequence] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """CUSUM (Cumulative Sum) control chart with an alarm threshold.

    Parameters
    ----------
    cusum_values : Sequence[float]
        Cumulative sum statistic over time.
    alarm_threshold : float
        Threshold value above which an alarm is raised.
    timestamps : Sequence, optional
        X-axis labels; defaults to integer indices.
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    set_style()
    n = len(cusum_values)
    x = timestamps if timestamps is not None else list(range(n))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, cusum_values, color=COLORS["primary"], linewidth=1.5, label="CUSUM")
    ax.axhline(alarm_threshold, color=COLORS["critical"], linestyle="--", linewidth=1.2, label=f"Alarm ({alarm_threshold})")

    # Shade alarm region.
    ax.fill_between(
        x,
        alarm_threshold,
        max(max(cusum_values), alarm_threshold * 1.3),
        alpha=0.08,
        color=COLORS["critical"],
    )

    # Mark alarm crossings.
    vals = np.asarray(cusum_values)
    alarm_idx = np.where(vals >= alarm_threshold)[0]
    if len(alarm_idx) > 0:
        alarm_x = [x[i] for i in alarm_idx]
        alarm_y = [cusum_values[i] for i in alarm_idx]
        ax.scatter(alarm_x, alarm_y, color=COLORS["critical"], s=30, zorder=5, label="Alarms")

    ax.set_xlabel("Time")
    ax.set_ylabel("CUSUM Statistic")
    ax.set_title("CUSUM Control Chart")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save_and_return(fig, save_path)
