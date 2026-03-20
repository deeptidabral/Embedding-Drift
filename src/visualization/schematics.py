"""Conceptual and architectural diagrams drawn with matplotlib.

These functions produce explanatory figures (block diagrams, conceptual
illustrations) that do not depend on any dataset.  They are intended for
documentation, presentations, and the project paper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Reuse the colour palette from plots.py.
_C = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "accent": "#E67E22",
    "nominal": "#27AE60",
    "warning": "#F39C12",
    "critical": "#C0392B",
    "light_grey": "#BDC3C7",
    "dark_grey": "#2C3E50",
    "white": "#FFFFFF",
    "llm": "#8E44AD",
}


def _save_and_return(fig: Figure, save_path: Optional[Union[str, Path]]) -> Figure:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        logger.info("Schematic saved to %s", path)
    return fig


# ---------------------------------------------------------------------------
# Pipeline architecture
# ---------------------------------------------------------------------------


def draw_pipeline_architecture(
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Block diagram: Transaction -> ML Model -> Decision Router -> RAG+LLM.

    Uses matplotlib FancyBboxPatch objects and FancyArrowPatch connectors to
    create a clean, left-to-right pipeline schematic.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Fraud Detection Pipeline Architecture", fontsize=14, fontweight="bold", pad=15)

    # Block definitions: (x_center, y_center, width, height, label, color)
    blocks = [
        (1.5, 2.5, 2.2, 1.6, "Transaction\nStream", _C["dark_grey"]),
        (4.8, 2.5, 2.2, 1.6, "Embedding\nModel", _C["primary"]),
        (8.1, 2.5, 2.2, 1.6, "ML Fraud\nScorer", _C["secondary"]),
        (11.4, 2.5, 2.2, 1.6, "Decision\nRouter", _C["accent"]),
        (14.5, 3.5, 2.0, 1.2, "RAG + LLM\nAnalysis", _C["llm"]),
        (14.5, 1.5, 2.0, 1.2, "Auto-\nDecision", _C["nominal"]),
    ]

    for x, y, w, h, label, color in blocks:
        box = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")

    # Arrows between blocks.
    arrow_kw = dict(
        arrowstyle="-|>",
        color=_C["dark_grey"],
        linewidth=1.8,
        mutation_scale=15,
    )
    arrow_pairs = [
        ((2.6, 2.5), (3.7, 2.5)),
        ((5.9, 2.5), (7.0, 2.5)),
        ((9.2, 2.5), (10.3, 2.5)),
        ((12.5, 2.9), (13.5, 3.5)),
        ((12.5, 2.1), (13.5, 1.5)),
    ]
    for (x1, y1), (x2, y2) in arrow_pairs:
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=_C["dark_grey"], lw=1.8),
        )

    # Drift monitor annotation.
    ax.annotate(
        "Drift\nMonitor",
        xy=(6.45, 1.0), fontsize=8, ha="center", va="center",
        fontweight="bold", color=_C["critical"],
        bbox=dict(boxstyle="round,pad=0.3", fc="#FDEDEC", ec=_C["critical"], lw=1.2),
    )
    ax.annotate(
        "", xy=(4.8, 1.7), xytext=(6.0, 1.0),
        arrowprops=dict(arrowstyle="-|>", color=_C["critical"], lw=1.0, linestyle="--"),
    )
    ax.annotate(
        "", xy=(8.1, 1.7), xytext=(6.9, 1.0),
        arrowprops=dict(arrowstyle="-|>", color=_C["critical"], lw=1.0, linestyle="--"),
    )

    fig.tight_layout()
    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# Drift types diagram
# ---------------------------------------------------------------------------


def draw_drift_types_diagram(
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """2x2 grid showing four canonical drift patterns as line charts.

    Patterns: sudden, gradual, incremental, recurring.

    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Types of Embedding Drift", fontsize=14, fontweight="bold", y=0.98)

    t = np.linspace(0, 10, 200)

    # -- Sudden drift --
    ax = axes[0, 0]
    y = np.where(t < 5, 0.2, 0.8) + np.random.default_rng(0).normal(0, 0.03, len(t))
    ax.plot(t, y, color=_C["primary"], linewidth=1.5)
    ax.axvline(5, color=_C["critical"], linestyle="--", alpha=0.7)
    ax.set_title("Sudden Drift", fontweight="bold")
    ax.set_ylabel("Drift Metric")
    ax.set_ylim(-0.05, 1.1)
    _shade_regions(ax, 5, len(t))

    # -- Gradual drift --
    ax = axes[0, 1]
    rng = np.random.default_rng(1)
    sigmoid = 1 / (1 + np.exp(-1.2 * (t - 5)))
    y = 0.2 + 0.6 * sigmoid + rng.normal(0, 0.03, len(t))
    ax.plot(t, y, color=_C["primary"], linewidth=1.5)
    ax.set_title("Gradual Drift", fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    _shade_regions(ax, 3, 7, gradual=True)

    # -- Incremental drift --
    ax = axes[1, 0]
    rng = np.random.default_rng(2)
    y = 0.2 + 0.06 * t + rng.normal(0, 0.03, len(t))
    ax.plot(t, y, color=_C["primary"], linewidth=1.5)
    ax.set_title("Incremental Drift", fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drift Metric")
    ax.set_ylim(-0.05, 1.1)

    # -- Recurring drift --
    ax = axes[1, 1]
    rng = np.random.default_rng(3)
    y = 0.5 + 0.3 * np.sin(1.2 * t) + rng.normal(0, 0.03, len(t))
    ax.plot(t, y, color=_C["primary"], linewidth=1.5)
    ax.set_title("Recurring Drift", fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylim(-0.05, 1.1)

    for row in axes:
        for a in row:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.set_xticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save_and_return(fig, save_path)


def _shade_regions(
    ax: plt.Axes,
    start: float,
    end: float,
    gradual: bool = False,
) -> None:
    """Helper to shade a drift region on an axis."""
    if gradual:
        ax.axvspan(start, end, alpha=0.08, color=_C["warning"])
    else:
        ax.axvspan(start, 10, alpha=0.08, color=_C["critical"])


# ---------------------------------------------------------------------------
# Threshold bands diagram
# ---------------------------------------------------------------------------


def draw_threshold_bands(
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Diagram showing nominal / warning / critical zones with a sample trajectory.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Drift Severity Threshold Bands", fontsize=13, fontweight="bold")

    warn_thresh = 0.4
    crit_thresh = 0.7
    y_max = 1.05

    # Zones.
    ax.axhspan(0, warn_thresh, alpha=0.15, color=_C["nominal"], label="Nominal Zone")
    ax.axhspan(warn_thresh, crit_thresh, alpha=0.15, color=_C["warning"], label="Warning Zone")
    ax.axhspan(crit_thresh, y_max, alpha=0.15, color=_C["critical"], label="Critical Zone")

    ax.axhline(warn_thresh, color=_C["warning"], linewidth=1.2, linestyle="--")
    ax.axhline(crit_thresh, color=_C["critical"], linewidth=1.2, linestyle="--")

    # Threshold labels.
    ax.text(0.5, warn_thresh + 0.02, "Warning Threshold", fontsize=8,
            color=_C["warning"], fontweight="bold")
    ax.text(0.5, crit_thresh + 0.02, "Critical Threshold", fontsize=8,
            color=_C["critical"], fontweight="bold")

    # Example metric trajectory.
    rng = np.random.default_rng(42)
    t = np.linspace(0, 30, 150)
    base = np.concatenate([
        np.full(40, 0.2),
        np.linspace(0.2, 0.55, 30),
        np.full(20, 0.55),
        np.linspace(0.55, 0.85, 25),
        np.full(20, 0.85),
        np.linspace(0.85, 0.3, 15),
    ])
    noise = rng.normal(0, 0.02, len(base))
    trajectory = np.clip(base + noise, 0, 1.0)

    ax.plot(t, trajectory, color=_C["dark_grey"], linewidth=1.8, label="Drift Metric")

    ax.set_xlim(0, 30)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Time Window")
    ax.set_ylabel("Drift Score")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# Dual-layer interaction matrix
# ---------------------------------------------------------------------------


def draw_dual_layer_interaction(
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """2x2 matrix showing four compound drift scenarios.

    Rows: ML feature drift (stable / drifted).
    Columns: Embedding drift (stable / drifted).

    Scenarios:
    1. Both stable        -- normal operations
    2. ML drift only      -- feature distribution shift
    3. Embedding drift    -- semantic shift
    4. Compound drift     -- both layers drifting

    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        "Dual-Layer Drift Interaction Matrix",
        fontsize=14, fontweight="bold", y=0.98,
    )

    scenarios = [
        {
            "title": "Both Stable",
            "color": _C["nominal"],
            "desc": "Normal operations.\nNo corrective action needed.",
            "ml_drift": False,
            "emb_drift": False,
        },
        {
            "title": "Embedding Drift Only",
            "color": _C["accent"],
            "desc": "Semantic shift detected.\nLLM context may be stale;\nupdate RAG knowledge base.",
            "ml_drift": False,
            "emb_drift": True,
        },
        {
            "title": "ML Feature Drift Only",
            "color": _C["warning"],
            "desc": "Feature distribution shift.\nRetrain or recalibrate\nthe ML fraud scorer.",
            "ml_drift": True,
            "emb_drift": False,
        },
        {
            "title": "Compound Drift",
            "color": _C["critical"],
            "desc": "Both layers affected.\nHighest severity; trigger\nfull pipeline review.",
            "ml_drift": True,
            "emb_drift": True,
        },
    ]

    for ax, scenario in zip(axes.flat, scenarios):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Background box.
        bg = mpatches.FancyBboxPatch(
            (0.3, 0.3), 9.4, 9.4,
            boxstyle="round,pad=0.3",
            facecolor=scenario["color"],
            alpha=0.12,
            edgecolor=scenario["color"],
            linewidth=2,
        )
        ax.add_patch(bg)

        ax.text(5, 8.5, scenario["title"], ha="center", va="center",
                fontsize=12, fontweight="bold", color=scenario["color"])

        ax.text(5, 5.5, scenario["desc"], ha="center", va="center",
                fontsize=9, color=_C["dark_grey"], linespacing=1.6)

        # Status indicators.
        ml_status = "DRIFTED" if scenario["ml_drift"] else "STABLE"
        emb_status = "DRIFTED" if scenario["emb_drift"] else "STABLE"
        ml_color = _C["critical"] if scenario["ml_drift"] else _C["nominal"]
        emb_color = _C["critical"] if scenario["emb_drift"] else _C["nominal"]

        ax.text(2.5, 2.0, f"ML: {ml_status}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc=ml_color, ec="none"))
        ax.text(7.5, 2.0, f"Emb: {emb_status}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc=emb_color, ec="none"))

    # Row / column labels.
    fig.text(0.02, 0.73, "ML Stable", rotation=90, va="center", fontsize=10,
             fontweight="bold", color=_C["dark_grey"])
    fig.text(0.02, 0.30, "ML Drifted", rotation=90, va="center", fontsize=10,
             fontweight="bold", color=_C["dark_grey"])
    fig.text(0.30, 0.01, "Embedding Stable", ha="center", fontsize=10,
             fontweight="bold", color=_C["dark_grey"])
    fig.text(0.73, 0.01, "Embedding Drifted", ha="center", fontsize=10,
             fontweight="bold", color=_C["dark_grey"])

    fig.tight_layout(rect=[0.05, 0.04, 1, 0.95])
    return _save_and_return(fig, save_path)
