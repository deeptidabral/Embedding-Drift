"""
Fraud Detection Demo -- Dual-layer pipeline with ML model as the primary
scorer and RAG+LLM as a complementary layer for gray zone analysis.

This demo works best with the Sparkov Credit Card Fraud Detection dataset
(https://www.kaggle.com/datasets/kartik2112/fraud-detection). When the
Sparkov CSV files are present in data/, the demo loads a sample of real
simulated transactions. Otherwise it falls back to a small set of
hand-crafted synthetic transactions so the demo runs without any external
data.

This script demonstrates the full lifecycle of transactions flowing
through the dual-layer architecture:

    1. Load transactions from the Sparkov dataset (or generate synthetic ones).
    2. Score ALL transactions with the ML model (XGBoost-style heuristic).
    3. Route transactions through the decision router:
       - Clear ML scores (< 0.3 or > 0.7) -> ML-only decision.
       - Gray zone scores (0.3--0.7) -> escalate to RAG+LLM.
       - High-value transactions (> $10k) -> always escalate to RAG+LLM.
    4. Display which transactions went ML-only vs ML+LLM.
    5. Show the analysis tier and scoring breakdown for each result.
    6. Plot the fraud score distribution across all processed transactions.

Run with:
    python examples/fraud_detection_demo.py
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is on sys.path so src imports work when running as a script
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

# ---------------------------------------------------------------------------
# Sparkov dataset loading (falls back to synthetic data if unavailable)
# ---------------------------------------------------------------------------

_SPARKOV_TRAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "fraudTrain.csv"
_SPARKOV_TEST_PATH = Path(__file__).resolve().parent.parent / "data" / "fraudTest.csv"


def _load_sparkov_transactions(
    n_samples: int = 20, seed: int = 42
) -> list[dict[str, Any]] | None:
    """Attempt to load a sample of transactions from the Sparkov dataset.

    Returns None if the CSV files are not present, allowing the caller to
    fall back to synthetic data.
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    csv_path = _SPARKOV_TRAIN_PATH if _SPARKOV_TRAIN_PATH.exists() else _SPARKOV_TEST_PATH
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, nrows=50_000)
    rng = np.random.default_rng(seed)

    # Sample a mix: some fraud, some legitimate, some high-value
    fraud_rows = df[df["is_fraud"] == 1]
    legit_rows = df[df["is_fraud"] == 0]
    high_value = df[df["amt"] > 10_000]

    n_fraud = min(max(n_samples // 4, 2), len(fraud_rows))
    n_high = min(max(n_samples // 5, 1), len(high_value))
    n_legit = n_samples - n_fraud - n_high

    sampled = pd.concat([
        fraud_rows.sample(n=n_fraud, random_state=seed) if n_fraud > 0 else fraud_rows.head(0),
        high_value.sample(n=n_high, random_state=seed) if n_high > 0 else high_value.head(0),
        legit_rows.sample(n=n_legit, random_state=seed) if n_legit > 0 else legit_rows.head(0),
    ]).drop_duplicates().head(n_samples)

    transactions: list[dict[str, Any]] = []
    for idx, row in sampled.iterrows():
        transactions.append({
            "transaction_id": f"spk_{int(row.get('Unnamed: 0', idx)):07d}",
            "amount": float(row["amt"]),
            "currency": "USD",
            "merchant_name": str(row.get("merchant", "Unknown")),
            "merchant_category_code": str(row.get("category", "misc")),
            "card_type": "visa",
            "card_last_four": str(row.get("cc_num", 0))[-4:],
            "city": str(row.get("city", "Unknown")),
            "country": "US",
            "timestamp": str(row.get("trans_date_trans_time", datetime.now(tz=timezone.utc).isoformat())),
            "channel": "online" if rng.random() > 0.5 else "in_store",
            "is_recurring": bool(rng.random() < 0.2),
            "_label": "sparkov_fraud" if row.get("is_fraud", 0) == 1 else "sparkov_legit",
        })

    return transactions if transactions else None


# ---------------------------------------------------------------------------
# Embedding generation for transactions
# ---------------------------------------------------------------------------

def _transaction_to_text(txn: dict[str, Any]) -> str:
    """Convert a transaction dict into a natural-language description for embedding."""
    return (
        f"{txn.get('merchant_name', 'Unknown')} "
        f"{txn.get('merchant_category_code', '')} "
        f"${txn.get('amount', 0):.2f} {txn.get('currency', 'USD')} "
        f"{txn.get('channel', 'unknown')} "
        f"{txn.get('city', '')}, {txn.get('country', '')} "
        f"{'recurring' if txn.get('is_recurring') else 'one-time'}"
    )


def generate_embeddings(transactions: list[dict[str, Any]]) -> np.ndarray:
    """Generate embeddings for a list of transactions.

    Uses the local sentence-transformer model (all-MiniLM-L6-v2, 384 dims)
    when available.  Falls back to random 384-dimensional vectors otherwise.
    """
    if _USE_REAL_EMBEDDINGS and _embedding_generator is not None:
        texts = [_transaction_to_text(txn) for txn in transactions]
        embeddings = _embedding_generator.encode(texts)
        print(f"  Generated real embeddings: {embeddings.shape} via all-MiniLM-L6-v2")
        return embeddings
    else:
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((len(transactions), 384)).astype(np.float32)
        print(f"  Generated random embeddings: {embeddings.shape} (fallback)")
        return embeddings


# ---------------------------------------------------------------------------
# 1. Synthetic transaction generation
# ---------------------------------------------------------------------------

def generate_transactions(seed: int = 42) -> list[dict[str, Any]]:
    """Create a diverse set of synthetic transactions covering all
    risk levels: low risk, gray zone, high risk, and high value.

    Returns a list of transaction dicts that mirror real payment
    processor authorization messages.
    """
    rng = np.random.default_rng(seed)
    now = datetime.now(tz=timezone.utc).isoformat()

    # Manually constructed to exercise each pipeline path.
    transactions: list[dict[str, Any]] = [
        # Low risk -- small recurring in-store purchase
        {
            "transaction_id": "txn_000001",
            "amount": 12.50,
            "currency": "USD",
            "merchant_name": "CornerCafe",
            "merchant_category_code": "5812",
            "card_type": "visa",
            "card_last_four": "4321",
            "city": "San Francisco",
            "country": "US",
            "timestamp": now,
            "channel": "in_store",
            "is_recurring": True,
            "_label": "low_risk",
        },
        # Moderate / gray zone -- online electronics, first time merchant
        {
            "transaction_id": "txn_000002",
            "amount": 849.99,
            "currency": "USD",
            "merchant_name": "ElectroMart",
            "merchant_category_code": "5732",
            "card_type": "mastercard",
            "card_last_four": "7890",
            "city": "New York",
            "country": "US",
            "timestamp": now,
            "channel": "online",
            "is_recurring": False,
            "_label": "gray_zone",
        },
        # High risk -- large amount, high-risk country, new merchant
        {
            "transaction_id": "txn_000003",
            "amount": 4800.00,
            "currency": "USD",
            "merchant_name": "QuickTransfer",
            "merchant_category_code": "4829",
            "card_type": "visa",
            "card_last_four": "5555",
            "city": "Lagos",
            "country": "NG",
            "timestamp": now,
            "channel": "online",
            "is_recurring": False,
            "_label": "high_risk",
        },
        # High value -- triggers LLM regardless of ML score
        {
            "transaction_id": "txn_000004",
            "amount": 18500.00,
            "currency": "USD",
            "merchant_name": "LuxuryWatches.com",
            "merchant_category_code": "5944",
            "card_type": "amex",
            "card_last_four": "0001",
            "city": "Zurich",
            "country": "CH",
            "timestamp": now,
            "channel": "online",
            "is_recurring": False,
            "_label": "high_value",
        },
        # Low risk but high value -- tests override
        {
            "transaction_id": "txn_000005",
            "amount": 12000.00,
            "currency": "USD",
            "merchant_name": "TravelBooker",
            "merchant_category_code": "4722",
            "card_type": "visa",
            "card_last_four": "9999",
            "city": "San Francisco",
            "country": "US",
            "timestamp": now,
            "channel": "mobile",
            "is_recurring": True,
            "_label": "high_value_low_risk",
        },
        # Gray zone -- ATM withdrawal, moderate amount
        {
            "transaction_id": "txn_000006",
            "amount": 500.00,
            "currency": "USD",
            "merchant_name": "ATM-Downtown",
            "merchant_category_code": "6011",
            "card_type": "visa",
            "card_last_four": "3333",
            "city": "Houston",
            "country": "US",
            "timestamp": now,
            "channel": "atm",
            "is_recurring": False,
            "_label": "gray_zone_atm",
        },
    ]

    return transactions


# ---------------------------------------------------------------------------
# 2. Simulated ML model scoring
# ---------------------------------------------------------------------------

def ml_score_transaction(txn: dict[str, Any]) -> dict[str, Any]:
    """Simulate the XGBoost-based ML fraud scorer.

    Returns a dict with the ML score, top risk factors, and feature
    importances.  In production this calls ``SimulatedMLScorer.predict``
    (or a real trained model).
    """
    amount = txn["amount"]
    channel = txn.get("channel", "online")
    country = txn.get("country", "US")
    is_recurring = txn.get("is_recurring", False)

    risk = 0.0
    factors: list[str] = []

    # Amount-based risk
    if amount > 10_000:
        risk += 0.25
        factors.append("very_high_amount")
    elif amount > 5_000:
        risk += 0.15
        factors.append("high_amount")
    elif amount > 1_000:
        risk += 0.05

    # Channel risk
    if channel == "online":
        risk += 0.10
        factors.append("online_channel")
    elif channel == "atm":
        risk += 0.07
        factors.append("atm_withdrawal")

    # Country risk
    high_risk_countries = {"NG", "PH", "RO", "UA", "BY"}
    if country in high_risk_countries:
        risk += 0.20
        factors.append("high_risk_country")

    # Recurring reduces risk
    if is_recurring:
        risk -= 0.10

    score = max(0.0, min(1.0, risk))

    return {
        "ml_score": round(score, 4),
        "top_risk_factors": factors[:3],
        "feature_importances": {
            "amount": 0.18,
            "channel": 0.10,
            "country": 0.12,
            "is_recurring": 0.05,
        },
    }


# ---------------------------------------------------------------------------
# 3. Decision router
# ---------------------------------------------------------------------------

GRAY_ZONE_LOW = 0.3
GRAY_ZONE_HIGH = 0.7
HIGH_VALUE_THRESHOLD = 10_000


def decision_router(
    ml_result: dict[str, Any],
    txn: dict[str, Any],
) -> dict[str, Any]:
    """Determine whether the transaction needs RAG+LLM analysis.

    Rules:
      - ML score in [0.3, 0.7] (gray zone) -> invoke LLM.
      - Transaction amount > $10k (high value) -> invoke LLM.
      - Otherwise -> ML-only decision is sufficient.

    Returns a dict with routing decision and reasoning.
    """
    ml_score = ml_result["ml_score"]
    amount = txn["amount"]

    is_gray_zone = GRAY_ZONE_LOW <= ml_score <= GRAY_ZONE_HIGH
    is_high_value = amount > HIGH_VALUE_THRESHOLD

    reasons: list[str] = []
    if is_gray_zone:
        reasons.append(f"ML score {ml_score:.2f} is in gray zone [{GRAY_ZONE_LOW}, {GRAY_ZONE_HIGH}]")
    if is_high_value:
        reasons.append(f"Amount ${amount:,.2f} exceeds ${HIGH_VALUE_THRESHOLD:,} threshold")

    invoke_llm = is_gray_zone or is_high_value

    return {
        "invoke_llm": invoke_llm,
        "is_gray_zone": is_gray_zone,
        "is_high_value": is_high_value,
        "routing_reasons": reasons,
    }


# ---------------------------------------------------------------------------
# 4. Simulated RAG+LLM assessment
# ---------------------------------------------------------------------------

_FRAUD_PATTERNS: list[dict[str, Any]] = [
    {"pattern_id": "FP001", "description": "High-value electronics from overseas IP", "risk_weight": 0.8},
    {"pattern_id": "FP002", "description": "Rapid successive ATM withdrawals", "risk_weight": 0.6},
    {"pattern_id": "FP003", "description": "First-time luxury goods purchase", "risk_weight": 0.7},
]


def llm_assess_fraud(
    txn: dict[str, Any],
    ml_result: dict[str, Any],
) -> dict[str, Any]:
    """Simulate a RAG+LLM fraud assessment.

    In production, this retrieves similar fraud patterns from ChromaDB
    and sends the transaction context to Ollama phi3:mini for structured analysis.
    """
    amount = txn["amount"]
    ml_score = ml_result["ml_score"]
    country = txn.get("country", "US")

    # LLM produces its own score informed by patterns and ML context
    llm_risk = ml_score * 0.4  # start from ML baseline
    if amount > 10_000:
        llm_risk += 0.20
    if country in {"NG", "PH", "RO", "UA", "BY"}:
        llm_risk += 0.15
    if txn.get("channel") == "online":
        llm_risk += 0.08

    llm_score = max(0.0, min(1.0, llm_risk))
    matched_patterns = _FRAUD_PATTERNS[:2] if llm_score > 0.3 else []

    reasoning = (
        f"LLM analysis considered {len(matched_patterns)} historical patterns. "
        f"ML baseline score: {ml_score:.2f}. "
        f"After contextual analysis, adjusted risk to {llm_score:.2f}."
    )

    return {
        "llm_score": round(llm_score, 4),
        "matched_patterns": [p["pattern_id"] for p in matched_patterns],
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# 5. Combined scoring
# ---------------------------------------------------------------------------

def compute_final_assessment(
    txn: dict[str, Any],
    ml_result: dict[str, Any],
    routing: dict[str, Any],
    llm_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Produce the final fraud assessment combining ML and optional LLM scores.

    When LLM was invoked, the final score is a weighted combination.
    """
    ml_score = ml_result["ml_score"]

    if routing["invoke_llm"] and llm_result is not None:
        llm_score = llm_result["llm_score"]
        # Weighted combination: ML gets 60%, LLM gets 40%
        final_score = round(0.6 * ml_score + 0.4 * llm_score, 4)
        analysis_tier = "ml_plus_llm"
    else:
        llm_score = None
        final_score = ml_score
        analysis_tier = "ml_only"

    if final_score >= 0.7:
        risk_level, action = "high", "decline"
    elif final_score >= 0.3:
        risk_level, action = "medium", "flag_for_review"
    else:
        risk_level, action = "low", "approve"

    return {
        "transaction_id": txn["transaction_id"],
        "ml_score": ml_score,
        "llm_score": llm_score,
        "final_score": final_score,
        "analysis_tier": analysis_tier,
        "risk_level": risk_level,
        "recommended_action": action,
        "risk_factors": ml_result.get("top_risk_factors", []),
        "matched_patterns": llm_result.get("matched_patterns", []) if llm_result else [],
        "reasoning": llm_result.get("reasoning", "ML model decision.") if llm_result else "ML model decision.",
    }


# ---------------------------------------------------------------------------
# 6. Output formatting
# ---------------------------------------------------------------------------

def print_assessment(assessment: dict[str, Any]) -> None:
    """Pretty-print a fraud assessment with dual-layer details."""
    border = "=" * 64
    print(border)
    print(f"  FRAUD ASSESSMENT -- {assessment['transaction_id']}")
    print(border)
    print(f"  Analysis Tier     : {assessment['analysis_tier'].upper()}")
    print(f"  ML Score          : {assessment['ml_score']:.4f}")
    if assessment["llm_score"] is not None:
        print(f"  LLM Score         : {assessment['llm_score']:.4f}")
    else:
        print(f"  LLM Score         : (not invoked)")
    print(f"  Final Score       : {assessment['final_score']:.4f}")
    print(f"  Risk Level        : {assessment['risk_level'].upper()}")
    print(f"  Recommended Action: {assessment['recommended_action']}")
    if assessment["risk_factors"]:
        print(f"  Risk Factors      : {', '.join(assessment['risk_factors'])}")
    if assessment["matched_patterns"]:
        print(f"  Matched Patterns  : {', '.join(assessment['matched_patterns'])}")
    print()
    print(textwrap.fill(
        f"  Reasoning: {assessment['reasoning']}",
        width=62,
        initial_indent="  ",
        subsequent_indent="    ",
    ))
    print(border)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the dual-layer fraud detection demo pipeline."""
    print()
    print("Dual-Layer Fraud Detection Pipeline Demo")
    print("ML Model (primary) + RAG+LLM (complementary)")
    print("-" * 64)
    print()

    # Step 1 -- Load from Sparkov dataset if available, else generate synthetic
    sparkov_txns = _load_sparkov_transactions(n_samples=12)
    if sparkov_txns is not None:
        transactions = sparkov_txns
        print(f"Loaded {len(transactions)} transactions from Sparkov dataset.\n")
    else:
        transactions = generate_transactions()
        print(f"Sparkov dataset not found in data/. Using {len(transactions)} synthetic transactions.")
        print("  (Download Sparkov: kaggle datasets download -d kartik2112/fraud-detection -p data/ --unzip)\n")

    # Step 1b -- Generate real embeddings for all transactions
    print("Generating transaction embeddings...")
    embeddings = generate_embeddings(transactions)
    print(f"  Embedding matrix shape: {embeddings.shape}")
    print()

    # Tracking for summary
    ml_only_count = 0
    ml_plus_llm_count = 0
    results: list[dict[str, Any]] = []

    for txn in transactions:
        label = txn.pop("_label", "unknown")

        # Step 2 -- ML model scores ALL transactions
        ml_result = ml_score_transaction(txn)

        # Step 3 -- Decision router determines if LLM is needed
        routing = decision_router(ml_result, txn)

        # Step 4 -- Conditionally invoke RAG+LLM
        llm_result = None
        if routing["invoke_llm"]:
            llm_result = llm_assess_fraud(txn, ml_result)
            ml_plus_llm_count += 1
        else:
            ml_only_count += 1

        # Step 5 -- Combine scores into final assessment
        assessment = compute_final_assessment(txn, ml_result, routing, llm_result)
        assessment["_label"] = label
        results.append(assessment)

        # Step 6 -- Display
        print(f"  [{label}]")
        if routing["routing_reasons"]:
            for reason in routing["routing_reasons"]:
                print(f"    -> Routing: {reason}")
        else:
            print(f"    -> Routing: ML score is clear, no LLM needed")
        print()
        print_assessment(assessment)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("=" * 64)
    print("  PIPELINE SUMMARY")
    print("=" * 64)
    print(f"  Total transactions processed : {len(results)}")
    print(f"  ML-only decisions            : {ml_only_count}")
    print(f"  ML+LLM decisions             : {ml_plus_llm_count}")
    print()
    print(f"  {'Transaction':<16s} {'ML Score':>9s} {'LLM Score':>10s} {'Tier':<14s} {'Action':<16s}")
    print(f"  {'-' * 65}")
    for r in results:
        llm_str = f"{r['llm_score']:.4f}" if r["llm_score"] is not None else "   --"
        print(
            f"  {r['transaction_id']:<16s} "
            f"{r['ml_score']:>9.4f} "
            f"{llm_str:>10s} "
            f"{r['analysis_tier']:<14s} "
            f"{r['recommended_action']:<16s}"
        )
    print()
    print("-" * 64)
    print("Demo complete.")
    print(
        "The dual-layer architecture ensures high-confidence ML decisions "
        "are fast,\nwhile uncertain or high-value transactions get deeper "
        "RAG+LLM analysis."
    )
    print()

    # ---------------------------------------------------------------------------
    # Visualization: fraud score distribution
    # ---------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        final_scores = [r["final_score"] for r in results]
        ml_scores = [r["ml_score"] for r in results]
        llm_scores = [r["llm_score"] for r in results if r["llm_score"] is not None]
        tiers = [r["analysis_tier"] for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: final score distribution colored by analysis tier
        ml_only_scores = [s for s, t in zip(final_scores, tiers) if t == "ml_only"]
        ml_llm_scores = [s for s, t in zip(final_scores, tiers) if t == "ml_plus_llm"]

        bins = np.linspace(0.0, 1.0, 15)
        axes[0].hist(ml_only_scores, bins=bins, alpha=0.7, label="ML-only", color="#2196F3", edgecolor="white")
        axes[0].hist(ml_llm_scores, bins=bins, alpha=0.7, label="ML+LLM", color="#FF9800", edgecolor="white")
        axes[0].axvline(x=0.3, color="gray", linestyle="--", linewidth=0.8, label="Gray zone bounds")
        axes[0].axvline(x=0.7, color="gray", linestyle="--", linewidth=0.8)
        axes[0].set_xlabel("Final Fraud Score")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Final Score Distribution by Analysis Tier")
        axes[0].legend()

        # Right: ML score vs LLM score for transactions that used both
        if llm_scores:
            ml_for_llm = [r["ml_score"] for r in results if r["llm_score"] is not None]
            axes[1].scatter(ml_for_llm, llm_scores, c="#FF5722", alpha=0.8, edgecolors="white", s=80)
            axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3, label="ML = LLM")
            axes[1].set_xlabel("ML Score")
            axes[1].set_ylabel("LLM Score")
            axes[1].set_title("ML vs LLM Scores (Escalated Transactions)")
            axes[1].legend()
            axes[1].set_xlim(-0.05, 1.05)
            axes[1].set_ylim(-0.05, 1.05)
        else:
            axes[1].text(0.5, 0.5, "No transactions escalated to LLM",
                         ha="center", va="center", fontsize=12)
            axes[1].set_title("ML vs LLM Scores")

        plt.tight_layout()
        out_path = Path(__file__).resolve().parent.parent / "reports" / "fraud_score_distribution.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Score distribution chart saved to {out_path}")
    except ImportError:
        print("matplotlib not installed -- skipping visualization.")
    print()


if __name__ == "__main__":
    main()
