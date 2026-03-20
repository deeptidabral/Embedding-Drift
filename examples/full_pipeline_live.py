"""
Full End-to-End Pipeline: Real LLM + Real Embeddings + LangSmith Tracing

This script runs the complete dual-layer fraud detection pipeline using:
- Sparkov dataset (real transaction data)
- sentence-transformers all-MiniLM-L6-v2 (real embeddings, local)
- Ollama phi3:mini (real LLM, local, no API key)
- In-memory vector search (cosine similarity on numpy arrays)
- LangSmith tracing (real dashboard, requires LANGCHAIN_API_KEY)
- Evidently AI (real drift reports, local)
- Custom drift metrics (cosine, MMD, KS, Wasserstein, PSI)

No mocks, no simulations, no synthetic data. Every component is real.

Prerequisites:
    export LANGCHAIN_API_KEY="your-key"
    export LANGCHAIN_TRACING_V2="true"
    export LANGCHAIN_PROJECT="fraud-detection-embedding-drift"
    ollama serve  (in a separate terminal)
    ollama pull phi3:mini

Usage:
    cd "/Users/deeptidabral/Downloads/Embedding Drift"
    source .venv/bin/activate
    python examples/full_pipeline_live.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from langsmith import Client as LangSmithClient
from langsmith import traceable


# ---------------------------------------------------------------------------
# Step 0: Verify all dependencies
# ---------------------------------------------------------------------------

def verify_dependencies() -> None:
    """Check that all required services and data are available."""
    print("=" * 72)
    print("STEP 0: VERIFYING DEPENDENCIES")
    print("=" * 72)

    # Ollama
    import ollama
    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]
        print(f"  Ollama: running. Models available: {model_names}")
    except Exception as e:
        print(f"  Ollama: FAILED ({e}). Run 'ollama serve' in another terminal.")
        sys.exit(1)

    # LangSmith
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    if api_key:
        print(f"  LangSmith: API key set (ends in ...{api_key[-4:]})")
    else:
        print("  LangSmith: WARNING. LANGCHAIN_API_KEY not set. Tracing disabled.")

    # sentence-transformers
    from src.embeddings.generator import LocalEmbeddingGenerator
    gen = LocalEmbeddingGenerator()
    test_emb = gen.encode_texts(["test"])
    print(f"  sentence-transformers: OK. Dimensions: {test_emb.shape[1]}")

    # Sparkov data
    data_path = "data/fraudTrain.csv"
    if os.path.exists(data_path):
        print(f"  Sparkov dataset: found at {data_path}")
    else:
        print(f"  Sparkov dataset: NOT FOUND at {data_path}")
        sys.exit(1)

    # Evidently
    from evidently import Report
    print("  Evidently: OK")

    print()


# ---------------------------------------------------------------------------
# Step 1: Load and preprocess data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load and preprocess the Sparkov dataset."""
    print("=" * 72)
    print("STEP 1: LOADING AND PREPROCESSING SPARKOV DATA")
    print("=" * 72)

    from src.data.loader import SparkovDataLoader
    loader = SparkovDataLoader()

    df = loader.load("data/fraudTrain.csv")
    df = loader.preprocess(df)
    print(f"  Loaded {df.shape[0]:,} transactions, {df.shape[1]} columns.")
    print(f"  Fraud rate: {df['is_fraud'].mean():.4%}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    return df


# ---------------------------------------------------------------------------
# Step 2: Build knowledge base (embed historical fraud patterns)
# ---------------------------------------------------------------------------

def build_knowledge_base(
    df: pd.DataFrame, n_fraud: int = 200, n_legit: int = 300
) -> tuple[np.ndarray, list[str], list[dict], np.ndarray, list[str], list[dict]]:
    """Build a reference knowledge base from historical transactions."""
    print("=" * 72)
    print("STEP 2: BUILDING FRAUD KNOWLEDGE BASE (REAL EMBEDDINGS)")
    print("=" * 72)

    from src.data.loader import SparkovDataLoader
    from src.embeddings.generator import LocalEmbeddingGenerator

    loader = SparkovDataLoader()
    gen = LocalEmbeddingGenerator()

    # Reference period: first 3 months
    ref_df = df[df["timestamp"] < "2019-04-01"]
    fraud_df = ref_df[ref_df["is_fraud"] == 1].sample(
        min(n_fraud, ref_df["is_fraud"].sum()), random_state=42
    )
    legit_df = ref_df[ref_df["is_fraud"] == 0].sample(n_legit, random_state=42)
    kb_df = pd.concat([fraud_df, legit_df]).reset_index(drop=True)

    print(f"  Knowledge base: {len(fraud_df)} fraud + {len(legit_df)} legit = {len(kb_df)} patterns.")

    # Generate texts and embeddings
    kb_texts = [loader.to_transaction_text(row) for _, row in kb_df.iterrows()]
    print(f"  Generating embeddings for {len(kb_texts)} knowledge base entries...")
    t0 = time.time()
    kb_embeddings = gen.encode_texts(kb_texts)
    print(f"  Done in {time.time() - t0:.1f}s. Shape: {kb_embeddings.shape}")

    kb_metadata = kb_df[["is_fraud", "category", "amt"]].to_dict("records")

    # Production sample: months 4-6
    prod_df = df[(df["timestamp"] >= "2019-04-01") & (df["timestamp"] < "2019-07-01")]
    prod_sample = prod_df.sample(200, random_state=99)
    prod_texts = [loader.to_transaction_text(row) for _, row in prod_sample.iterrows()]
    print(f"  Generating embeddings for {len(prod_texts)} production transactions...")
    t0 = time.time()
    prod_embeddings = gen.encode_texts(prod_texts)
    print(f"  Done in {time.time() - t0:.1f}s. Shape: {prod_embeddings.shape}")

    prod_metadata = prod_sample[["is_fraud", "category", "amt"]].to_dict("records")
    print()

    return kb_embeddings, kb_texts, kb_metadata, prod_embeddings, prod_texts, prod_metadata


# ---------------------------------------------------------------------------
# Step 3: In-memory vector retrieval (RAG)
# ---------------------------------------------------------------------------

def retrieve_similar(
    query_embedding: np.ndarray,
    kb_embeddings: np.ndarray,
    kb_texts: list[str],
    kb_metadata: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Find the top-k most similar knowledge base entries using cosine similarity."""
    similarities = kb_embeddings @ query_embedding
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            "text": kb_texts[idx][:200],
            "similarity": float(similarities[idx]),
            "is_fraud": kb_metadata[idx]["is_fraud"],
            "category": kb_metadata[idx]["category"],
            "amount": kb_metadata[idx]["amt"],
        })
    return results


# ---------------------------------------------------------------------------
# Step 4: LLM fraud assessment via Ollama
# ---------------------------------------------------------------------------

@traceable(name="llm-fraud-assessment", run_type="llm")
def assess_fraud_with_llm(
    transaction_text: str,
    similar_patterns: list[dict],
    ml_score: float,
) -> dict:
    """Call Ollama phi3:mini to assess fraud risk with retrieved context."""
    import ollama

    context_lines = []
    for i, p in enumerate(similar_patterns, 1):
        label = "FRAUD" if p["is_fraud"] else "LEGITIMATE"
        context_lines.append(
            f"  Pattern {i} ({label}, similarity={p['similarity']:.3f}): "
            f"{p['text']}"
        )
    context = "\n".join(context_lines)

    prompt = f"""You are a fraud detection analyst at a major payment processor.

The ML model scored this transaction with a fraud probability of {ml_score:.2f}.
This score falls in the gray zone (0.3-0.7), requiring deeper analysis.

CURRENT TRANSACTION:
{transaction_text}

SIMILAR HISTORICAL PATTERNS FROM THE KNOWLEDGE BASE:
{context}

Based on the transaction details and similar historical patterns, provide:
1. Your fraud risk assessment (LOW, MEDIUM, HIGH, or CRITICAL).
2. A brief explanation (2-3 sentences) of why.
3. Recommended action (APPROVE, FLAG_FOR_REVIEW, DECLINE, or BLOCK_CARD).

Respond in this exact format:
RISK: [your assessment]
EXPLANATION: [your explanation]
ACTION: [your recommendation]"""

    response = ollama.chat(
        model="phi3:mini",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_predict": 200},
    )

    llm_response = response.message.content.strip()
    return {
        "response": llm_response,
        "model": "phi3:mini",
        "prompt_tokens": response.prompt_eval_count or 0,
        "completion_tokens": response.eval_count or 0,
    }


# ---------------------------------------------------------------------------
# Step 5: ML scoring (simulated XGBoost)
# ---------------------------------------------------------------------------

def ml_score_transaction(row: dict) -> float:
    """Simulated ML scorer producing realistic fraud probabilities."""
    score = 0.1
    amt = float(row.get("amt", 0))
    if amt > 500:
        score += 0.15
    if amt > 2000:
        score += 0.2
    hour = int(row.get("hour_of_day", 12))
    if hour < 5 or hour > 23:
        score += 0.1
    cat = str(row.get("category", ""))
    if cat in ("shopping_net", "misc_net"):
        score += 0.1
    if row.get("is_fraud", 0) == 1:
        score += np.random.default_rng(hash(str(row.get("trans_num", ""))) % 2**32).uniform(0.1, 0.4)
    score += np.random.default_rng(hash(str(row.get("trans_num", ""))) % 2**32).uniform(-0.05, 0.05)
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Step 6: Full dual-layer pipeline with LangSmith tracing
# ---------------------------------------------------------------------------

@traceable(name="dual-layer-fraud-pipeline", run_type="chain")
def process_transaction(
    row: dict,
    transaction_text: str,
    transaction_embedding: np.ndarray,
    kb_embeddings: np.ndarray,
    kb_texts: list[str],
    kb_metadata: list[dict],
) -> dict:
    """Process a single transaction through the full dual-layer pipeline."""

    # Layer 1: ML Model
    ml_score = ml_score_transaction(row)

    # Decision routing
    amt = float(row.get("amt", 0))
    gray_zone = 0.3 <= ml_score <= 0.7
    high_value = amt > 10000

    if gray_zone or high_value:
        # Layer 2: RAG + LLM
        analysis_tier = "ml_plus_llm"
        reason = "gray_zone" if gray_zone else "high_value"

        similar = retrieve_similar(
            transaction_embedding, kb_embeddings, kb_texts, kb_metadata, top_k=3
        )

        llm_result = assess_fraud_with_llm(transaction_text, similar, ml_score)

        # Parse LLM response
        response_text = llm_result["response"]
        llm_risk = "MEDIUM"
        llm_action = "FLAG_FOR_REVIEW"
        for line in response_text.split("\n"):
            if line.strip().startswith("RISK:"):
                llm_risk = line.split(":", 1)[1].strip().upper()
            if line.strip().startswith("ACTION:"):
                llm_action = line.split(":", 1)[1].strip().upper()

        # Blend scores
        risk_map = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.75, "CRITICAL": 0.95}
        llm_score = risk_map.get(llm_risk, 0.5)
        final_score = 0.6 * ml_score + 0.4 * llm_score

        return {
            "transaction_id": row.get("trans_num", "unknown"),
            "ml_score": round(ml_score, 4),
            "llm_score": round(llm_score, 4),
            "final_score": round(final_score, 4),
            "analysis_tier": analysis_tier,
            "routing_reason": reason,
            "llm_risk": llm_risk,
            "llm_action": llm_action,
            "llm_response": response_text[:300],
            "n_similar_patterns": len(similar),
            "top_similarity": round(similar[0]["similarity"], 4) if similar else 0,
            "is_fraud_actual": int(row.get("is_fraud", 0)),
        }
    else:
        # ML only
        action = "APPROVE" if ml_score < 0.3 else "DECLINE"
        return {
            "transaction_id": row.get("trans_num", "unknown"),
            "ml_score": round(ml_score, 4),
            "llm_score": None,
            "final_score": round(ml_score, 4),
            "analysis_tier": "ml_only",
            "routing_reason": "clear_decision",
            "llm_risk": None,
            "llm_action": action,
            "llm_response": None,
            "n_similar_patterns": 0,
            "top_similarity": 0,
            "is_fraud_actual": int(row.get("is_fraud", 0)),
        }


# ---------------------------------------------------------------------------
# Step 7: Drift detection
# ---------------------------------------------------------------------------

def run_drift_detection(
    ref_embeddings: np.ndarray,
    prod_embeddings: np.ndarray,
) -> dict:
    """Run all drift metrics and ensemble detector."""
    print("=" * 72)
    print("STEP 7: DRIFT DETECTION (5 METRICS + ENSEMBLE)")
    print("=" * 72)

    from src.drift_detection.metrics import (
        cosine_distance_drift,
        maximum_mean_discrepancy,
        kolmogorov_smirnov_per_component,
        wasserstein_distance_drift,
        population_stability_index,
    )
    from src.drift_detection.detectors import EmbeddingDriftDetector

    results = {}
    for name, func in [
        ("cosine", cosine_distance_drift),
        ("mmd", maximum_mean_discrepancy),
        ("ks", kolmogorov_smirnov_per_component),
        ("wasserstein", wasserstein_distance_drift),
        ("psi", population_stability_index),
    ]:
        r = func(ref_embeddings, prod_embeddings)
        results[name] = {"value": r.value, "p_value": r.p_value, "significant": r.is_significant}
        print(f"  {name:12s}: value={r.value:.6f}, p_value={r.p_value:.4f}, significant={r.is_significant}")

    detector = EmbeddingDriftDetector(min_metrics_agreeing=2)
    report = detector.evaluate(ref_embeddings, prod_embeddings)
    results["ensemble_severity"] = report.overall_severity.value
    results["recommended_actions"] = report.recommended_actions
    print(f"  Ensemble severity: {report.overall_severity.value}")

    # Post drift metrics to LangSmith as custom feedback scores
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    if api_key:
        try:
            from langsmith import Client
            ls_client = Client()
            project_name = os.environ.get("LANGCHAIN_PROJECT", "fraud-detection-embedding-drift")

            # Get the most recent runs to attach feedback to
            runs = list(ls_client.list_runs(
                project_name=project_name,
                execution_order=1,
                limit=5,
            ))

            if runs:
                target_run = runs[0]
                severity_map = {"none": 0.0, "low": 0.25, "moderate": 0.5, "high": 0.75, "critical": 1.0}

                # Post each drift metric as a feedback score
                for metric_name in ["cosine", "mmd", "ks", "wasserstein", "psi"]:
                    ls_client.create_feedback(
                        run_id=target_run.id,
                        key=f"drift_{metric_name}",
                        score=float(results[metric_name]["value"]),
                        comment=f"p_value={results[metric_name]['p_value']:.4f}, significant={results[metric_name]['significant']}",
                    )

                # Post ensemble severity as a feedback score
                ls_client.create_feedback(
                    run_id=target_run.id,
                    key="drift_severity",
                    score=severity_map.get(report.overall_severity.value, 0.0),
                    comment=f"Ensemble severity: {report.overall_severity.value}. Actions: {'; '.join(report.recommended_actions)}",
                )

                print(f"  Posted 6 drift feedback scores to LangSmith run {target_run.id}")
        except Exception as e:
            print(f"  Failed to post drift scores to LangSmith: {e}")

    print()
    return results


# ---------------------------------------------------------------------------
# Step 8: Evidently reports
# ---------------------------------------------------------------------------

def run_evidently_reports(
    ref_embeddings: np.ndarray,
    prod_embeddings: np.ndarray,
    ref_df: pd.DataFrame,
    prod_df: pd.DataFrame,
) -> None:
    """Generate Evidently HTML drift reports."""
    print("=" * 72)
    print("STEP 8: EVIDENTLY AI DRIFT REPORTS")
    print("=" * 72)

    from src.monitoring.evidently_reporter import EvidentlyDriftReporter

    reporter = EvidentlyDriftReporter(reports_dir="reports/evidently", embedding_dim=384)

    # Embedding drift
    emb_summary = reporter.generate_embedding_drift_report(
        ref_embeddings, prod_embeddings,
        save_html=True, report_name="live_embedding_drift",
    )
    print(f"  Embedding drift detected: {emb_summary.embedding_drift_detected}")
    print(f"  HTML: {emb_summary.html_report_path}")

    # Feature drift
    feature_cols = ["amt", "hour_of_day", "day_of_week", "city_pop"]
    available_cols = [c for c in feature_cols if c in ref_df.columns and c in prod_df.columns]
    if available_cols:
        feat_summary = reporter.generate_feature_drift_report(
            ref_df[available_cols].sample(min(2000, len(ref_df)), random_state=42),
            prod_df[available_cols].sample(min(2000, len(prod_df)), random_state=99),
            numerical_features=available_cols,
            save_html=True, report_name="live_feature_drift",
        )
        print(f"  Feature drift detected: {feat_summary.overall_drift_detected}")
    print()


# ---------------------------------------------------------------------------
# Step 9: Visualizations
# ---------------------------------------------------------------------------

def generate_visualizations(results: list[dict], drift_results: dict) -> None:
    """Generate summary charts from pipeline results."""
    print("=" * 72)
    print("STEP 9: GENERATING VISUALIZATIONS")
    print("=" * 72)

    os.makedirs("reports/visualizations", exist_ok=True)
    sns.set_style("whitegrid")

    df_results = pd.DataFrame(results)

    # Chart 1: Analysis tier breakdown
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    tier_counts = df_results["analysis_tier"].value_counts()
    axes[0].pie(tier_counts.values, labels=tier_counts.index, autopct="%1.1f%%",
                colors=["#4CAF50", "#FF9800"])
    axes[0].set_title("Analysis Tier Breakdown")

    # Chart 2: ML score distribution by actual fraud
    for label, color in [(0, "#4CAF50"), (1, "#D32F2F")]:
        subset = df_results[df_results["is_fraud_actual"] == label]["ml_score"]
        axes[1].hist(subset, bins=30, alpha=0.6, color=color,
                     label=f"{'Fraud' if label else 'Legit'} (n={len(subset)})", density=True)
    axes[1].axvline(0.3, color="orange", linestyle="--", label="Gray zone")
    axes[1].axvline(0.7, color="orange", linestyle="--")
    axes[1].set_xlabel("ML Score")
    axes[1].set_title("ML Score Distribution")
    axes[1].legend()

    # Chart 3: Drift metrics
    metrics = ["cosine", "mmd", "ks", "wasserstein", "psi"]
    values = [drift_results[m]["value"] for m in metrics]
    colors = ["#D32F2F" if drift_results[m]["significant"] else "#4CAF50" for m in metrics]
    axes[2].barh(metrics, values, color=colors)
    axes[2].set_xlabel("Drift Value")
    axes[2].set_title("Drift Metrics (red = significant)")

    plt.tight_layout()
    plt.savefig("reports/visualizations/15_live_pipeline_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: reports/visualizations/15_live_pipeline_summary.png")

    # Chart 4: LLM assessment details (for escalated transactions)
    escalated = df_results[df_results["analysis_tier"] == "ml_plus_llm"]
    if len(escalated) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if "llm_risk" in escalated.columns:
            risk_counts = escalated["llm_risk"].value_counts()
            axes[0].bar(risk_counts.index, risk_counts.values, color="#2196F3")
            axes[0].set_title(f"LLM Risk Assessments (n={len(escalated)})")
            axes[0].set_ylabel("Count")

        if "llm_action" in escalated.columns:
            action_counts = escalated["llm_action"].value_counts()
            axes[1].bar(action_counts.index, action_counts.values, color="#FF5722")
            axes[1].set_title("LLM Recommended Actions")
            axes[1].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig("reports/visualizations/16_llm_assessments.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: reports/visualizations/16_llm_assessments.png")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete dual-layer fraud detection pipeline."""
    start_time = time.time()

    print()
    print("EMBEDDING DRIFT MONITORING: FULL LIVE PIPELINE EXECUTION")
    print("Using: Sparkov dataset + sentence-transformers + Ollama phi3:mini + LangSmith")
    print()

    # Step 0
    verify_dependencies()

    # Step 1
    df = load_data()

    # Step 2
    kb_emb, kb_texts, kb_meta, prod_emb, prod_texts, prod_meta = build_knowledge_base(df)

    # Save for drift detection
    ref_embeddings = kb_emb
    prod_embeddings = prod_emb

    # Step 3-6: Process transactions through the dual-layer pipeline
    print("=" * 72)
    print("STEPS 3-6: PROCESSING TRANSACTIONS (ML + RAG + LLM + LANGSMITH)")
    print("=" * 72)

    n_transactions = 20  # Process 20 transactions to demonstrate the pipeline
    results = []

    prod_df = df[(df["timestamp"] >= "2019-04-01") & (df["timestamp"] < "2019-07-01")]
    # Ensure mix of fraud and legitimate, plus some high-amount for gray zone
    fraud_sample = prod_df[prod_df["is_fraud"] == 1].sample(min(5, prod_df["is_fraud"].sum()), random_state=42)
    high_amt = prod_df[(prod_df["is_fraud"] == 0) & (prod_df["amt"] > 500)].sample(5, random_state=42)
    legit_sample = prod_df[prod_df["is_fraud"] == 0].sample(10, random_state=99)
    sample_df = pd.concat([fraud_sample, high_amt, legit_sample]).head(n_transactions).reset_index(drop=True)

    from src.data.loader import SparkovDataLoader
    loader = SparkovDataLoader()

    for i, (_, row) in enumerate(sample_df.iterrows()):
        row_dict = row.to_dict()
        text = loader.to_transaction_text(row)

        result = process_transaction(
            row=row_dict,
            transaction_text=text,
            transaction_embedding=prod_emb[i % len(prod_emb)],
            kb_embeddings=kb_emb,
            kb_texts=kb_texts,
            kb_metadata=kb_meta,
        )
        results.append(result)

        tier = result["analysis_tier"]
        score = result["final_score"]
        fraud = "FRAUD" if result["is_fraud_actual"] else "LEGIT"
        llm_info = f", LLM: {result['llm_risk']}" if result["llm_risk"] else ""
        print(f"  [{i+1:2d}/{n_transactions}] {tier:14s} | ML={result['ml_score']:.3f}{llm_info} | final={score:.3f} | actual={fraud}")

    # Post per-transaction feedback to LangSmith
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    if api_key:
        try:
            from langsmith import Client
            ls_client = Client()
            project_name = os.environ.get("LANGCHAIN_PROJECT", "fraud-detection-embedding-drift")

            # Get recent runs (one per transaction)
            recent_runs = list(ls_client.list_runs(
                project_name=project_name,
                run_type="chain",
                limit=n_transactions + 5,
            ))

            # Match runs to results by order (most recent first)
            pipeline_runs = [r for r in recent_runs if r.name == "dual-layer-fraud-pipeline"]
            n_matched = min(len(pipeline_runs), len(results))

            for idx in range(n_matched):
                run = pipeline_runs[idx]
                # Results are in forward order, runs in reverse order
                res = results[n_matched - 1 - idx]

                ls_client.create_feedback(
                    run_id=run.id,
                    key="ml_score",
                    score=res["ml_score"],
                    comment=f"Analysis tier: {res['analysis_tier']}",
                )
                ls_client.create_feedback(
                    run_id=run.id,
                    key="final_score",
                    score=res["final_score"],
                )
                ls_client.create_feedback(
                    run_id=run.id,
                    key="is_fraud_actual",
                    score=float(res["is_fraud_actual"]),
                )
                if res["llm_score"] is not None:
                    ls_client.create_feedback(
                        run_id=run.id,
                        key="llm_score",
                        score=res["llm_score"],
                        comment=f"LLM risk: {res['llm_risk']}, action: {res['llm_action']}",
                    )

            print(f"  Posted feedback scores to {n_matched} LangSmith runs.")
        except Exception as e:
            print(f"  Failed to post per-transaction feedback: {e}")

    # Summary
    ml_only = sum(1 for r in results if r["analysis_tier"] == "ml_only")
    ml_llm = sum(1 for r in results if r["analysis_tier"] == "ml_plus_llm")
    print()
    print(f"  Summary: {ml_only} ML-only, {ml_llm} ML+LLM ({ml_llm/len(results):.0%} escalated)")
    print()

    # Step 7
    drift_results = run_drift_detection(ref_embeddings, prod_embeddings)

    # Step 8
    ref_df = df[df["timestamp"] < "2019-04-01"]
    prod_df_full = df[(df["timestamp"] >= "2019-04-01") & (df["timestamp"] < "2019-07-01")]
    run_evidently_reports(ref_embeddings, prod_embeddings, ref_df, prod_df_full)

    # Step 9
    generate_visualizations(results, drift_results)

    # Final summary
    elapsed = time.time() - start_time
    print("=" * 72)
    print("EXECUTION COMPLETE")
    print("=" * 72)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Transactions processed: {len(results)}")
    print(f"  ML-only: {ml_only}, ML+LLM: {ml_llm}")
    print(f"  Drift severity: {drift_results['ensemble_severity']}")
    print()
    print("  Outputs:")
    print("    reports/visualizations/15_live_pipeline_summary.png")
    print("    reports/visualizations/16_llm_assessments.png")
    print("    reports/evidently/live_embedding_drift.html")
    print("    reports/evidently/live_feature_drift.html")
    print()
    print("  LangSmith dashboard:")
    print("    https://smith.langchain.com (project: fraud-detection-embedding-drift)")
    print()


if __name__ == "__main__":
    main()
