"""
Dual-layer fraud detection pipeline with drift-aware fallback.

Architecture
------------
1. **Primary scorer** -- an ML model (XGBoost / gradient boosted trees)
   produces a fraud probability at high speed for every transaction.
2. **Complementary layer** -- RAG+LLM is invoked selectively for
   gray-zone transactions, high-value transactions, explainability,
   novel-pattern detection, and audit-trail generation.
3. **Drift monitoring** -- continuous embedding drift detection guards
   the quality of the RAG retrieval layer.

Decision routing
~~~~~~~~~~~~~~~~
- ML score < gray_zone_lower  --> approve (ML only)
- ML score > gray_zone_upper  --> decline (ML only)
- gray_zone_lower <= score <= gray_zone_upper  --> invoke RAG+LLM
- amount > high_value_threshold --> always invoke RAG+LLM

Supports async processing for concurrent transaction evaluation.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.drift_detection.detectors import DriftReport, DriftSeverity, EmbeddingDriftDetector
from src.embeddings.generator import TransactionEmbeddingGenerator
from src.embeddings.store import EmbeddingStore
from src.fraud_detection.ml_scorer import MLFraudScorer, MLScoringResult
from src.fraud_detection.rag_retriever import FraudPatternRetriever
from src.fraud_detection.transaction_processor import (
    EnrichedTransaction,
    FraudAssessment,
    Transaction,
    TransactionProcessor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Runtime configuration for the dual-layer fraud detection pipeline."""

    fraud_threshold: float = 0.7
    retrieval_top_k: int = 5
    drift_evaluation_interval: int = 100
    fallback_on_critical_drift: bool = True
    model_version: str = "v1.0.0"
    rule_based_threshold: float = 0.85

    # Decision-routing thresholds
    gray_zone_lower: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="ML scores at or above this value enter the gray zone.",
    )
    gray_zone_upper: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="ML scores at or above this value are clear fraud.",
    )
    high_value_threshold: float = Field(
        default=10_000.0,
        ge=0.0,
        description=(
            "Transactions whose amount exceeds this threshold always "
            "receive complementary RAG+LLM analysis."
        ),
    )

    # Weight given to the ML score when combining with the LLM score.
    ml_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight of ML score in the blended ML+LLM score.",
    )


class PipelineState(BaseModel):
    """Mutable state tracked across pipeline invocations."""

    transactions_since_drift_check: int = 0
    last_drift_report: DriftReport | None = None
    production_embeddings_buffer: list[list[float]] = Field(default_factory=list)
    total_processed: int = 0
    total_fallback: int = 0
    total_ml_only: int = 0
    total_ml_plus_llm: int = 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class FraudDetectionPipeline:
    """Dual-layer, drift-aware fraud detection pipeline.

    The ML model is the **primary** scorer.  The RAG+LLM layer is
    **complementary** and invoked only when deeper analysis is warranted.

    Parameters
    ----------
    ml_scorer:
        Primary ML fraud scorer (XGBoost / gradient boosted trees).
    embedding_generator:
        Generates dense vectors from transaction data.
    embedding_store:
        Persists and queries transaction embeddings.
    retriever:
        Retrieves similar historical fraud patterns for RAG.
    drift_detector:
        Ensemble drift detector for monitoring distributional shift.
    config:
        Runtime configuration.
    llm_assessor:
        Optional callable ``(enriched_transaction, patterns) -> float``
        used as the complementary LLM scoring layer.  When *None*, a
        similarity-based heuristic is used instead.
    """

    def __init__(
        self,
        ml_scorer: MLFraudScorer,
        embedding_generator: TransactionEmbeddingGenerator,
        embedding_store: EmbeddingStore,
        retriever: FraudPatternRetriever,
        drift_detector: EmbeddingDriftDetector,
        config: PipelineConfig | None = None,
        llm_assessor: Any | None = None,
    ) -> None:
        self._ml_scorer = ml_scorer
        self._generator = embedding_generator
        self._store = embedding_store
        self._retriever = retriever
        self._drift_detector = drift_detector
        self._config = config or PipelineConfig()
        self._llm_assessor = llm_assessor
        self._processor = TransactionProcessor()
        self._state = PipelineState()

    # ------------------------------------------------------------------
    # Synchronous entry point
    # ------------------------------------------------------------------

    def process_transaction(
        self,
        transaction: dict[str, Any],
        history: dict[str, Any] | None = None,
    ) -> FraudAssessment:
        """Process a single transaction through the dual-layer pipeline.

        Steps:
          1. Validate and enrich.
          2. ML model scoring (primary).
          3. Generate embedding (for RAG and drift monitoring).
          4. Periodic drift evaluation.
          5. If critical drift and fallback enabled, use rule-based engine.
          6. Decision routing -- invoke RAG+LLM only when required.
          7. Store production embedding.

        Parameters
        ----------
        transaction:
            Raw transaction dictionary.
        history:
            Optional historical statistics for enrichment.

        Returns
        -------
        FraudAssessment
        """
        # Step 1 -- validate and enrich
        txn = self._processor.validate(transaction)
        enriched = self._processor.enrich(txn, history)

        # Step 2 -- primary ML scoring
        ml_result = self._ml_scorer.predict(enriched)
        logger.debug(
            "ML score for %s: %.4f (top factors: %s)",
            txn.transaction_id,
            ml_result.score,
            ml_result.top_risk_factors,
        )

        # Step 3 -- generate embedding (needed for RAG retrieval and drift)
        txn_dict = TransactionProcessor.to_embedding_text(enriched)
        embed_result = self._generator.generate_single(txn_dict)
        embedding = embed_result.embedding

        # Buffer for drift checks
        self._state.production_embeddings_buffer.append(embedding)
        self._state.transactions_since_drift_check += 1
        self._state.total_processed += 1

        # Step 4 -- periodic drift evaluation
        if (
            self._state.transactions_since_drift_check
            >= self._config.drift_evaluation_interval
        ):
            self._run_drift_evaluation()

        # Step 5 -- fallback on critical drift
        current_severity = self._current_drift_severity()
        if (
            self._config.fallback_on_critical_drift
            and current_severity == DriftSeverity.CRITICAL
        ):
            logger.warning(
                "Critical drift detected -- using rule-based fallback for %s",
                txn.transaction_id,
            )
            self._state.total_fallback += 1
            return self._rule_based_assessment(enriched, current_severity, ml_result)

        # Step 6 -- decision routing
        needs_llm = self._requires_llm_analysis(ml_result, enriched)

        if needs_llm:
            assessment = self._ml_plus_llm_assessment(
                txn, enriched, embedding, ml_result, current_severity,
            )
            self._state.total_ml_plus_llm += 1
        else:
            assessment = self._ml_only_assessment(
                txn, enriched, ml_result, current_severity,
            )
            self._state.total_ml_only += 1

        # Step 7 -- store production embedding
        self._store.add_embeddings(
            ids=[txn.transaction_id],
            embeddings=[embedding],
            metadatas=[
                {
                    "merchant_category_code": txn.merchant_category_code,
                    "amount_band": enriched.amount_band.value,
                    "country": txn.country,
                    "channel": txn.channel.value,
                    "fraud_score": assessment.fraud_score,
                    "analysis_tier": assessment.analysis_tier,
                }
            ],
        )

        return assessment

    # ------------------------------------------------------------------
    # Decision routing
    # ------------------------------------------------------------------

    def _requires_llm_analysis(
        self,
        ml_result: MLScoringResult,
        enriched: EnrichedTransaction,
    ) -> bool:
        """Determine whether the complementary RAG+LLM layer should be
        invoked for this transaction.

        Returns ``True`` when any of the following hold:
        - The ML score falls in the gray zone
          (``gray_zone_lower <= score <= gray_zone_upper``).
        - The transaction amount exceeds ``high_value_threshold``.
        """
        cfg = self._config
        score = ml_result.score

        # Gray-zone: ML model is uncertain.
        if cfg.gray_zone_lower <= score <= cfg.gray_zone_upper:
            logger.debug(
                "Gray-zone score (%.4f) -- routing to RAG+LLM", score,
            )
            return True

        # High-value transactions always need deeper analysis.
        if enriched.transaction.amount > cfg.high_value_threshold:
            logger.debug(
                "High-value transaction ($%.2f) -- routing to RAG+LLM",
                enriched.transaction.amount,
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Assessment paths
    # ------------------------------------------------------------------

    def _ml_only_assessment(
        self,
        txn: Transaction,
        enriched: EnrichedTransaction,
        ml_result: MLScoringResult,
        severity: DriftSeverity,
    ) -> FraudAssessment:
        """Build a ``FraudAssessment`` using only the ML model score."""
        score = ml_result.score
        is_fraud = score >= self._config.gray_zone_upper

        factors_str = ", ".join(ml_result.top_risk_factors) if ml_result.top_risk_factors else "none"
        decision = "fraudulent" if is_fraud else "legitimate"
        explanation = (
            f"ML model assessed transaction as {decision} "
            f"(score={score:.4f}). Top risk factors: {factors_str}."
        )

        return FraudAssessment(
            transaction_id=txn.transaction_id,
            fraud_score=score,
            is_fraud=is_fraud,
            confidence=self._score_to_confidence(score),
            explanation=explanation,
            drift_severity=severity.value,
            model_version=self._config.model_version,
            ml_score=score,
            llm_score=None,
            analysis_tier="ml_only",
        )

    def _ml_plus_llm_assessment(
        self,
        txn: Transaction,
        enriched: EnrichedTransaction,
        embedding: list[float],
        ml_result: MLScoringResult,
        severity: DriftSeverity,
    ) -> FraudAssessment:
        """Build a ``FraudAssessment`` that combines ML scoring with
        complementary RAG+LLM analysis.
        """
        # RAG retrieval
        patterns = self._retriever.retrieve(
            transaction_embedding=embedding,
            top_k=self._config.retrieval_top_k,
            query_id=txn.transaction_id,
        )

        # LLM scoring (complementary layer)
        llm_score = self._compute_llm_score(enriched, patterns)

        # Blend ML and LLM scores
        w = self._config.ml_weight
        blended = w * ml_result.score + (1.0 - w) * llm_score
        blended = max(0.0, min(1.0, blended))

        is_fraud = blended >= self._config.fraud_threshold

        # Build rich explanation
        explanation = self._build_dual_layer_explanation(
            ml_result, llm_score, patterns, blended, is_fraud,
        )

        return FraudAssessment(
            transaction_id=txn.transaction_id,
            fraud_score=blended,
            is_fraud=is_fraud,
            confidence=self._score_to_confidence(blended),
            explanation=explanation,
            similar_fraud_ids=[p.transaction_id for p in patterns[:3]],
            drift_severity=severity.value,
            model_version=self._config.model_version,
            ml_score=ml_result.score,
            llm_score=llm_score,
            analysis_tier="ml_plus_llm",
        )

    # ------------------------------------------------------------------
    # Async entry points
    # ------------------------------------------------------------------

    async def process_transaction_async(
        self,
        transaction: dict[str, Any],
        history: dict[str, Any] | None = None,
    ) -> FraudAssessment:
        """Async wrapper that delegates to the synchronous pipeline
        via an executor, allowing concurrent transaction processing.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.process_transaction, transaction, history,
        )

    async def process_batch_async(
        self,
        transactions: list[dict[str, Any]],
        histories: list[dict[str, Any] | None] | None = None,
    ) -> list[FraudAssessment]:
        """Process multiple transactions concurrently."""
        if histories is None:
            histories = [None] * len(transactions)

        tasks = [
            self.process_transaction_async(txn, hist)
            for txn, hist in zip(transactions, histories)
        ]
        return await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # Drift evaluation
    # ------------------------------------------------------------------

    def _run_drift_evaluation(self) -> None:
        """Evaluate drift using buffered production embeddings."""
        if len(self._state.production_embeddings_buffer) < 2:
            return

        reference = self._store.get_reference_distribution(limit=5_000)
        if reference.size == 0:
            logger.warning("No reference distribution available -- skipping drift check")
            self._state.transactions_since_drift_check = 0
            return

        production = np.asarray(
            self._state.production_embeddings_buffer, dtype=np.float32,
        )

        try:
            report = self._drift_detector.evaluate(reference, production)
            self._state.last_drift_report = report
            logger.info(
                "Drift evaluation: severity=%s (n_ref=%d, n_prod=%d)",
                report.overall_severity.value,
                reference.shape[0],
                production.shape[0],
            )
        except Exception:
            logger.exception("Drift evaluation failed")

        # Reset buffer
        self._state.production_embeddings_buffer.clear()
        self._state.transactions_since_drift_check = 0

    def _current_drift_severity(self) -> DriftSeverity:
        if self._state.last_drift_report is not None:
            return self._state.last_drift_report.overall_severity
        return DriftSeverity.NONE

    # ------------------------------------------------------------------
    # LLM scoring (complementary layer)
    # ------------------------------------------------------------------

    def _compute_llm_score(
        self, enriched: EnrichedTransaction, patterns: list[Any],
    ) -> float:
        """Compute a fraud score using the LLM assessor or a heuristic.

        This is the *complementary* scoring layer, invoked selectively
        for gray-zone and high-value transactions.
        """
        if self._llm_assessor is not None:
            try:
                score = float(self._llm_assessor(enriched, patterns))
                return max(0.0, min(1.0, score))
            except Exception:
                logger.exception("LLM assessor failed -- using heuristic")

        # Heuristic fallback: weighted average of similarity scores
        if not patterns:
            return 0.1

        similarities = [p.similarity_score for p in patterns]
        weighted = sum(s ** 2 for s in similarities) / sum(similarities)
        return max(0.0, min(1.0, weighted))

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_assessment(
        self,
        enriched: EnrichedTransaction,
        severity: DriftSeverity,
        ml_result: MLScoringResult | None = None,
    ) -> FraudAssessment:
        """Simple rule-based fallback when drift is critical.

        If an ``ml_result`` is available its score is recorded but the
        final decision is driven by the conservative rule engine.
        """
        txn = enriched.transaction
        risk_score = 0.0

        if enriched.is_high_risk_country:
            risk_score += 0.3
        if enriched.is_new_merchant:
            risk_score += 0.2
        if txn.amount > 5_000:
            risk_score += 0.2
        if txn.channel.value == "online":
            risk_score += 0.1
        if enriched.txn_count_30d is not None and enriched.txn_count_30d < 3:
            risk_score += 0.1

        risk_score = min(1.0, risk_score)
        is_fraud = risk_score >= self._config.rule_based_threshold

        return FraudAssessment(
            transaction_id=txn.transaction_id,
            fraud_score=risk_score,
            is_fraud=is_fraud,
            confidence=0.4,  # Low confidence for rule-based
            explanation="Rule-based fallback due to critical embedding drift.",
            drift_severity=severity.value,
            model_version=f"{self._config.model_version}-fallback",
            ml_score=ml_result.score if ml_result is not None else None,
            llm_score=None,
            analysis_tier="rule_based_fallback",
        )

    # ------------------------------------------------------------------
    # Explanation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_confidence(score: float) -> float:
        """Map a fraud score to a confidence value.

        Scores near 0 or 1 are high-confidence; scores near 0.5 are
        low-confidence.
        """
        return abs(2.0 * score - 1.0)

    @staticmethod
    def _build_dual_layer_explanation(
        ml_result: MLScoringResult,
        llm_score: float,
        patterns: list[Any],
        blended_score: float,
        is_fraud: bool,
    ) -> str:
        """Build a human-readable explanation combining both scoring layers."""
        decision = "fraudulent" if is_fraud else "legitimate"
        factors_str = (
            ", ".join(ml_result.top_risk_factors)
            if ml_result.top_risk_factors
            else "none identified"
        )
        n_similar = len(patterns)
        pattern_info = (
            f"RAG retrieval found {n_similar} similar historical pattern(s)."
            if n_similar > 0
            else "No similar historical patterns found."
        )

        return (
            f"Dual-layer analysis assessed transaction as {decision} "
            f"(blended score={blended_score:.4f}). "
            f"ML model score: {ml_result.score:.4f} "
            f"(top risk factors: {factors_str}). "
            f"LLM complementary score: {llm_score:.4f}. "
            f"{pattern_info}"
        )
