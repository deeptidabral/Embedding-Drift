"""
Fraud detection pipeline with async LLM investigation and drift-aware fallback.

Architecture
------------
1. **Real-time ML scorer** -- an ML model (XGBoost / gradient boosted
   trees) produces a fraud probability at high speed for every
   transaction.  This is the only component in the synchronous
   authorization path.
2. **Async LLM investigation** -- RAG+LLM is invoked *asynchronously*
   (post-transaction) for flagged transactions: gray-zone scores,
   high-value transactions, novel-pattern detection, and audit-trail
   generation.  It does **not** block the authorization decision.
3. **Drift monitoring** -- continuous MMD-based embedding drift
   detection guards the quality of the embedding space.

Decision routing (synchronous, real-time)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ML score < gray_zone_lower  --> approve
- ML score > gray_zone_upper  --> decline
- gray_zone_lower <= score <= gray_zone_upper  --> flag for review,
  queue for async LLM investigation

The LLM/RAG layer is never in the synchronous authorization path.
It runs asynchronously after the transaction has been authorized,
declined, or flagged.
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
    """Runtime configuration for the fraud detection pipeline."""

    fraud_threshold: float = 0.7
    retrieval_top_k: int = 5
    drift_evaluation_interval: int = 100
    fallback_on_critical_drift: bool = True
    model_version: str = "v1.0.0"
    rule_based_threshold: float = 0.85

    # Decision-routing thresholds (synchronous authorization)
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
            "Transactions whose amount exceeds this threshold are flagged "
            "and queued for async RAG+LLM investigation."
        ),
    )


class PipelineState(BaseModel):
    """Mutable state tracked across pipeline invocations."""

    transactions_since_drift_check: int = 0
    last_drift_report: DriftReport | None = None
    production_embeddings_buffer: list[list[float]] = Field(default_factory=list)
    total_processed: int = 0
    total_fallback: int = 0
    total_approved: int = 0
    total_declined: int = 0
    total_flagged: int = 0
    total_async_llm_queued: int = 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class FraudDetectionPipeline:
    """ML-first fraud detection pipeline with async LLM investigation.

    The ML model is the **sole** real-time scorer in the authorization
    path.  The RAG+LLM layer runs **asynchronously** for flagged
    transactions and does not block authorization.

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
        MMD-based drift detector for monitoring distributional shift.
    config:
        Runtime configuration.
    llm_assessor:
        Optional callable ``(enriched_transaction, patterns) -> float``
        used as the async LLM investigation layer.  When *None*, a
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
    # Synchronous entry point (real-time authorization)
    # ------------------------------------------------------------------

    def process_transaction(
        self,
        transaction: dict[str, Any],
        history: dict[str, Any] | None = None,
    ) -> FraudAssessment:
        """Process a single transaction through the real-time pipeline.

        Steps:
          1. Validate and enrich.
          2. ML model scoring (real-time authorization).
          3. Generate embedding (for drift monitoring).
          4. Periodic drift evaluation.
          5. If critical drift and fallback enabled, use rule-based engine.
          6. Authorize / decline / flag based on ML score alone.
          7. If flagged, queue for async LLM investigation.
          8. Store production embedding.

        The LLM/RAG layer is NOT in this synchronous path.  It is
        triggered asynchronously for flagged transactions only.

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

        # Step 2 -- real-time ML scoring (authorization path)
        ml_result = self._ml_scorer.predict(enriched)
        logger.debug(
            "ML score for %s: %.4f (top factors: %s)",
            txn.transaction_id,
            ml_result.score,
            ml_result.top_risk_factors,
        )

        # Step 3 -- generate embedding (for drift monitoring)
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

        # Step 6 -- authorize / decline / flag (ML score only)
        should_flag = self._should_flag_for_review(ml_result, enriched)
        assessment = self._ml_authorization(
            txn, enriched, ml_result, current_severity, flagged=should_flag,
        )

        if should_flag:
            self._state.total_flagged += 1
            # Step 7 -- queue async LLM investigation for flagged transactions
            self._queue_async_llm_investigation(txn, enriched, embedding, ml_result)
        elif assessment.is_fraud:
            self._state.total_declined += 1
        else:
            self._state.total_approved += 1

        # Step 8 -- store production embedding
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
    # Decision routing (synchronous)
    # ------------------------------------------------------------------

    def _should_flag_for_review(
        self,
        ml_result: MLScoringResult,
        enriched: EnrichedTransaction,
    ) -> bool:
        """Determine whether this transaction should be flagged for review
        and queued for async LLM investigation.

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
                "Gray-zone score (%.4f) -- flagging for async LLM investigation",
                score,
            )
            return True

        # High-value transactions always need deeper (async) analysis.
        if enriched.transaction.amount > cfg.high_value_threshold:
            logger.debug(
                "High-value transaction ($%.2f) -- flagging for async LLM investigation",
                enriched.transaction.amount,
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Assessment (synchronous -- ML only)
    # ------------------------------------------------------------------

    def _ml_authorization(
        self,
        txn: Transaction,
        enriched: EnrichedTransaction,
        ml_result: MLScoringResult,
        severity: DriftSeverity,
        flagged: bool = False,
    ) -> FraudAssessment:
        """Build a ``FraudAssessment`` using the ML model score.

        This is the real-time authorization decision.  If the transaction
        is flagged, it is provisionally authorized (or declined) and
        separately queued for async LLM investigation.
        """
        score = ml_result.score
        is_fraud = score >= self._config.gray_zone_upper

        factors_str = ", ".join(ml_result.top_risk_factors) if ml_result.top_risk_factors else "none"

        if flagged:
            decision = "flagged for review"
            tier = "ml_flagged_for_async_investigation"
        elif is_fraud:
            decision = "declined (fraudulent)"
            tier = "ml_only"
        else:
            decision = "approved (legitimate)"
            tier = "ml_only"

        explanation = (
            f"Real-time ML authorization: {decision} "
            f"(score={score:.4f}). Top risk factors: {factors_str}."
        )
        if flagged:
            explanation += " Queued for async LLM/RAG investigation."

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
            analysis_tier=tier,
        )

    # ------------------------------------------------------------------
    # Async LLM investigation (post-transaction, non-blocking)
    # ------------------------------------------------------------------

    def _queue_async_llm_investigation(
        self,
        txn: Transaction,
        enriched: EnrichedTransaction,
        embedding: list[float],
        ml_result: MLScoringResult,
    ) -> None:
        """Queue a flagged transaction for async LLM/RAG investigation.

        This does NOT block the authorization decision.  The investigation
        runs in the background and its results are stored for analyst
        review.
        """
        self._state.total_async_llm_queued += 1
        logger.info(
            "Queued transaction %s for async LLM investigation (ML score=%.4f)",
            txn.transaction_id,
            ml_result.score,
        )
        # In production this would enqueue to a message broker (e.g. Kafka,
        # SQS).  The async consumer calls _run_llm_investigation().
        # For demonstration, we fire-and-forget if an event loop is running.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._run_llm_investigation_async(txn, enriched, embedding, ml_result)
            )
        except RuntimeError:
            # No event loop -- log for later batch processing.
            logger.debug(
                "No event loop available; transaction %s queued for batch LLM processing",
                txn.transaction_id,
            )

    async def _run_llm_investigation_async(
        self,
        txn: Transaction,
        enriched: EnrichedTransaction,
        embedding: list[float],
        ml_result: MLScoringResult,
    ) -> dict[str, Any]:
        """Perform the actual LLM/RAG investigation asynchronously.

        This runs outside the authorization path and stores the result
        for analyst review.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._run_llm_investigation,
            txn, enriched, embedding, ml_result,
        )
        return result

    def _run_llm_investigation(
        self,
        txn: Transaction,
        enriched: EnrichedTransaction,
        embedding: list[float],
        ml_result: MLScoringResult,
    ) -> dict[str, Any]:
        """Synchronous LLM/RAG investigation logic.

        Called from the async wrapper.  Retrieves similar patterns,
        runs the LLM assessor, and returns the investigation result.
        """
        patterns = self._retriever.retrieve(
            transaction_embedding=embedding,
            top_k=self._config.retrieval_top_k,
            query_id=txn.transaction_id,
        )

        llm_score = self._compute_llm_score(enriched, patterns)

        investigation = {
            "transaction_id": txn.transaction_id,
            "ml_score": ml_result.score,
            "llm_score": llm_score,
            "n_similar_patterns": len(patterns),
            "similar_fraud_ids": [p.transaction_id for p in patterns[:3]],
            "investigated_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Async LLM investigation complete for %s: llm_score=%.4f",
            txn.transaction_id,
            llm_score,
        )
        return investigation

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
    # LLM scoring (async investigation layer)
    # ------------------------------------------------------------------

    def _compute_llm_score(
        self, enriched: EnrichedTransaction, patterns: list[Any],
    ) -> float:
        """Compute a fraud score using the LLM assessor or a heuristic.

        This is the *async investigation* scoring layer, invoked
        post-transaction for flagged transactions only.
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
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_confidence(score: float) -> float:
        """Map a fraud score to a confidence value.

        Scores near 0 or 1 are high-confidence; scores near 0.5 are
        low-confidence.
        """
        return abs(2.0 * score - 1.0)
