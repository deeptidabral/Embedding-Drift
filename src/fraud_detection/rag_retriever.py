"""
RAG-based retrieval of historical fraud patterns.

Queries the vector store for transactions similar to the current one,
optionally reranks results with a cross-encoder, and tracks retrieval
quality metrics for drift correlation analysis.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.embeddings.store import EmbeddingStore, QueryResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FraudPattern(BaseModel):
    """A historical fraud pattern retrieved from the vector store."""

    transaction_id: str
    similarity_score: float
    rerank_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class RetrievalMetrics(BaseModel):
    """Quality metrics for a single retrieval operation."""

    query_id: str
    top_k: int
    n_results: int
    mean_similarity: float
    max_similarity: float
    latency_ms: float
    reranked: bool = False


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class FraudPatternRetriever:
    """Retrieve similar historical fraud patterns via vector search.

    Parameters
    ----------
    store:
        An ``EmbeddingStore`` instance connected to the fraud pattern
        collection.
    reranker:
        Optional callable implementing a cross-encoder reranker.  It
        should accept ``(query_embedding, candidate_embeddings)`` and
        return a numpy array of scores.
    default_top_k:
        Default number of candidates to retrieve before reranking.
    rerank_top_k:
        Number of results to keep after reranking.  Ignored when no
        reranker is provided.
    """

    def __init__(
        self,
        store: EmbeddingStore,
        reranker: Any | None = None,
        default_top_k: int = 20,
        rerank_top_k: int = 5,
    ) -> None:
        self._store = store
        self._reranker = reranker
        self._default_top_k = default_top_k
        self._rerank_top_k = rerank_top_k
        self._metrics_buffer: list[RetrievalMetrics] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        transaction_embedding: list[float],
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        query_id: str = "",
    ) -> list[FraudPattern]:
        """Retrieve the most similar historical fraud patterns.

        Parameters
        ----------
        transaction_embedding:
            Dense vector for the current transaction.
        top_k:
            Number of results to return.  When a reranker is configured,
            this controls the final count after reranking.
        where:
            Optional ChromaDB metadata filter.
        query_id:
            Identifier for this retrieval (used in metrics tracking).

        Returns
        -------
        list[FraudPattern]
            Fraud patterns sorted by relevance (highest first).
        """
        effective_top_k = top_k or self._default_top_k
        fetch_k = max(effective_top_k, self._default_top_k)

        start = time.perf_counter()
        raw_results: list[QueryResult] = self._store.query_similar(
            query_embedding=transaction_embedding,
            top_k=fetch_k,
            where=where,
            from_reference=True,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        patterns = self._to_fraud_patterns(raw_results)

        # Rerank if a cross-encoder is available
        reranked = False
        if self._reranker is not None and patterns:
            patterns = self._apply_reranking(
                transaction_embedding, patterns, self._rerank_top_k
            )
            reranked = True
        else:
            patterns = patterns[:effective_top_k]

        # Record metrics
        similarities = [p.similarity_score for p in patterns]
        metrics = RetrievalMetrics(
            query_id=query_id,
            top_k=effective_top_k,
            n_results=len(patterns),
            mean_similarity=float(np.mean(similarities)) if similarities else 0.0,
            max_similarity=float(np.max(similarities)) if similarities else 0.0,
            latency_ms=latency_ms,
            reranked=reranked,
        )
        self._metrics_buffer.append(metrics)
        logger.debug(
            "Retrieved %d patterns in %.1fms (reranked=%s)",
            len(patterns),
            latency_ms,
            reranked,
        )
        return patterns

    def get_metrics(self, clear: bool = True) -> list[RetrievalMetrics]:
        """Return and optionally clear the accumulated retrieval metrics."""
        metrics = list(self._metrics_buffer)
        if clear:
            self._metrics_buffer.clear()
        return metrics

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_fraud_patterns(results: list[QueryResult]) -> list[FraudPattern]:
        """Convert raw query results to ``FraudPattern`` objects."""
        patterns: list[FraudPattern] = []
        for r in results:
            # ChromaDB cosine distance is in [0, 2]; convert to similarity.
            similarity = max(0.0, 1.0 - r.distance)
            patterns.append(
                FraudPattern(
                    transaction_id=r.id,
                    similarity_score=similarity,
                    metadata=r.metadata,
                    embedding=r.embedding,
                )
            )
        return patterns

    def _apply_reranking(
        self,
        query_embedding: list[float],
        patterns: list[FraudPattern],
        top_k: int,
    ) -> list[FraudPattern]:
        """Rerank candidate patterns using the cross-encoder."""
        candidate_embeddings = []
        for p in patterns:
            if p.embedding is not None:
                candidate_embeddings.append(p.embedding)
            else:
                candidate_embeddings.append([0.0] * len(query_embedding))

        query_arr = np.asarray(query_embedding, dtype=np.float32)
        cand_arr = np.asarray(candidate_embeddings, dtype=np.float32)

        try:
            scores: np.ndarray = self._reranker(query_arr, cand_arr)
        except Exception:
            logger.exception("Reranker failed -- falling back to initial ranking")
            return patterns[:top_k]

        for i, pattern in enumerate(patterns):
            pattern.rerank_score = float(scores[i])

        patterns.sort(key=lambda p: p.rerank_score or 0.0, reverse=True)
        return patterns[:top_k]
