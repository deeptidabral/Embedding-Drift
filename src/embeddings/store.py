"""
In-memory vector store using numpy cosine similarity.

Supports collection versioning so that reference distributions can be
snapshotted independently of the live production collection, and
metadata filtering for segment-level drift analysis.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class QueryResult(BaseModel):
    """A single nearest-neighbour result from the vector store."""

    id: str
    distance: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Internal in-memory collection
# ---------------------------------------------------------------------------


class _InMemoryCollection:
    """Simple in-memory collection storing embeddings, ids, and metadata."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._ids: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._metadatas: list[dict[str, Any]] = []

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or update embeddings by id."""
        for i, uid in enumerate(ids):
            vec = np.asarray(embeddings[i], dtype=np.float32)
            meta = metadatas[i] if metadatas else {}
            if uid in self._ids:
                idx = self._ids.index(uid)
                self._embeddings[idx] = vec
                self._metadatas[idx] = meta
            else:
                self._ids.append(uid)
                self._embeddings.append(vec)
                self._metadatas.append(meta)

    def _matches_filter(self, meta: dict[str, Any], where: dict[str, Any]) -> bool:
        """Evaluate a metadata filter against a single metadata dict.

        Supports a limited subset of filter syntax:
        - ``{"field": value}`` -- exact match
        - ``{"field": {"$gte": v}}`` / ``{"$lte": v}`` -- range comparison
        - ``{"$and": [filter, ...]}`` -- conjunction
        """
        for key, condition in where.items():
            if key == "$and":
                if not all(self._matches_filter(meta, sub) for sub in condition):
                    return False
            elif isinstance(condition, dict):
                val = meta.get(key)
                if val is None:
                    return False
                for op, threshold in condition.items():
                    if op == "$gte" and not (val >= threshold):
                        return False
                    elif op == "$lte" and not (val <= threshold):
                        return False
                    elif op == "$eq" and not (val == threshold):
                        return False
                    elif op == "$in" and val not in threshold:
                        return False
            else:
                if meta.get(key) != condition:
                    return False
        return True

    def get_embeddings(
        self,
        limit: int,
        where: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return stored embeddings as an (N, D) numpy array."""
        if not self._embeddings:
            return np.empty((0, 0), dtype=np.float32)

        if where is not None:
            indices = [
                i for i, m in enumerate(self._metadatas)
                if self._matches_filter(m, where)
            ]
        else:
            indices = list(range(len(self._embeddings)))

        indices = indices[:limit]
        if not indices:
            return np.empty((0, 0), dtype=np.float32)

        return np.stack([self._embeddings[i] for i in indices]).astype(np.float32)

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any], list[float]]]:
        """Return the top_k nearest neighbours by cosine distance.

        Returns a list of (id, cosine_distance, metadata, embedding) tuples
        sorted by ascending distance.
        """
        if not self._embeddings:
            return []

        # Filter candidates
        if where is not None:
            indices = [
                i for i, m in enumerate(self._metadatas)
                if self._matches_filter(m, where)
            ]
        else:
            indices = list(range(len(self._embeddings)))

        if not indices:
            return []

        candidates = np.stack([self._embeddings[i] for i in indices]).astype(np.float32)
        q = query_embedding.astype(np.float32)

        # Cosine similarity: dot(a,b) / (||a|| * ||b||)
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        c_norms = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-10)
        similarities = c_norms @ q_norm

        # Cosine distance = 1 - cosine_similarity
        distances = 1.0 - similarities

        # Sort by ascending distance and take top_k
        sorted_local = np.argsort(distances)[:top_k]

        results = []
        for local_idx in sorted_local:
            original_idx = indices[local_idx]
            results.append((
                self._ids[original_idx],
                float(distances[local_idx]),
                self._metadatas[original_idx],
                self._embeddings[original_idx].tolist(),
            ))
        return results


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class EmbeddingStore:
    """In-memory vector store for transaction embeddings using numpy
    cosine similarity.

    Parameters
    ----------
    persist_directory:
        Ignored (kept for API compatibility). All data is held in memory.
    collection_name:
        Base name for the default collection.
    reference_version:
        Optional version tag appended to the collection name when
        operating on a reference distribution snapshot.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = "transaction_embeddings",
        reference_version: str | None = None,
    ) -> None:
        self._base_name = collection_name
        self._reference_version = reference_version

        # Create both production and reference collections in memory.
        self._production = _InMemoryCollection(name=self._base_name)
        ref_name = self._versioned_name(reference_version)
        self._reference = _InMemoryCollection(name=ref_name)
        logger.info(
            "EmbeddingStore initialised (in-memory) -- production=%s, reference=%s",
            self._base_name,
            ref_name,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def add_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        to_reference: bool = False,
    ) -> None:
        """Insert embeddings into either the production or reference collection.

        Parameters
        ----------
        ids:
            Unique identifiers for each embedding (typically transaction IDs).
        embeddings:
            Dense vectors to store.
        metadatas:
            Optional per-embedding metadata dicts (merchant category,
            amount band, timestamp, etc.).
        to_reference:
            When ``True`` the embeddings are written to the reference
            distribution collection instead of production.
        """
        if metadatas is None:
            metadatas = [{} for _ in ids]

        # Inject ingestion timestamp when not already present.
        now_iso = datetime.now(timezone.utc).isoformat()
        for meta in metadatas:
            meta.setdefault("ingested_at", now_iso)

        collection = self._reference if to_reference else self._production

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.debug(
            "Upserted %d embeddings into %s", len(ids), collection.name
        )

    def query_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        *,
        from_reference: bool = False,
    ) -> list[QueryResult]:
        """Return the *top_k* nearest neighbours for *query_embedding*.

        An optional *where* filter narrows results by metadata fields
        such as ``merchant_category_code`` or ``amount_band``.
        """
        collection = self._reference if from_reference else self._production
        q = np.asarray(query_embedding, dtype=np.float32)

        raw = collection.query(query_embedding=q, top_k=top_k, where=where)
        results: list[QueryResult] = []

        for uid, dist, meta, emb in raw:
            results.append(
                QueryResult(
                    id=uid,
                    distance=dist,
                    metadata=meta,
                    embedding=emb,
                )
            )
        return results

    def get_reference_distribution(
        self,
        limit: int = 10_000,
        where: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Load the reference embedding distribution as a numpy array.

        Returns an ``(N, D)`` array where *N* is at most *limit*.
        """
        return self._reference.get_embeddings(limit=limit, where=where)

    def get_production_window(
        self,
        start_iso: str,
        end_iso: str,
        limit: int = 10_000,
        where: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Load production embeddings ingested within a time window.

        The window is defined by ISO-8601 strings which are compared
        lexicographically against the ``ingested_at`` metadata field.
        """
        time_filter: dict[str, Any] = {
            "$and": [
                {"ingested_at": {"$gte": start_iso}},
                {"ingested_at": {"$lte": end_iso}},
            ]
        }
        if where is not None:
            combined: dict[str, Any] = {"$and": [time_filter, where]}
        else:
            combined = time_filter

        return self._production.get_embeddings(limit=limit, where=combined)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _versioned_name(self, version: str | None) -> str:
        if version is not None:
            return f"{self._base_name}_ref_{version}"
        return f"{self._base_name}_reference"
