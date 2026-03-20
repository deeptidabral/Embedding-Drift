"""
Persistent embedding store backed by ChromaDB.

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

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMADB_AVAILABLE = True
except ImportError:
    _CHROMADB_AVAILABLE = False

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
# Store
# ---------------------------------------------------------------------------


class EmbeddingStore:
    """ChromaDB-backed store for transaction embeddings.

    Parameters
    ----------
    persist_directory:
        Filesystem path where ChromaDB persists data.  When *None* an
        ephemeral in-memory client is used (useful for tests).
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
        if persist_directory is not None:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False),
            )

        self._base_name = collection_name
        self._reference_version = reference_version

        # Eagerly create / retrieve both production and reference collections.
        self._production = self._client.get_or_create_collection(
            name=self._base_name,
            metadata={"hnsw:space": "cosine"},
        )
        ref_name = self._versioned_name(reference_version)
        self._reference = self._client.get_or_create_collection(
            name=ref_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "EmbeddingStore initialised -- production=%s, reference=%s",
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

        An optional *where* filter (ChromaDB ``$eq`` / ``$in`` syntax)
        narrows results by metadata fields such as
        ``merchant_category_code`` or ``amount_band``.
        """
        collection = self._reference if from_reference else self._production
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["distances", "metadatas", "embeddings"],
        }
        if where is not None:
            kwargs["where"] = where

        raw = collection.query(**kwargs)
        results: list[QueryResult] = []

        for idx in range(len(raw["ids"][0])):
            results.append(
                QueryResult(
                    id=raw["ids"][0][idx],
                    distance=raw["distances"][0][idx],  # type: ignore[index]
                    metadata=raw["metadatas"][0][idx] if raw["metadatas"] else {},  # type: ignore[index]
                    embedding=raw["embeddings"][0][idx] if raw["embeddings"] else None,  # type: ignore[index]
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
        return self._get_embeddings(self._reference, limit=limit, where=where)

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

        return self._get_embeddings(self._production, limit=limit, where=combined)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _versioned_name(self, version: str | None) -> str:
        if version is not None:
            return f"{self._base_name}_ref_{version}"
        return f"{self._base_name}_reference"

    @staticmethod
    def _get_embeddings(
        collection: Any,
        limit: int,
        where: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Retrieve embeddings from a collection and return as ndarray."""
        kwargs: dict[str, Any] = {
            "include": ["embeddings"],
            "limit": limit,
        }
        if where is not None:
            kwargs["where"] = where

        result = collection.get(**kwargs)
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return np.empty((0, 0), dtype=np.float32)

        return np.asarray(embeddings, dtype=np.float32)
