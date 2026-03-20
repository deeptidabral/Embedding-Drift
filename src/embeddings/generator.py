"""
Transaction embedding generation using sentence-transformers.

Backend: sentence-transformers with all-MiniLM-L6-v2 (384 dimensions).
This runs locally, requires no API key, and is free.

Converts raw transaction dictionaries into dense vector representations
suitable for similarity search and drift monitoring.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: sentence-transformers
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIMENSIONS = 384
MAX_BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# Template used to serialise a transaction dict into natural-language text
# ---------------------------------------------------------------------------

_TRANSACTION_TEMPLATE = (
    "Transaction ID: {transaction_id}\n"
    "Timestamp: {timestamp}\n"
    "Amount: {currency} {amount:.2f}\n"
    "Merchant: {merchant_name} (MCC {merchant_category_code})\n"
    "Card: {card_type} ending {card_last_four}\n"
    "Location: {city}, {country}\n"
    "Channel: {channel}\n"
    "Recurring: {is_recurring}\n"
    "Historical context: {historical_summary}"
)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class EmbeddingResult(BaseModel):
    """Container for a single embedding vector together with its metadata."""

    transaction_id: str
    embedding: list[float]
    model: str = DEFAULT_MODEL
    dimensions: int = DEFAULT_DIMENSIONS
    token_count: int = 0


# ---------------------------------------------------------------------------
# Local embedding generator (default) -- sentence-transformers
# ---------------------------------------------------------------------------


class LocalEmbeddingGenerator:
    """Generate embeddings for payment transactions using sentence-transformers.

    This is the default embedding backend. It runs entirely on the local
    machine, requires no API key, and is free.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier for sentence-transformers.
        Defaults to ``all-MiniLM-L6-v2`` (384 dimensions).
    device:
        PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).  When *None*
        the library selects the best available device automatically.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required for LocalEmbeddingGenerator. "
                "Install it with: pip install sentence-transformers>=2.2.0"
            )
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._dimensions = self._model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        return self._dimensions

    def generate_single(self, transaction: dict[str, Any]) -> EmbeddingResult:
        """Generate an embedding for a single transaction.

        Returns an ``EmbeddingResult`` containing the dense vector and
        associated metadata.
        """
        text = self._format_transaction(transaction)
        vector = self._model.encode([text], show_progress_bar=False)[0]
        return EmbeddingResult(
            transaction_id=str(transaction.get("transaction_id", "unknown")),
            embedding=vector.tolist(),
            model=self._model_name,
            dimensions=self._dimensions,
            token_count=0,
        )

    def generate_batch(
        self,
        transactions: list[dict[str, Any]],
        batch_size: int = MAX_BATCH_SIZE,
    ) -> list[EmbeddingResult]:
        """Generate embeddings for a list of transactions.

        Transactions are processed in batches of *batch_size*.
        """
        results: list[EmbeddingResult] = []
        texts = [self._format_transaction(t) for t in transactions]

        for start in range(0, len(texts), batch_size):
            chunk_texts = texts[start : start + batch_size]
            chunk_txns = transactions[start : start + batch_size]

            vectors = self._model.encode(chunk_texts, show_progress_bar=False)

            for vec, txn in zip(vectors, chunk_txns):
                results.append(
                    EmbeddingResult(
                        transaction_id=str(txn.get("transaction_id", "unknown")),
                        embedding=vec.tolist(),
                        model=self._model_name,
                        dimensions=self._dimensions,
                        token_count=0,
                    )
                )
            logger.info(
                "Embedded batch %d-%d of %d transactions",
                start,
                start + len(chunk_texts),
                len(texts),
            )

        return results

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of raw text strings into embedding vectors.

        Parameters
        ----------
        texts : list[str]
            Texts to encode.
        batch_size : int
            Number of texts per forward pass.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(texts), self.dimensions)`` with dtype float32.
        """
        embeddings = self._model.encode(
            texts, batch_size=batch_size, show_progress_bar=False
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Alias for :meth:`encode` for backward compatibility."""
        return self.encode(texts, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _format_transaction(transaction: dict[str, Any]) -> str:
        """Convert a transaction dictionary into a text string for embedding.

        Missing keys are replaced with sensible defaults so that the
        template never raises a ``KeyError``.
        """
        defaults: dict[str, Any] = {
            "transaction_id": "N/A",
            "timestamp": "N/A",
            "currency": "USD",
            "amount": 0.0,
            "merchant_name": "Unknown",
            "merchant_category_code": "0000",
            "card_type": "Unknown",
            "card_last_four": "XXXX",
            "city": "Unknown",
            "country": "Unknown",
            "channel": "Unknown",
            "is_recurring": False,
            "historical_summary": "No prior history available.",
        }
        merged = {**defaults, **transaction}
        return _TRANSACTION_TEMPLATE.format(**merged)


# ---------------------------------------------------------------------------
# Backward compatibility alias
# ---------------------------------------------------------------------------

TransactionEmbeddingGenerator = LocalEmbeddingGenerator
