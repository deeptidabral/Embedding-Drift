"""
Transaction embedding generation using sentence-transformers (local) or OpenAI.

Default backend: sentence-transformers with all-MiniLM-L6-v2 (384 dimensions).
This runs locally, requires no API key, and is free. OpenAI text-embedding-3-large
is available as an optional backend for users who have an API key.

Converts raw transaction dictionaries into dense vector representations
suitable for similarity search and drift monitoring.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: sentence-transformers (default backend)
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Optional dependency: OpenAI (alternative backend)
# ---------------------------------------------------------------------------

try:
    from openai import OpenAI, APIError, RateLimitError

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore[assignment,misc]
    APIError = None  # type: ignore[assignment,misc]
    RateLimitError = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIMENSIONS = 384
OPENAI_DEFAULT_MODEL = "text-embedding-3-large"
OPENAI_DEFAULT_DIMENSIONS = 3072
MAX_BATCH_SIZE = 128
MAX_RETRIES = 4
INITIAL_BACKOFF_SECONDS = 1.0


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
# OpenAI embedding generator (optional backend)
# ---------------------------------------------------------------------------


class OpenAIEmbeddingGenerator:
    """Generate embeddings for payment transactions using OpenAI.

    This is the optional backend for production use cases where OpenAI
    text-embedding-3-large quality is required.  An API key is needed.

    Parameters
    ----------
    api_key:
        OpenAI API key.  When *None* the client falls back to the
        ``OPENAI_API_KEY`` environment variable.
    model:
        Embedding model identifier.
    dimensions:
        Desired output dimensionality (supported by text-embedding-3-*).
    max_retries:
        Maximum number of retry attempts on transient API errors.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = OPENAI_DEFAULT_MODEL,
        dimensions: int = OPENAI_DEFAULT_DIMENSIONS,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        if not _HAS_OPENAI:
            raise ImportError(
                "openai is required for OpenAIEmbeddingGenerator. "
                "Install it with: pip install openai>=1.0.0"
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_single(self, transaction: dict[str, Any]) -> EmbeddingResult:
        """Generate an embedding for a single transaction.

        Returns an ``EmbeddingResult`` containing the dense vector and
        associated metadata.
        """
        text = self._format_transaction(transaction)
        response = self._call_api_with_retry([text])
        data = response.data[0]
        return EmbeddingResult(
            transaction_id=str(transaction.get("transaction_id", "unknown")),
            embedding=data.embedding,
            model=self._model,
            dimensions=self._dimensions,
            token_count=response.usage.total_tokens,
        )

    def generate_batch(
        self,
        transactions: list[dict[str, Any]],
        batch_size: int = MAX_BATCH_SIZE,
    ) -> list[EmbeddingResult]:
        """Generate embeddings for a list of transactions.

        Transactions are processed in batches of *batch_size* to respect
        API payload limits.  Each batch is retried independently on
        transient failures.
        """
        results: list[EmbeddingResult] = []
        texts = [self._format_transaction(t) for t in transactions]

        for start in range(0, len(texts), batch_size):
            chunk_texts = texts[start : start + batch_size]
            chunk_txns = transactions[start : start + batch_size]

            response = self._call_api_with_retry(chunk_texts)

            for datum, txn in zip(response.data, chunk_txns):
                results.append(
                    EmbeddingResult(
                        transaction_id=str(txn.get("transaction_id", "unknown")),
                        embedding=datum.embedding,
                        model=self._model,
                        dimensions=self._dimensions,
                        token_count=response.usage.total_tokens // len(chunk_texts),
                    )
                )
            logger.info(
                "Embedded batch %d-%d of %d transactions",
                start,
                start + len(chunk_texts),
                len(texts),
            )

        return results

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

    def _call_api_with_retry(self, texts: list[str]) -> Any:
        """Call the OpenAI embeddings endpoint with exponential back-off.

        WARNING: This method uses ``time.sleep()`` for retry backoff,
        which is a blocking synchronous call. If this code runs inside
        an async event loop (asyncio, FastAPI, Celery), use the
        ``AsyncOpenAI`` client with ``await asyncio.sleep()`` instead.
        This synchronous version is intended for batch/offline pipelines.
        """
        backoff = INITIAL_BACKOFF_SECONDS
        last_exception: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return self._client.embeddings.create(
                    input=texts,
                    model=self._model,
                    dimensions=self._dimensions,
                )
            except RateLimitError as exc:
                last_exception = exc
                logger.warning(
                    "Rate-limited on attempt %d/%d. Backing off %.1fs.",
                    attempt,
                    self._max_retries,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2
            except APIError as exc:
                last_exception = exc
                if exc.status_code is not None and exc.status_code < 500:
                    raise
                logger.warning(
                    "Transient API error on attempt %d/%d: %s",
                    attempt,
                    self._max_retries,
                    exc,
                )
                time.sleep(backoff)
                backoff *= 2

        raise RuntimeError(
            f"OpenAI embedding request failed after {self._max_retries} attempts"
        ) from last_exception


# ---------------------------------------------------------------------------
# Backward compatibility alias
# ---------------------------------------------------------------------------

TransactionEmbeddingGenerator = LocalEmbeddingGenerator
