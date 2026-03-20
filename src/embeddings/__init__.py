"""
Embedding generation and storage utilities for transaction data.

This module provides the ``LocalEmbeddingGenerator`` (aliased as
``TransactionEmbeddingGenerator``) for turning raw transaction
dictionaries into dense vector representations via sentence-transformers
(all-MiniLM-L6-v2), and the ``EmbeddingStore`` for persisting and
querying those vectors with an in-memory vector store.
"""

from src.embeddings.generator import (
    LocalEmbeddingGenerator,
    TransactionEmbeddingGenerator,
)

__all__ = [
    "LocalEmbeddingGenerator",
    "TransactionEmbeddingGenerator",
]

try:
    from src.embeddings.store import EmbeddingStore
    __all__.append("EmbeddingStore")
except ImportError:
    pass
