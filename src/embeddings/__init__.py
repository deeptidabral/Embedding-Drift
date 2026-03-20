"""
Embedding generation and storage utilities for transaction data.

This module provides the ``TransactionEmbeddingGenerator`` for turning
raw transaction dictionaries into dense vector representations via
OpenAI, the ``LocalEmbeddingGenerator`` for offline embedding via
sentence-transformers (all-MiniLM-L6-v2), and the ``EmbeddingStore``
for persisting and querying those vectors with ChromaDB.
"""

from src.embeddings.generator import (
    LocalEmbeddingGenerator,
    OpenAIEmbeddingGenerator,
    TransactionEmbeddingGenerator,
)

__all__ = [
    "LocalEmbeddingGenerator",
    "OpenAIEmbeddingGenerator",
    "TransactionEmbeddingGenerator",
]

try:
    from src.embeddings.store import EmbeddingStore
    __all__.append("EmbeddingStore")
except ImportError:
    pass
