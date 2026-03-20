"""
Fraud detection pipeline components.

Orchestrates embedding generation, RAG-based pattern retrieval,
LLM assessment, and drift-aware fallback logic for real-time
transaction fraud scoring.
"""

from src.fraud_detection.pipeline import FraudDetectionPipeline
from src.fraud_detection.rag_retriever import FraudPatternRetriever
from src.fraud_detection.transaction_processor import (
    Transaction,
    EnrichedTransaction,
    FraudAssessment,
    TransactionProcessor,
)

__all__ = [
    "FraudDetectionPipeline",
    "FraudPatternRetriever",
    "Transaction",
    "EnrichedTransaction",
    "FraudAssessment",
    "TransactionProcessor",
]
