"""
Transaction ingestion, validation, and enrichment.

Provides Pydantic models for the transaction lifecycle and a processor
that validates raw input, enriches it with historical context, and
serialises it for embedding generation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Channel(str, Enum):
    """Transaction origination channel."""

    ONLINE = "online"
    IN_STORE = "in_store"
    ATM = "atm"
    MOBILE = "mobile"
    PHONE = "phone"


class AmountBand(str, Enum):
    """Coarse amount bucket for stratified analysis."""

    MICRO = "micro"        # < 10
    LOW = "low"            # 10 - 100
    MEDIUM = "medium"      # 100 - 1000
    HIGH = "high"          # 1000 - 10000
    VERY_HIGH = "very_high"  # > 10000


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Transaction(BaseModel):
    """Raw incoming transaction before validation or enrichment."""

    transaction_id: str
    timestamp: str
    amount: float
    currency: str = "USD"
    merchant_name: str
    merchant_category_code: str
    card_type: str
    card_last_four: str
    city: str
    country: str
    channel: Channel
    is_recurring: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Transaction amount must be non-negative")
        return v

    @field_validator("card_last_four")
    @classmethod
    def card_last_four_length(cls, v: str) -> str:
        if len(v) != 4 or not v.isdigit():
            raise ValueError("card_last_four must be exactly 4 digits")
        return v


class EnrichedTransaction(BaseModel):
    """Transaction augmented with historical and derived features."""

    transaction: Transaction
    amount_band: AmountBand
    historical_summary: str
    days_since_last_txn: float | None = None
    avg_amount_30d: float | None = None
    txn_count_30d: int | None = None
    same_merchant_count_90d: int | None = None
    is_new_merchant: bool = False
    is_high_risk_country: bool = False
    enriched_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class FraudAssessment(BaseModel):
    """Final fraud scoring output for a transaction.

    Attributes
    ----------
    ml_score:
        Fraud probability produced by the primary ML model (XGBoost).
        ``None`` only when the ML model was bypassed (e.g. rule-based
        fallback during critical drift).
    llm_score:
        Fraud probability from the complementary RAG+LLM layer.
        ``None`` when the transaction was handled by the ML model alone.
    analysis_tier:
        ``"ml_only"`` when the ML score was sufficient, or
        ``"ml_plus_llm"`` when the RAG+LLM layer was also invoked.
        ``"rule_based_fallback"`` for drift-triggered fallback.
    """

    transaction_id: str
    fraud_score: float = Field(ge=0.0, le=1.0)
    is_fraud: bool
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = ""
    similar_fraud_ids: list[str] = Field(default_factory=list)
    drift_severity: str = "none"
    model_version: str = ""
    ml_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Primary ML model fraud probability.",
    )
    llm_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Complementary RAG+LLM fraud probability (when invoked).",
    )
    analysis_tier: str = Field(
        default="ml_only",
        description='One of "ml_only", "ml_plus_llm", or "rule_based_fallback".',
    )
    assessed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class TransactionProcessor:
    """Validate, enrich, and serialise transactions.

    Parameters
    ----------
    high_risk_countries:
        ISO country codes considered high-risk for fraud.
    """

    def __init__(
        self,
        high_risk_countries: set[str] | None = None,
    ) -> None:
        self._high_risk_countries = high_risk_countries or {
            "NG", "PH", "RO", "UA", "BY",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, raw: dict[str, Any]) -> Transaction:
        """Parse and validate a raw transaction dictionary.

        Raises ``pydantic.ValidationError`` on invalid input.
        """
        return Transaction(**raw)

    def enrich(
        self,
        transaction: Transaction,
        history: dict[str, Any] | None = None,
    ) -> EnrichedTransaction:
        """Augment a validated transaction with derived features.

        Parameters
        ----------
        transaction:
            A validated ``Transaction``.
        history:
            Optional dict of historical statistics fetched from the
            data warehouse.  Expected keys: ``days_since_last_txn``,
            ``avg_amount_30d``, ``txn_count_30d``,
            ``same_merchant_count_90d``.
        """
        history = history or {}
        amount_band = self._classify_amount_band(transaction.amount)
        is_new_merchant = history.get("same_merchant_count_90d", 0) == 0
        is_high_risk = transaction.country in self._high_risk_countries

        summary_parts: list[str] = []
        if history.get("txn_count_30d") is not None:
            summary_parts.append(
                f"{history['txn_count_30d']} transactions in last 30 days"
            )
        if history.get("avg_amount_30d") is not None:
            summary_parts.append(
                f"average amount {transaction.currency} {history['avg_amount_30d']:.2f}"
            )
        if is_new_merchant:
            summary_parts.append("first transaction at this merchant")
        if is_high_risk:
            summary_parts.append("originating from high-risk country")

        historical_summary = "; ".join(summary_parts) if summary_parts else (
            "No prior history available."
        )

        return EnrichedTransaction(
            transaction=transaction,
            amount_band=amount_band,
            historical_summary=historical_summary,
            days_since_last_txn=history.get("days_since_last_txn"),
            avg_amount_30d=history.get("avg_amount_30d"),
            txn_count_30d=history.get("txn_count_30d"),
            same_merchant_count_90d=history.get("same_merchant_count_90d"),
            is_new_merchant=is_new_merchant,
            is_high_risk_country=is_high_risk,
        )

    @staticmethod
    def to_embedding_text(enriched: EnrichedTransaction) -> dict[str, Any]:
        """Convert an enriched transaction into a dict suitable for
        ``TransactionEmbeddingGenerator._format_transaction``.
        """
        txn = enriched.transaction
        return {
            "transaction_id": txn.transaction_id,
            "timestamp": txn.timestamp,
            "amount": txn.amount,
            "currency": txn.currency,
            "merchant_name": txn.merchant_name,
            "merchant_category_code": txn.merchant_category_code,
            "card_type": txn.card_type,
            "card_last_four": txn.card_last_four,
            "city": txn.city,
            "country": txn.country,
            "channel": txn.channel.value,
            "is_recurring": txn.is_recurring,
            "historical_summary": enriched.historical_summary,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_amount_band(amount: float) -> AmountBand:
        if amount < 10:
            return AmountBand.MICRO
        if amount < 100:
            return AmountBand.LOW
        if amount < 1_000:
            return AmountBand.MEDIUM
        if amount < 10_000:
            return AmountBand.HIGH
        return AmountBand.VERY_HIGH
