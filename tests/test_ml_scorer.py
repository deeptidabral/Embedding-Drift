"""
Test suite for the ML fraud scorer module.

Covers the SimulatedMLScorer heuristic-based scoring, feature extraction
from enriched transactions, MLScoringResult model validation, and
verification that high-risk indicators produce higher scores.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from src.fraud_detection.ml_scorer import (
    FEATURE_COLUMNS,
    MLFraudScorer,
    MLScoringResult,
    SimulatedMLScorer,
)
from src.fraud_detection.transaction_processor import (
    AmountBand,
    Channel,
    EnrichedTransaction,
    Transaction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(tz=timezone.utc).isoformat()


def _make_transaction(overrides: dict[str, Any] | None = None) -> Transaction:
    """Build a Transaction with sensible defaults, applying *overrides*."""
    base: dict[str, Any] = {
        "transaction_id": "txn_test_001",
        "timestamp": _NOW,
        "amount": 250.00,
        "currency": "USD",
        "merchant_name": "CornerCafe",
        "merchant_category_code": "5812",
        "card_type": "visa",
        "card_last_four": "1234",
        "city": "San Francisco",
        "country": "US",
        "channel": "in_store",
        "is_recurring": False,
    }
    if overrides:
        base.update(overrides)
    return Transaction(**base)


def _make_enriched(
    txn_overrides: dict[str, Any] | None = None,
    enrichment_overrides: dict[str, Any] | None = None,
) -> EnrichedTransaction:
    """Build an EnrichedTransaction for testing."""
    txn = _make_transaction(txn_overrides)
    enrichment: dict[str, Any] = {
        "transaction": txn,
        "amount_band": AmountBand.MEDIUM,
        "historical_summary": "No prior history available.",
        "is_new_merchant": False,
        "is_high_risk_country": False,
        "avg_amount_30d": 200.0,
        "txn_count_30d": 15,
        "days_since_last_txn": 2.0,
    }
    if enrichment_overrides:
        enrichment.update(enrichment_overrides)
    return EnrichedTransaction(**enrichment)


# ===================================================================
# SimulatedMLScorer -- scores in [0, 1]
# ===================================================================


@pytest.mark.unit
class TestSimulatedMLScorerRange:
    """Verify that SimulatedMLScorer always produces scores in [0, 1]."""

    def test_baseline_score_in_range(self) -> None:
        """A normal transaction should yield a score between 0 and 1."""
        scorer = SimulatedMLScorer()
        enriched = _make_enriched()
        result = scorer.predict(enriched)
        assert 0.0 <= result.score <= 1.0

    def test_high_risk_score_in_range(self) -> None:
        """Even a maximally risky transaction should not exceed 1.0."""
        scorer = SimulatedMLScorer()
        enriched = _make_enriched(
            txn_overrides={
                "amount": 50_000.0,
                "channel": "online",
                "timestamp": "2026-03-18T03:00:00+00:00",
            },
            enrichment_overrides={
                "amount_band": AmountBand.VERY_HIGH,
                "is_new_merchant": True,
                "is_high_risk_country": True,
                "avg_amount_30d": 100.0,
                "txn_count_30d": 1,
            },
        )
        result = scorer.predict(enriched)
        assert 0.0 <= result.score <= 1.0

    def test_low_risk_score_in_range(self) -> None:
        """A recurring, in-store, low-amount transaction should be low risk
        but still >= 0.0."""
        scorer = SimulatedMLScorer()
        enriched = _make_enriched(
            txn_overrides={
                "amount": 15.0,
                "channel": "in_store",
                "is_recurring": True,
            },
            enrichment_overrides={
                "amount_band": AmountBand.LOW,
                "avg_amount_30d": 20.0,
                "txn_count_30d": 50,
            },
        )
        result = scorer.predict(enriched)
        assert 0.0 <= result.score <= 1.0

    def test_many_random_transactions(self) -> None:
        """Score a batch of varied transactions and verify all in range."""
        scorer = SimulatedMLScorer()
        rng = np.random.default_rng(42)
        channels = ["online", "in_store", "atm", "mobile"]

        for i in range(20):
            enriched = _make_enriched(
                txn_overrides={
                    "transaction_id": f"txn_batch_{i:03d}",
                    "amount": float(rng.uniform(5.0, 30_000.0)),
                    "channel": channels[int(rng.integers(0, len(channels)))],
                },
                enrichment_overrides={
                    "amount_band": AmountBand.MEDIUM,
                    "is_new_merchant": bool(rng.integers(0, 2)),
                    "is_high_risk_country": bool(rng.integers(0, 2)),
                },
            )
            result = scorer.predict(enriched)
            assert 0.0 <= result.score <= 1.0, (
                f"Transaction {i}: score {result.score} out of range"
            )


# ===================================================================
# Feature extraction
# ===================================================================


@pytest.mark.unit
class TestFeatureExtraction:
    """Verify feature extraction from EnrichedTransaction."""

    def test_feature_vector_length(self) -> None:
        """The feature vector should match the number of FEATURE_COLUMNS."""
        scorer = SimulatedMLScorer()
        enriched = _make_enriched()
        features = scorer.extract_features(enriched)
        assert features.shape == (len(FEATURE_COLUMNS),)

    def test_feature_vector_dtype(self) -> None:
        """Features should be float64 numpy array."""
        scorer = SimulatedMLScorer()
        enriched = _make_enriched()
        features = scorer.extract_features(enriched)
        assert features.dtype == np.float64

    def test_amount_is_first_feature(self) -> None:
        """The first feature should be the transaction amount."""
        scorer = SimulatedMLScorer()
        enriched = _make_enriched(txn_overrides={"amount": 999.99})
        features = scorer.extract_features(enriched)
        assert features[0] == pytest.approx(999.99)

    def test_missing_history_uses_defaults(self) -> None:
        """When historical fields are None, defaults should be used."""
        scorer = SimulatedMLScorer()
        enriched = _make_enriched(
            enrichment_overrides={
                "days_since_last_txn": None,
                "avg_amount_30d": None,
                "txn_count_30d": None,
            },
        )
        features = scorer.extract_features(enriched)
        # Should not raise and should produce valid numbers
        assert not np.any(np.isnan(features))

    def test_boolean_features_are_float(self) -> None:
        """Boolean fields (is_new_merchant, is_high_risk_country) should
        be encoded as 0.0 or 1.0."""
        scorer = SimulatedMLScorer()
        enriched = _make_enriched(
            enrichment_overrides={
                "is_new_merchant": True,
                "is_high_risk_country": True,
            },
        )
        features = scorer.extract_features(enriched)
        # is_new_merchant is at index 10, is_high_risk_country at 11
        assert features[10] == 1.0
        assert features[11] == 1.0


# ===================================================================
# High-risk indicators produce higher scores
# ===================================================================


@pytest.mark.unit
class TestHighRiskIndicators:
    """Verify that transactions with high-risk indicators score higher."""

    def test_high_amount_increases_score(self) -> None:
        """Very high amounts should increase the fraud score."""
        scorer = SimulatedMLScorer()
        low_amount = _make_enriched(txn_overrides={"amount": 50.0})
        high_amount = _make_enriched(
            txn_overrides={"amount": 15_000.0},
            enrichment_overrides={"amount_band": AmountBand.VERY_HIGH},
        )
        assert scorer.predict(high_amount).score > scorer.predict(low_amount).score

    def test_high_risk_country_increases_score(self) -> None:
        """Transactions from high-risk countries should score higher."""
        scorer = SimulatedMLScorer()
        normal = _make_enriched()
        risky = _make_enriched(
            enrichment_overrides={"is_high_risk_country": True},
        )
        assert scorer.predict(risky).score > scorer.predict(normal).score

    def test_new_merchant_increases_score(self) -> None:
        """First-time merchant transactions should score higher."""
        scorer = SimulatedMLScorer()
        known = _make_enriched(enrichment_overrides={"is_new_merchant": False})
        new = _make_enriched(enrichment_overrides={"is_new_merchant": True})
        assert scorer.predict(new).score > scorer.predict(known).score

    def test_online_channel_increases_score(self) -> None:
        """Online channel should carry more risk than in-store."""
        scorer = SimulatedMLScorer()
        in_store = _make_enriched(txn_overrides={"channel": "in_store"})
        online = _make_enriched(txn_overrides={"channel": "online"})
        assert scorer.predict(online).score > scorer.predict(in_store).score

    def test_recurring_reduces_score(self) -> None:
        """Recurring transactions should reduce the fraud score."""
        scorer = SimulatedMLScorer()
        one_off = _make_enriched(txn_overrides={"is_recurring": False})
        recurring = _make_enriched(txn_overrides={"is_recurring": True})
        assert scorer.predict(recurring).score <= scorer.predict(one_off).score


# ===================================================================
# MLScoringResult model validation
# ===================================================================


@pytest.mark.unit
class TestMLScoringResultModel:
    """Validate the MLScoringResult pydantic model."""

    def test_valid_construction(self) -> None:
        """A well-formed result should be accepted."""
        result = MLScoringResult(
            score=0.42,
            feature_importances={"amount": 0.18, "channel": 0.10},
            top_risk_factors=["online_channel", "new_merchant"],
        )
        assert result.score == 0.42
        assert len(result.feature_importances) == 2
        assert len(result.top_risk_factors) == 2

    def test_score_out_of_range_raises(self) -> None:
        """Scores outside [0, 1] should be rejected."""
        with pytest.raises((ValueError, ValidationError)):
            MLScoringResult(score=1.5)
        with pytest.raises((ValueError, ValidationError)):
            MLScoringResult(score=-0.1)

    def test_defaults(self) -> None:
        """Default values should be empty collections."""
        result = MLScoringResult(score=0.5)
        assert result.feature_importances == {}
        assert result.top_risk_factors == []
        assert result.scored_at  # should have a timestamp

    def test_top_risk_factors_populated_by_scorer(self) -> None:
        """SimulatedMLScorer should populate top_risk_factors for
        risky transactions."""
        scorer = SimulatedMLScorer()
        enriched = _make_enriched(
            txn_overrides={"amount": 20_000.0, "channel": "online"},
            enrichment_overrides={
                "amount_band": AmountBand.VERY_HIGH,
                "is_new_merchant": True,
                "is_high_risk_country": True,
            },
        )
        result = scorer.predict(enriched)
        assert len(result.top_risk_factors) > 0
        assert all(isinstance(f, str) for f in result.top_risk_factors)


# ===================================================================
# MLFraudScorer base class
# ===================================================================


@pytest.mark.unit
class TestMLFraudScorerBase:
    """Test the base MLFraudScorer class behaviour."""

    def test_predict_without_model_raises(self) -> None:
        """Calling predict without loading a model should raise RuntimeError."""
        scorer = MLFraudScorer()
        enriched = _make_enriched()
        with pytest.raises(RuntimeError, match="No model loaded"):
            scorer.predict(enriched)

    def test_load_model_missing_file_raises(self) -> None:
        """Loading a non-existent model file should raise FileNotFoundError."""
        scorer = MLFraudScorer()
        with pytest.raises(FileNotFoundError):
            scorer.load_model("/nonexistent/model.joblib")

    def test_simulated_scorer_load_model_noop(self) -> None:
        """SimulatedMLScorer.load_model should be a no-op."""
        scorer = SimulatedMLScorer()
        # Should not raise
        scorer.load_model("/any/path.joblib")
        # Should still be able to predict
        result = scorer.predict(_make_enriched())
        assert 0.0 <= result.score <= 1.0
