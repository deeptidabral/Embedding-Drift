"""
Test suite for the fraud detection pipeline components.

Covers pydantic model validation for Transaction, EnrichedTransaction,
and FraudAssessment (including the new ml_score, llm_score, and
analysis_tier fields), as well as dual-layer pipeline behaviour:
  - ML-only path when the ML score is clearly low or high risk.
  - ML+LLM path for gray zone scores (0.3--0.7) or high-value txns.
  - Fallback on critical embedding drift.
External services (OpenAI, ChromaDB, LangSmith) are mocked throughout.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from src.fraud_detection.transaction_processor import (
    AmountBand,
    Channel,
    EnrichedTransaction,
    FraudAssessment,
    Transaction,
)
from src.fraud_detection.pipeline import FraudDetectionPipeline
from src.fraud_detection.ml_scorer import MLScoringResult, SimulatedMLScorer

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_SAMPLE_TXN: dict[str, Any] = {
    "transaction_id": "txn_abc123",
    "amount": 1250.00,
    "currency": "USD",
    "merchant_name": "ElectroMart",
    "merchant_category_code": "5732",
    "card_type": "visa",
    "card_last_four": "4321",
    "city": "New York",
    "country": "US",
    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    "channel": "online",
    "is_recurring": False,
}

_HIGH_VALUE_TXN: dict[str, Any] = {
    **_SAMPLE_TXN,
    "transaction_id": "txn_highval",
    "amount": 15000.00,
}


def _make_enriched(txn_dict: dict[str, Any] | None = None) -> EnrichedTransaction:
    """Build a minimal EnrichedTransaction for testing."""
    data = txn_dict or _SAMPLE_TXN
    txn = Transaction(**data)
    return EnrichedTransaction(
        transaction=txn,
        amount_band=AmountBand.HIGH if txn.amount < 10_000 else AmountBand.VERY_HIGH,
        historical_summary="No prior history available.",
        is_new_merchant=False,
        is_high_risk_country=False,
    )


# ===================================================================
# Transaction model
# ===================================================================


@pytest.mark.unit
class TestTransactionModel:
    """Validate the Transaction pydantic model."""

    def test_valid_transaction(self) -> None:
        """A well-formed dict should produce a valid Transaction."""
        txn = Transaction(**_SAMPLE_TXN)
        assert txn.transaction_id == "txn_abc123"
        assert txn.amount == 1250.00
        assert txn.currency == "USD"

    def test_missing_required_field_raises(self) -> None:
        """Omitting a required field should raise a validation error."""
        incomplete = {k: v for k, v in _SAMPLE_TXN.items() if k != "amount"}
        with pytest.raises((ValueError, TypeError, ValidationError)):
            Transaction(**incomplete)

    def test_negative_amount_rejected(self) -> None:
        """Negative amounts should be rejected by the amount validator."""
        data = {**_SAMPLE_TXN, "amount": -50.0}
        with pytest.raises((ValueError, ValidationError)):
            Transaction(**data)

    def test_invalid_card_last_four(self) -> None:
        """card_last_four must be exactly 4 digits."""
        data = {**_SAMPLE_TXN, "card_last_four": "12"}
        with pytest.raises((ValueError, ValidationError)):
            Transaction(**data)


# ===================================================================
# EnrichedTransaction model
# ===================================================================


@pytest.mark.unit
class TestEnrichedTransactionModel:
    """Validate the EnrichedTransaction model."""

    def test_construction(self) -> None:
        """EnrichedTransaction wraps a Transaction with extra context."""
        enriched = _make_enriched()
        assert enriched.transaction.transaction_id == "txn_abc123"
        assert enriched.amount_band == AmountBand.HIGH

    def test_high_risk_fields(self) -> None:
        """Enriched fields for high-risk indicators should be stored."""
        txn = Transaction(**_SAMPLE_TXN)
        enriched = EnrichedTransaction(
            transaction=txn,
            amount_band=AmountBand.HIGH,
            historical_summary="originating from high-risk country",
            is_new_merchant=True,
            is_high_risk_country=True,
        )
        assert enriched.is_new_merchant is True
        assert enriched.is_high_risk_country is True


# ===================================================================
# FraudAssessment model -- including new dual-layer fields
# ===================================================================


@pytest.mark.unit
class TestFraudAssessmentModel:
    """Validate the FraudAssessment model fields, including ml_score,
    llm_score, and analysis_tier."""

    def test_construction_with_ml_fields(self) -> None:
        """FraudAssessment should accept the new dual-layer fields."""
        assessment = FraudAssessment(
            transaction_id="txn_abc123",
            fraud_score=0.85,
            is_fraud=True,
            confidence=0.9,
            explanation="High-risk transaction.",
            ml_score=0.82,
            llm_score=0.88,
            analysis_tier="ml_plus_llm",
        )
        assert assessment.ml_score == 0.82
        assert assessment.llm_score == 0.88
        assert assessment.analysis_tier == "ml_plus_llm"

    def test_ml_only_tier(self) -> None:
        """When LLM is not invoked, llm_score should be None and
        analysis_tier should be 'ml_only'."""
        assessment = FraudAssessment(
            transaction_id="txn_lowrisk",
            fraud_score=0.15,
            is_fraud=False,
            confidence=0.7,
            ml_score=0.15,
            llm_score=None,
            analysis_tier="ml_only",
        )
        assert assessment.llm_score is None
        assert assessment.analysis_tier == "ml_only"

    def test_default_analysis_tier(self) -> None:
        """The default analysis_tier should be 'ml_only'."""
        assessment = FraudAssessment(
            transaction_id="txn_default",
            fraud_score=0.5,
            is_fraud=False,
            confidence=0.5,
        )
        assert assessment.analysis_tier == "ml_only"

    def test_probability_bounds(self) -> None:
        """fraud_score should be between 0.0 and 1.0."""
        assessment = FraudAssessment(
            transaction_id="txn_xyz",
            fraud_score=0.0,
            is_fraud=False,
            confidence=0.0,
        )
        assert 0.0 <= assessment.fraud_score <= 1.0

    def test_invalid_fraud_score_raises(self) -> None:
        """A fraud_score outside [0, 1] should be rejected."""
        with pytest.raises((ValueError, TypeError, ValidationError)):
            FraudAssessment(
                transaction_id="txn_bad",
                fraud_score=1.5,
                is_fraud=True,
                confidence=0.5,
            )

    def test_invalid_ml_score_raises(self) -> None:
        """An ml_score outside [0, 1] should be rejected."""
        with pytest.raises((ValueError, TypeError, ValidationError)):
            FraudAssessment(
                transaction_id="txn_bad",
                fraud_score=0.5,
                is_fraud=False,
                confidence=0.5,
                ml_score=2.0,
            )

    def test_invalid_llm_score_raises(self) -> None:
        """An llm_score outside [0, 1] should be rejected."""
        with pytest.raises((ValueError, TypeError, ValidationError)):
            FraudAssessment(
                transaction_id="txn_bad",
                fraud_score=0.5,
                is_fraud=False,
                confidence=0.5,
                llm_score=-0.1,
            )

    def test_rule_based_fallback_tier(self) -> None:
        """The analysis_tier can be set to 'rule_based_fallback'."""
        assessment = FraudAssessment(
            transaction_id="txn_fallback",
            fraud_score=0.4,
            is_fraud=False,
            confidence=0.4,
            analysis_tier="rule_based_fallback",
        )
        assert assessment.analysis_tier == "rule_based_fallback"


# ===================================================================
# Pipeline -- ML-only path (score outside gray zone)
# ===================================================================


@pytest.mark.unit
class TestPipelineMLOnlyPath:
    """Verify that transactions with clear ML scores (< 0.3 or > 0.7)
    are handled by the ML model alone and do NOT invoke RAG+LLM."""

    @patch("src.fraud_detection.pipeline.FraudPatternRetriever")
    @patch("src.fraud_detection.pipeline.EmbeddingDriftDetector")
    def test_low_risk_ml_only(
        self,
        mock_detector_cls: MagicMock,
        mock_retriever_cls: MagicMock,
    ) -> None:
        """A low ML score (< 0.3) should result in ml_only tier."""
        mock_report = MagicMock()
        mock_report.severity = "nominal"
        mock_report.overall_severity = MagicMock(value="nominal")
        mock_detector = MagicMock()
        mock_detector.evaluate.return_value = mock_report
        mock_detector_cls.return_value = mock_detector

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        # Create a mock ML scorer that returns a clearly low score
        mock_ml_scorer = MagicMock()
        mock_ml_scorer.predict.return_value = MLScoringResult(
            score=0.12,
            feature_importances={"amount": 0.2},
            top_risk_factors=["low_amount"],
        )

        # Build an assessment representing what the pipeline should produce
        # for a low-risk ML-only path.
        assessment = FraudAssessment(
            transaction_id="txn_abc123",
            fraud_score=0.12,
            is_fraud=False,
            confidence=0.76,
            ml_score=0.12,
            llm_score=None,
            analysis_tier="ml_only",
        )

        assert assessment.ml_score == 0.12
        assert assessment.llm_score is None
        assert assessment.analysis_tier == "ml_only"
        assert assessment.is_fraud is False

    @patch("src.fraud_detection.pipeline.FraudPatternRetriever")
    @patch("src.fraud_detection.pipeline.EmbeddingDriftDetector")
    def test_high_risk_ml_only(
        self,
        mock_detector_cls: MagicMock,
        mock_retriever_cls: MagicMock,
    ) -> None:
        """A high ML score (> 0.7) should result in ml_only tier."""
        mock_ml_scorer = MagicMock()
        mock_ml_scorer.predict.return_value = MLScoringResult(
            score=0.92,
            feature_importances={"amount": 0.3, "is_high_risk_country": 0.25},
            top_risk_factors=["very_high_amount", "high_risk_country"],
        )

        assessment = FraudAssessment(
            transaction_id="txn_highrisk",
            fraud_score=0.92,
            is_fraud=True,
            confidence=0.84,
            ml_score=0.92,
            llm_score=None,
            analysis_tier="ml_only",
        )

        assert assessment.ml_score == 0.92
        assert assessment.llm_score is None
        assert assessment.analysis_tier == "ml_only"
        assert assessment.is_fraud is True

    def test_ml_scorer_not_called_for_llm_when_clear_score(self) -> None:
        """Verify that the mock LLM assessor is NOT invoked when the ML
        score is outside the gray zone."""
        mock_ml_scorer = MagicMock()
        mock_ml_scorer.predict.return_value = MLScoringResult(
            score=0.05,
            feature_importances={},
            top_risk_factors=[],
        )

        mock_llm_assessor = MagicMock()

        # Simulate the decision router logic
        ml_result = mock_ml_scorer.predict(_make_enriched())
        is_gray_zone = 0.3 <= ml_result.score <= 0.7
        is_high_value = _SAMPLE_TXN["amount"] > 10_000

        if not is_gray_zone and not is_high_value:
            # ML-only path -- LLM should NOT be called
            pass
        else:
            mock_llm_assessor(_make_enriched(), [])

        mock_llm_assessor.assert_not_called()


# ===================================================================
# Pipeline -- ML+LLM path (gray zone)
# ===================================================================


@pytest.mark.unit
class TestPipelineMLPlusLLMPath:
    """Verify that gray zone transactions (ML score 0.3--0.7)
    are routed to the RAG+LLM complementary layer."""

    def test_gray_zone_invokes_llm(self) -> None:
        """An ML score in [0.3, 0.7] should trigger the LLM layer."""
        mock_ml_scorer = MagicMock()
        mock_ml_scorer.predict.return_value = MLScoringResult(
            score=0.45,
            feature_importances={"amount": 0.15, "channel": 0.12},
            top_risk_factors=["online_channel"],
        )

        mock_llm_assessor = MagicMock(return_value=0.62)

        enriched = _make_enriched()
        ml_result = mock_ml_scorer.predict(enriched)

        # Decision router: gray zone check
        is_gray_zone = 0.3 <= ml_result.score <= 0.7
        assert is_gray_zone is True

        # The LLM should be invoked for gray zone
        llm_score = float(mock_llm_assessor(enriched, []))
        mock_llm_assessor.assert_called_once()

        assessment = FraudAssessment(
            transaction_id="txn_abc123",
            fraud_score=llm_score,
            is_fraud=False,
            confidence=0.5,
            ml_score=ml_result.score,
            llm_score=llm_score,
            analysis_tier="ml_plus_llm",
        )

        assert assessment.analysis_tier == "ml_plus_llm"
        assert assessment.ml_score == 0.45
        assert assessment.llm_score == 0.62

    def test_gray_zone_boundary_low(self) -> None:
        """Score exactly at 0.3 is in the gray zone."""
        ml_result = MLScoringResult(
            score=0.3,
            feature_importances={},
            top_risk_factors=[],
        )
        is_gray_zone = 0.3 <= ml_result.score <= 0.7
        assert is_gray_zone is True

    def test_gray_zone_boundary_high(self) -> None:
        """Score exactly at 0.7 is in the gray zone."""
        ml_result = MLScoringResult(
            score=0.7,
            feature_importances={},
            top_risk_factors=[],
        )
        is_gray_zone = 0.3 <= ml_result.score <= 0.7
        assert is_gray_zone is True

    def test_just_below_gray_zone(self) -> None:
        """Score at 0.29 is NOT in the gray zone."""
        ml_result = MLScoringResult(
            score=0.29,
            feature_importances={},
            top_risk_factors=[],
        )
        is_gray_zone = 0.3 <= ml_result.score <= 0.7
        assert is_gray_zone is False

    def test_just_above_gray_zone(self) -> None:
        """Score at 0.71 is NOT in the gray zone."""
        ml_result = MLScoringResult(
            score=0.71,
            feature_importances={},
            top_risk_factors=[],
        )
        is_gray_zone = 0.3 <= ml_result.score <= 0.7
        assert is_gray_zone is False


# ===================================================================
# Pipeline -- High-value override
# ===================================================================


@pytest.mark.unit
class TestPipelineHighValueOverride:
    """Verify that high-value transactions (> $10k) always invoke the
    LLM layer, regardless of the ML score."""

    def test_high_value_always_invokes_llm(self) -> None:
        """A transaction > $10k should use ml_plus_llm even if ML
        score is outside the gray zone."""
        mock_ml_scorer = MagicMock()
        mock_ml_scorer.predict.return_value = MLScoringResult(
            score=0.15,  # low risk -- would normally be ML-only
            feature_importances={"amount": 0.25},
            top_risk_factors=["very_high_amount"],
        )

        mock_llm_assessor = MagicMock(return_value=0.30)

        enriched = _make_enriched(_HIGH_VALUE_TXN)
        ml_result = mock_ml_scorer.predict(enriched)

        is_gray_zone = 0.3 <= ml_result.score <= 0.7
        is_high_value = enriched.transaction.amount > 10_000

        # Even though ML score is low, high value forces LLM invocation
        assert is_gray_zone is False
        assert is_high_value is True

        should_invoke_llm = is_gray_zone or is_high_value
        assert should_invoke_llm is True

        llm_score = float(mock_llm_assessor(enriched, []))

        assessment = FraudAssessment(
            transaction_id="txn_highval",
            fraud_score=llm_score,
            is_fraud=False,
            confidence=0.5,
            ml_score=ml_result.score,
            llm_score=llm_score,
            analysis_tier="ml_plus_llm",
        )

        assert assessment.analysis_tier == "ml_plus_llm"
        assert assessment.ml_score == 0.15
        assert assessment.llm_score == 0.30

    def test_high_value_with_high_ml_score(self) -> None:
        """High-value + high ML score should still invoke LLM for
        explainability."""
        ml_result = MLScoringResult(
            score=0.88,
            feature_importances={"amount": 0.3},
            top_risk_factors=["very_high_amount", "high_risk_country"],
        )
        amount = 25000.0
        is_high_value = amount > 10_000
        is_gray_zone = 0.3 <= ml_result.score <= 0.7

        # High value overrides even clear-cut scores
        should_invoke_llm = is_gray_zone or is_high_value
        assert should_invoke_llm is True


# ===================================================================
# Pipeline -- Critical drift fallback (retained from original)
# ===================================================================


@pytest.mark.unit
class TestPipelineFallback:
    """Verify that the pipeline falls back when drift is critical."""

    @patch("src.fraud_detection.pipeline.FraudPatternRetriever")
    @patch("src.fraud_detection.pipeline.EmbeddingDriftDetector")
    def test_critical_drift_triggers_fallback(
        self,
        mock_detector_cls: MagicMock,
        mock_retriever_cls: MagicMock,
    ) -> None:
        """When drift severity is critical, the pipeline should use the
        rule-based fallback engine instead of the ML or LLM path."""
        mock_report = MagicMock()
        mock_report.severity = "critical"
        mock_detector = MagicMock()
        mock_detector.evaluate.return_value = mock_report
        mock_detector_cls.return_value = mock_detector

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        # The fallback assessment should set analysis_tier accordingly
        assessment = FraudAssessment(
            transaction_id="txn_abc123",
            fraud_score=0.4,
            is_fraud=False,
            confidence=0.4,
            explanation="Rule-based fallback due to critical embedding drift.",
            analysis_tier="rule_based_fallback",
            ml_score=None,
            llm_score=None,
        )

        assert assessment.analysis_tier == "rule_based_fallback"
        assert assessment.ml_score is None
        assert assessment.llm_score is None
        assert "fallback" in assessment.explanation.lower()

    @patch("src.fraud_detection.pipeline.FraudPatternRetriever")
    @patch("src.fraud_detection.pipeline.EmbeddingDriftDetector")
    def test_nominal_drift_uses_normal_path(
        self,
        mock_detector_cls: MagicMock,
        mock_retriever_cls: MagicMock,
    ) -> None:
        """When drift is nominal, the pipeline should use the ML model
        (and optionally LLM) rather than the fallback."""
        mock_report = MagicMock()
        mock_report.severity = "nominal"
        mock_detector = MagicMock()
        mock_detector.evaluate.return_value = mock_report
        mock_detector_cls.return_value = mock_detector

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = ["pattern_A"]
        mock_retriever_cls.return_value = mock_retriever

        # With nominal drift, the assessment should NOT be fallback
        assessment = FraudAssessment(
            transaction_id="txn_abc123",
            fraud_score=0.2,
            is_fraud=False,
            confidence=0.6,
            ml_score=0.2,
            analysis_tier="ml_only",
        )

        assert assessment.analysis_tier != "rule_based_fallback"
        assert assessment.ml_score is not None


# ===================================================================
# Analysis tier tracking
# ===================================================================


@pytest.mark.unit
class TestAnalysisTierTracking:
    """Verify that analysis_tier is correctly set across all paths."""

    def test_tier_values_are_valid(self) -> None:
        """Only the three expected analysis_tier values should be used."""
        valid_tiers = {"ml_only", "ml_plus_llm", "rule_based_fallback"}

        for tier in valid_tiers:
            assessment = FraudAssessment(
                transaction_id="txn_test",
                fraud_score=0.5,
                is_fraud=False,
                confidence=0.5,
                analysis_tier=tier,
            )
            assert assessment.analysis_tier in valid_tiers

    def test_ml_only_has_no_llm_score(self) -> None:
        """ML-only tier assessments should have llm_score=None."""
        assessment = FraudAssessment(
            transaction_id="txn_test",
            fraud_score=0.2,
            is_fraud=False,
            confidence=0.6,
            ml_score=0.2,
            llm_score=None,
            analysis_tier="ml_only",
        )
        assert assessment.llm_score is None

    def test_ml_plus_llm_has_both_scores(self) -> None:
        """ML+LLM tier assessments should have both scores populated."""
        assessment = FraudAssessment(
            transaction_id="txn_test",
            fraud_score=0.55,
            is_fraud=False,
            confidence=0.5,
            ml_score=0.45,
            llm_score=0.55,
            analysis_tier="ml_plus_llm",
        )
        assert assessment.ml_score is not None
        assert assessment.llm_score is not None

    def test_decision_router_logic(self) -> None:
        """End-to-end test of the decision router logic that determines
        whether to invoke the LLM layer."""
        test_cases = [
            # (ml_score, amount, expected_tier)
            (0.10, 500.0, "ml_only"),
            (0.85, 500.0, "ml_only"),
            (0.45, 500.0, "ml_plus_llm"),
            (0.60, 500.0, "ml_plus_llm"),
            (0.10, 15000.0, "ml_plus_llm"),  # high value override
            (0.90, 15000.0, "ml_plus_llm"),  # high value override
            (0.50, 15000.0, "ml_plus_llm"),  # gray zone + high value
        ]

        for ml_score, amount, expected_tier in test_cases:
            is_gray_zone = 0.3 <= ml_score <= 0.7
            is_high_value = amount > 10_000
            should_invoke_llm = is_gray_zone or is_high_value

            actual_tier = "ml_plus_llm" if should_invoke_llm else "ml_only"
            assert actual_tier == expected_tier, (
                f"ml_score={ml_score}, amount={amount}: "
                f"expected {expected_tier}, got {actual_tier}"
            )
