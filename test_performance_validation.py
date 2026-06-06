from __future__ import annotations

import unittest

from performance_validation import (
    CorporateActionStatus,
    IdentityStatus,
    LabelQuality,
    ValidationStatus,
    build_seed_row,
    evaluate_training_gate,
)


def verified_row(**overrides):
    row = {
        "validation_status": ValidationStatus.VERIFIED_EXACT,
        "identity_status": IdentityStatus.SAME_SECURITY,
        "corporate_action_status": CorporateActionStatus.ADJUSTED_BY_PROVIDER,
        "label_quality": LabelQuality.HIGH,
        "identity_confidence": 0.95,
        "return_confidence": 0.9,
        "agent_b_result": {
            "reviewer_status": "pass",
            "calculation_reproduced": True,
            "identity_passed": True,
            "source_quality_passed": True,
            "matching_label": True,
        },
        "corporate_action_timeline": [
            {
                "date": "2020-08-31",
                "type": "split",
                "description": "AAPL 4:1 split",
                "materiality": "high",
                "source_ids": ["provider:yahoo", "company_ir:aapl"],
            }
        ],
        "sources": [
            {
                "source_id": "provider:yahoo",
                "url": "https://query1.finance.yahoo.com/v8/finance/chart/AAPL",
                "publisher": "Yahoo Finance",
                "source_type": "data_vendor",
                "supports": "price",
                "quote_or_fact": "Adjusted close series includes split adjustment.",
            },
            {
                "source_id": "company_ir:aapl",
                "url": "https://investor.apple.com/",
                "publisher": "Apple Investor Relations",
                "source_type": "company_ir",
                "supports": "corporate_action",
                "quote_or_fact": "4-for-1 split effective August 31, 2020.",
            },
        ],
    }
    row.update(overrides)
    return row


class PerformanceValidationGateTests(unittest.TestCase):
    def test_aapl_split_adjusted_control_can_pass(self):
        result = evaluate_training_gate(verified_row())

        self.assertTrue(result.allowed)
        self.assertEqual(result.reasons, [])

    def test_raw_yahoo_math_alone_cannot_pass(self):
        result = evaluate_training_gate(
            verified_row(
                sources=[
                    {
                        "source_id": "provider:yahoo",
                        "url": "https://query1.finance.yahoo.com/v8/finance/chart/AAPL",
                        "publisher": "Yahoo Finance",
                        "source_type": "data_vendor",
                        "supports": "price",
                        "quote_or_fact": "Adjusted close series.",
                    }
                ]
            )
        )

        self.assertFalse(result.allowed)
        self.assertIn("insufficient_sources", result.reasons)

    def test_agent_a_pass_agent_b_fail_excludes_row(self):
        result = evaluate_training_gate(
            verified_row(agent_b_result={"reviewer_status": "fail", "reason_code": "WRONG_SECURITY"})
        )

        self.assertFalse(result.allowed)
        self.assertIn("agent_b_not_pass", result.reasons)

    def test_sfi_ticker_change_without_lineage_evidence_fails(self):
        result = evaluate_training_gate(
            verified_row(
                validation_status=ValidationStatus.NEEDS_MANUAL_REVIEW,
                identity_status=IdentityStatus.TICKER_REUSE_SUSPECTED,
                corporate_action_status=CorporateActionStatus.UNKNOWN,
                identity_confidence=0.4,
                return_confidence=0.2,
                failure_modes=["ticker_change_requires_lineage_proof"],
            )
        )

        self.assertFalse(result.allowed)
        self.assertIn("validation_status_not_training_eligible", result.reasons)
        self.assertIn("corporate_action_status_not_training_eligible", result.reasons)

    def test_pqe_identity_break_cannot_enter_training(self):
        result = evaluate_training_gate(
            verified_row(
                validation_status=ValidationStatus.IDENTITY_CONFLICT,
                identity_status=IdentityStatus.UNKNOWN,
                corporate_action_status=CorporateActionStatus.CONFLICTING_ACTION_DATA,
            )
        )

        self.assertFalse(result.allowed)
        self.assertIn("validation_status_not_training_eligible", result.reasons)

    def test_ksw_fake_post_acquisition_endpoint_fails(self):
        result = evaluate_training_gate(
            verified_row(
                validation_status=ValidationStatus.NEEDS_MANUAL_REVIEW,
                identity_status=IdentityStatus.ACQUIRED_CASH,
                corporate_action_status=CorporateActionStatus.MISSING_MATERIAL_ACTION,
                corporate_action_timeline=[
                    {
                        "date": "2012-01-01",
                        "type": "merger_cash",
                        "description": "Acquisition before modeled 5y endpoint.",
                        "materiality": "high",
                        "source_ids": [],
                    }
                ],
            )
        )

        self.assertFalse(result.allowed)
        self.assertIn("unresolved_material_corporate_action", result.reasons)

    def test_large_dividend_or_spinoff_must_be_modelled(self):
        result = evaluate_training_gate(
            verified_row(
                validation_status=ValidationStatus.VERIFIED_WITH_CORPORATE_ACTION,
                corporate_action_status=CorporateActionStatus.PARTIALLY_MODELED,
                corporate_action_timeline=[
                    {
                        "date": "2015-01-01",
                        "type": "spinoff",
                        "description": "Material spin-off value not fully modeled.",
                        "materiality": "high",
                        "source_ids": ["company_ir:test"],
                    }
                ],
            )
        )

        self.assertFalse(result.allowed)
        self.assertIn("corporate_action_status_not_training_eligible", result.reasons)
        self.assertIn("unresolved_material_corporate_action", result.reasons)

    def test_seed_row_records_provider_split_evidence_but_does_not_verify(self):
        seed = build_seed_row(
            {
                "idea_id": "aapl-row",
                "raw_symbol": "AAPL",
                "yahoo_symbol": "AAPL",
                "company_name": "Apple Inc.",
                "publication_date": "2020-08-30",
                "position_type": "long",
                "split_events": [
                    {
                        "date": "2020-08-31",
                        "numerator": 4,
                        "denominator": 1,
                        "splitRatio": "4:1",
                    }
                ],
                "dividend_events": [],
            }
        )

        self.assertEqual(seed["validation_status"], ValidationStatus.UNREVIEWED)
        self.assertFalse(seed["include_in_training"])
        self.assertTrue(seed["split_adjusted"])
        self.assertEqual(seed["corporate_action_status"], CorporateActionStatus.ADJUSTED_BY_PROVIDER)
        self.assertEqual(seed["corporate_action_timeline"][0]["type"], "split")


if __name__ == "__main__":
    unittest.main()
