import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from manual_review_validation import (
    PricePoint,
    ProviderSeries,
    provider_returns,
    relative_diff,
    review_row,
    select_rows,
)


class ManualReviewValidationTests(unittest.TestCase):
    def test_provider_returns_use_adjusted_close(self):
        series = ProviderSeries(
            "fixture",
            "AAA.US",
            [
                PricePoint(dt.date(2020, 1, 1), 100, 50, "fixture"),
                PricePoint(dt.date(2021, 1, 1), 110, 110, "fixture"),
            ],
            [],
        )

        returns = provider_returns(series, dt.date(2020, 1, 1))

        self.assertAlmostEqual(returns["1y"]["multiplier"], 2.2)

    def test_relative_diff_is_symmetric_and_bounded(self):
        self.assertAlmostEqual(relative_diff(10, 11), relative_diff(11, 10))
        self.assertAlmostEqual(relative_diff(10, 10), 0)
        self.assertIsNone(relative_diff(None, 10))

    def test_mcem_like_adjustment_conflict_rejects(self):
        # MCEM failure pattern: EODHD start adjusted close is tiny while the
        # verifier sees a normal adjusted/close relationship.
        eodhd = ProviderSeries(
            "eodhd",
            "MCEM.US",
            [
                PricePoint(dt.date(2014, 1, 31), 25, 0.0281, "eodhd"),
                PricePoint(dt.date(2019, 1, 31), 64, 48.441, "eodhd"),
            ],
            [],
        )
        yahoo = ProviderSeries(
            "yahoo",
            "MCEM",
            [
                PricePoint(dt.date(2014, 1, 31), 25, 16.3063, "yahoo"),
                PricePoint(dt.date(2019, 1, 31), 64, 48.4398, "yahoo"),
            ],
            [],
        )

        # Avoid network and file I/O by monkey-patching the module functions.
        import manual_review_validation as mrv

        original_load = mrv.load_eodhd_series
        original_fetch = mrv.fetch_yahoo_series
        try:
            mrv.load_eodhd_series = lambda symbol, cache_dir: eodhd
            mrv.fetch_yahoo_series = lambda symbol, start, end: yahoo
            review = review_row(
                {
                    "idea_id": "fixture",
                    "raw_symbol": "MCEM",
                    "eodhd_symbol": "MCEM.US",
                    "publication_date": "2014-01-31",
                }
            )
        finally:
            mrv.load_eodhd_series = original_load
            mrv.fetch_yahoo_series = original_fetch

        self.assertEqual(review["review_status"], "reject")
        self.assertIn("provider_adjustment_factor_conflict_with_return_mismatch", review["row_failures"])
        self.assertEqual(review["horizon_reviews"]["5y"]["verdict"], "fail")

    def test_clean_extreme_reproduced_needs_qualitative_evidence(self):
        eodhd = ProviderSeries(
            "eodhd",
            "AAPL.US",
            [
                PricePoint(dt.date(2003, 4, 22), 13.5, 0.2022, "eodhd"),
                PricePoint(dt.date(2004, 4, 22), 13.5, 0.4, "eodhd"),
                PricePoint(dt.date(2006, 4, 24), 13.5, 2.0, "eodhd"),
                PricePoint(dt.date(2008, 4, 22), 160.2, 4.7956, "eodhd"),
            ],
            [],
        )
        yahoo = ProviderSeries(
            "yahoo",
            "AAPL",
            [
                PricePoint(dt.date(2003, 4, 22), 13.5, 0.20221, "yahoo"),
                PricePoint(dt.date(2004, 4, 22), 13.5, 0.4001, "yahoo"),
                PricePoint(dt.date(2006, 4, 24), 13.5, 2.001, "yahoo"),
                PricePoint(dt.date(2008, 4, 22), 160.2, 4.7957, "yahoo"),
            ],
            [],
        )

        import manual_review_validation as mrv

        original_load = mrv.load_eodhd_series
        original_fetch = mrv.fetch_yahoo_series
        try:
            mrv.load_eodhd_series = lambda symbol, cache_dir: eodhd
            mrv.fetch_yahoo_series = lambda symbol, start, end: yahoo
            review = review_row(
                {
                    "idea_id": "fixture",
                    "raw_symbol": "AAPL",
                    "eodhd_symbol": "AAPL.US",
                    "publication_date": "2003-04-22",
                }
            )
        finally:
            mrv.load_eodhd_series = original_load
            mrv.fetch_yahoo_series = original_fetch

        self.assertEqual(review["review_status"], "manual_review")
        self.assertIn("extreme_price_reproduced", review["horizon_reviews"]["5y"]["reasons"])
        self.assertIn("qualitative_evidence_required_for_extreme_return", review["horizon_reviews"]["5y"]["reasons"])

    def test_missing_requested_yahoo_cross_check_does_not_pass(self):
        eodhd = ProviderSeries(
            "eodhd",
            "AAPL.US",
            [
                PricePoint(dt.date(2020, 1, 2), 10, 10, "eodhd"),
                PricePoint(dt.date(2021, 1, 4), 12, 12, "eodhd"),
                PricePoint(dt.date(2023, 1, 3), 13, 13, "eodhd"),
                PricePoint(dt.date(2025, 1, 2), 15, 15, "eodhd"),
            ],
            [],
        )
        yahoo = ProviderSeries("yahoo", "AAPL", [], ["network:blocked"])

        import manual_review_validation as mrv

        original_load = mrv.load_eodhd_series
        original_fetch = mrv.fetch_yahoo_series
        try:
            mrv.load_eodhd_series = lambda symbol, cache_dir: eodhd
            mrv.fetch_yahoo_series = lambda symbol, start, end: yahoo
            review = review_row(
                {
                    "idea_id": "fixture",
                    "raw_symbol": "AAPL",
                    "eodhd_symbol": "AAPL.US",
                    "publication_date": "2020-01-02",
                    "validation_status": "verified_candidate_provider_adjusted",
                    "math_validation_status": "math_reproduced",
                    "review_stage": "math_reproduced_low_risk",
                    "training_readiness": "candidate_low_risk",
                }
            )
        finally:
            mrv.load_eodhd_series = original_load
            mrv.fetch_yahoo_series = original_fetch

        self.assertEqual(review["review_status"], "manual_review")
        self.assertIn("yahoo_cross_check_unavailable", review["horizon_reviews"]["1y"]["reasons"])

    def test_clean_extreme_with_qualitative_evidence_can_pass(self):
        eodhd = ProviderSeries(
            "eodhd",
            "AAPL.US",
            [
                PricePoint(dt.date(2003, 4, 22), 13.5, 0.2022, "eodhd"),
                PricePoint(dt.date(2004, 4, 22), 13.5, 0.4, "eodhd"),
                PricePoint(dt.date(2006, 4, 24), 13.5, 2.0, "eodhd"),
                PricePoint(dt.date(2008, 4, 22), 160.2, 4.7956, "eodhd"),
            ],
            [],
        )
        yahoo = ProviderSeries(
            "yahoo",
            "AAPL",
            [
                PricePoint(dt.date(2003, 4, 22), 13.5, 0.20221, "yahoo"),
                PricePoint(dt.date(2004, 4, 22), 13.5, 0.4001, "yahoo"),
                PricePoint(dt.date(2006, 4, 24), 13.5, 2.001, "yahoo"),
                PricePoint(dt.date(2008, 4, 22), 160.2, 4.7957, "yahoo"),
            ],
            [],
        )

        import manual_review_validation as mrv

        original_load = mrv.load_eodhd_series
        original_fetch = mrv.fetch_yahoo_series
        try:
            mrv.load_eodhd_series = lambda symbol, cache_dir: eodhd
            mrv.fetch_yahoo_series = lambda symbol, start, end: yahoo
            review = review_row(
                {
                    "idea_id": "fixture",
                    "raw_symbol": "AAPL",
                    "eodhd_symbol": "AAPL.US",
                    "publication_date": "2003-04-22",
                },
                qualitative_evidence={
                    "AAPL|2003-04-22": {
                        "verdict": "pass",
                        "outcome_type": "extreme_winner",
                        "business_explanation": "Product-cycle growth supports the return.",
                        "sources": [{"source_id": "fixture:annual_report", "source_type": "filing"}],
                        "confidence": 0.9,
                    }
                },
            )
        finally:
            mrv.load_eodhd_series = original_load
            mrv.fetch_yahoo_series = original_fetch

        self.assertEqual(review["review_status"], "pass")
        self.assertIn("qualitative_evidence_supports_extreme_return", review["horizon_reviews"]["5y"]["reasons"])
        self.assertEqual(review["agent_c_qualitative"]["reviewer_status"], "pass")

    def test_fundamentals_cache_is_attached_to_agent_c_review(self):
        eodhd = ProviderSeries(
            "eodhd",
            "AAPL.US",
            [
                PricePoint(dt.date(2003, 4, 22), 13.5, 0.2, "eodhd"),
                PricePoint(dt.date(2008, 4, 22), 160.2, 4.8, "eodhd"),
            ],
            [],
        )
        yahoo = ProviderSeries(
            "yahoo",
            "AAPL",
            [
                PricePoint(dt.date(2003, 4, 22), 13.5, 0.2, "yahoo"),
                PricePoint(dt.date(2008, 4, 22), 160.2, 4.8, "yahoo"),
            ],
            [],
        )

        import manual_review_validation as mrv

        original_load = mrv.load_eodhd_series
        original_fetch = mrv.fetch_yahoo_series
        with tempfile.TemporaryDirectory() as tmp:
            fundamentals_dir = Path(tmp)
            (fundamentals_dir / "AAPL.US.json").write_text(
                json.dumps(
                    {
                        "symbol": "AAPL.US",
                        "payload": {
                            "General": {
                                "Code": "AAPL",
                                "Type": "Common Stock",
                                "Name": "Apple Inc",
                                "Sector": "Technology",
                                "Industry": "Consumer Electronics",
                            },
                            "Highlights": {"MarketCapitalization": 3000000000000, "RevenueTTM": 390000000000},
                            "Financials": {
                                "Income_Statement": {
                                    "yearly": {
                                        "2020-09-26": {"totalRevenue": "274515000000", "netIncome": "57411000000"}
                                    }
                                }
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            try:
                mrv.load_eodhd_series = lambda symbol, cache_dir: eodhd
                mrv.fetch_yahoo_series = lambda symbol, start, end: yahoo
                review = review_row(
                    {
                        "idea_id": "fixture",
                        "raw_symbol": "AAPL",
                        "eodhd_symbol": "AAPL.US",
                        "publication_date": "2003-04-22",
                    },
                    fundamentals_cache_dir=fundamentals_dir,
                )
            finally:
                mrv.load_eodhd_series = original_load
                mrv.fetch_yahoo_series = original_fetch

        self.assertEqual(review["review_status"], "manual_review")
        self.assertEqual(review["agent_c_qualitative"]["reason"], "fundamentals_available_needs_qualitative_synthesis")
        self.assertEqual(review["agent_c_fundamentals"]["fundamentals_name"], "Apple Inc")
        self.assertEqual(review["agent_c_fundamentals"]["fundamentals_revenue_ttm"], 390000000000)

    def test_select_rows_uses_priority_review_order(self):
        rows = [
            {
                "idea_id": "ordinary",
                "raw_symbol": "AAA",
                "eodhd_symbol": "AAA.US",
                "publication_date": "2020-01-01",
                "validation_status": "verified_candidate_provider_adjusted",
                "perf_5y": "1.5",
                "failure_modes": "",
                "warning_modes": "",
            },
            {
                "idea_id": "delisted",
                "raw_symbol": "BBB",
                "eodhd_symbol": "BBB.US",
                "publication_date": "2020-01-01",
                "validation_status": "needs_manual_review",
                "perf_5y": "0.8",
                "failure_modes": "",
                "warning_modes": "symbol_in_delisted_cache",
            },
            {
                "idea_id": "extreme",
                "raw_symbol": "CCC",
                "eodhd_symbol": "CCC.US",
                "publication_date": "2020-01-01",
                "validation_status": "needs_manual_review",
                "perf_5y": "22",
                "failure_modes": "",
                "warning_modes": "extreme_return_requires_stronger_evidence",
            },
        ]

        selected = select_rows(rows, symbols=None, limit=None)

        self.assertEqual([row["idea_id"] for row in selected], ["extreme", "delisted", "ordinary"])


if __name__ == "__main__":
    unittest.main()
