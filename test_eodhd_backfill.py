from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

import eodhd_backfill as eb
from eodhd_backfill import (
    PricePoint,
    build_delisted_index,
    calculate_row,
    delisted_metadata,
    extract_fundamentals_summary,
    output_manual_review_queue,
)


class EodhdDelistedArchiveTests(unittest.TestCase):
    def test_delisted_index_uses_exchange_qualified_symbols(self):
        index = build_delisted_index(
            {
                "US": [{"Code": "TWTR", "Exchange": "US", "Name": "Twitter Inc.", "Type": "Common Stock"}],
                "HK": [{"Code": "0100", "Exchange": "HK", "Name": "Clear Media Limited"}],
            }
        )

        self.assertIn("TWTR.US", index)
        self.assertIn("TWTR", index)
        self.assertIn("0100.HK", index)
        self.assertNotIn("0100", index)

    def test_calculate_row_flags_exact_delisted_archive_match(self):
        prices = [
            PricePoint(dt.date(2017, 1, 3), 16.44, 16.44, {"date": "2017-01-03", "close": 16.44, "adjusted_close": 16.44}),
            PricePoint(dt.date(2018, 1, 3), 24.45, 24.45, {"date": "2018-01-03", "close": 24.45, "adjusted_close": 24.45}),
            PricePoint(dt.date(2020, 1, 3), 31.52, 31.52, {"date": "2020-01-03", "close": 31.52, "adjusted_close": 31.52}),
            PricePoint(dt.date(2022, 1, 3), 42.66, 42.66, {"date": "2022-01-03", "close": 42.66, "adjusted_close": 42.66}),
        ]
        delisted_index = build_delisted_index(
            {"US": [{"Code": "TWTR", "Exchange": "US", "Name": "Twitter Inc.", "Type": "Common Stock"}]}
        )

        row = calculate_row(
            {
                "idea_id": "twtr-row",
                "raw_symbol": "TWTR",
                "eodhd_symbol": "TWTR.US",
                "company_name": "Twitter Inc.",
                "publication_date": "2017-01-03",
            },
            prices,
            splits=[],
            dividends=[],
            delisted_index=delisted_index,
        )

        self.assertTrue(row["is_in_delisted_cache"])
        self.assertEqual(row["delisted_provider_code"], "TWTR")
        self.assertIn("symbol_in_delisted_cache", row["warning_modes"])
        self.assertEqual(row["review_stage"], "provider_warning")
        self.assertEqual(row["training_readiness"], "manual_review_required")
        self.assertEqual(row["delisted_provider_record"]["Name"], "Twitter Inc.")

    def test_reverse_split_is_not_low_risk_training_candidate(self):
        prices = [
            PricePoint(dt.date(2020, 1, 2), 10, 10, {"date": "2020-01-02", "adjusted_close": 10}),
            PricePoint(dt.date(2021, 1, 4), 11, 11, {"date": "2021-01-04", "adjusted_close": 11}),
            PricePoint(dt.date(2023, 1, 3), 12, 12, {"date": "2023-01-03", "adjusted_close": 12}),
            PricePoint(dt.date(2025, 1, 2), 13, 13, {"date": "2025-01-02", "adjusted_close": 13}),
        ]

        row = calculate_row(
            {
                "idea_id": "reverse-split-row",
                "raw_symbol": "RSPL",
                "eodhd_symbol": "RSPL.US",
                "company_name": "Reverse Split Co.",
                "publication_date": "2020-01-02",
            },
            prices,
            splits=[{"date": "2022-01-03", "split": "1/10"}],
            dividends=[],
            delisted_index={},
        )

        self.assertIn("reverse_split_provider_adjusted", row["warning_modes"])
        self.assertIn("high_risk_warning_needs_review", row["failure_modes"])
        self.assertEqual(row["math_validation_status"], "math_reproduced")
        self.assertEqual(row["review_stage"], "provider_warning")
        self.assertEqual(row["training_readiness"], "manual_review_required")
        self.assertEqual(row["manual_review_reason"], "reverse_split_provider_adjusted")

    def test_manual_review_queue_sorts_high_risk_rows(self):
        rows = [
            {
                "idea_id": "delisted",
                "raw_symbol": "BBB",
                "eodhd_symbol": "BBB.US",
                "publication_date": "2020-01-01",
                "review_stage": "provider_warning",
                "math_validation_status": "math_reproduced",
                "training_readiness": "manual_review_required",
                "manual_review_priority": 30,
                "manual_review_reason": "delisted_or_early_ended_history",
                "manual_review_tags": ["delisted_or_early_ended_history"],
                "horizons": {"5y": {"multiplier": 0.8}},
                "failure_modes": ["high_risk_warning_needs_review"],
                "warning_modes": ["symbol_in_delisted_cache"],
            },
            {
                "idea_id": "extreme",
                "raw_symbol": "AAA",
                "eodhd_symbol": "AAA.US",
                "publication_date": "2020-01-01",
                "review_stage": "provider_warning",
                "math_validation_status": "math_reproduced",
                "training_readiness": "manual_review_required",
                "manual_review_priority": 10,
                "manual_review_reason": "extreme_winner_gt_15x",
                "manual_review_tags": ["extreme_winner"],
                "horizons": {"5y": {"multiplier": 22}},
                "failure_modes": ["high_risk_warning_needs_review"],
                "warning_modes": ["extreme_return_requires_stronger_evidence"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            queue_path = Path(tmp) / "manual_review_queue.csv"
            output_manual_review_queue(rows, queue_path)
            lines = queue_path.read_text(encoding="utf-8").splitlines()

        self.assertIn("manual_review_priority", lines[0])
        self.assertIn("extreme", lines[1])
        self.assertIn("delisted", lines[2])

    def test_delisted_metadata_matches_non_us_qualified_symbol(self):
        index = build_delisted_index({"HK": [{"Code": "0100", "Exchange": "HK", "Name": "Clear Media Limited"}]})

        metadata = delisted_metadata("100 HK", "0100.HK", index)

        self.assertTrue(metadata["is_in_delisted_cache"])
        self.assertEqual(metadata["delisted_provider_exchange"], "HK")

    def test_extract_fundamentals_summary_includes_business_reality_fields(self):
        summary = extract_fundamentals_summary(
            "AAPL.US",
            {
                "symbol": "AAPL.US",
                "payload": {
                    "General": {
                        "Code": "AAPL",
                        "Type": "Common Stock",
                        "Name": "Apple Inc",
                        "Exchange": "NASDAQ",
                        "IsDelisted": False,
                        "Sector": "Technology",
                        "Industry": "Consumer Electronics",
                    },
                    "Highlights": {
                        "MarketCapitalization": 3000000000000,
                        "RevenueTTM": 390000000000,
                        "EBITDA": 130000000000,
                        "ProfitMargin": 0.24,
                    },
                    "Financials": {
                        "Income_Statement": {
                            "yearly": {
                                "2020-09-26": {"totalRevenue": "274515000000", "netIncome": "57411000000"},
                                "2024-09-28": {"totalRevenue": "391035000000", "netIncome": "93736000000"},
                            },
                            "quarterly": {
                                "2024-12-28": {"totalRevenue": "124300000000", "netIncome": "36330000000"}
                            },
                        }
                    },
                },
            },
        )

        self.assertEqual(summary["fundamentals_status"], "fetched")
        self.assertEqual(summary["fundamentals_type"], "Common Stock")
        self.assertEqual(summary["fundamentals_sector"], "Technology")
        self.assertEqual(summary["fundamentals_yearly_revenue_first"], 274515000000)
        self.assertEqual(summary["fundamentals_yearly_revenue_last"], 391035000000)
        self.assertTrue(summary["fundamentals_has_financials"])

    def test_bulk_fundamentals_materializes_symbol_cache_files(self):
        original_request_json = eb.request_json
        try:
            eb.request_json = lambda path, params: (
                {
                    "0": {
                        "General": {
                            "Code": "AAPL",
                            "PrimaryTicker": "AAPL.US",
                            "Type": "Common Stock",
                            "Name": "Apple Inc",
                        },
                        "Highlights": {"RevenueTTM": 390000000000},
                    },
                    "1": {
                        "General": {
                            "Code": "MSFT",
                            "PrimaryTicker": "MSFT.US",
                            "Type": "Common Stock",
                            "Name": "Microsoft Corporation",
                        },
                        "Highlights": {"RevenueTTM": 250000000000},
                    },
                },
                [],
            )
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bundles = eb.load_fundamentals_bundles_bulk(
                    ["AAPL.US", "MSFT.US"],
                    root / "symbols",
                    root / "bulk",
                    refresh=True,
                    workers=1,
                )

                self.assertIn("AAPL.US", bundles)
                self.assertTrue((root / "symbols" / "AAPL.US.json").exists())
                self.assertTrue(list((root / "bulk").glob("us_*.json")))
                summary = extract_fundamentals_summary("AAPL.US", bundles["AAPL.US"])
                self.assertEqual(summary["fundamentals_name"], "Apple Inc")
        finally:
            eb.request_json = original_request_json


if __name__ == "__main__":
    unittest.main()
