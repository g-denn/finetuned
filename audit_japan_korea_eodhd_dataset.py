#!/usr/bin/env python3
"""Audit whether the Japan/Korea EODHD dataset satisfies the collection goal."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("EODHD_JK_OUT_DIR", str(ROOT / "eodhd_output" / "japan_korea_fundamentals")))
RAW_DIR = DATASET_DIR / "raw"
NORMALIZED_DIR = DATASET_DIR / "normalized"
SCREEN_DIR = DATASET_DIR / "screening"


REQUIRED_NORMALIZED = [
    "companies.csv",
    "income_statement.csv",
    "balance_sheet.csv",
    "cash_flow.csv",
    "income_statement_latest_5y.csv",
    "balance_sheet_latest_5y.csv",
    "cash_flow_latest_5y.csv",
    "earnings.csv",
    "fundamentals_raw_payloads.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-screening", action="store_true", help="Require screening output files.")
    parser.add_argument("--require-upload-status", action="store_true", help="Require Hugging Face upload status.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def count_jsonl_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def manifest_symbols(manifest: Any) -> list[dict[str, Any]]:
    return manifest if isinstance(manifest, list) else []


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    issues: list[str] = []
    warnings: list[str] = []

    manifest = manifest_symbols(read_json(DATASET_DIR / "stock_pull_manifest.json"))
    progress = read_json(DATASET_DIR / "progress_summary.json") or {}
    transcript = read_json(DATASET_DIR / "earnings_transcript_availability.json") or {}
    normalization = read_json(DATASET_DIR / "normalization_summary.json") or {}
    country_availability = read_json(DATASET_DIR / "country_availability.json") or {}

    if not manifest:
        issues.append("missing or empty stock_pull_manifest.json")
    country_counts = Counter(str(row.get("country_key") or "") for row in manifest)
    for country_key in ["JP", "KR"]:
        if country_counts[country_key] == 0:
            availability = country_availability.get(country_key, {}) if isinstance(country_availability, dict) else {}
            if availability.get("available_in_exchanges_list") is False:
                warnings.append(f"manifest has no {country_key} symbols because EODHD exchanges-list returned no {country_key} exchange")
            else:
                issues.append(f"manifest has no {country_key} symbols")

    raw_saved = 0
    raw_missing: list[str] = []
    raw_invalid: list[str] = []
    statement_coverage = {
        "Income_Statement": 0,
        "Balance_Sheet": 0,
        "Cash_Flow": 0,
    }
    five_year_payload_coverage = {
        "Income_Statement": 0,
        "Balance_Sheet": 0,
        "Cash_Flow": 0,
    }
    error_files = []

    for row in manifest:
        symbol = str(row.get("symbol") or "")
        if not symbol:
            continue
        symbol_dir = RAW_DIR / symbol.replace("/", "_").replace("\\", "_").replace(":", "_")
        raw_path = symbol_dir / "fundamentals.json"
        error_files.extend(str(path.relative_to(DATASET_DIR)) for path in symbol_dir.glob("*.error.json"))
        payload = read_json(raw_path)
        if payload is None:
            raw_missing.append(symbol)
            continue
        if not isinstance(payload, dict):
            raw_invalid.append(symbol)
            continue
        raw_saved += 1
        financials = payload.get("Financials", {})
        for statement in statement_coverage:
            yearly = financials.get(statement, {}).get("yearly", {}) if isinstance(financials, dict) else {}
            if isinstance(yearly, dict) and yearly:
                statement_coverage[statement] += 1
                if len(yearly) >= 5:
                    five_year_payload_coverage[statement] += 1

    if raw_missing:
        issues.append(f"missing raw fundamentals for {len(raw_missing)} symbols")
    if raw_invalid:
        issues.append(f"invalid raw fundamentals payloads for {len(raw_invalid)} symbols")
    if error_files:
        warnings.append(f"{len(error_files)} raw error files present")

    normalized_counts: dict[str, int | None] = {}
    for filename in REQUIRED_NORMALIZED:
        path = NORMALIZED_DIR / filename
        if filename.endswith(".jsonl"):
            row_count = count_jsonl_rows(path)
        else:
            row_count = count_csv_rows(path)
        normalized_counts[filename] = row_count
        if row_count is None:
            issues.append(f"missing normalized/{filename}")

    for filename in ["income_statement_latest_5y.csv", "balance_sheet_latest_5y.csv", "cash_flow_latest_5y.csv"]:
        count = normalized_counts.get(filename)
        if count is not None and manifest and count < len(manifest):
            warnings.append(f"normalized/{filename} has {count} rows for {len(manifest)} manifest symbols")

    if not transcript:
        issues.append("missing earnings_transcript_availability.json")
    elif transcript.get("documented_in_eodhd_fundamentals_or_calendar_docs") is not False:
        warnings.append("transcript documentation status is not explicitly recorded as unavailable")

    screening_files = [
        SCREEN_DIR / "shareholder_yield_screen_all.csv",
        SCREEN_DIR / "shareholder_yield_screen_top.csv",
        SCREEN_DIR / "screening_summary.json",
    ]
    if args.require_screening:
        for path in screening_files:
            if not path.exists():
                issues.append(f"missing screening/{path.name}")

    upload_status = read_json(ROOT / "eodhd_output" / "hf_japan_korea_fundamentals_upload_status.json")
    if args.require_upload_status:
        if not upload_status:
            issues.append("missing Hugging Face upload status")
        elif upload_status.get("stage") != "complete":
            issues.append("Hugging Face upload status is not complete")
        elif upload_status.get("missing_expected_files"):
            issues.append("Hugging Face upload reports missing expected files")

    report = {
        "dataset_dir": str(DATASET_DIR),
        "complete": not issues,
        "issues": issues,
        "warnings": warnings,
        "manifest_symbols": len(manifest),
        "manifest_country_counts": dict(country_counts),
        "progress_summary": progress,
        "normalization_summary": normalization,
        "country_availability": country_availability,
        "raw_fundamentals_saved": raw_saved,
        "raw_missing_sample": raw_missing[:20],
        "raw_invalid_sample": raw_invalid[:20],
        "statement_coverage_symbols": statement_coverage,
        "five_year_payload_coverage_symbols": five_year_payload_coverage,
        "normalized_row_counts": normalized_counts,
        "transcript_status": transcript,
        "screening_required": bool(args.require_screening),
        "upload_required": bool(args.require_upload_status),
    }
    return report, 0 if not issues else 1


def main() -> int:
    args = parse_args()
    report, exit_code = audit(args)
    out_path = DATASET_DIR / "dataset_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
