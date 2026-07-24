from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_hf_stage" / "data"
OUT_DIR = ROOT / "eodhd_output" / "alpha_vantage_transcripts"
RAW_DIR = OUT_DIR / "raw"
ROW_COVERAGE_CSV = OUT_DIR / "vic_transcript_row_coverage.csv"
SUMMARY_JSON = OUT_DIR / "transcript_enrichment_summary.json"
MANIFEST_CSV = OUT_DIR / "alpha_vantage_transcript_manifest.csv"

ALPHA_URL = "https://www.alphavantage.co/query"


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def quarter_for(value: date) -> str:
    quarter = (value.month - 1) // 3 + 1
    return f"{value.year}Q{quarter}"


def add_quarters(start: date, count: int) -> date:
    month_index = (start.year * 12 + start.month - 1) + count * 3
    return date(month_index // 12, month_index % 12 + 1, 1)


def publication_to_3y_quarters(publication_date: date) -> list[str]:
    first = date(publication_date.year, ((publication_date.month - 1) // 3) * 3 + 1, 1)
    quarters = []
    cursor = first
    end = date(publication_date.year + 3, publication_date.month, min(publication_date.day, 28))
    while cursor <= end:
        quarters.append(quarter_for(cursor))
        cursor = add_quarters(cursor, 1)
    return quarters


def alpha_symbol(row: dict[str, Any]) -> str | None:
    eodhd_symbol = str(row.get("eodhd_symbol") or "").strip()
    if not eodhd_symbol.endswith(".US"):
        return None
    symbol = eodhd_symbol[:-3]
    if not symbol or "-" in symbol or "/" in symbol:
        return None
    return symbol.replace(".", "-")


def iter_rows():
    for path in sorted(SOURCE_DIR.glob("vic_pitch_financial_context-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    yield path.name, line_number, json.loads(line)


def cache_path(symbol: str, quarter: str) -> Path:
    return RAW_DIR / symbol / f"{quarter}.json"


def fetch_transcript(symbol: str, quarter: str, api_key: str, sleep_seconds: float) -> dict[str, Any]:
    path = cache_path(symbol, quarter)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    params = {
        "function": "EARNINGS_CALL_TRANSCRIPT",
        "symbol": symbol,
        "quarter": quarter,
        "apikey": api_key,
    }
    url = f"{ALPHA_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if sleep_seconds:
        time.sleep(sleep_seconds)
    return payload


def is_valid_transcript(payload: dict[str, Any]) -> bool:
    transcript = payload.get("transcript")
    return isinstance(transcript, list) and len(transcript) > 0


def build_tasks(rows: list[dict[str, Any]], max_rows: int | None) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    selected = rows[:max_rows] if max_rows else rows
    tasks: set[tuple[str, str]] = set()
    for row in selected:
        symbol = alpha_symbol(row)
        pub_date = parse_date(row.get("publication_date"))
        if not symbol or not pub_date:
            continue
        for quarter in publication_to_3y_quarters(pub_date):
            tasks.add((symbol, quarter))
    return selected, tasks


def write_row_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ROW_COVERAGE_CSV.parent.mkdir(parents=True, exist_ok=True)
    coverage_rows = []
    for row in rows:
        symbol = alpha_symbol(row)
        pub_date = parse_date(row.get("publication_date"))
        quarters = publication_to_3y_quarters(pub_date) if symbol and pub_date else []
        available = []
        missing = []
        for quarter in quarters:
            path = cache_path(symbol, quarter)
            if path.exists() and is_valid_transcript(json.loads(path.read_text(encoding="utf-8"))):
                available.append(quarter)
            else:
                missing.append(quarter)
        coverage_rows.append(
            {
                "idea_id": row.get("idea_id"),
                "eodhd_symbol": row.get("eodhd_symbol"),
                "alpha_vantage_symbol": symbol,
                "company_name": row.get("company_name"),
                "publication_date": row.get("publication_date"),
                "quarters_expected_publication_to_3y": len(quarters),
                "quarters_with_transcripts": len(available),
                "quarters_missing_transcripts": len(missing),
                "available_quarters": ";".join(available),
                "missing_quarters": ";".join(missing),
            }
        )

    fieldnames = list(coverage_rows[0].keys()) if coverage_rows else []
    with ROW_COVERAGE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(coverage_rows)

    return {
        "row_coverage_csv": str(ROW_COVERAGE_CSV),
        "rows": len(coverage_rows),
        "rows_with_any_transcript": sum(1 for row in coverage_rows if row["quarters_with_transcripts"] > 0),
        "rows_with_all_expected_transcripts": sum(
            1
            for row in coverage_rows
            if row["quarters_expected_publication_to_3y"] > 0
            and row["quarters_missing_transcripts"] == 0
        ),
        "total_expected_row_quarters": sum(row["quarters_expected_publication_to_3y"] for row in coverage_rows),
        "total_available_row_quarters": sum(row["quarters_with_transcripts"] for row in coverage_rows),
    }


def write_manifest(tasks: set[tuple[str, str]]) -> dict[str, Any]:
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["alpha_vantage_symbol", "quarter"])
        writer.writeheader()
        for symbol, quarter in sorted(tasks):
            writer.writerow({"alpha_vantage_symbol": symbol, "quarter": quarter})
    symbols = {symbol for symbol, _quarter in tasks}
    return {
        "manifest_csv": str(MANIFEST_CSV),
        "unique_symbols": len(symbols),
        "unique_symbol_quarter_tasks": len(tasks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=12.5)
    parser.add_argument("--demo-smoke", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.demo_smoke:
        api_key = "demo"
        payload = fetch_transcript("IBM", "2024Q1", api_key, 0)
        summary = {
            "mode": "demo-smoke",
            "symbol": "IBM",
            "quarter": "2024Q1",
            "valid_transcript": is_valid_transcript(payload),
            "turns": len(payload.get("transcript") or []),
            "raw_path": str(cache_path("IBM", "2024Q1")),
        }
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    rows = [row for _source, _line, row in iter_rows()]
    selected_rows, tasks = build_tasks(rows, args.max_rows)
    manifest = write_manifest(tasks)
    if args.manifest_only:
        coverage = write_row_coverage(selected_rows)
        summary = {
            "mode": "manifest-only",
            "selected_rows": len(selected_rows),
            "manifest": manifest,
            "coverage_from_existing_cache": coverage,
        }
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise SystemExit("Set ALPHAVANTAGE_API_KEY, or run --manifest-only / --demo-smoke.")

    ordered_tasks = sorted(tasks)
    if args.max_requests is not None:
        ordered_tasks = ordered_tasks[: args.max_requests]

    fetched = 0
    valid = 0
    errors = []
    for symbol, quarter in ordered_tasks:
        try:
            payload = fetch_transcript(symbol, quarter, api_key, args.sleep_seconds)
            fetched += 1
            if is_valid_transcript(payload):
                valid += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"symbol": symbol, "quarter": quarter, "error": str(exc)})

    coverage = write_row_coverage(selected_rows)
    summary = {
        "mode": "full-or-limited",
        "selected_rows": len(selected_rows),
        "manifest": manifest,
        "unique_symbol_quarter_tasks": len(tasks),
        "attempted_requests": len(ordered_tasks),
        "fetched_or_cached": fetched,
        "valid_transcripts": valid,
        "errors": errors[:50],
        "error_count": len(errors),
        "coverage": coverage,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
