from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
CANONICAL_CSV = ROOT / "data" / "processed" / "investment_canonical.csv"
RAW_DIR = ROOT / "eodhd_output" / "dataset_financial_pull" / "raw"
OUT_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context"
OUT_JSONL = OUT_DIR / "vic_pitch_financial_context.jsonl"
OUT_CSV = OUT_DIR / "vic_pitch_financial_context_preview.csv"
OUT_SUMMARY = OUT_DIR / "dataset_summary.json"
OUT_README = OUT_DIR / "README.md"

STATEMENTS = {
    "Balance_Sheet": "balance_sheet",
    "Cash_Flow": "cash_flow",
    "Income_Statement": "income_statement",
}
PERIODS = ("yearly", "quarterly")

PITCH_COLUMNS = [
    "idea_id",
    "raw_symbol",
    "eodhd_symbol",
    "company_name",
    "publication_date",
    "is_short",
    "is_contest_winner",
    "link",
    "author_user_id",
    "description",
    "catalyst",
    "fundamentals_sector",
    "fundamentals_industry",
    "fundamentals_market_cap",
    "fundamentals_revenue_ttm",
    "fundamentals_profit_margin",
    "math_validation_status",
    "review_status",
    "reviewed_at",
    "raw_perf_1y",
    "directional_perf_1y",
    "outcome_1y",
    "raw_perf_3y",
    "directional_perf_3y",
    "outcome_3y",
    "raw_perf_5y",
    "directional_perf_5y",
    "outcome_5y",
    "raw_perf_10y",
    "directional_perf_10y",
    "outcome_10y",
    "raw_perf_20y",
    "directional_perf_20y",
    "outcome_20y",
    "primary_horizon",
    "primary_outcome",
]


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NONE", "NULL"}:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "statement": record["statement"],
        "period": record["period"],
        "fiscal_date": record["fiscal_date"],
        "filing_date": record["filing_date"],
        "currency_symbol": record["data"].get("currency_symbol"),
        "data": json_ready(record["data"]),
    }


def extract_statement_records(fundamentals: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    financials = fundamentals.get("Financials", {})
    if not isinstance(financials, dict):
        return records

    for source_key, statement_name in STATEMENTS.items():
        statement = financials.get(source_key, {})
        if not isinstance(statement, dict):
            continue
        for period in PERIODS:
            period_records = statement.get(period, {})
            if not isinstance(period_records, dict):
                continue
            for record_key, payload in period_records.items():
                if not isinstance(payload, dict):
                    continue
                fiscal_date = parse_date(payload.get("date") or record_key)
                filing_date = parse_date(payload.get("filing_date"))
                records.append(
                    {
                        "statement": statement_name,
                        "period": period,
                        "fiscal_date_obj": fiscal_date,
                        "filing_date_obj": filing_date,
                        "fiscal_date": fiscal_date.isoformat() if fiscal_date else None,
                        "filing_date": filing_date.isoformat() if filing_date else None,
                        "data": payload,
                    }
                )
    return sorted(
        records,
        key=lambda item: (
            item["filing_date_obj"] or date.min,
            item["fiscal_date_obj"] or date.min,
            item["statement"],
            item["period"],
        ),
    )


def load_symbol_financials(symbol: str) -> dict[str, Any]:
    path = RAW_DIR / symbol / "fundamentals.json"
    fundamentals = read_json(path)
    if not isinstance(fundamentals, dict):
        return {
            "has_fundamentals_file": False,
            "fundamentals_path": f"raw/{symbol}/fundamentals.json",
            "records": [],
        }
    general = fundamentals.get("General", {})
    return {
        "has_fundamentals_file": True,
        "fundamentals_path": f"raw/{symbol}/fundamentals.json",
        "general": json_ready(general) if isinstance(general, dict) else {},
        "records": extract_statement_records(fundamentals),
    }


def latest_by_statement(records: list[dict[str, Any]], period: str, pub_date: date) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        filing_date = record["filing_date_obj"]
        if record["period"] != period or filing_date is None or filing_date > pub_date:
            continue
        statement = record["statement"]
        current = latest.get(statement)
        if current is None or (
            filing_date,
            record["fiscal_date_obj"] or date.min,
        ) > (
            current["filing_date_obj"],
            current["fiscal_date_obj"] or date.min,
        ):
            latest[statement] = record
    return [compact_record(latest[key]) for key in sorted(latest)]


def window_records(
    records: list[dict[str, Any]],
    start: date,
    end: date,
    *,
    include_start: bool,
    include_end: bool,
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        filing_date = record["filing_date_obj"]
        if filing_date is None:
            continue
        left_ok = filing_date >= start if include_start else filing_date > start
        right_ok = filing_date <= end if include_end else filing_date < end
        if left_ok and right_ok:
            output.append(compact_record(record))
    return output


def full_pitch_text(description: str, catalyst: str) -> str:
    parts = []
    if description:
        parts.append("Description:\n" + description)
    if catalyst:
        parts.append("Catalyst:\n" + catalyst)
    return "\n\n".join(parts)


def make_row(pitch: dict[str, Any], symbol_cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    symbol = str(pitch.get("eodhd_symbol") or "").strip()
    pub_date = parse_date(pitch.get("publication_date"))
    symbol_data = symbol_cache.setdefault(symbol, load_symbol_financials(symbol)) if symbol else {
        "has_fundamentals_file": False,
        "fundamentals_path": None,
        "records": [],
    }
    records = symbol_data["records"]

    trailing_5y = []
    forward_3y = []
    forward_5y = []
    latest_annual = []
    latest_quarterly = []
    if pub_date:
        trailing_5y = window_records(
            records,
            pub_date - timedelta(days=365 * 5),
            pub_date,
            include_start=True,
            include_end=True,
        )
        forward_3y = window_records(
            records,
            pub_date,
            pub_date + timedelta(days=365 * 3),
            include_start=False,
            include_end=True,
        )
        forward_5y = window_records(
            records,
            pub_date,
            pub_date + timedelta(days=365 * 5),
            include_start=False,
            include_end=True,
        )
        latest_annual = latest_by_statement(records, "yearly", pub_date)
        latest_quarterly = latest_by_statement(records, "quarterly", pub_date)

    description = str(pitch.get("description") or "")
    catalyst = str(pitch.get("catalyst") or "")
    row = {
        "idea_id": pitch.get("idea_id"),
        "raw_symbol": pitch.get("raw_symbol"),
        "eodhd_symbol": symbol,
        "company_name": pitch.get("company_name"),
        "publication_date": pitch.get("publication_date"),
        "is_short": pitch.get("is_short"),
        "is_contest_winner": pitch.get("is_contest_winner"),
        "link": pitch.get("link"),
        "author_user_id": pitch.get("author_user_id"),
        "pitch_description_full": description,
        "pitch_catalyst_full": catalyst,
        "full_stock_pitch_text": full_pitch_text(description, catalyst),
        "performance": {
            "raw_perf_1y": pitch.get("raw_perf_1y"),
            "directional_perf_1y": pitch.get("directional_perf_1y"),
            "outcome_1y": pitch.get("outcome_1y"),
            "raw_perf_3y": pitch.get("raw_perf_3y"),
            "directional_perf_3y": pitch.get("directional_perf_3y"),
            "outcome_3y": pitch.get("outcome_3y"),
            "raw_perf_5y": pitch.get("raw_perf_5y"),
            "directional_perf_5y": pitch.get("directional_perf_5y"),
            "outcome_5y": pitch.get("outcome_5y"),
            "raw_perf_10y": pitch.get("raw_perf_10y"),
            "directional_perf_10y": pitch.get("directional_perf_10y"),
            "outcome_10y": pitch.get("outcome_10y"),
            "raw_perf_20y": pitch.get("raw_perf_20y"),
            "directional_perf_20y": pitch.get("directional_perf_20y"),
            "outcome_20y": pitch.get("outcome_20y"),
            "primary_horizon": pitch.get("primary_horizon"),
            "primary_outcome": pitch.get("primary_outcome"),
        },
        "raw_perf_3y": pitch.get("raw_perf_3y"),
        "outcome_3y": pitch.get("outcome_3y"),
        "raw_perf_5y": pitch.get("raw_perf_5y"),
        "outcome_5y": pitch.get("outcome_5y"),
        "fundamentals_snapshot_from_original_dataset": {
            "sector": pitch.get("fundamentals_sector"),
            "industry": pitch.get("fundamentals_industry"),
            "market_cap": pitch.get("fundamentals_market_cap"),
            "revenue_ttm": pitch.get("fundamentals_revenue_ttm"),
            "profit_margin": pitch.get("fundamentals_profit_margin"),
        },
        "eodhd_general": symbol_data.get("general", {}),
        "raw_fundamentals_path": symbol_data.get("fundamentals_path"),
        "financials_trailing_5y_asof_pitch": trailing_5y,
        "financials_latest_annual_asof_pitch": latest_annual,
        "financials_latest_quarterly_asof_pitch": latest_quarterly,
        "financials_forward_3y_after_pitch": forward_3y,
        "financials_forward_5y_after_pitch": forward_5y,
        "financial_context_counts": {
            "all_statement_records_available": len(records),
            "trailing_5y_asof_pitch_records": len(trailing_5y),
            "latest_annual_asof_pitch_records": len(latest_annual),
            "latest_quarterly_asof_pitch_records": len(latest_quarterly),
            "forward_3y_after_pitch_records": len(forward_3y),
            "forward_5y_after_pitch_records": len(forward_5y),
            "has_point_in_time_financials": bool(trailing_5y or latest_annual or latest_quarterly),
            "has_forward_3y_financials": bool(forward_3y),
            "has_forward_5y_financials": bool(forward_5y),
        },
        "leakage_note": (
            "Use only financials_trailing_5y_asof_pitch, "
            "financials_latest_annual_asof_pitch, and "
            "financials_latest_quarterly_asof_pitch for prediction inputs. "
            "The forward_3y/forward_5y columns are post-pitch outcome/context "
            "windows and should not be used as prediction features."
        ),
    }
    return row


def write_readme(summary: dict[str, Any]) -> None:
    OUT_README.write_text(
        f"""---
license: other
pretty_name: VIC Pitch Financial Context EODHD
task_categories:
- text-classification
- tabular-regression
language:
- en
tags:
- finance
- eodhd
- value-investing
- financial-statements
- stock-pitches
private-dataset: true
---

# VIC Pitch Financial Context EODHD

Private pitch-level dataset joining VIC stock pitches to EODHD financial
statement records.

## Files

- `vic_pitch_financial_context.jsonl`: full dataset, one JSON object per pitch.
- `vic_pitch_financial_context_preview.csv`: compact preview with counts and
  performance fields.
- `dataset_summary.json`: row counts and coverage summary.

## Main Columns

- `full_stock_pitch_text`: full pitch text from the source dataset, preserving
  the full description and catalyst text.
- `performance`: 1y, 3y, 5y, 10y, and 20y raw/directional performance and
  outcome labels where available.
- `financials_trailing_5y_asof_pitch`: full EODHD statement records with
  `filing_date` from five years before the pitch through the publication date.
- `financials_latest_annual_asof_pitch`: latest annual balance sheet, cash flow,
  and income statement filed on or before the pitch.
- `financials_latest_quarterly_asof_pitch`: latest quarterly balance sheet, cash
  flow, and income statement filed on or before the pitch.
- `financials_forward_3y_after_pitch`: statement records filed after the pitch
  through three years after the pitch.
- `financials_forward_5y_after_pitch`: statement records filed after the pitch
  through five years after the pitch.

## Leakage Note

For prediction features, use only the `asof_pitch` columns. The forward columns
are included for outcome analysis and should not be used as model input features.

## Summary

```json
{json.dumps(summary, indent=2, sort_keys=True)}
```
""",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pitches = pd.read_csv(CANONICAL_CSV, dtype=str, keep_default_na=False, usecols=PITCH_COLUMNS)
    symbol_cache: dict[str, dict[str, Any]] = {}
    rows = []
    preview_rows = []

    with OUT_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for pitch in pitches.to_dict(orient="records"):
            row = make_row(pitch, symbol_cache)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts = row["financial_context_counts"]
            preview_rows.append(
                {
                    "idea_id": row["idea_id"],
                    "eodhd_symbol": row["eodhd_symbol"],
                    "company_name": row["company_name"],
                    "publication_date": row["publication_date"],
                    "raw_perf_3y": row["raw_perf_3y"],
                    "outcome_3y": row["outcome_3y"],
                    "raw_perf_5y": row["raw_perf_5y"],
                    "outcome_5y": row["outcome_5y"],
                    "pitch_text_chars": len(row["full_stock_pitch_text"]),
                    "has_point_in_time_financials": counts["has_point_in_time_financials"],
                    "trailing_5y_asof_pitch_records": counts["trailing_5y_asof_pitch_records"],
                    "latest_annual_asof_pitch_records": counts["latest_annual_asof_pitch_records"],
                    "latest_quarterly_asof_pitch_records": counts["latest_quarterly_asof_pitch_records"],
                    "forward_3y_after_pitch_records": counts["forward_3y_after_pitch_records"],
                    "forward_5y_after_pitch_records": counts["forward_5y_after_pitch_records"],
                    "link": row["link"],
                    "raw_fundamentals_path": row["raw_fundamentals_path"],
                }
            )
            rows.append(row)

    preview = pd.DataFrame(preview_rows)
    preview.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "rows": len(rows),
        "unique_symbols": len(symbol_cache),
        "jsonl_path": str(OUT_JSONL),
        "preview_csv_path": str(OUT_CSV),
        "jsonl_bytes": OUT_JSONL.stat().st_size,
        "preview_csv_bytes": OUT_CSV.stat().st_size,
        "rows_with_point_in_time_financials": int(preview["has_point_in_time_financials"].sum()),
        "rows_with_forward_3y_financials": int((preview["forward_3y_after_pitch_records"] > 0).sum()),
        "rows_with_forward_5y_financials": int((preview["forward_5y_after_pitch_records"] > 0).sum()),
        "rows_with_3y_performance": int((preview["raw_perf_3y"] != "").sum()),
        "rows_with_5y_performance": int((preview["raw_perf_5y"] != "").sum()),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_readme(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
