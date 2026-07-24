from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CANONICAL_CSV = ROOT / "data" / "processed" / "investment_canonical.csv"
RAW_DIR = ROOT / "eodhd_output" / "dataset_financial_pull" / "raw"
OUT_DIR = ROOT / "eodhd_output" / "financial_statement_date_repair_audit"
OUT_CSV = OUT_DIR / "financial_statement_date_repair_rows.csv"
OUT_JSON = OUT_DIR / "financial_statement_date_repair_summary.json"

STATEMENTS = {
    "Balance_Sheet": "balance_sheet",
    "Cash_Flow": "cash_flow",
    "Income_Statement": "income_statement",
}
PERIODS = ("yearly", "quarterly")
FALLBACK_LAGS = {
    "quarterly": 60,
    "yearly": 120,
}


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


def extract_records(symbol: str) -> list[dict[str, Any]]:
    path = RAW_DIR / symbol / "fundamentals.json"
    if not path.exists():
        return []
    try:
        fundamentals = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    financials = fundamentals.get("Financials", {})
    if not isinstance(financials, dict):
        return []

    records: list[dict[str, Any]] = []
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
                fallback_date = fiscal_date + timedelta(days=FALLBACK_LAGS[period]) if fiscal_date else None
                effective_date = filing_date or fallback_date
                if not fiscal_date and not effective_date:
                    continue
                records.append(
                    {
                        "statement": statement_name,
                        "period": period,
                        "fiscal_date": fiscal_date,
                        "filing_date": filing_date,
                        "effective_date": effective_date,
                        "date_source": "filing_date" if filing_date else "estimated_from_fiscal_date",
                    }
                )
    return records


def latest_by_statement(records: list[dict[str, Any]], period: str, pub_date: date) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        effective_date = record["effective_date"]
        if record["period"] != period or effective_date is None or effective_date > pub_date:
            continue
        statement = record["statement"]
        current = latest.get(statement)
        current_key = (current["effective_date"], current["fiscal_date"] or date.min) if current else None
        candidate_key = (effective_date, record["fiscal_date"] or date.min)
        if current_key is None or candidate_key > current_key:
            latest[statement] = record
    return latest


def window_records(records: list[dict[str, Any]], start: date, end: date) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["effective_date"] is not None and start <= record["effective_date"] <= end
    ]


def has_estimated(records: list[dict[str, Any]] | dict[str, dict[str, Any]]) -> bool:
    values = records.values() if isinstance(records, dict) else records
    return any(record["date_source"] == "estimated_from_fiscal_date" for record in values)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbol_cache: dict[str, list[dict[str, Any]]] = {}
    rows = []
    with CANONICAL_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for pitch in reader:
            symbol = str(pitch.get("eodhd_symbol") or "").strip()
            pub_date = parse_date(pitch.get("publication_date"))
            records = symbol_cache.setdefault(symbol, extract_records(symbol)) if symbol else []
            trailing = []
            latest_annual = {}
            latest_quarterly = {}
            if pub_date:
                trailing = window_records(records, pub_date - timedelta(days=365 * 5), pub_date)
                latest_annual = latest_by_statement(records, "yearly", pub_date)
                latest_quarterly = latest_by_statement(records, "quarterly", pub_date)
            model_ready = bool(trailing) and len(latest_annual) >= 3 and len(latest_quarterly) >= 3
            rows.append(
                {
                    "idea_id": pitch.get("idea_id"),
                    "eodhd_symbol": symbol,
                    "company_name": pitch.get("company_name"),
                    "publication_date": pitch.get("publication_date"),
                    "statement_records": len(records),
                    "trailing_5y_records_repaired": len(trailing),
                    "latest_annual_records_repaired": len(latest_annual),
                    "latest_quarterly_records_repaired": len(latest_quarterly),
                    "model_ready_financials_repaired": model_ready,
                    "uses_estimated_dates": has_estimated(trailing) or has_estimated(latest_annual) or has_estimated(latest_quarterly),
                    "estimated_latest_annual_count": sum(
                        1 for item in latest_annual.values() if item["date_source"] == "estimated_from_fiscal_date"
                    ),
                    "estimated_latest_quarterly_count": sum(
                        1 for item in latest_quarterly.values() if item["date_source"] == "estimated_from_fiscal_date"
                    ),
                    "estimated_trailing_count": sum(
                        1 for item in trailing if item["date_source"] == "estimated_from_fiscal_date"
                    ),
                }
            )

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": len(rows),
        "audit_csv": str(OUT_CSV),
        "fallback_lags_days": FALLBACK_LAGS,
        "rows_with_statement_records": sum(1 for row in rows if row["statement_records"] > 0),
        "rows_with_repaired_model_ready_financials": sum(1 for row in rows if row["model_ready_financials_repaired"]),
        "rows_using_estimated_dates": sum(1 for row in rows if row["uses_estimated_dates"]),
        "note": (
            "This is a repair audit only. Missing filing dates are estimated from fiscal dates "
            "using conservative period lags and must be labeled if used in a dataset."
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
